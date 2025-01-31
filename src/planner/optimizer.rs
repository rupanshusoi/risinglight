// Copyright 2024 RisingLight Project Authors. Licensed under Apache-2.0.

use std::sync::{LazyLock, Mutex};

use egg::CostFunction;
use lazy_static::lazy_static;

use super::*;
use crate::catalog::RootCatalogRef;

lazy_static! {
    static ref INC_EGRAPH: Mutex<Option<EGraph>> = Mutex::new(Some(EGraph::default()));
}

/// Plan optimizer.
#[derive(Clone)]
pub struct Optimizer {
    analysis: ExprAnalysis,
}

/// Optimizer configurations.
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub enable_range_filter_scan: bool,
    pub table_is_sorted_by_primary_key: bool,
}

impl Optimizer {
    /// Creates a new optimizer.
    pub fn new(catalog: RootCatalogRef, stat: Statistics, config: Config) -> Self {
        Self {
            analysis: ExprAnalysis {
                catalog,
                config,
                stat,
            },
        }
    }

    /// Optimize the given expression.
    pub fn optimize(&self, mut expr: RecExpr) -> RecExpr {
        let mut cost = f32::MAX;

        // define extra rules for some configurations
        let mut extra_rules = vec![];
        if self.analysis.config.enable_range_filter_scan {
            extra_rules.append(&mut rules::range::filter_scan_rule());
        }

        // 1. pushdown apply
        self.optimize_stage(&mut expr, &mut cost, STAGE1_RULES.iter(), 2, 6);
        // 2. pushdown predicate and projection
        let rules = STAGE2_RULES.iter().chain(&extra_rules);
        self.optimize_stage(&mut expr, &mut cost, rules, 4, 6);
        // 3. join reorder and hashjoin
        self.optimize_stage(&mut expr, &mut cost, STAGE3_RULES.iter(), 3, 8);
        expr
    }

    /// Optimize the expression with the given rules in multiple iterations.
    /// In each iteration, the best expression is selected as the input of the next iteration.
    fn optimize_stage<'a>(
        &self,
        expr: &mut RecExpr,
        cost: &mut f32,
        rules: impl IntoIterator<Item = &'a Rewrite> + Clone,
        iteration: usize,
        iter_limit: usize,
    ) {
        for _ in 0..iteration {
            let mut guard = INC_EGRAPH.try_lock().unwrap();
            let mut inc_egraph = guard.take().unwrap();
            inc_egraph.inc_version();

            let mut runner = egg::Runner::<_, _, ()>::new(self.analysis.clone())
                .with_egraph(inc_egraph)
                .with_expr(expr)
                .with_iter_limit(iter_limit)
                .run(rules.clone());
            eprintln!("Here");
            println!("{}", runner.report());
            runner.egraph.rebuild();
            let cost_fn = cost::CostFn {
                egraph: &runner.egraph,
            };
            let extractor = egg::Extractor::new(&runner.egraph, cost_fn);
            let cost0;
            (cost0, *expr) = extractor.find_best(runner.roots[0]);
            *guard = Some(runner.egraph);
            if cost0 >= *cost {
                break;
            }
            *cost = cost0;
        }
    }

    /// Returns the cost for each node in the expression.
    pub fn costs(&self, expr: &RecExpr) -> Vec<f32> {
        let mut egraph = EGraph::new(self.analysis.clone());
        // NOTE: we assume Expr node has the same Id in both EGraph and RecExpr.
        egraph.add_expr(expr);
        let mut cost_fn = cost::CostFn { egraph: &egraph };
        let mut costs = vec![0.0; expr.as_ref().len()];
        for (i, node) in expr.as_ref().iter().enumerate() {
            let cost = cost_fn.cost(node, |i| costs[usize::from(i)]);
            costs[i] = cost;
        }
        costs
    }

    /// Returns the estimated row for each node in the expression.
    pub fn rows(&self, expr: &RecExpr) -> Vec<f32> {
        let mut egraph = EGraph::new(self.analysis.clone());
        // NOTE: we assume Expr node has the same Id in both EGraph and RecExpr.
        egraph.add_expr(expr);
        (0..expr.as_ref().len())
            .map(|i| egraph[i.into()].data.rows)
            .collect()
    }

    /// Returns the catalog.
    pub fn catalog(&self) -> &RootCatalogRef {
        &self.analysis.catalog
    }
}

/// Stage1 rules in the optimizer.
/// - pushdown apply and turn into join
static STAGE1_RULES: LazyLock<Vec<Rewrite>> = LazyLock::new(|| {
    let mut rules = vec![];
    rules.append(&mut rules::expr::and_rules());
    rules.append(&mut rules::plan::always_better_rules());
    rules.append(&mut rules::plan::subquery_rules());
    rules
});

/// Stage2 rules in the optimizer.
/// - pushdown predicate and projection
static STAGE2_RULES: LazyLock<Vec<Rewrite>> = LazyLock::new(|| {
    let mut rules = vec![];
    rules.append(&mut rules::expr::rules());
    rules.append(&mut rules::plan::always_better_rules());
    rules.append(&mut rules::plan::predicate_pushdown_rules());
    rules.append(&mut rules::plan::projection_pushdown_rules());
    rules
});

/// Stage3 rules in the optimizer.
/// - join reorder and hashjoin
static STAGE3_RULES: LazyLock<Vec<Rewrite>> = LazyLock::new(|| {
    let mut rules = vec![];
    rules.append(&mut rules::expr::and_rules());
    rules.append(&mut rules::plan::always_better_rules());
    rules.append(&mut rules::plan::join_reorder_rules());
    rules.append(&mut rules::plan::hash_join_rules());
    rules.append(&mut rules::plan::predicate_pushdown_rules());
    rules.append(&mut rules::plan::projection_pushdown_rules());
    rules.append(&mut rules::order::order_rules());
    rules
});
