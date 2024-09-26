use crate::goal_graph::{GoalGraph, GoalInfo, GraphProveStatus};
use colored::Colorize;
use egg::*;
use itertools::Itertools;
use log::warn;

use core::panic;
use std::collections::{BTreeMap, VecDeque};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Display;
use std::iter::zip;
use std::str::FromStr;
use std::time::{Duration, Instant};
use symbolic_expressions::{parser, Sexp};

use crate::analysis::{
  cvecs_equal, print_cvec, CanonicalForm, CanonicalFormAnalysis, CycleggAnalysis,
};
use crate::ast::*;
use crate::config::*;
use crate::egraph::*;
use crate::utils::*;

// We will use SymbolLang for now
pub type Eg = EGraph<SymbolLang, CycleggAnalysis>;
pub type Rw = Rewrite<SymbolLang, CycleggAnalysis>;
pub type CvecRw = Rewrite<SymbolLang, ()>;
type IH = (
  ConditionalSearcher<SoundnessWithType, Pattern<SymbolLang>>,
  ConditionalSearcher<SoundnessWithType, Pattern<SymbolLang>>,
);

/// A special scrutinee name used to signal that case split bound has been exceeded
pub const LEMMA_PREFIX: &str = "lemma";
pub const CC_LEMMA_PREFIX: &str = "cc-lemma";
pub const IH_EQUALITY_PREFIX: &str = "ih-equality-"; // TODO: remove

/// Condition that checks whether it is sound to apply a lemma
#[derive(Clone)]
pub struct Soundness {
  /// A substitution from lemma's free variables
  /// to the original e-classes these variables came from
  pub free_vars: IdSubst,
  /// All premises that must hold for this lemma to apply,
  /// expressed in terms of the free variables
  pub premises: Vec<ETermEquation>,
}

impl Soundness {
  /// Substitution as a string, for debugging purposes
  fn _pretty_subst(subst: &[(Symbol, Expr, Expr)]) -> String {
    let strings: Vec<String> = subst
      .iter()
      .map(|(x, orig, new)| {
        format!(
          "{} = {} -> {}",
          &x.to_string(),
          &orig.to_string(),
          &new.to_string()
        )
      })
      .collect();
    strings.join(", ")
  }

  /// Are the canonical forms of the e-classes in new_subst strictly smaller than those in orig_subst?
  /// For now implements a sound but incomplete measure,
  /// where all forms need to be no larger, and at least one has to be strictly smaller.
  fn smaller_tuple(
    &self,
    triples: &Vec<(Symbol, Expr, Expr)>,
    _blocking_vars: &BTreeSet<Symbol>,
  ) -> bool {
    let mut has_strictly_smaller = false;
    // // If all free vars are non-blocking, we can skip the soundness check
    // if triples.iter().all(|(x, _, _)| !blocking_vars.contains(x)) {
    //   return true;
    // }
    for (_, orig, new) in triples {
      match is_subterm(new, orig) {
        StructuralComparison::LT => {
          has_strictly_smaller = true;
        }
        StructuralComparison::Incomparable => {
          return false;
        }
        _ => {}
      }
    }
    has_strictly_smaller
  }

  /// Apply subst to self.premise (if any)
  /// and check whether the resulting terms are equal in the egraph
  fn check_premise(premise: &ETermEquation, triples: &[(Symbol, Expr, Expr)], egraph: &Eg) -> bool {
    // let info = SmallerVar::pretty_subst(triples);
    // println!("checking premise {} = {} for {}", premise.lhs.sexp, premise.rhs.sexp, info);

    // TODO: it's annoying having to convert everything to s-expressions and back
    // but substituting over RecExprs is too much of a pain
    // Convert triples to a substitution over s-expressions
    let subst: SSubst = triples
      .iter()
      .map(|(var, _, new_expr)| {
        (
          var.to_string(),
          // FIXME: we give an empty expression if the var is not blocking.
          // Right now, we just substitute the var for itself, but we should instead
          // find the correct expression to give.
          if new_expr.as_ref().is_empty() {
            Sexp::String(var.to_string())
          } else {
            symbolic_expressions::parser::parse_str(&new_expr.to_string()).unwrap()
          },
        )
      })
      .collect();

    // Perform the substitution
    let lhs: Expr = resolve_sexp(&premise.lhs.sexp, &subst)
      .to_string()
      .parse()
      .unwrap();
    let rhs: Expr = resolve_sexp(&premise.rhs.sexp, &subst)
      .to_string()
      .parse()
      .unwrap();

    // Lookup the sides of the new premise in the egraph;
    // they must be there, since we added them during grounding
    if let Some(lhs_id) = egraph.lookup_expr(&lhs) {
      if let Some(rhs_id) = egraph.lookup_expr(&rhs) {
        // println!("{} == {}", lhs_id, rhs_id);
        return lhs_id == rhs_id;
      }
    }
    // This cannot happen in uncyclic mode, because we have grounded all the premises,
    // but it can happen in cyclic mode
    // panic!("premise {:?} = {:?} not found in egraph", lhs, rhs);
    false
  }

  /// Check all of the premises of this condition
  fn check_premises(&self, triples: &[(Symbol, Expr, Expr)], egraph: &Eg) -> bool {
    self
      .premises
      .iter()
      .all(|premise| Soundness::check_premise(premise, triples, egraph))
  }
}

impl SearchCondition<SymbolLang, CycleggAnalysis> for Soundness {
  // FIXME: needs to be updated to accurately handle dealing with cases where
  // we can skip the soundness check on some variables because they are not blocking
  /// Returns true if the substitution is into a smaller tuple of variables
  fn check(&self, egraph: &Eg, _eclass: Id, subst: &Subst) -> bool {
    // Create an iterator over triples: (variable, old canonical form, new canonical form)

    let triples = self
      .free_vars
      .iter()
      .map(|(x, orig_id)| {
        // Exit early with something guaranteed to be LE if this var is not blocking
        if CONFIG.better_termination && !egraph.analysis.case_split_vars.contains(x) {
          // FIXME: we need to give the actual value here
          return Some((*x, vec![].into(), vec![].into()));
        }
        let v = to_wildcard(x);
        // Subst must have all lemma variables defined
        // because we did the filtering when creating SmallerVars
        // if CONFIG.verbose {
        //   println!("subst {:?}", subst);
        //   println!("var: {}", v);
        //   println!("eclass: {}", _eclass);
        //   dump_eclass_exprs(egraph, _eclass);
        // }
        let new_id = subst.get(v)?;
        // If the actual argument of the lemma is not canonical, give up
        let new_canonical = CanonicalFormAnalysis::extract_canonical(egraph, *new_id)?;
        // Same for the original argument:
        // it might not be canonical if it's inconsistent, in which case there's no point applying any lemmas
        let orig_canonical = CanonicalFormAnalysis::extract_canonical(egraph, *orig_id)?;
        Some((*x, orig_canonical, new_canonical))
      })
      .collect::<Option<Vec<(Symbol, Expr, Expr)>>>();

    match triples {
      None => false, // All actual arguments must be canonical in order to be comparable to the formals
      Some(triples) => {
        // Check that the actuals are smaller than the formals
        // and that the actual premise holds
        let terminates = self.smaller_tuple(&triples, &egraph.analysis.case_split_vars);
        // Let's not check the premises if the termination check doesn't hold:

        //println!("  {}", res);
        terminates && self.check_premises(&triples, egraph)
      }
    }
  }
}

#[derive(Clone)]
struct TypeRestriction {
  ty: Symbol,
  ctx: Context,
}

impl<'a> SearchCondition<SymbolLang, CycleggAnalysis> for TypeRestriction {
  fn check(
    &self,
    egraph: &EGraph<SymbolLang, CycleggAnalysis>,
    eclass: Id,
    _subst: &Subst,
  ) -> bool {
    let mut res = true;
    let op = egraph[eclass].nodes.first().unwrap().op;
    if let Some(op_type) = self.ctx.get(&op) {
      res = get_output_type_name(op_type) == self.ty
    } else if let Some(var_type) = egraph.analysis.local_ctx.get(&op) {
      res = get_output_type_name(var_type) == self.ty
    } else if op.to_string().starts_with("g_") {
      res = self.ty == BOOL_TYPE.parse().unwrap()
    } else {
      // println!("unknown type of variable {} {:?}", op, egraph.analysis.local_ctx);
    }
    // if CONFIG.verbose && !res {
    //   println!(
    //     "reject rewrite on {} {}",
    //     egraph[eclass].nodes.first().unwrap().op,
    //     self.ty
    //   );
    // }
    res
  }
}

#[derive(Clone)]
pub struct SoundnessWithType {
  soundness: Option<Soundness>,
  type_cons: Option<TypeRestriction>,
}

impl SearchCondition<SymbolLang, CycleggAnalysis> for SoundnessWithType {
  fn check(&self, egraph: &EGraph<SymbolLang, CycleggAnalysis>, eclass: Id, subst: &Subst) -> bool {
    (self.soundness.is_none()
      || self
        .soundness
        .as_ref()
        .unwrap()
        .check(egraph, eclass, subst))
      && (self.type_cons.is_none()
        || self
          .type_cons
          .as_ref()
          .unwrap()
          .check(egraph, eclass, subst))
  }
}

/// A term inside the egraph;
/// we store multiple representations because they are useful for different purposes.
#[derive(Debug, Clone)]
pub struct ETerm {
  /// Term as a symbolic expression
  pub sexp: Sexp,
  /// E-class id of the term in the egraph
  id: Id,
  /// Terms as egg's RecExpr
  pub expr: Expr,
}

impl ETerm {
  /// Create a new term from a symbolic expression
  /// and add it to the egraph
  fn new(sexp: &Sexp, egraph: &mut Eg) -> ETerm {
    let expr = sexp.to_string().parse().unwrap();
    egraph.add_expr(&expr);
    let id = egraph.lookup_expr(&expr).unwrap();
    Self {
      sexp: sexp.clone(),
      id,
      expr,
    }
  }

  fn new_from_expr(expr: &Expr, egraph: &mut Eg) -> ETerm {
    let sexp = parser::parse_str(&expr.to_string()).unwrap();
    egraph.add_expr(expr);
    let id = egraph.lookup_expr(expr).unwrap();
    Self {
      sexp,
      id,
      expr: expr.clone(),
    }
  }

  fn from_expr(expr: Expr, egraph: &Eg) -> Self {
    let id = egraph.lookup_expr(&expr).unwrap();
    let sexp = parser::parse_str(&expr.to_string()).unwrap();
    Self { sexp, id, expr }
  }

  /// Update variables in my expressions with their canonical forms
  fn update_variables(&self, subst: &IdSubst, egraph: &Eg) -> Self {
    let ssubst: SSubst = subst
      .iter()
      .map(|(x, id)| {
        let expr = CanonicalFormAnalysis::extract_canonical(egraph, *id).unwrap();
        (
          x.to_string(),
          symbolic_expressions::parser::parse_str(&expr.to_string()).unwrap(),
        )
      })
      .collect();
    let new_sexp = resolve_sexp(&self.sexp, &ssubst);
    let new_expr = new_sexp.to_string().parse().unwrap();
    Self {
      sexp: new_sexp,
      id: egraph.lookup_expr(&new_expr).unwrap(),
      expr: new_expr,
    }
  }
}

impl Display for ETerm {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.sexp)
  }
}

/// As opposed to the Equation in ast.rs, an ETermEquation additionally records
/// an e-class id for the LHS and RHS.
#[derive(Debug, Clone)]
pub struct ETermEquation {
  pub lhs: ETerm,
  pub rhs: ETerm,
}

impl Display for ETermEquation {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{} =?= {}", self.lhs.sexp, self.rhs.sexp)
  }
}

fn find_generalizations_prop(
  prop: &Prop,
  global_context: &Context,
  local_context: &Context,
  renamed_params: &BTreeMap<String, String>,
  fresh_name: String,
) -> Vec<Prop> {
  // println!("Prop: {}", prop);
  let lhs_nontrivial_subexprs = nontrivial_sexp_subexpressions_containing_vars(&prop.eq.lhs);
  let rhs_nontrivial_subexprs = nontrivial_sexp_subexpressions_containing_vars(&prop.eq.rhs);
  let mut output = vec![];
  // println!(
  //   "lhs_nontrivial_subexprs: {:#?}",
  //   lhs_nontrivial_subexprs.keys()
  // );
  // println!(
  //   "rhs_nontrivial_subexprs: {:#?}",
  //   rhs_nontrivial_subexprs.keys()
  // );
  for (rhs_subexpr_str, subexpr) in &rhs_nontrivial_subexprs {
    // should be the same subexpr so we don't need to bind it
    if lhs_nontrivial_subexprs.get(rhs_subexpr_str).is_some() {
      // println!("Generalizing: {}", rhs_subexpr_str);
      let op = match subexpr {
        Sexp::Empty => unreachable!(),
        // This shouldn't happen unless we generalize a constant
        Sexp::String(s) => s,
        Sexp::List(list) => list.first().unwrap().string().unwrap(),
      };
      // HACK: Skip partial applications because they have no type
      if op == "$" {
        continue;
      }
      // println!("Local context: {:#?}", local_context);
      // println!("Renamed params: {:#?}", renamed_params);
      let op_ty = &global_context
        .get(&Symbol::new(op))
        .or_else(|| local_context.get(&Symbol::new(&renamed_params[op])))
        .unwrap();
      // Again, we assume that the expression here is fully applied, i.e. it is not a $
      let (_, ty) = op_ty.args_ret();
      let var_symb = Symbol::new(&fresh_name);
      let generalized_var = Sexp::String(fresh_name.clone());
      if CONFIG.subset_generalization {
        let new_lhs_sexps = substitute_sexp_subsets(&prop.eq.lhs, subexpr, &generalized_var);
        let new_rhs_sexps = substitute_sexp_subsets(&prop.eq.rhs, subexpr, &generalized_var);
        // println!(
        //   "New lhs sexps: {:#?}",
        //   new_lhs_sexps
        //     .iter()
        //     .map(|s| s.to_string())
        //     .collect::<Vec<_>>()
        // );
        // println!(
        //   "New rhs sexps: {:#?}",
        //   new_rhs_sexps
        //     .iter()
        //     .map(|s| s.to_string())
        //     .collect::<Vec<_>>()
        // );
        // TODO: Narrow down
        for (new_lhs, new_rhs) in new_lhs_sexps.into_iter().zip(new_rhs_sexps) {
          // FIXME: hacky way to find variables
          let lhs_vars = sexp_leaves(&new_lhs);
          let rhs_vars = sexp_leaves(&new_rhs);
          let mut new_params = prop.params.clone();
          // Only keep the vars that remain after substituting.
          new_params.retain(|(var, _)| {
            lhs_vars.contains(&var.to_string()) || rhs_vars.contains(&var.to_string())
          });
          new_params.push((var_symb, ty.clone()));
          // println!("Generalization candidate: {} = {}", new_lhs, new_rhs);
          output.push(Prop::new(Equation::new(new_lhs, new_rhs), new_params).0);
        }
      } else {
        let new_lhs = substitute_sexp(&prop.eq.lhs, subexpr, &generalized_var);
        let new_rhs = substitute_sexp(&prop.eq.rhs, subexpr, &generalized_var);
        // FIXME: hacky way to find variables
        let lhs_vars = sexp_leaves(&new_lhs);
        let rhs_vars = sexp_leaves(&new_rhs);
        let mut new_params = prop.params.clone();
        // Only keep the vars that remain after substituting.
        new_params.retain(|(var, _)| {
          lhs_vars.contains(&var.to_string()) || rhs_vars.contains(&var.to_string())
        });
        new_params.push((var_symb, ty));
        // println!("Generalization candidate: {} = {}", new_lhs, new_rhs);
        output.push(Prop::new(Equation::new(new_lhs, new_rhs), new_params).0);
      }
    }
  }
  output
}

impl ETermEquation {
  /// Add both sides of a raw equation to the egraph,
  /// producing an equation;
  /// if assume is true, also union the the two sides
  fn new(eq: &Equation, egraph: &mut Eg, assume: bool) -> Self {
    let lhs = ETerm::new(&eq.lhs, egraph);
    let rhs = ETerm::new(&eq.rhs, egraph);
    if assume {
      // Assume the premise
      egraph.union_trusted(lhs.id, rhs.id, format!("premise {}={}", lhs.sexp, rhs.sexp));
      egraph.rebuild();
    }
    Self { lhs, rhs }
  }

  /// Update variables in my expressions with their canonical forms
  fn update_variables(&self, subst: &IdSubst, egraph: &Eg) -> Self {
    Self {
      lhs: self.lhs.update_variables(subst, egraph),
      rhs: self.rhs.update_variables(subst, egraph),
    }
  }
}

/// When we make a new lemma and rewrites out of it, this tracks the the
/// rewrites we made as well as the information about the lemma.
#[derive(Clone)]
pub struct LemmaRewrite<A> {
  pub lhs_to_rhs: Option<(String, Rewrite<SymbolLang, A>)>,
  pub rhs_to_lhs: Option<(String, Rewrite<SymbolLang, A>)>,
  pub lemma_number: usize,
  pub lemma_prop: Prop,
  pub renamed_params: BTreeMap<String, String>,
}

impl<A: Analysis<SymbolLang> + Clone> LemmaRewrite<A> {
  pub fn new(
    lhs_to_rhs: Option<(String, Rewrite<SymbolLang, A>)>,
    rhs_to_lhs: Option<(String, Rewrite<SymbolLang, A>)>,
    lemma_number: usize,
    lemma_prop: Prop,
  ) -> Self {
    Self {
      lhs_to_rhs,
      rhs_to_lhs,
      lemma_number,
      lemma_prop,
      renamed_params: BTreeMap::new(),
    }
  }

  pub fn lemma_name(&self) -> String {
    format!("lemma_{}", self.lemma_number)
  }

  pub fn names_and_rewrites(&self) -> Vec<(String, Rewrite<SymbolLang, A>)> {
    self
      .lhs_to_rhs
      .iter()
      .chain(self.rhs_to_lhs.iter())
      .cloned()
      .collect()
  }

  pub fn rewrites(&self) -> Vec<Rewrite<SymbolLang, A>> {
    self
      .names_and_rewrites()
      .into_iter()
      .map(|(_, rw)| rw)
      .collect()
  }

  pub fn add_to_rewrites(&self, rewrites: &mut BTreeMap<String, Rewrite<SymbolLang, A>>) {
    if let Some((name, rw)) = self.lhs_to_rhs.as_ref() {
      rewrites.entry(name.clone()).or_insert(rw.clone());
    }
    if let Some((name, rw)) = self.rhs_to_lhs.as_ref() {
      rewrites.entry(name.clone()).or_insert(rw.clone());
    }
  }
}

#[derive(Debug, Clone)]
enum ScrutineeType {
  Guard,
  Var,
}

#[derive(Debug, Clone)]
struct Scrutinee {
  pub name: Symbol,
  pub depth: usize,
  pub scrutinee_type: ScrutineeType,
}

impl Scrutinee {
  /// Creates a new var scrutinee
  pub fn new_var(name: Symbol, depth: usize) -> Self {
    Self {
      name,
      depth,
      scrutinee_type: ScrutineeType::Var,
    }
  }
  /// Creates a new guard scrutinee (a scrutinee that will split a conditional
  /// expression).
  ///
  /// This will have depth 0 since it will always be fresh.
  pub fn new_guard(name: Symbol) -> Self {
    Self {
      name,
      depth: 0,
      scrutinee_type: ScrutineeType::Guard,
    }
  }
}

/// These are all values that will not be modified throughout the course of
/// proving the goal which we will thread through new goals we create from it.
///
/// Contains things such as the global context or environment.
///
/// This is copyable because it only refers to shared references.
///
// TODO: Can we wrap most of this state in a lazy_static! block at initialization?
// - reductions and cvec_reductions might need to be cloned
//   * Or we could make the full set of reductions and cvec_reductions lazy_static!
//     and pass around a small set that corresponds to the keys which we can obtain
//     the rewrites from the global map. Cloning that shouldn't be too bad.
// - defns doesn't need to be threaded through in the first place.
// - searchers could be copied since it's for debugging
// - The rest should be safe to keep in a truly global block.
// TODO: This can probably be moved into the ProofState.
#[derive(Copy, Clone)]
pub struct GlobalSearchState<'a> {
  /// Environment
  pub env: &'a Env,
  /// Global context (i.e. constructors and top-level bindings)
  pub context: &'a Context,
  /// Rewrites are split into reductions (invertible rules) and lemmas
  /// (non-invertible rules). Lemmas may (and often will) change between goals,
  /// but reductions will always be the same.
  pub reductions: &'a Vec<Rw>,
  /// HACK: an identical copy to the reductions used for the cvec egraph.
  /// This is because of type system stuff.
  pub cvec_reductions: &'a Vec<CvecRw>,
  /// Definitions in a form amenable to proof emission
  pub defns: &'a Defns,
  /// Searchers for whether the LHS and RHS of some rewrite appears in our
  /// e-graph.
  pub searchers: &'a Vec<ConditionalSearcher<Pattern<SymbolLang>, Pattern<SymbolLang>>>,
}

impl<'a> GlobalSearchState<'a> {
  pub fn new(
    env: &'a Env,
    context: &'a Context,
    reductions: &'a Vec<Rw>,
    cvec_reductions: &'a Vec<CvecRw>,
    defns: &'a Defns,
    searchers: &'a Vec<ConditionalSearcher<Pattern<SymbolLang>, Pattern<SymbolLang>>>,
  ) -> Self {
    Self {
      env,
      context,
      reductions,
      cvec_reductions,
      defns,
      searchers,
    }
  }
}

fn rewrite_expr(expr: &Equation, name: &String, term: &Sexp) -> Equation {
  fn replace(exp: &Sexp, name: &String, term: &Sexp) -> Sexp {
    match exp {
      Sexp::String(s) => {
        if *s == *name {
          term.clone()
        } else {
          exp.clone()
        }
      }
      Sexp::List(subs) => Sexp::List(subs.iter().map(|sub| replace(sub, name, term)).collect()),
      Sexp::Empty => Sexp::Empty,
    }
  }

  Equation {
    lhs: replace(&expr.lhs, name, term),
    rhs: replace(&expr.rhs, name, term),
  }
}

fn get_top_symbol(expr: &Expr) -> Symbol {
  expr.as_ref().last().unwrap().op
}

fn get_output_type_name(expr: &Type) -> Symbol {
  let output_type = match &expr.repr {
    Sexp::List(tokens) if tokens.first().unwrap().to_string() == "->" => {
      tokens.last().unwrap().clone()
    }
    _ => expr.repr.clone(),
  };
  match output_type {
    Sexp::List(tokens) => tokens.first().unwrap().to_string().into(),
    Sexp::String(s) => s.into(),
    _ => panic!(),
  }
}

/// Proof goal
#[derive(Clone)]
pub struct Goal<'a> {
  /// Goal name
  pub name: String,
  /// Equivalences we already proved
  pub egraph: Eg,
  /// Rewrites are split into reductions (invertible rules) and lemmas (non-invertible rules)
  /// Rewrites are split into reductions (invertible rules) and lemmas
  /// (non-invertible rules). Reductions - being unchanging - live in
  /// global_search_state.
  lemmas: BTreeMap<String, Rw>,
  /// Mapping from all universally-quantified variables of the goal to their types
  /// (note this includes both current and old variables, which have been case-split away)
  pub local_context: Context,
  /// Mapping from all universally-quantified variables of the goal to the e-classes they are stored in
  /// (note this includes both current and old variables, which have been case-split away)
  pub var_classes: IdSubst,
  /// The top-level parameters to the goal
  pub top_level_params: Vec<Symbol>,
  /// Variables we can case-split
  /// (i.e. the subset of local_context that have datatype types)
  scrutinees: VecDeque<Scrutinee>,
  /// Variables that we have already case split
  pub case_split_vars: BTreeSet<Symbol>,
  // TODO: Check this out
  /// Instantiations of the induction hypothesis that are in the egraph
  grounding_instantiations: Vec<IdSubst>,
  /// The equation we are trying to prove
  pub eq: ETermEquation,
  /// If this is a conditional prop, the premises
  pub premises: Vec<ETermEquation>,
  /// Stores the expression each guard variable maps to
  guard_exprs: BTreeMap<String, Expr>,
  /// The global search state.
  pub global_search_state: GlobalSearchState<'a>,
  /// added by Ruyi: to get the size after case split
  pub full_expr: Equation,
  /// Induction hypothesis
  pub ih: Option<Vec<IH>>,
}

impl<'a> Goal<'a> {
  /// Create top-level goal
  pub fn top(
    name: &str,
    prop: &Prop,
    premise: &Option<Equation>,
    global_search_state: GlobalSearchState<'a>,
  ) -> Self {
    let mut egraph: Eg = EGraph::default().with_explanations_enabled();
    egraph.analysis.global_ctx = global_search_state.context.clone();
    let eq = ETermEquation::new(&prop.eq, &mut egraph, false);
    let premise = premise
      .as_ref()
      .map(|eq| ETermEquation::new(eq, &mut egraph, true));
    let var_classes = lookup_vars(&egraph, prop.params.iter().map(|(x, _)| x));

    let mut res = Self {
      name: name.to_string(),
      // The only instantiation we have so far is where the parameters map to themselves
      var_classes: var_classes.clone(),
      grounding_instantiations: vec![var_classes.clone()],
      egraph,
      lemmas: BTreeMap::new(),
      local_context: Context::new(),
      top_level_params: prop.params.iter().map(|(x, _)| *x).collect(),
      case_split_vars: BTreeSet::new(),
      guard_exprs: BTreeMap::new(),
      scrutinees: VecDeque::new(),
      eq,
      // Convert to a singleton list if the Option is Some, else the empty list
      premises: premise.into_iter().collect(),
      global_search_state,
      full_expr: prop.eq.clone(),
      ih: None,
    };
    // TODO: this could really also be a reference. Probably not necessary
    // for efficiency reason but yeah.
    res.egraph.analysis.cvec_analysis.reductions = global_search_state.cvec_reductions.clone();
    for (name, ty) in &prop.params {
      res.add_scrutinee(*name, ty, 0);
      res.local_context.insert(*name, ty.clone());
    }
    res.egraph.analysis.local_ctx = res.local_context.clone();
    res.build_cvecs();
    res
  }

  /// Construct cvecs for the goal's parameters. We need type information in order
  /// to construct these, so they cannot be created automatically.
  // FIXME: This currently does not work for any goal other than the top-level goal.
  // So we don't use it elsewhere.
  fn build_cvecs(&mut self) {
    // Update the timestamp so that we ensure the new cvecs are applied.
    self.egraph.analysis.cvec_analysis.current_timestamp += 1;
    // Annoyingly, we need to collect these values before we iterate over them
    // to avoid mutably borrowing self. I think it's worth it so that we can
    // factor out the add_cvec_for_class function which is used elsewhere.
    let var_tys: Vec<(Id, Type)> = self
      .top_level_params
      .iter()
      .map(|param| {
        let ty = self.local_context[param].clone();
        let var_id = self.var_classes[param];
        (var_id, ty)
      })
      .collect();
    for (var_id, ty) in var_tys {
      self.add_cvec_for_class(var_id, &ty);
    }
    self.egraph.analysis.cvec_analysis.saturate();
    self.egraph.rebuild();
  }

  /// Constructs a cvec for the class at id with type ty.
  ///
  /// It's important to update the current timestamp before calling this function.
  ///
  /// Returns whether it made a cvec (we don't make cvecs for arrow types
  /// because we don't know how to make arbitrary functions).
  fn add_cvec_for_class(&mut self, id: Id, ty: &Type) -> bool {
    if ty.is_arrow() {
      return false;
    }
    let cvec = self.egraph.analysis.cvec_analysis.make_cvec_for_type(
      ty,
      self.global_search_state.env,
      self.global_search_state.context,
    );
    let mut analysis = self.egraph[id].data.clone();
    analysis.timestamp = self.egraph.analysis.cvec_analysis.current_timestamp;
    analysis.cvec_data = cvec;
    self.egraph.set_analysis_data(id, analysis);
    true
  }

  pub fn cvecs_valid(&mut self) -> Option<bool> {
    self.egraph.analysis.cvec_analysis.saturate();
    let lhs_cvec = &self.egraph[self.eq.lhs.id].data.cvec_data;
    let rhs_cvec = &self.egraph[self.eq.rhs.id].data.cvec_data;
    // print_cvec(&self.egraph.analysis.cvec_analysis, lhs_cvec);
    // print_cvec(&self.egraph.analysis.cvec_analysis, rhs_cvec);
    cvecs_equal(&self.egraph.analysis.cvec_analysis, lhs_cvec, rhs_cvec)
  }

  /// Saturate the goal by applying all available rewrites
  pub fn saturate(&mut self, top_lemmas: &BTreeMap<String, Rw>) -> Eg {
    let mut rewrites = self
      .global_search_state
      .reductions
      .iter()
      .chain(top_lemmas.values())
      .collect::<Vec<_>>();
    if !CONFIG.ripple_mode {
      rewrites.extend(self.lemmas.values());
    }
    if CONFIG.verbose {
      println!("Saturate using lemmas: {:#?}", top_lemmas.keys());
    }
    let lhs_id = self.eq.lhs.id;
    let rhs_id = self.eq.rhs.id;
    let runner = Runner::default()
      .with_explanations_enabled()
      .with_egraph(self.egraph.to_owned())
      .with_hook(move |runner| {
        // Stop iteration if we have proven lhs == rhs
        if runner.egraph.find(lhs_id) == runner.egraph.find(rhs_id) {
          Err("Goal proven".to_string())
        } else {
          Ok(())
        }
      })
      .run(rewrites);
    // self.egraph = runner.egraph;
    runner.egraph
  }

  /// Look to see if we have proven the goal somehow. Note that this does not
  /// perform the actual proof search, it simply checks if the proof exists.
  pub fn find_proof(&mut self) -> Option<ProofLeaf> {
    let resolved_lhs_id = self.egraph.find(self.eq.lhs.id);
    let resolved_rhs_id = self.egraph.find(self.eq.rhs.id);
    if CONFIG.verbose {
      println!("Find proof of goal:");
      dump_eclass_exprs(&self.egraph, resolved_lhs_id);
      println!("=?=");
      dump_eclass_exprs(&self.egraph, resolved_rhs_id);
    }

    // Have we proven LHS == RHS?
    if resolved_lhs_id == resolved_rhs_id {
      if self.egraph.lookup_expr(&self.eq.lhs.expr).is_none()
        || self.egraph.lookup_expr(&self.eq.rhs.expr).is_none()
      {
        println!("goal: {}", self.name);
        panic!(
          "One of {} or {} was removed from the e-graph! We can't emit a proof",
          self.eq.lhs.expr, self.eq.rhs.expr
        );
      }
      if CONFIG.verbose {
        println!("Proof by reflexivity");
      }
      return Some(ProofLeaf::Refl(
        self
          .egraph
          .explain_equivalence(&self.eq.lhs.expr, &self.eq.rhs.expr),
      ));
    }

    if CONFIG.ripple_mode {
      if let Some(ihs) = &self.ih {
        for (lhs_ih, rhs_ih) in ihs {
          if CONFIG.verbose {
            println!("IH: {} == {}", lhs_ih.searcher, rhs_ih.searcher);
          }
          if let Some(lhs_matches) = lhs_ih.search_eclass(&self.egraph, resolved_lhs_id) {
            if let Some(rhs_matches) = rhs_ih.search_eclass(&self.egraph, resolved_rhs_id) {
              for (lhs_subst, rhs_subst) in lhs_matches
                .substs
                .iter()
                .cartesian_product(&rhs_matches.substs)
              {
                let common_vars_consistent = var_set(&rhs_ih.searcher)
                  .intersection(&var_set(&lhs_ih.searcher))
                  .into_iter()
                  .all(|&var| lhs_subst.get(var) == rhs_subst.get(var));
                if common_vars_consistent {
                  if CONFIG.verbose {
                    println!("Proof by strong fertilization");
                  }
                  return Some(ProofLeaf::StrongFertilization(None));
                }
              }
            }
          }
        }
      }
    }

    // Check if this case in unreachable (i.e. if there are any inconsistent
    // e-classes in the e-graph)

    // TODO: Right now we only look for contradictions using the canonical
    // form analysis. We currently don't generate contradictions from the
    // cvec analysis, but we should be able to. However, even if we find
    // a cvec contradiction, it isn't as easy to use in our proof.
    //
    // If we find a contradiction from the cvecs, we need to first find which
    // enodes the cvecs came from, then we need to explain why those nodes are
    // equal, then we need to provide the concrete values that cause them to
    // be unequal. This will probably require us to update the Cvec analysis
    // to track enodes, which is a little unfortunate.
    let inconsistent_exprs = self.egraph.classes().find_map(|eclass| {
      if let CanonicalForm::Inconsistent(n1, n2) = &eclass.data.canonical_form_data {
        // println!("Proof by contradiction {} != {}", n1, n2);

        // FIXME: these nodes might have been removed, we'll need to be
        // careful about how we generate this proof. Perhaps we can generate
        // the proof when we discover the contradiction, since we hopefully
        // will not have finished removing the e-node at this point.
        if self.egraph.lookup(n1.clone()).is_none() || self.egraph.lookup(n2.clone()).is_none() {
          println!(
            "One of {} or {} was removed from the e-graph! We can't emit a proof",
            n1, n2
          );
          None
        } else {
          // This is here only for the purpose of proof generation:
          let extractor = Extractor::new(&self.egraph, AstSize);
          let expr1 = extract_with_node(n1, &extractor);
          let expr2 = extract_with_node(n2, &extractor);
          if CONFIG.verbose {
            println!("{}: {} = {}", "UNREACHABLE".bright_red(), expr1, expr2);
          }
          Some((expr1, expr2))
        }
      } else {
        None
      }
    });
    if let Some((expr1, expr2)) = inconsistent_exprs {
      let explanation = self.egraph.explain_equivalence(&expr1, &expr2);
      Some(ProofLeaf::Contradiction(explanation))
    } else {
      None
    }
  }

  /// Check whether an expression is reducible using this goal's reductions
  /// NOTE: This is largely not necessary when we have destructive rewrites
  /// enabled (the default). This is why it is by default disabled.
  pub fn is_reducible(&self, expr: &Expr) -> bool {
    let mut local_graph: Eg = Default::default();
    local_graph.add_expr(expr);
    local_graph.rebuild();
    for reduction in self.global_search_state.reductions {
      if !reduction.search(&local_graph).is_empty() {
        return true;
      }
    }
    false
  }

  /// Creates rewrites from all of the expressions in lhs_id to all of the
  /// expressions in rhs_id.
  ///
  /// Returns a hashmap of the rewrites as well as a vector of LemmaRewrites
  /// describing each rewrite.
  ///
  /// The rewrites will have a termination (soundness) check if
  /// add_termination_check is true, otherwise they will not.
  ///
  /// The rewrites will each be named lemma_n.
  fn make_lemma_rewrites_from_all_exprs(
    &self,
    mut lhs_id: Id,
    mut rhs_id: Id,
    premises: Vec<ETermEquation>,
    timer: &Timer,
    lemmas_state: &mut LemmasState,
    add_termination_check: bool,
    exclude_wildcards: bool,
    canonicalize: bool,
  ) -> (BTreeMap<String, Rw>, Vec<LemmaRewrite<CycleggAnalysis>>) {
    if canonicalize {
      lhs_id = self.egraph.find(lhs_id);
      rhs_id = self.egraph.find(rhs_id);
    }
    let exprs = get_all_expressions(&self.egraph, vec![lhs_id, rhs_id]);
    let is_var = |v| self.local_context.contains_key(v);
    let mut rewrites = self.lemmas.clone();
    let mut lemma_rws = vec![];
    for lhs_expr in &exprs[&lhs_id] {
      let lhs: Pattern<SymbolLang> = to_pattern(lhs_expr, is_var);
      // TODO: Check whether is_reducible is needed to constrain lemmas/rewrites. Probably not
      if (CONFIG.irreducible_only && self.is_reducible(lhs_expr)) || has_guard_wildcards(&lhs) {
        continue;
      }
      for rhs_expr in &exprs[&rhs_id] {
        if timer.timeout() {
          return (rewrites, lemma_rws);
        }
        let lemma_number = lemmas_state.fresh_lemma();

        // println!("found {} {}", lhs_expr, rhs_expr);
        let lemma_rw_opt = if add_termination_check {
          self
            .make_lemma_rewrite(
              lhs_expr,
              rhs_expr,
              &premises,
              lemma_number,
              exclude_wildcards,
            )
            .0
        } else {
          self.make_lemma_rewrite_type_only(lhs_expr, rhs_expr, lemma_number, exclude_wildcards)
        };

        if let Some(lemma_rw) = lemma_rw_opt {
          // Add the rewrites (if lemma_rw is some at least one is guaranteed to exist).
          lemma_rw.add_to_rewrites(&mut rewrites);
          lemma_rws.push(lemma_rw);
        }
      }
    }
    (rewrites, lemma_rws)
  }

  fn get_expected_type(&self, lhs: &Expr, rhs: &Expr) -> Option<Symbol> {
    for operator in [get_top_symbol(lhs), get_top_symbol(rhs)] {
      if let Some(ty) = self.global_search_state.context.get(&operator) {
        let type_name = get_output_type_name(ty);
        if self.global_search_state.env.contains_key(&type_name) {
          return Some(type_name);
        }
      }
    }
    None
  }

  fn make_lemma_rewrite(
    &self,
    lhs_expr: &Expr,
    rhs_expr: &Expr,
    premises: &Vec<ETermEquation>,
    lemma_number: usize,
    exclude_wildcards: bool,
  ) -> (Option<LemmaRewrite<CycleggAnalysis>>, Option<IH>) {
    let is_var = |v| self.local_context.contains_key(v);

    // NOTE: (CK) Before we would not recreate the lhs from lhs_expr every time
    // we made a lemma rewrite since we did nested for loops
    //
    // for lhs_expr {
    //   let lhs = ...
    //   for rhs_expr {
    //   ...
    //
    // which meant we just needed to clone it.
    //
    // I don't think this is a huge hit to efficiency though. If we cared, we
    // could instead loop over all lhs and rhs exprs first and precompute their
    // patterns + figure out which ones we don't need to consider.
    let lhs: Pattern<SymbolLang> = to_pattern(lhs_expr, is_var);
    if (CONFIG.irreducible_only && self.is_reducible(lhs_expr))
      || (exclude_wildcards && has_guard_wildcards(&lhs))
    {
      return (None, None);
    }

    let rhs: Pattern<SymbolLang> = to_pattern(rhs_expr, is_var);
    if (CONFIG.irreducible_only && self.is_reducible(rhs_expr))
      || (exclude_wildcards && has_guard_wildcards(&rhs))
    {
      return (None, None);
    }

    let lhs_vars = var_set(&lhs);
    let rhs_vars = var_set(&rhs);
    // println!("lhs vars: {:?}", lhs_vars);
    // println!("rhs vars: {:?}", rhs_vars);
    let lemma_vars = lhs_vars.union(&rhs_vars).cloned().collect();
    // println!("trying to make lemma rewrite forall {:?}. {} = {}", lemma_vars, lhs, rhs);

    // If any of my premises contain variables that are not present in lhs or rhs,
    // skip because we don't know how to check such a premise
    if !premises.iter().all(|eq| {
      let premise_lhs_vars = var_set(&to_pattern(&eq.lhs.expr, is_var));
      let premise_rhs_vars = var_set(&to_pattern(&eq.rhs.expr, is_var));
      let premise_vars: BTreeSet<Var> =
        premise_lhs_vars.union(&premise_rhs_vars).cloned().collect();
      premise_vars.is_subset(&lemma_vars)
    }) {
      return (None, None);
    }

    // Pick out those variables that occur in the lemma
    let lemma_var_classes: IdSubst = self
      .var_classes
      .iter()
      .filter(|(x, _)| lemma_vars.contains(&to_wildcard(x)))
      .map(|(x, id)| (*x, *id))
      .collect();
    let params: Vec<(Symbol, Type)> = lemma_var_classes
      .keys()
      .map(|var| (*var, self.local_context.get(var).unwrap().clone()))
      .collect();

    let mut condition = SoundnessWithType {
      soundness: Some(Soundness {
        free_vars: lemma_var_classes,
        premises: premises.clone(),
      }),
      type_cons: None,
    };
    if let Some(ty) = self.get_expected_type(lhs_expr, rhs_expr) {
      condition.type_cons = Some(TypeRestriction {
        ty,
        ctx: self.global_search_state.context.clone(),
      })
    }

    let rewrite_eq = Equation::from_exprs(lhs_expr, rhs_expr);
    // println!("make lemma {} {}", rewrite_eq, params.iter().map(|(name, ty)| format!("{}[{}]", name, ty)).join(" "));
    let (lemma_prop, renamed_params) = Prop::new(rewrite_eq, params.clone());
    let mut lemma_rw = LemmaRewrite {
      lhs_to_rhs: None,
      rhs_to_lhs: None,
      lemma_number,
      lemma_prop,
      renamed_params,
    };
    let lemma_name = lemma_rw.lemma_name();
    if rhs_vars.is_subset(&lhs_vars) {
      // if rhs has no extra wildcards, create a lemma lhs => rhs
      let lhs_to_rhs = Goal::make_rewrite_with_type_condition(
        lhs.clone(),
        rhs.clone(),
        condition.clone(),
        lemma_name.clone(),
      );
      lemma_rw.lhs_to_rhs = Some(lhs_to_rhs);

      if CONFIG.single_rhs {
        return (Some(lemma_rw), None);
      };
    }
    if lhs_vars.is_subset(&rhs_vars) {
      // if lhs has no extra wildcards, create a lemma rhs => lhs;
      // NOTE: (CK) This below comment is no longer true when our termination check is more complicated.
      // in non-cyclic mode, a single direction of IH is always sufficient
      // (because grounding adds all instantiations we could possibly care about).
      let rhs_to_lhs = Goal::make_rewrite_with_type_condition(
        rhs.clone(),
        lhs.clone(),
        condition.clone(),
        lemma_name.clone(),
      );
      lemma_rw.rhs_to_lhs = Some(rhs_to_lhs);
    }
    let has_lemma_rw = lemma_rw.lhs_to_rhs.is_some() || lemma_rw.rhs_to_lhs.is_some();
    if !has_lemma_rw {
      warn!("cannot create a lemma from {} and {}", lhs, rhs);
      println!("cannot create a lemma from {} and {}", lhs, rhs);
      (None, None)
    } else {
      (
        Some(lemma_rw),
        Some((
          ConditionalSearcher {
            condition: condition.clone(),
            searcher: lhs,
          },
          ConditionalSearcher {
            condition,
            searcher: rhs,
          },
        )),
      )
    }
  }

  /// Creates a lemma rewrite that does not check for soundness before applying.
  ///
  /// TODO: There's a lot of code duplication here because the checked rewrite
  /// cannot be generic over the EGraph analysis.
  ///
  /// Also this should probably not live in Goal. Suggestion: take is_var as a
  /// generic parameter and throw this somewhere else.
  fn make_lemma_rewrite_type_only(
    &self,
    lhs_expr: &Expr,
    rhs_expr: &Expr,
    lemma_number: usize,
    exclude_wildcards: bool,
  ) -> Option<LemmaRewrite<CycleggAnalysis>> {
    let is_var = |v| self.local_context.contains_key(v);

    // NOTE: (CK) Before we would not recreate the lhs from lhs_expr every time
    // we made a lemma rewrite since we did nested for loops
    //
    // for lhs_expr {
    //   let lhs = ...
    //   for rhs_expr {
    //   ...
    //
    // which meant we just needed to clone it.
    //
    // I don't think this is a huge hit to efficiency though. If we cared, we
    // could instead loop over all lhs and rhs exprs first and precompute their
    // patterns + figure out which ones we don't need to consider.
    let lhs: Pattern<SymbolLang> = to_pattern(lhs_expr, is_var);
    if (CONFIG.irreducible_only && self.is_reducible(lhs_expr))
      || (exclude_wildcards && has_guard_wildcards(&lhs))
    {
      return None;
    }

    let rhs: Pattern<SymbolLang> = to_pattern(rhs_expr, is_var);
    if (CONFIG.irreducible_only && self.is_reducible(rhs_expr))
      || (exclude_wildcards && has_guard_wildcards(&rhs))
    {
      return None;
    }

    let lhs_vars = var_set(&lhs);
    let rhs_vars = var_set(&rhs);
    let lemma_vars: BTreeSet<Var> = lhs_vars.union(&rhs_vars).cloned().collect();

    // Pick out those variables that occur in the lemma
    let lemma_var_classes: IdSubst = self
      .var_classes
      .iter()
      .filter(|(x, _)| lemma_vars.contains(&to_wildcard(x)))
      .map(|(x, id)| (*x, *id))
      .collect();
    let params: Vec<(Symbol, Type)> = lemma_var_classes
      .keys()
      .map(|var| (*var, self.local_context.get(var).unwrap().clone()))
      .collect();

    let mut condition = SoundnessWithType {
      soundness: None,
      type_cons: None,
    };
    if let Some(ty) = self.get_expected_type(lhs_expr, rhs_expr) {
      condition.type_cons = Some(TypeRestriction {
        ty,
        ctx: self.global_search_state.context.clone(),
      })
    }

    let rewrite_eq = Equation::from_exprs(lhs_expr, rhs_expr);
    // println!("make lemma {} {}", rewrite_eq, params.iter().map(|(name, ty)| format!("{}[{}]", name, ty)).join(" "));
    let (lemma_prop, renamed_params) = Prop::new(rewrite_eq, params.clone());
    let mut lemma_rw = LemmaRewrite {
      lhs_to_rhs: None,
      rhs_to_lhs: None,
      lemma_number,
      lemma_prop,
      renamed_params,
    };
    let lemma_name = lemma_rw.lemma_name();
    if rhs_vars.is_subset(&lhs_vars) {
      // if rhs has no extra wildcards, create a lemma lhs => rhs
      let lhs_to_rhs = Goal::make_rewrite_with_type_condition(
        lhs.clone(),
        rhs.clone(),
        condition.clone(),
        lemma_name.clone(),
      );
      lemma_rw.lhs_to_rhs = Some(lhs_to_rhs);

      if CONFIG.single_rhs {
        return Some(lemma_rw);
      };
    }
    if lhs_vars.is_subset(&rhs_vars) {
      // if lhs has no extra wildcards, create a lemma rhs => lhs;
      // NOTE: (CK) This below comment is no longer true when our termination check is more complicated.
      // in non-cyclic mode, a single direction of IH is always sufficient
      // (because grounding adds all instantiations we could possibly care about).
      let rhs_to_lhs = Goal::make_rewrite_with_type_condition(
        rhs.clone(),
        lhs.clone(),
        condition.clone(),
        lemma_name.clone(),
      );
      lemma_rw.rhs_to_lhs = Some(rhs_to_lhs);
    }
    let has_lemma_rw = lemma_rw.lhs_to_rhs.is_some() || lemma_rw.rhs_to_lhs.is_some();
    if !has_lemma_rw {
      warn!("cannot create a lemma from {} and {}", lhs, rhs);
      None
    } else {
      Some(lemma_rw)
    }
  }

  fn make_lemma_rewrite_unchecked<A: Analysis<SymbolLang> + Clone>(
    &self,
    lhs_expr: &Expr,
    rhs_expr: &Expr,
    lemma_number: usize,
    exclude_wildcards: bool,
  ) -> Option<LemmaRewrite<A>> {
    let is_var = |v| self.local_context.contains_key(v);

    // NOTE: (CK) Before we would not recreate the lhs from lhs_expr every time we made a lemma rewrite since we did nested for loops
    // for lhs_expr {
    //   let lhs = ...
    //   for rhs_expr {
    //   ...
    //
    // which meant we just needed to clone it.
    //
    // I don't think this is a huge hit to efficiency though. If we cared, we
    // could instead loop over all lhs and rhs exprs first and precompute their
    // patterns + figure out which ones we don't need to consider.
    let lhs: Pattern<SymbolLang> = to_pattern(lhs_expr, is_var);
    if (CONFIG.irreducible_only && self.is_reducible(lhs_expr))
      || (exclude_wildcards && has_guard_wildcards(&lhs))
    {
      return None;
    }

    let rhs: Pattern<SymbolLang> = to_pattern(rhs_expr, is_var);
    if (CONFIG.irreducible_only && self.is_reducible(rhs_expr))
      || (exclude_wildcards && has_guard_wildcards(&rhs))
    {
      return None;
    }

    let lhs_vars = var_set(&lhs);
    let rhs_vars = var_set(&rhs);
    let lemma_vars: BTreeSet<Var> = lhs_vars.union(&rhs_vars).cloned().collect();

    // Pick out those variables that occur in the lemma
    let lemma_var_classes: IdSubst = self
      .var_classes
      .iter()
      .filter(|(x, _)| lemma_vars.contains(&to_wildcard(x)))
      .map(|(x, id)| (*x, *id))
      .collect();
    let params: Vec<(Symbol, Type)> = lemma_var_classes
      .keys()
      .map(|var| (*var, self.local_context.get(var).unwrap().clone()))
      .collect();

    let rewrite_eq = Equation::from_exprs(lhs_expr, rhs_expr);
    let (lemma_prop, renamed_params) = Prop::new(rewrite_eq, params.clone());
    let mut lemma_rw = LemmaRewrite {
      lhs_to_rhs: None,
      rhs_to_lhs: None,
      lemma_number,
      lemma_prop,
      renamed_params,
    };
    let lemma_name = lemma_rw.lemma_name();
    if rhs_vars.is_subset(&lhs_vars) {
      // if rhs has no extra wildcards, create a lemma lhs => rhs
      let lhs_to_rhs = Goal::make_rewrite_unchecked(lhs.clone(), rhs.clone(), lemma_name.clone());
      lemma_rw.lhs_to_rhs = Some(lhs_to_rhs);

      if CONFIG.single_rhs {
        return Some(lemma_rw);
      };
    }
    if lhs_vars.is_subset(&rhs_vars) {
      // if lhs has no extra wildcards, create a lemma rhs => lhs;
      // NOTE: (CK) This below comment is no longer true when our termination check is more complicated.
      // in non-cyclic mode, a single direction of IH is always sufficient
      // (because grounding adds all instantiations we could possibly care about).
      let rhs_to_lhs = Goal::make_rewrite_unchecked(rhs.clone(), lhs.clone(), lemma_name.clone());
      lemma_rw.rhs_to_lhs = Some(rhs_to_lhs);
    }
    let has_lemma_rw = lemma_rw.lhs_to_rhs.is_some() || lemma_rw.rhs_to_lhs.is_some();
    if !has_lemma_rw {
      warn!("cannot create a lemma from {} and {}", lhs, rhs);
      None
    } else {
      Some(lemma_rw)
    }
  }

  /// Before creating a lemma with premises, we need to update the variables
  /// in the premises with their canonical forms in terms of the current goal
  /// variables
  fn update_premises(&self) -> Vec<ETermEquation> {
    self
      .premises
      .iter()
      .map(|eq| eq.update_variables(&self.var_classes, &self.egraph))
      .collect()
  }

  /// Create a rewrite `lhs => rhs` which will serve as the lemma ("induction hypothesis") for a cycle in the proof;
  /// here lhs and rhs are patterns, created by replacing all scrutinees with wildcards;
  /// soundness requires that the pattern only apply to variable tuples smaller than the current scrutinee tuple.
  fn add_lemma_rewrites(
    &mut self,
    timer: &Timer,
    lemmas_state: &mut LemmasState,
    ih_lemma_number: usize,
  ) -> BTreeMap<String, Rw> {
    // Special case: the first time we add lemmas (i.e. when there are no
    // previous lemmas), we will make lemma rewrites out of the lhs and rhs only
    // and we will use the special IH name.
    if self.lemmas.is_empty() {
      let premises = self.update_premises();
      let mut rewrites = self.lemmas.clone();
      let mut ihs = vec![];
      let lhs = self.egraph.find(self.eq.lhs.id);
      let rhs = self.egraph.find(self.eq.rhs.id);
      if CONFIG.eqsat_ih {
        // Make IHs from saturated e-classes instead of initial exprs
        let lhs_exprs = collect_expressions_with_loops(&self.egraph, lhs);
        let rhs_exprs = collect_expressions_with_loops(&self.egraph, rhs);
        for (lhs_expr, rhs_expr) in lhs_exprs.iter().cartesian_product(&rhs_exprs) {
          let (lemma_rw, ih_searchers) =
            self.make_lemma_rewrite(lhs_expr, &rhs_expr, &premises, ih_lemma_number, false);
          if lemma_rw.is_some() {
            if let Some((lhs_ih, rhs_ih)) = &ih_searchers {
              ihs.push((lhs_ih.clone(), rhs_ih.clone()));
            }
            lemma_rw.unwrap().add_to_rewrites(&mut rewrites);
          }
        }
      } else {
        // In the non-cyclic case, only use the original LHS and RHS
        // and only if no other lemmas have been added yet
        let (lemma_rw, ih_searchers) = self.make_lemma_rewrite(
          &self.eq.lhs.expr,
          &self.eq.rhs.expr,
          &premises,
          ih_lemma_number,
          false,
        );
        if lemma_rw.is_none() {
          println!(
            "{}: {} == {}. params: {:?}",
            self.name, self.eq.lhs.sexp, self.eq.rhs.sexp, self.top_level_params
          );
          panic!()
        }
        if lemma_rw.is_some() {
          if let Some((lhs_ih, rhs_ih)) = &ih_searchers {
            ihs.push((lhs_ih.clone(), rhs_ih.clone()));
          }
          lemma_rw.unwrap().add_to_rewrites(&mut rewrites);
        }
      }
      if self.ih.is_none() && !ihs.is_empty() {
        if CONFIG.verbose {
          println!(
            "No IHs found, create {}:\n{}",
            if CONFIG.eqsat_ih { "them" } else { "it" },
            ihs
              .iter()
              .map(|(lhs_ih, rhs_ih)| format!("{} == {}", lhs_ih.searcher, rhs_ih.searcher))
              .join("\n")
          );
        }
        self.ih = Some(ihs);
      }
      return rewrites;
    }
    // Otherwise, we only create lemmas when we are operating in the cyclic mode
    if CONFIG.is_cyclic() {
      self.make_cyclic_lemma_rewrites(timer, lemmas_state, true).0
    } else {
      self.lemmas.clone()
    }
  }

  /// Creates cyclic lemmas from the current goal.
  fn make_cyclic_lemma_rewrites(
    &self,
    timer: &Timer,
    lemmas_state: &mut LemmasState,
    add_termination_check: bool,
  ) -> (BTreeMap<String, Rw>, Vec<LemmaRewrite<CycleggAnalysis>>) {
    let lhs_id = self.egraph.find(self.eq.lhs.id);
    let rhs_id = self.egraph.find(self.eq.rhs.id);

    let premises = self.update_premises();
    self.make_lemma_rewrites_from_all_exprs(
      lhs_id,
      rhs_id,
      premises,
      timer,
      lemmas_state,
      add_termination_check,
      true,
      true,
    )
  }

  fn make_rewrite_with_type_condition(
    lhs: Pat,
    rhs: Pat,
    cond: SoundnessWithType,
    lemma_name: String,
  ) -> (String, Rw) {
    let name = format!("{}-{}={}", lemma_name, lhs, rhs);
    warn!("creating lemma: {} => {}", lhs, rhs);
    let rw = Rewrite::new(
      &name,
      ConditionalSearcher {
        condition: cond,
        searcher: lhs,
      },
      rhs,
    )
    .unwrap();
    (name, rw)
  }

  fn make_rewrite_unchecked<A: Analysis<SymbolLang> + Clone>(
    lhs: Pat,
    rhs: Pat,
    lemma_name: String,
  ) -> (String, Rewrite<SymbolLang, A>) {
    // TODO: I think we should put just the lemma name and rewrite direction now
    // that we can record information about the lemmas elsewhere.
    let name = format!("{}-{}={}", lemma_name, lhs, rhs);
    warn!("creating unchecked lemma: {} => {}", lhs, rhs);
    let rw = Rewrite::new(&name, lhs, rhs).unwrap();
    (name, rw)
  }

  /// Add var as a scrutinee if its type `ty` is a datatype
  fn add_scrutinee(&mut self, var: Symbol, ty: &Type, depth: usize) {
    if let Ok((dt, _)) = ty.datatype() {
      if self.global_search_state.env.contains_key(&Symbol::from(dt)) {
        self.scrutinees.push_back(Scrutinee::new_var(var, depth));
      }
    }
  }

  /// If the egraph contains ITEs whose condition is "irreducible"
  /// (i.e. not equivalent to a constant or a scrutinee variable),
  /// add a fresh scrutinee to its eclass, so that we can match on it.
  fn split_ite(&mut self) {
    if CONFIG.verbose {
      println!("=== split_ite ===");
    }
    let guard_var = "?g".parse().unwrap();
    // Pattern "(ite ?g ?x ?y)"
    let searcher: Pattern<SymbolLang> = format!("({} {} ?x ?y)", *ITE, guard_var).parse().unwrap();
    if CONFIG.verbose {
      println!("search e-graph for ites using {searcher}");
    }
    let matches = searcher.search(&self.egraph);
    // Collects class IDs of all stuck guards;
    // it's a map because the same guard can match more than once, but we only want to add a new scrutinee once
    let mut stuck_guards = BTreeMap::new();
    for m in matches {
      if CONFIG.verbose {
        println!("found a match, go through substs:");
      }
      for subst in m.substs {
        if CONFIG.verbose {
          println!("subst: {subst:?}");
        }
        let guard_id = *subst.get(guard_var).unwrap();
        if CONFIG.verbose {
          println!("guard e-class:");
          print_expressions_in_eclass(&self.egraph, guard_id);
        }
        if let CanonicalForm::Stuck(_) = self.egraph[guard_id].data.canonical_form_data {
          if CONFIG.verbose {
            println!("guard is stuck");
          }
          stuck_guards.insert(guard_id, subst);
        }
      }
    }
    if CONFIG.verbose {
      println!("stuck guards: {stuck_guards:?}");
    }
    // Iterate over all stuck guard eclasses and add a new scrutinee to each
    for (guard_id, subst) in stuck_guards {
      if CONFIG.verbose {
        println!("guard ID: {guard_id}");
        println!("guard subst: {subst:?}");
      }
      let fresh_var = Symbol::from(format!("{}{}", GUARD_PREFIX, guard_id));
      if CONFIG.verbose {
        println!("fresh var: {fresh_var}");
      }
      // This is here only for logging purposes
      let expr = Extractor::new(&self.egraph, AstSize).find_best(guard_id).1;
      let add_scrutinee_message =
        format!("adding scrutinee {} to split condition {}", fresh_var, expr);
      if CONFIG.verbose {
        println!("{add_scrutinee_message}");
        warn!("{}", add_scrutinee_message);
      }
      self
        .local_context
        .insert(fresh_var, BOOL_TYPE.parse().unwrap());
      // We are adding the new scrutinee to the front of the deque,
      // because we want to split conditions first, since they don't introduce new variables
      if CONFIG.verbose {
        println!(
          "make scrutinee for {fresh_var}: {:?}",
          Scrutinee::new_guard(fresh_var)
        );
      }
      self.scrutinees.push_front(Scrutinee::new_guard(fresh_var));
      let new_node = SymbolLang::leaf(fresh_var);
      let new_pattern_ast = vec![ENodeOrVar::ENode(new_node.clone())].into();
      if CONFIG.verbose {
        println!("new_pattern_ast: {new_pattern_ast}");
      }
      let guard_var_pattern_ast = vec![ENodeOrVar::Var(guard_var)].into();
      if CONFIG.verbose {
        println!("guard_var_pattern_ast: {guard_var_pattern_ast}");
      }
      self.guard_exprs.insert(fresh_var.to_string(), expr);
      if CONFIG.verbose {
        println!("union_instantiations({guard_var_pattern_ast}, {new_pattern_ast}, {subst:?})");
      }
      let (union_id, union_happened) = self.egraph.union_instantiations(
        &guard_var_pattern_ast,
        &new_pattern_ast,
        &subst,
        add_scrutinee_message,
      );
      if CONFIG.verbose {
        println!("union e-class:");
        dump_eclass_exprs(&self.egraph, union_id);
        println!("union_happened: {union_happened}");
      }
    }
    if CONFIG.verbose {
      println!("*** split_ite ***");
    }
    self.egraph.rebuild();
  }

  /// Consume this goal and add its case splits to the proof state
  fn case_split(
    mut self,
    scrutinee: Scrutinee,
    timer: &Timer,
    lemmas_state: &mut LemmasState,
    ih_lemma_number: usize,
  ) -> (ProofTerm, Vec<Goal<'a>>) {
    let new_lemmas = self.add_lemma_rewrites(timer, lemmas_state, ih_lemma_number);

    let var_str = scrutinee.name.to_string();
    warn!("case-split on {}", scrutinee.name);
    let var_node = SymbolLang::leaf(scrutinee.name);
    let var_pattern_ast: RecExpr<ENodeOrVar<SymbolLang>> =
      vec![ENodeOrVar::ENode(var_node.clone())].into();
    // Get the type of the variable, and then remove the variable
    let ty = match self.local_context.get(&scrutinee.name) {
      Some(ty) => ty,
      None => panic!("{} not in local context", scrutinee.name),
    };
    // Convert to datatype name
    let dt = Symbol::from(ty.datatype().unwrap().0);
    // Get the constructors of the datatype
    let (_, cons) = self.global_search_state.env.get(&dt).unwrap_or_else(|| {
      println!("unexpected datatype {} for variable {}", dt, var_str);
      println!("params: {:?}, prop: {}", self.top_level_params, self.eq);
      panic!()
    });
    // We will add this to state.proof to describe the case split.
    let mut instantiated_cons_and_goals: Vec<(String, String)> = vec![];
    // These are the new goals generated from the case split
    let mut goals = vec![];

    // Create a new goal for each constructor we can case split to and add it to
    // the proof state.
    //
    // (We process constructors in reverse order so that base case ends up at
    // the top of the stack - this is due to how we typically define the orders
    // for our datatypes in our definitions files. It's not a very principled
    // iteration order)
    for &con in cons.iter().rev() {
      if scrutinee.depth >= CONFIG.max_split_depth {
        continue;
      }
      let mut new_goal = self.clone();
      new_goal.case_split_vars.insert(scrutinee.name);
      new_goal.egraph.analysis.case_split_vars = new_goal.case_split_vars.clone();
      new_goal.lemmas = new_lemmas.clone();

      // Get the arguments of the constructor.
      let con_args = self.instantiate_constructor(&con, ty);
      // For each argument: we will create a fresh variable that we can use as a
      // scrutinee.
      let mut fresh_vars = vec![];

      // We will update the timestamp of the cvec analysis so that we enforce
      // the update when we generate a cvec.
      new_goal.egraph.analysis.cvec_analysis.current_timestamp += 1;
      // let _pre_expr = self.full_expr.clone();

      for (i, arg_type) in con_args.iter().enumerate() {
        let fresh_var_name = format!("{}_{}{}", scrutinee.name, self.egraph.total_size(), i);
        let fresh_var = Symbol::from(fresh_var_name.clone());
        fresh_vars.push(fresh_var);
        // Add new variable to context
        new_goal.local_context.insert(fresh_var, arg_type.clone());
        // The depth of a scrutinee tracks how many ancestors are between it and
        // a top-level parameter, so we add 1 when we case split.
        new_goal.add_scrutinee(fresh_var, arg_type, scrutinee.depth + 1);
        let id = new_goal.egraph.add(SymbolLang::leaf(fresh_var));
        // The class corresponding to this var is its class in the e-graph.
        new_goal.var_classes.insert(fresh_var, id);
        // Generate a cvec for the fresh_var.
        new_goal.add_cvec_for_class(id, arg_type);

        if CONFIG.add_grounding && ty == arg_type {
          // This is a recursive constructor parameter:
          // add new grounding instantiations replacing var with fresh_var
          new_goal.add_grounding(scrutinee.name, fresh_var);
        }
      }
      new_goal.egraph.analysis.local_ctx = new_goal.local_context.clone();

      // Create an application of the constructor to the fresh vars

      let con_app_string = if fresh_vars.is_empty() {
        format!("({})", con)
      } else {
        format!(
          "({} {})",
          con,
          fresh_vars
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
        )
      };
      if CONFIG.verbose {
        println!("Fill constructor: {con_app_string}");
      }
      let con_app: Expr = con_app_string.parse().unwrap();

      new_goal.name = format!("{}_{}={}", new_goal.name, scrutinee.name, con_app);
      // This is tracked for proof emission.
      instantiated_cons_and_goals.push((con_app_string, new_goal.name.clone()));

      // Add con_app to the new goal's egraph and union it with var
      new_goal.egraph.add_expr(&con_app);
      let con_expr = symbolic_expressions::parser::parse_str(&con_app.to_string()).unwrap();
      new_goal.full_expr =
        rewrite_expr(&new_goal.full_expr, &scrutinee.name.to_string(), &con_expr);
      new_goal.egraph.union_instantiations(
        &var_pattern_ast,
        &rec_expr_to_pattern_ast(con_app.clone()),
        &Subst::default(),
        format!("case-split:{}", new_goal.name),
      );
      // Remove old variable from the egraph and context
      remove_node(&mut new_goal.egraph, &var_node);

      new_goal.egraph.rebuild();

      // if CONFIG.verbose {
      //   println!("Case-split subgoal:");
      //   dump_eclass_exprs(&new_goal.egraph, new_goal.eq.lhs.id);
      //   println!("=?=");
      //   dump_eclass_exprs(&new_goal.egraph, new_goal.eq.rhs.id);
      // }

      // In cyclic mode: add the guard to premises,
      if CONFIG.is_cyclic()
        && var_str.starts_with(GUARD_PREFIX)
        && self.guard_exprs.contains_key(&var_str)
      {
        let lhs = ETerm::from_expr(self.guard_exprs[&var_str].clone(), &new_goal.egraph);
        let rhs = ETerm::from_expr(con_app, &new_goal.egraph);
        let eq = ETermEquation { lhs, rhs };
        new_goal.premises.push(eq);
      }

      // Add the subgoal to the proof state
      goals.push(new_goal);
    }
    // We split on var into the various instantiated constructors and subgoals.
    //
    // If the var is an ITE split, we will add the condition that was split on
    // to our proof term. This is necessary because for ITE splits we introduce
    // a new variable that we bind an expression to.
    let proof_term = match scrutinee.scrutinee_type {
      ScrutineeType::Guard => {
        // There should only be two cases.
        assert_eq!(instantiated_cons_and_goals.len(), 2);
        ProofTerm::ITESplit(
          var_str.clone(),
          self.guard_exprs[&var_str].to_string(),
          instantiated_cons_and_goals,
        )
      }
      ScrutineeType::Var => ProofTerm::CaseSplit(var_str, instantiated_cons_and_goals),
    };
    (proof_term, goals)
  }

  fn find_blocking(&self, timer: &Timer) -> (BTreeSet<Symbol>, BTreeSet<Id>) {
    let mut blocking_vars: BTreeSet<_> = BTreeSet::default();
    let mut blocking_exprs: BTreeSet<Id> = BTreeSet::default();

    let mut lhs_descendents = BTreeSet::default();
    // TODO: Canonicalize?
    self.compute_descendents(self.eq.lhs.id, &mut lhs_descendents);

    let mut rhs_descendents = BTreeSet::default();
    self.compute_descendents(self.eq.rhs.id, &mut rhs_descendents);

    for reduction in self.global_search_state.reductions {
      let x = reduction.searcher.get_pattern_ast().unwrap();
      let sexp = symbolic_expressions::parser::parse_str(&x.to_string()).unwrap();

      // Hack to dedup the new patterns (sexps) we generated
      let mut new_sexps: Vec<Sexp> = Goal::analyze_sexp_for_blocking_vars(&sexp)
        .into_iter()
        .map(|x| x.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|x| symbolic_expressions::parser::parse_str(x.as_str()).unwrap())
        .collect();

      // the patterns we generated contained only ? instead of ?var, so we go and add fresh variable names everywhere
      for ns in new_sexps.iter_mut() {
        *ns = Goal::gen_fresh_vars(ns.clone(), 1);
      }

      // use these patterns to search over the egraph
      for new_sexp in new_sexps {
        if timer.timeout() {
          return (blocking_vars, blocking_exprs);
        }
        let mod_searcher: Pattern<SymbolLang> = new_sexp.to_string().parse().unwrap();

        // for each new pattern, find the pattern variables in blocking positions so that we can use them to look up the substs later
        let bvs: Vec<Var> = mod_searcher
          .vars()
          .iter()
          .filter(|&x| x.to_string().contains("block_"))
          .cloned()
          .collect();

        let matches = mod_searcher.search(&self.egraph);

        // let extractor = Extractor::new(&self.egraph, AstSize);

        // look at the e-class analysis for each matched e-class, if any of them has a variable, use it
        for m in matches {
          for subst in m.substs {
            for v in &bvs[0..] {
              if let Some(&ecid) = subst.get(*v) {
                match &self.egraph[ecid].data.canonical_form_data {
                  CanonicalForm::Var(n) => {
                    blocking_vars.insert(n.op);
                  }
                  CanonicalForm::Stuck(_) | CanonicalForm::Const(_) => {
                    if lhs_descendents.contains(&ecid) && rhs_descendents.contains(&ecid) {
                      blocking_exprs.insert(ecid);
                      // let expr = extractor.find_best(ecid).1;
                      // blocking_exprs.insert(expr.to_string());
                    }
                  }
                  _ => (),
                }
              }
            }
          }
        }
      }
    }
    (blocking_vars, blocking_exprs)
  }

  fn compute_descendents(&self, class: Id, descendents: &mut BTreeSet<Id>) {
    if descendents.contains(&class) {
      return;
    }
    descendents.insert(class);
    for node in self.egraph[class].nodes.iter() {
      for child in node.children() {
        self.compute_descendents(*child, descendents);
      }
    }
  }

  /// Gets the next variable to case split on using the blocking var analysis
  fn next_scrutinee(&mut self, mut blocking_vars: BTreeSet<Symbol>) -> Option<Scrutinee> {
    let is_goal_var = match (
      &self.egraph[self.eq.lhs.id].data.canonical_form_data,
      &self.egraph[self.eq.rhs.id].data.canonical_form_data,
    ) {
      (CanonicalForm::Var(_), CanonicalForm::Const(_))
      | (CanonicalForm::Const(_), CanonicalForm::Var(_)) => true,
      _ => false,
    };
    let blocking = if is_goal_var {
      self.scrutinees.iter().next().map(|s| (0, s))
    } else {
      self
        .scrutinees
        .iter()
        .find_position(|s| blocking_vars.contains(&s.name))
    };

    // Add the vars we already have case split on, since those were previously
    // blocking. This is important for our soundness check, since we skip
    // checking any variables which are not blocking.
    blocking_vars.extend(&self.case_split_vars);
    // Record into the e-graph analysis so that we can
    // use this infromation in the soundness check
    self.egraph.analysis.blocking_vars = blocking_vars;

    blocking?;

    let var_idx = blocking.unwrap().0;
    self.scrutinees.remove(var_idx)
  }

  // FIXME: factor this out somehow
  // (it relies on the magic string "?" so I'm not sure how to)
  fn gen_fresh_vars(sexp: Sexp, mut idx: i32) -> Sexp {
    match sexp {
      Sexp::String(s) if s == "?" => Sexp::String(format!("?block_{}", idx)),
      Sexp::List(v) => Sexp::List(
        v.iter()
          .map(|x| {
            idx += 1;
            Goal::gen_fresh_vars(x.clone(), idx + 1)
          })
          .collect(),
      ),
      _ => sexp,
    }
  }

  /// Looks at an sexp representing a rewrite (or part of a rewrite) to determine where blocking vars might be
  /// e.g. if we have a rule that looks like `foo Z (Cons Z ?xs)` => ...)
  /// then we want to generate patterns like
  ///   1. `foo ?fresh1 (Cons Z ?xs)`
  ///   2. `foo ?fresh1 ?fresh2`
  ///   3. `foo ?fresh1 (Cons ?fresh2 ?xs)`
  ///   4. `foo Z ?fresh2`
  ///   5. `foo Z (Cons ?fresh1 ?xs)`
  // TODO: factor this out somehow
  fn analyze_sexp_for_blocking_vars(sexp: &Sexp) -> Vec<Sexp> {
    let mut new_exps: Vec<Sexp> = vec![];
    new_exps.push(sexp.clone());

    // If this sexp is a constructor application, replace it by ?
    if sexp_is_constructor(sexp) {
      // for now, just indicate by "?" each position where we could have a blocking var, and later go and replace them with fresh vars
      let fresh_var_indicator = "?";
      new_exps.push(Sexp::String(fresh_var_indicator.to_string()));
    }

    // also recursively analyse its children to find other potential blocking arguments
    match sexp {
      Sexp::List(v) => {
        let head = &v[0];
        let mut all_replacements: Vec<Vec<Sexp>> = vec![];
        for (_, sub_arg) in v[1..].iter().enumerate() {
          all_replacements.push(Goal::analyze_sexp_for_blocking_vars(sub_arg));
        }
        // get all possible subsets of the replacements (i.e. every subset of constructor applications replaced by fresh pattern vars)
        let all_combinations = cartesian_product(&all_replacements);
        for mut combination in all_combinations {
          combination.insert(0, head.clone());
          new_exps.push(Sexp::List(combination));
        }
      }
      _ => {}
    };

    new_exps
  }

  /// Save e-graph to file
  fn save_egraph(&self) {
    let filename = CONFIG.output_directory.join(format!("{}.png", self.name));
    let verbosity = format!("-q{}", CONFIG.log_level as usize);
    let dot = self.egraph.dot();
    dot
      .run_dot([
        "-Tpng",
        verbosity.as_str(),
        "-o",
        &filename.to_string_lossy(),
      ])
      .unwrap();
  }

  /// Given a polymorphic constructor and a concrete instantiation of a
  /// datatype, return the concrete types of the constructor's arguments.
  fn instantiate_constructor(&self, con: &Symbol, actual: &Type) -> Vec<Type> {
    let con_ty = self.global_search_state.context.get(con).unwrap();
    let (args, ret) = con_ty.args_ret();
    let instantiations = find_instantiations(&ret.repr, &actual.repr, is_var).unwrap();
    let ret = args
      .iter()
      .map(|arg| Type::new(resolve_sexp(&arg.repr, &instantiations)))
      .collect();
    ret
  }

  /// Add new grounding instantiations
  /// that replace parent with child in previous instantiations
  fn add_grounding(&mut self, parent: Symbol, child: Symbol) {
    // First gather all the terms we want to instantiate:
    // take both sides of the equation and all the premises
    let mut sides = vec![&self.eq.lhs, &self.eq.rhs];
    for premise in self.premises.iter() {
      sides.push(&premise.lhs);
      sides.push(&premise.rhs);
    }

    // Now create new instantiations from existing ones
    let mut new_instantiations = vec![];
    for inst in self.grounding_instantiations.iter() {
      let replaced_canonicals: Vec<(Symbol, ETerm, bool)> = self
        .top_level_params
        .iter()
        .map(|x| {
          // Which class was this param instantiated to?
          let id = inst.get(x).unwrap();
          // Parameters must be canonical (at least in a clean state)
          let canonical = CanonicalFormAnalysis::extract_canonical(&self.egraph, *id).unwrap();
          // Try replacing the case-split variable with its child
          let (new_expr, replaced) = replace_var(&canonical, parent, child);
          let eterm = if replaced {
            ETerm::new_from_expr(&new_expr, &mut self.egraph)
          } else {
            ETerm::from_expr(new_expr, &self.egraph)
          };
          (*x, eterm, replaced)
        })
        .collect();
      // If any of the canonical forms had a replacement, add a new instantiation:
      if replaced_canonicals.iter().any(|(_, _, b)| *b) {
        let new_instantiation = replaced_canonicals
          .iter()
          .map(|(x, e, _)| (x.to_string(), e.sexp.clone()))
          .collect();
        // For each new instantiation, instantiate all the sides and add them to the egraph
        for term in sides.iter() {
          let new_term = resolve_sexp(&term.sexp, &new_instantiation);
          ETerm::new(&new_term, &mut self.egraph);
        }
        // Add the new instantiation to the list of grounding instantiations
        let new_subst = replaced_canonicals
          .iter()
          .map(|(x, e, _)| (*x, e.id))
          .collect();
        new_instantiations.push(new_subst);
      }
    }

    // Add the new instantiations to the list of grounding instantiations
    self.grounding_instantiations.extend(new_instantiations);
  }

  /// Search for cc (concrete correspondence) lemmas.
  ///
  /// These are lemmas we propose from subterms in the e-graph that our concrete
  /// analysis deems equal on some set of random terms.
  fn search_for_cc_lemmas(&mut self, timer: &Timer, lemmas_state: &mut LemmasState) -> Vec<Prop> {
    let mut lemmas = vec![];
    self.egraph.analysis.cvec_analysis.saturate();
    let resolved_lhs_id = self.egraph.find(self.eq.lhs.id);
    let resolved_rhs_id = self.egraph.find(self.eq.rhs.id);
    if CONFIG.verbose {
      println!("LHS: ");
      print_expressions_in_eclass(&self.egraph, resolved_lhs_id);
      println!("RHS: ");
      print_expressions_in_eclass(&self.egraph, resolved_rhs_id);
      // print_all_expressions_in_egraph(&self.egraph, 7);
    }
    let class_ids: Vec<Id> = self.egraph.classes().map(|c| c.id).collect();

    for class_1_id in &class_ids {
      for class_2_id in &class_ids {
        if timer.timeout() {
          return lemmas;
        }
        // Resolve the ids because as we union things, we might make more
        // equalities.
        let class_1_id = self.egraph.find(*class_1_id);
        let class_2_id = self.egraph.find(*class_2_id);
        // Don't try to union two of the same e-class.
        //
        // Also, only try pairs (id1, id2) where id1 < id2.
        // Since unioning doesn't care about order, we can avoid
        // trying classes redundantly.
        if class_1_id >= class_2_id {
          continue;
        }

        // NOTE: We no longer skip here because we need to generalize sometimes
        // Don't try unioning the LHS and RHS, we've seen those already.
        // if class_1_id == resolved_lhs_id && class_2_id == resolved_rhs_id
        //   || class_1_id == resolved_rhs_id && class_2_id == resolved_lhs_id {
        //     continue
        // }

        if let Some(true) = cvecs_equal(
          &self.egraph.analysis.cvec_analysis,
          &self.egraph[class_1_id].data.cvec_data,
          &self.egraph[class_2_id].data.cvec_data,
        ) {
          let class_1_canonical = &self.egraph[class_1_id].data.canonical_form_data;
          let class_2_canonical = &self.egraph[class_2_id].data.canonical_form_data;
          match (class_1_canonical, class_2_canonical) {
            (CanonicalForm::Const(c1_node), CanonicalForm::Const(c2_node)) => {
              let num_differing_children: usize = zip(c1_node.children(), c2_node.children())
                .map(|(child_1, child_2)| if child_1 != child_2 { 0 } else { 1 })
                .sum();
              //* They also do anti-unification, but in a very limited way:
              // There is a simpler CC lemma to prove.
              //
              // Consider for example the case when the canonical forms are
              //   c1: (S (plus x x))
              //   c2: (S (double x))
              // In this case, the number of differing children is only 1.
              // The differing children are
              //   (plus x x) and (double x)
              // However, we can be sure that if the cvec analysis deemed c1 and
              // c2 equal, then it will deem these two differing children equal.
              //
              // Thus we won't waste our time trying to prove c1 == c2 when
              // we could instead prove (plus x x) == (double x), which implies
              // by congruence that c1 == c2.
              if num_differing_children <= 1 {
                continue;
              }
            }
            _ => {}
          }

          // println!("Found cc lemma:");
          // dump_eclass_exprs(&self.egraph, self.egraph.find(class_1_id));
          // println!("=?=");
          // dump_eclass_exprs(&self.egraph, self.egraph.find(class_2_id));

          // println!("equal_pair {} {}", class_1_id, class_2_id);
          // println!("found candidate cc lemma: making rewrites");
          let (_rewrites, rewrite_infos) = self.make_lemma_rewrites_from_all_exprs(
            class_1_id,
            class_2_id,
            vec![],
            timer,
            lemmas_state,
            false,
            false,
            true,
          );
          // println!("made rewrites");
          let new_rewrite_eqs = rewrite_infos
            .into_iter()
            .map(|rw_info| (rw_info.lemma_prop, rw_info.renamed_params))
            .collect::<Vec<_>>();
          // println!("rewrites: {:#?}", new_rewrite_eqs);
          // We used to check the egraph to see if the lemma helped us, but now
          // we just throw it into our list. We do that check in try_prove_lemmas.
          if new_rewrite_eqs.is_empty() {
            continue;
          }

          if CONFIG.cc_lemmas_generalization {
            let fresh_name = format!("fresh_{}_{}", self.name, self.egraph.total_size());
            for (new_rewrite_eq, renamed_params) in &new_rewrite_eqs {
              if timer.timeout() {
                return lemmas;
              }
              lemmas.extend(find_generalizations_prop(
                new_rewrite_eq,
                self.global_search_state.context,
                &self.local_context,
                renamed_params,
                fresh_name.clone(),
              ));
            }
          }

          // Optimization: skip adding any lemmas that would be subsumed by a cyclic lemma
          //for lemma in new_rewrite_eqs.iter() {
          //  println!("raw lemmas {}", lemma)
          //}
          if !CONFIG.only_generalize
            && !(class_1_id == resolved_lhs_id && class_2_id == resolved_rhs_id
              || class_1_id == resolved_rhs_id && class_2_id == resolved_lhs_id)
          {
            lemmas.extend::<Vec<_>>(new_rewrite_eqs.into_iter().unzip::<_, _, _, Vec<_>>().0);
          }
        }
      }
    }
    lemmas
  }

  // fn _ripple_alt(&mut self) {
  //   if CONFIG.verbose {
  //     println!("=== ripple_alt ===");
  //   }
  //   if let Some((lhs_ih, rhs_ih)) = &self.ih() {
  //     if CONFIG.verbose {
  //       println!("IH: {lhs_ih} == {rhs_ih}");
  //     }
  //     let lhs = self.egraph.find(self.eq.lhs.id);
  //     let rhs = self.egraph.find(self.eq.rhs.id);
  //     if CONFIG.verbose {
  //       println!("LHS:");
  //       dump_eclass_exprs(&self.egraph, lhs);
  //       println!("RHS:");
  //       dump_eclass_exprs(&self.egraph, rhs);
  //       println!("Ripple LHS:");
  //     }
  //     search_wave_fronts(&mut self.egraph, lhs_ih, lhs);
  //     if CONFIG.verbose {
  //       println!("Ripple RHS:");
  //     }
  //     search_wave_fronts(&mut self.egraph, rhs_ih, rhs);
  //   } else {
  //     if CONFIG.verbose {
  //       println!("Cannot ripple");
  //     }
  //   }
  //   if CONFIG.verbose {
  //     println!("*** ripple_alt ***");
  //   }
  // }

  fn ripple_out(&mut self) -> Option<Vec<Goal<'a>>> {
    if CONFIG.verbose {
      println!("=== ripple_out ===");
    }
    // Only ripple when an IH exists, i.e., we are at a goal already case-split
    if let Some(ihs) = self.ih.clone() {
      let lhs = self.egraph.find(self.eq.lhs.id);
      let rhs = self.egraph.find(self.eq.rhs.id);
      if CONFIG.verbose {
        println!("Full expr: {}", self.full_expr);
        println!("Goal:");
        dump_eclass_exprs(&self.egraph, lhs);
        println!("=?=");
        dump_eclass_exprs(&self.egraph, rhs);
      }

      let mut new_goals = vec![];
      for (lhs_ih, rhs_ih) in ihs {
        if CONFIG.verbose {
          println!("IH: {} == {}", lhs_ih.searcher, rhs_ih.searcher);
        }

        let (rippled_rhs_set, _) = self.get_rippled_exprs(&lhs_ih, &rhs_ih, lhs);
        let (rippled_lhs_set, _) = self.get_rippled_exprs(&rhs_ih, &lhs_ih, rhs);

        self.egraph.rebuild();
        self.egraph.analysis.cvec_analysis.saturate();

        if rippled_rhs_set.is_empty() {
          if CONFIG.verbose {
            println!("Could not ripple RHS");
          }
        } else {
          for rippled_rhs in rippled_rhs_set {
            if rhs == rippled_rhs {
              if CONFIG.verbose {
                println!("Skipping RHS == rippled RHS");
              }
              continue;
            }
            // if let Some(true) = cvecs_equal(
            //   &self.egraph.analysis.cvec_analysis,
            //   &self.egraph[rhs].data.cvec_data,
            //   &self.egraph[rippled_rhs].data.cvec_data,
            // ) {
            if CONFIG.verbose {
              println!("Rippled RHS:");
              dump_eclass_exprs(&self.egraph, rippled_rhs);
            }
            let rippled_rhs_exprs = collect_expressions_with_loops(&self.egraph, rippled_rhs);
            let smallest_rippled_rhs_expr = get_smallest_expr(&rippled_rhs_exprs);
            let mut new_goal = self.clone();
            new_goal.name = format!(
              "rippled_rhs_{}={}",
              smallest_rippled_rhs_expr, new_goal.eq.rhs.expr
            );
            new_goal.full_expr =
              Equation::from_exprs(&smallest_rippled_rhs_expr, &new_goal.eq.rhs.expr);
            // new_goal.eq.lhs.sexp = symbolic_expressions::parser::parse_str(
            //   smallest_rippled_rhs_expr.to_string().as_str(),
            // )
            // .unwrap();
            new_goal.eq.lhs.expr = smallest_rippled_rhs_expr;
            new_goal.eq.lhs.id = rippled_rhs;
            // new_goal.ih = Some((rhs_ih.clone(), rhs_ih.clone()));
            if CONFIG.verbose {
              println!("New goal:");
              dump_eclass_exprs(&new_goal.egraph, new_goal.eq.lhs.id);
              println!("=?=");
              dump_eclass_exprs(&new_goal.egraph, new_goal.eq.rhs.id);
            }
            new_goals.push(new_goal);
            // } else {
            //   if CONFIG.verbose {
            //     println!("skipping cvecs(rippled RHS) != cvecs(RHS):");
            //     dump_eclass_exprs(&self.egraph, rippled_rhs);
            //     println!("=?=");
            //     dump_eclass_exprs(&self.egraph, rhs);
            //   }
            // }
          }
        }

        if rippled_lhs_set.is_empty() {
          if CONFIG.verbose {
            println!("Could not ripple LHS");
          }
        } else {
          for rippled_lhs in rippled_lhs_set {
            if lhs == rippled_lhs {
              if CONFIG.verbose {
                println!("Skipping LHS == rippled LHS");
              }
              continue;
            }
            // if let Some(true) = cvecs_equal(
            //   &self.egraph.analysis.cvec_analysis,
            //   &self.egraph[lhs].data.cvec_data,
            //   &self.egraph[rippled_lhs].data.cvec_data,
            // ) {
            if CONFIG.verbose {
              println!("Rippled LHS:");
              dump_eclass_exprs(&self.egraph, rippled_lhs);
            }
            let rippled_lhs_exprs = collect_expressions_with_loops(&self.egraph, rippled_lhs);
            let smallest_rippled_lhs_expr = get_smallest_expr(&rippled_lhs_exprs);
            let mut new_goal = self.clone();
            new_goal.name = format!(
              "rippled_lhs_{}={}",
              new_goal.eq.lhs.expr, smallest_rippled_lhs_expr
            );
            new_goal.full_expr =
              Equation::from_exprs(&new_goal.eq.lhs.expr, &smallest_rippled_lhs_expr);
            // new_goal.eq.lhs.sexp = symbolic_expressions::parser::parse_str(
            //   smallest_rippled_lhs_expr.to_string().as_str(),
            // )
            // .unwrap();
            new_goal.eq.rhs.expr = smallest_rippled_lhs_expr;
            new_goal.eq.rhs.id = rippled_lhs;
            // new_goal.ih = Some((lhs_ih.clone(), lhs_ih.clone()));
            if CONFIG.verbose {
              println!("New goal:");
              dump_eclass_exprs(&new_goal.egraph, new_goal.eq.lhs.id);
              println!("=?=");
              dump_eclass_exprs(&new_goal.egraph, new_goal.eq.rhs.id);
            }
            new_goals.push(new_goal);
            // } else {
            //   if CONFIG.verbose {
            //     println!("skipping cvecs(LHS) != cvecs(rippled LHS):");
            //     dump_eclass_exprs(&self.egraph, lhs);
            //     println!("=?=");
            //     dump_eclass_exprs(&self.egraph, rippled_lhs);
            //   }
            // }
          }
        }
      }
      if new_goals.is_empty() {
        if CONFIG.verbose {
          println!("Failed");
          println!("*** ripple_out ***");
        }
        None
      } else {
        Some(new_goals)
      }
    } else {
      if CONFIG.verbose {
        println!("Failed");
        println!("*** ripple_out ***");
      }

      None
    }
  }

  fn get_rippled_exprs(
    &mut self,
    lhs_ih: &ConditionalSearcher<SoundnessWithType, Pattern<SymbolLang>>,
    rhs_ih: &ConditionalSearcher<SoundnessWithType, Pattern<SymbolLang>>,
    lhs: Id,
  ) -> (HashSet<Id>, Vec<Id>) {
    let mut rippled_rhs = HashSet::new();
    let mut ih_replacements = vec![];

    let mut cache = HashMap::new();
    rippled_rhs.extend(self.pattern_replace_in_eclass_with_analysis_help(
      &mut cache,
      &mut ih_replacements,
      lhs,
      lhs_ih,
      rhs_ih,
    ));
    rippled_rhs.remove(&self.egraph.find(lhs));
    // if CONFIG.verbose {
    //   println!("rippled: {}", rippled_rhs.len() > 0);
    //   let _ = rippled_rhs
    //     .clone()
    //     .into_iter()
    //     .for_each(|x| println!("{}", self.egraph.id_to_expr(x)));
    // }
    // self.egraph.analysis.cvec_analysis.saturate();

    (rippled_rhs, ih_replacements)
  }

  fn pattern_replace_in_eclass_with_analysis_help(
    &mut self,
    cache: &mut HashMap<Id, HashSet<Id>>,
    ih_replacement_vec: &mut Vec<Id>,
    lhs: Id,
    lhs_ih: &ConditionalSearcher<SoundnessWithType, Pattern<SymbolLang>>,
    rhs_ih: &ConditionalSearcher<SoundnessWithType, Pattern<SymbolLang>>,
  ) -> HashSet<Id> {
    if let Some(rippled_rhs) = cache.get(&lhs) {
      return rippled_rhs.clone();
    }
    cache.insert(lhs, HashSet::from_iter([lhs]));
    let lhs_eclass = self.egraph[lhs].clone();
    self.egraph.rebuild();
    if false && CONFIG.verbose {
      println!("Search for {} in LHS e-class:", lhs_ih.searcher);
      dump_eclass_exprs(&self.egraph, lhs);
    }
    if let Some(matches) = lhs_ih.search_eclass(&self.egraph, lhs) {
      // println!("Found match:");
      for subst in matches.substs {
        // if CONFIG.verbose {
        // lhs_ih.searcher.vars().into_iter().for_each(|var| {
        //   if subst.get(var).is_none() {
        //     println!("{}: none", var)
        //   } else {
        //     println!(
        //       "{}: {}",
        //       var,
        //       self.egraph.id_to_expr(*subst.get(var).unwrap())
        //     )
        //   }
        // });
        // }
        let new_id = self.add_instantiation_with_var_if_necessary(&rhs_ih.searcher, subst);
        if false && CONFIG.verbose {
          println!("After adding instantiation:");
          dump_eclass_exprs(&self.egraph, new_id);
        }
        cache.get_mut(&lhs).unwrap().insert(new_id);
      }
      ih_replacement_vec.push(lhs);
    } else {
      let limit = 5;
      let mut j = 0;
      if false && CONFIG.verbose {
        println!("No match, go over LHS enodes");
      }
      for lhs_enode in lhs_eclass.nodes.iter() {
        let mut new_children = vec![vec![]];
        if false && CONFIG.verbose {
          println!("LHS enode: {lhs_enode}");
          println!("Go over enode children");
        }
        for &child in &lhs_enode.children {
          if false && CONFIG.verbose {
            println!("child:");
            dump_eclass_exprs(&self.egraph, child);
          }
          let mut temp = vec![];
          let rippled_ids = self.pattern_replace_in_eclass_with_analysis_help(
            cache,
            ih_replacement_vec,
            child,
            lhs_ih,
            rhs_ih,
          );
          if false && CONFIG.verbose {
            println!("Rippled IHs:");
            for id in &rippled_ids {
              dump_eclass_exprs(&self.egraph, *id);
            }
          }
          for id_list in new_children {
            // println!("Go over rippled IDs");
            for id in &rippled_ids {
              // println!("rippled ID:");
              // dump_eclass_exprs(&self.egraph, *id);
              let mut id_list_mod = id_list.clone();
              id_list_mod.push(id.clone());
              temp.push(id_list_mod);
            }
          }
          new_children = temp;
        }
        // self.egraph.analysis.cvec_analysis.current_timestamp += 1;
        for id_list in new_children {
          let enode = SymbolLang::new(lhs_enode.op, id_list);
          // let _type = self
          //   .global_search_state
          //   .context
          //   .get(&enode.op)
          //   .or_else(|| self.local_context.get(&enode.op))
          //   .and_then(|t| Some(t.args_ret().1));
          let new_id = self.egraph.add(enode);
          // TODO:
          // self.add_cvec_for_class(id, ty);
          // if let Some(t) = _type {
          //   if CONFIG.verbose {
          //     println!("type: {t}");
          //   }
          //   self.add_cvec_for_class(new_id, &t);
          // }
          if false && CONFIG.verbose {
            println!("New ID:");
            dump_eclass_exprs(&self.egraph, new_id);
          }
          cache.get_mut(&lhs).unwrap().insert(new_id);
          if j >= limit {
            break;
          }
          j += 1;
        }
      }
    }

    cache.get(&lhs).unwrap().clone()
  }

  fn add_instantiation_with_var_if_necessary(
    &mut self,
    rhs_ih: &Pattern<SymbolLang>,
    mut lhs_subst: Subst,
  ) -> Id {
    for var in rhs_ih.vars() {
      if lhs_subst.get(var).is_none() {
        let free_var = var.to_string().chars().skip(1).collect::<String>();
        let enode = SymbolLang::leaf(free_var);
        lhs_subst.insert(var, self.egraph.add(enode));
      }
    }
    self.egraph.add_instantiation(&rhs_ih.ast, &lhs_subst)
  }

  fn syntactic_decomp(&mut self) -> Result<Vec<(Goal<'a>, usize)>, Vec<(Goal<'a>, usize)>> {
    let lhs = self.egraph.find(self.eq.lhs.id);
    let rhs = self.egraph.find(self.eq.rhs.id);
    let mut cache = BTreeMap::new();
    let mut depths = BTreeMap::new();
    let aus = self
      .au_eclass(&mut cache, &mut depths, (lhs, rhs), 0)
      .into_iter()
      .filter(|au| match au {
        AU::Hole(_) => false,
        _ => true,
      })
      .fold(BTreeSet::new(), |acc, au| {
        acc.union(&au.extract_holes()).copied().collect()
      });
    let mut new_goals = vec![];
    if CONFIG.verbose {
      println!("=== syntactic_decomp ===");
    }
    if aus.is_empty() {
      return Err(new_goals);
    }
    if CONFIG.verbose {
      println!("Full expr: {}", self.full_expr);
      println!("Goal:");
      dump_eclass_exprs(&self.egraph, lhs);
      println!("=?=");
      dump_eclass_exprs(&self.egraph, rhs);
    }
    for au @ (au_lhs, au_rhs) in &aus {
      if let Some(true) = cvecs_equal(
        &self.egraph.analysis.cvec_analysis,
        &self.egraph[*au_lhs].data.cvec_data,
        &self.egraph[*au_rhs].data.cvec_data,
      ) {
        let depth = depths[au];
        if CONFIG.verbose {
          println!("New goal (at depth {}):", depth);
          dump_eclass_exprs(&self.egraph, *au_lhs);
          println!("=?=");
          dump_eclass_exprs(&self.egraph, *au_rhs);
        }
        let exprs = get_all_expressions(&self.egraph, vec![*au_lhs, *au_rhs]);
        let smallest_au_lhs_expr = get_smallest_expr(&exprs[&au_lhs]);
        let smallest_au_rhs_expr = get_smallest_expr(&exprs[&au_rhs]);
        let mut new_goal = self.clone();
        // TODO: Fix according to case_split
        new_goal.name = format!(
          "syntactic-decomp:{}={}",
          smallest_au_lhs_expr, smallest_au_rhs_expr
        );
        new_goal.full_expr = Equation::from_exprs(&smallest_au_lhs_expr, &smallest_au_rhs_expr);
        // new_goal.eq.lhs.sexp =
        //   symbolic_expressions::parser::parse_str(smallest_au_lhs_expr.to_string().as_str())
        //     .unwrap();
        // new_goal.eq.rhs.sexp =
        //   symbolic_expressions::parser::parse_str(smallest_au_rhs_expr.to_string().as_str())
        //     .unwrap();
        new_goal.eq.lhs.expr = smallest_au_lhs_expr;
        new_goal.eq.rhs.expr = smallest_au_rhs_expr;
        new_goal.eq.lhs.id = *au_lhs;
        new_goal.eq.rhs.id = *au_rhs;
        new_goals.push((new_goal, depth));
      } else {
        if CONFIG.verbose {
          println!("Skipping cvec(au_lhs) != cvec(au_rhs)");
        }
      }
    }
    if new_goals.len() == aus.len() {
      if CONFIG.verbose {
        println!("*** syntactic_decomp ***");
      }
      Ok(new_goals)
    } else {
      Err(new_goals)
    }
  }

  pub fn au_eclass(
    &self,
    cache: &mut BTreeMap<(Id, Id), BTreeSet<AU<Symbol>>>,
    depths: &mut BTreeMap<(Id, Id), usize>,
    state @ (c1, c2): (Id, Id),
    depth: usize,
  ) -> BTreeSet<AU<Symbol>> {
    if cache.contains_key(&state) {
      return cache[&state].clone();
    }
    cache.insert(state, BTreeSet::new());

    let mut aus = BTreeSet::new();
    for (n1, n2) in self.egraph[c1]
      .nodes
      .iter()
      .cartesian_product(&self.egraph[c2].nodes)
    {
      aus = aus
        .union(&self.au_enode(cache, depths, state, depth, n1, n2))
        .cloned()
        .collect();
    }

    *cache.get_mut(&state).unwrap() = aus.clone();
    aus
  }

  fn au_enode(
    &self,
    cache: &mut BTreeMap<(Id, Id), BTreeSet<AU<Symbol>>>,
    depths: &mut BTreeMap<(Id, Id), usize>,
    state: (Id, Id),
    depth: usize,
    n1: &SymbolLang,
    n2: &SymbolLang,
  ) -> BTreeSet<AU<Symbol>> {
    let mut aus = BTreeSet::new();

    if n1.op != n2.op || n1.children.len() != n2.children.len() {
      let au = AU::Hole(state);
      aus.insert(au);
      depths.insert(state, depth);
      return aus;
    }

    if n1.children.is_empty() {
      aus.insert(AU::Node(n1.op, vec![]));
    } else {
      let mut child_aus = vec![];
      for (&c1, &c2) in n1.children.iter().zip_eq(&n2.children) {
        child_aus.push(self.au_eclass(cache, depths, (c1, c2), depth + 1));
      }
      for child_aus_prod in child_aus.iter().multi_cartesian_product() {
        aus.insert(AU::Node(
          n1.op,
          child_aus_prod.into_iter().cloned().collect(),
        ));
      }
    }

    aus
  }

  fn semantic_decomp(&mut self, timer: &Timer) -> Option<Vec<Goal<'a>>> {
    if CONFIG.verbose {
      println!("=== semantic_decomp ===");
    }
    // Can be super slow:
    // self.egraph.analysis.cvec_analysis.saturate();
    let lemmas = self.get_descendent_pairs_with_matching_cvecs(timer);
    if lemmas.is_empty() {
      if CONFIG.verbose {
        println!("Failed");
        println!("*** semantic_decomp ***");
      }
      return None;
    }
    let mut new_goals = vec![];
    for (lhs_desc, rhs_desc) in lemmas {
      // TODO: May have to turn off:
      if timer.timeout() {
        if CONFIG.verbose {
          println!("*** semantic_decomp ***");
        }
        return Some(new_goals);
      }

      // get_all_expressions_with_loop fails to extract more often
      let exprs = get_all_expressions(&self.egraph, vec![lhs_desc, rhs_desc]);
      // println!(
      //   "exprs: {:#?}",
      //   exprs
      //     .values()
      //     .map(|v| v.iter().map(|e| e.to_string()).collect::<Vec<_>>())
      //     .collect::<Vec<_>>()
      // );

      let smallest_lhs_desc_expr = get_smallest_expr(&exprs[&lhs_desc]);
      let smallest_rhs_desc_expr = get_smallest_expr(&exprs[&rhs_desc]);
      let mut new_goal = self.clone();
      new_goal.name = format!(
        "semantic-decomp:{}={}",
        smallest_lhs_desc_expr, smallest_rhs_desc_expr
      );
      new_goal.full_expr = Equation::from_exprs(&smallest_lhs_desc_expr, &smallest_rhs_desc_expr);
      // new_goal.eq.lhs.sexp =
      //   symbolic_expressions::parser::parse_str(smallest_lhs_desc_expr.to_string().as_str())
      //     .unwrap();
      // new_goal.eq.rhs.sexp =
      //   symbolic_expressions::parser::parse_str(smallest_rhs_desc_expr.to_string().as_str())
      //     .unwrap();
      new_goal.eq.lhs.expr = smallest_lhs_desc_expr;
      new_goal.eq.rhs.expr = smallest_rhs_desc_expr;
      new_goal.eq.lhs.id = lhs_desc;
      new_goal.eq.rhs.id = rhs_desc;
      // TODO: More fields? (Reference case_split)
      if CONFIG.verbose {
        println!("New goal:");
        dump_eclass_exprs(&new_goal.egraph, new_goal.eq.lhs.id);
        println!("=?=");
        dump_eclass_exprs(&new_goal.egraph, new_goal.eq.rhs.id);
      }
      new_goals.push(new_goal);
      // new_goals.push(self.clone());
    }
    if CONFIG.verbose {
      println!("*** semantic_decomp ***");
    }
    Some(new_goals)
  }

  fn get_descendent_pairs_with_matching_cvecs(&mut self, timer: &Timer) -> Vec<(Id, Id)> {
    // if CONFIG.verbose {
    //   println!("=== get_descendent_pairs_with_matching_cvecs ===");
    // }
    let lhs = self.egraph.find(self.eq.lhs.id);
    let rhs = self.egraph.find(self.eq.rhs.id);
    let mut lemmas = vec![];
    // Relax the restriction that the outer e-classes are different to generate more lemmas:
    // if lhs == rhs {
    //   return lemmas;
    // }
    if let Some(true) = cvecs_equal(
      &self.egraph.analysis.cvec_analysis,
      &self.egraph[lhs].data.cvec_data,
      &self.egraph[rhs].data.cvec_data,
    ) {
      let mut lhs_descendents = BTreeSet::new();
      let mut rhs_descendents = BTreeSet::new();
      self.compute_descendents(lhs, &mut lhs_descendents);
      self.compute_descendents(rhs, &mut rhs_descendents);
      for (&lhs_desc, &rhs_desc) in lhs_descendents
        .iter()
        .cartesian_product(rhs_descendents.iter())
      {
        // TODO: May have to turn off:
        if timer.timeout() {
          if CONFIG.verbose {
            println!("*** get_descendent_pairs_with_matching_cvecs ***");
          }
          return lemmas;
        }
        if (lhs_desc != lhs && rhs_desc != rhs) && lhs_desc != rhs_desc {
          if let Some(true) = cvecs_equal(
            &self.egraph.analysis.cvec_analysis,
            &self.egraph[lhs_desc].data.cvec_data,
            &self.egraph[rhs_desc].data.cvec_data,
          ) {
            // lhs = lhs_desc_outer lhs_desc
            // rhs = rhs_desc_outer rhs_desc
            //   eclass(lhs_desc != rhs_desc)
            // but cvec(lhs_desc == rhs_desc)
            //   eclass(lhs_desc_outer != rhs_desc_outer)
            // but cvec(lhs_desc_outer == rhs_desc_outer)
            // if CONFIG.verbose {
            //   println!("Record descendent pair as inferred lemma:");
            //   dump_eclass_exprs(&self.egraph, lhs_desc);
            //   println!("=?=");
            //   dump_eclass_exprs(&self.egraph, rhs_desc);
            // }
            // lemmas.push((self.egraph.find(lhs_desc), self.egraph.find(rhs_desc)));
            lemmas.push((lhs_desc, rhs_desc));
          }
        }
      }
    }
    // if CONFIG.verbose {
    //   println!("*** get_descendent_pairs_with_matching_cvecs ***");
    // }
    lemmas
  }

  fn _replace_subexpr_with_fresh_var(&mut self, id: Id, id_desc: Id, egraph_size: usize) -> Id {
    if false && CONFIG.verbose {
      println!("=== replace_subexpr_with_fresh_var ===");
    }
    let expr_temp = self.egraph.id_to_expr(id_desc);
    let expr_ref = expr_temp.as_ref();
    let head_symbol = &expr_ref[expr_ref.len() - 1];
    if false && CONFIG.verbose {
      println!("descendent expr: {expr_temp}");
      println!("head symbol:      {head_symbol}");
      println!("descendent exprs:");
      dump_eclass_exprs(&self.egraph, id_desc);
    }
    let _type = self
      .global_search_state
      .context
      .get(&head_symbol.op)
      .or_else(|| self.local_context.get(&head_symbol.op))
      .and_then(|_t| Some(_t.args_ret().1));
    if let Some(t) = _type {
      if false && CONFIG.verbose {
        println!("Type: {t}");
      }
      let var_name = format!("{}_decomp_{}", t.to_string().to_lowercase(), egraph_size);
      if false && CONFIG.verbose {
        println!("Replace descendent with new var {var_name}");
      }
      let var_node = SymbolLang::leaf(var_name);
      let var_name_symbol = var_node.op.into();
      let var_id = self.egraph.add(var_node);
      // TODO: More fields? (Reference case_split)
      self.local_context.insert(var_name_symbol, t.clone());
      self.var_classes.insert(var_name_symbol, var_id);
      self.egraph.analysis.cvec_analysis.current_timestamp += 1;
      self.add_cvec_for_class(var_id, &t);
      self.egraph.analysis.cvec_analysis.saturate();
      self.egraph.rebuild();
      let mut cache = HashMap::new();
      let res = self.pattern_replace_in_eclass(
        id,
        &Pattern::from(&self.egraph.id_to_expr(id_desc)),
        vec![Pattern::from(&self.egraph.id_to_expr(var_id))],
        &mut cache,
        1,
      );
      if false && CONFIG.verbose {
        println!("After replacement: {res:?}");
        println!("After replacement (len: {}):", res.len());
        for v in &res {
          for &id in v {
            dump_eclass_exprs(&self.egraph, id);
          }
        }
      }

      // TODO
      if res[0].len() > 1 {
        // if res[0].len() > 0 {
        if false && CONFIG.verbose {
          println!("*** replace_subexpr_with_fresh_var ***");
        }
        return res[0][1];
        // return res[0][0];
      }
      // return id;
    } else {
      if false && CONFIG.verbose {
        println!("No type info");
      }
    }
    if false && CONFIG.verbose {
      println!("*** replace_subexpr_with_fresh_var ***");
    }
    id_desc
    // remember to apply rules to our newly created element, after this
  }

  // TODO: Timeout here?
  pub fn pattern_replace_in_eclass(
    &mut self,
    mut base_id: Id,
    pat_from: &Pattern<SymbolLang>,
    pat_to: Vec<Pattern<SymbolLang>>,
    cache: &mut HashMap<Id, Vec<Vec<Id>>>,
    inner_len: usize,
  ) -> Vec<Vec<Id>> {
    let n_pats = pat_to.len();
    base_id = self.egraph.find(base_id);
    if let Some(inner) = cache.get(&base_id) {
      return inner.clone();
    }
    let mut inner = vec![vec![]; inner_len];
    for i in 0..n_pats {
      inner[i].push(base_id);
    }
    cache.insert(base_id, inner);
    self.egraph.rebuild();
    if let Some(matches) = pat_from.search_eclass(&self.egraph, base_id) {
      for subst in matches.substs {
        let new_id_rep = self.add_instantiation_with_var_if_necessary(&pat_to[0], subst.clone());
        let should_add_rep =
          !cache.get(&base_id).unwrap()[0].contains(&self.egraph.find(new_id_rep)); // TODO fix maybe unsound
        if should_add_rep {
          for (i, p) in pat_to.clone().into_iter().enumerate() {
            let new_id = self.add_instantiation_with_var_if_necessary(&p, subst.clone());
            cache.get_mut(&base_id).unwrap()[i].push(self.egraph.find(new_id));
          }
        }
      }
    } else {
      for node in self.egraph[base_id].nodes.clone() {
        // heuristic that these are useless - how do we classify all useless ones?
        if (node.op.to_string() == "ite" || node.op.to_string() == "ite2")
          && (self.egraph.id_to_expr(node.children[0]).to_string() == "true"
            || self.egraph.id_to_expr(node.children[0]).to_string() == "false")
        {
          break;
        }
        let mut new_child_ids = vec![vec![vec![]]; n_pats];
        self.egraph.rebuild();
        for old_child_id in &node.children {
          let cur_ids = self.pattern_replace_in_eclass(
            *old_child_id,
            pat_from,
            pat_to.clone(),
            cache,
            inner_len,
          );
          for i in 0..n_pats {
            let mut temp = vec![];
            for id_list in &new_child_ids[i] {
              for id in &cur_ids[i] {
                let mut id_list_mod = id_list.clone();
                id_list_mod.push(id.clone());
                temp.push(id_list_mod);
              }
            }
            new_child_ids[i] = temp;
          }
        }
        let mut should_add_rep = vec![false; new_child_ids[0].len()];
        for i in 0..n_pats {
          let limit = 5;
          for (j, id_list) in new_child_ids[i].clone().into_iter().enumerate() {
            let new_id = self.egraph.add(SymbolLang::new(node.op, id_list));
            if i == 0 && !cache.get(&base_id).unwrap()[0].contains(&self.egraph.find(new_id)) {
              should_add_rep[j] = true;
            }
            if should_add_rep[j] {
              cache.get_mut(&base_id).unwrap()[i].push(self.egraph.find(new_id));
              if j >= limit {
                break;
              }
            }
          }
        }
      }
    }

    // self.egraph.rebuild();
    cache.get(&base_id).unwrap().clone()
  }

  /// Used for debugging.
  fn _print_lhs_rhs(&self) {
    let lhs_id = self.egraph.find(self.eq.lhs.id);
    let rhs_id = self.egraph.find(self.eq.rhs.id);

    let exprs = get_all_expressions(&self.egraph, vec![lhs_id, rhs_id]);

    println!("LHS Exprs:");
    for lhs_expr in exprs.get(&lhs_id).unwrap() {
      if CONFIG.irreducible_only && self.is_reducible(lhs_expr) {
        continue;
      }
      println!("{}", lhs_expr);
    }

    println!("RHS Exprs:");
    for rhs_expr in exprs.get(&rhs_id).unwrap() {
      if CONFIG.irreducible_only && self.is_reducible(rhs_expr) {
        continue;
      }
      println!("{}", rhs_expr);
    }
  }

  /// Poorly-named helper function for extract_generalized_expr. See that
  /// function for how it works.
  fn compute_parents(
    &self,
    class: Id,
    parents_map: &mut BTreeMap<Id, BTreeSet<(Id, usize)>>,
    seen: &mut BTreeSet<Id>,
  ) {
    if seen.contains(&class) {
      return;
    }
    seen.insert(class);
    for (i, node) in self.egraph[class].nodes.iter().enumerate() {
      for child in node.children() {
        parents_map
          .entry(*child)
          .and_modify(|e| {
            e.insert((class, i));
          })
          .or_insert(vec![(class, i)].into_iter().collect());
        self.compute_parents(*child, parents_map, seen);
      }
    }
  }

  /// Poorly-named helper function for extract_generalized_expr. See that
  /// function for how it works.
  fn all_parents(
    &self,
    start_class: Id,
    child_to_parent: &BTreeMap<Id, BTreeSet<(Id, usize)>>,
    parent_to_child_index: &mut BTreeMap<Id, usize>,
    seen: &mut BTreeSet<Id>,
  ) {
    if seen.contains(&start_class) {
      return;
    }
    seen.insert(start_class);

    if let Some(parents) = child_to_parent.get(&start_class) {
      parents
        .iter()
        .for_each(|(parent_class, parent_node_index)| {
          parent_to_child_index.insert(*parent_class, *parent_node_index);
          self.all_parents(*parent_class, child_to_parent, parent_to_child_index, seen);
        });
    }
  }

  fn extract_generalized_expr_helper(
    &self,
    gen_class: Id,
    gen_fresh_sym: Symbol,
    extract_class: Id,
    parent_to_child_index: &BTreeMap<Id, usize>,
    cache: &mut BTreeMap<Id, Option<Expr>>,
  ) -> Expr {
    // We handle cycles by using a cache. The cache contains an Option.
    match cache.get(&extract_class) {
      // If the Option is Some, that means we have successfully computed a value
      // and can reuse it.
      Some(Some(expr)) => {
        return expr.clone();
      }
      // If the Option is None, that means we have followed a cycle to this
      // class. If we contined trying to extract normally, we would infinitely
      // loop. So instead, we give up and ask it to do a regular extraction
      // using AstSize.
      Some(None) => {
        let extractor = Extractor::new(&self.egraph, AstSize);
        let expr = extractor.find_best(extract_class).1;
        cache.insert(extract_class, Some(expr.clone()));
        return expr;
      }
      _ => {}
    }

    // If this class can lead us to gen_class, it will be in
    // the parent_to_child_index map.
    parent_to_child_index
      .get(&extract_class)
      .map(|node_index| {
        // Get the node we need to follow.
        let node = &self.egraph[extract_class].nodes[*node_index];
        // Record that we have seen this class once so that we don't infinitely loop.
        cache.insert(extract_class, None);
        // Extract an expression for it.
        let expr = node.join_recexprs(|child_class| {
          self.extract_generalized_expr_helper(
            gen_class,
            gen_fresh_sym,
            child_class,
            parent_to_child_index,
            cache,
          )
        });
        cache.insert(extract_class, Some(expr.clone()));
        expr
        // If this class can't lead us to gen_class, we don't care
        // about it and we can just extract whatever.
      })
      .unwrap_or_else(|| {
        let extractor = Extractor::new(&self.egraph, AstSize);
        let expr = extractor.find_best(extract_class).1;
        cache.insert(extract_class, Some(expr.clone()));
        expr
      })
  }

  /// Extracts an expr from extract_class where all occurrences of gen_class are
  /// replaced by gen_fresh_sym. gen_class is assumed to be contained in the
  /// egraph rooted at extract_class.
  // NOTE (CK): This probably can be more effectively accomplished by using a
  // custom extractor that prioritizes the class we want to generalize - but we
  // would then need to figure out what parts of the returned expression
  // correspond to that class so we can generalize them.
  // TODO
  fn extract_generalized_expr(
    &self,
    gen_class: Id,
    gen_fresh_sym: Symbol,
    extract_class: Id,
  ) -> Expr {
    // println!("extracting generalized expr ({}) for {}", gen_class, extract_class);
    let mut parent_map = BTreeMap::default();
    let mut parents = BTreeSet::default();
    // Compute a map from each eclass to its parent enodes in the egraph rooted
    // at extract_class.
    self.compute_parents(extract_class, &mut parent_map, &mut parents);
    // println!("parent map: {:?}", parent_map);
    let mut parent_to_child_index = BTreeMap::default();

    // Computes a map from parent eclass to the index of the enode that will
    // lead the parent to gen_class. If there are multiple indices, one (I
    // believe the largest) is chosen arbitrarily.
    self.all_parents(
      gen_class,
      &parent_map,
      &mut parent_to_child_index,
      &mut BTreeSet::default(),
    );
    // println!("parent to child index: {:?}", parent_to_child_index);
    let mut cache = BTreeMap::default();
    cache.insert(
      gen_class,
      Some(vec![SymbolLang::leaf(gen_fresh_sym)].into()),
    );
    // FIXME: skip extraction if gen_class isn't contained in either the LHS and
    // RHS. I think it's fine to keep it as is for now because the generalized
    // lemma will just be the LHS = RHS and it will probably be rejected by our
    // lemma set since we seed it with LHS = RHS.
    self.extract_generalized_expr_helper(
      gen_class,
      gen_fresh_sym,
      extract_class,
      &parent_to_child_index,
      &mut cache,
    )
  }

  fn make_generalized_goal(&self, class: Id) -> Option<(Prop, Goal)> {
    // Get an op (function/constructor/var) that is a representative of this class.
    let class_op = self.egraph[class].data.canonical_form_data.get_enode().op;
    // HACK: Skip partial applications because we don't know how to find their type.
    if class_op == "$".into() {
      return None;
    }
    // NOTE We're assuming that we don't have to deal with higher-order
    // functions for generalizations, because we never will inspect a function's
    // value when pattern matching. However, a more correct analysis would take
    // into consideration how many arguments there are in the enode and from
    // those construct the appropriate (possibly higher-order) type.
    let (_, class_ty) = self
      .global_search_state
      .context
      .get(&class_op)
      .or_else(|| self.local_context.get(&class_op))
      .unwrap()
      .args_ret();
    // println!("generalizing {} with type {}", class_op, class_ty);
    let fresh_var = format!("fresh_{}_{}", self.name, self.egraph.total_size());
    let fresh_symb = Symbol::from(&fresh_var);
    let lhs_id = self.egraph.find(self.eq.lhs.id);
    let rhs_id = self.egraph.find(self.eq.rhs.id);

    let is_var = |v: &str| {
      self
        .local_context
        .contains_key(&Symbol::from_str(v).unwrap())
    };

    // println!("starting generalization {} {} {}", lhs_id, rhs_id, class);
    let lhs_expr = self.extract_generalized_expr(class, fresh_symb, lhs_id);
    let rhs_expr = self.extract_generalized_expr(class, fresh_symb, rhs_id);

    let params: Vec<(Symbol, Type)> = get_vars(&lhs_expr, is_var)
      .union(&get_vars(&rhs_expr, is_var))
      .flat_map(|var| {
        let var_ty_opt = if var == &fresh_symb {
          Some(class_ty.clone())
        } else if self.local_context.contains_key(var) {
          Some(self.local_context.get(var).unwrap().clone())
        } else {
          warn!("Leaf of term that isn't a known variable: {}", var);
          None
        };
        var_ty_opt.map(|var_ty| (*var, var_ty))
      })
      .collect();
    let eq = Equation::from_exprs(&lhs_expr, &rhs_expr);
    if sexp_size(&eq.lhs) > CONFIG.extraction_max_size
      || sexp_size(&eq.rhs) > CONFIG.extraction_max_size
    {
      return None;
    }

    let prop = Prop::new(eq.clone(), params.clone()).0;
    let mut new_goal = Goal::top(
      &format!("{}_gen", self.name),
      &prop,
      &None,
      self.global_search_state,
    );
    if new_goal.cvecs_valid() == Some(true) {
      // println!("generalizing {} === {}", lhs_expr, rhs_expr);
      Some((Prop::new(eq, params).0, new_goal))
    } else {
      // println!("cvecs disagree for {} === {}", lhs_expr, rhs_expr);
      None
    }
  }

  /// Return generalizations of the current goal found by generalizing a
  /// blocking_expr.
  fn find_generalized_goals(&self, blocking_exprs: &BTreeSet<Id>) -> Vec<Prop> {
    blocking_exprs
      .iter()
      .flat_map(|blocking_expr| {
        self
          .make_generalized_goal(*blocking_expr)
          .map(|(generalized_prop, _new_goal)| generalized_prop)
      })
      .collect()
  }

  /// Debug function to search for a pair of patterns in the e-graph
  fn _debug_search_for_patterns_in_egraph(&self) {
    self.global_search_state.searchers.iter().for_each(
      |searcher: &ConditionalSearcher<Pattern<SymbolLang>, Pattern<SymbolLang>>| {
        let results = searcher.search(&self.egraph);
        let extractor = Extractor::new(&self.egraph, AstSize);
        if !results.is_empty() {
          // println!(
          //   "Found search result for {} =?> {}",
          //   searcher.searcher, searcher.condition
          // );
          for result in results {
            println!("Result eclass: {}", result.eclass);
            result.substs.iter().for_each(|subst| {
              for var in searcher.searcher.vars().iter() {
                let exp = extractor.find_best(subst[*var]).1;
                println!("Var {} = {}", var, exp);
              }
            });
            let result_cvec = &self.egraph[result.eclass].data.cvec_data;
            for eclass in self.egraph.classes() {
              if eclass.id == result.eclass {
                continue;
              }
              if let Some(true) = cvecs_equal(
                &self.egraph.analysis.cvec_analysis,
                result_cvec,
                &self.egraph[eclass.id].data.cvec_data,
              ) {
                let exp = extractor.find_best(eclass.id).1;
                println!(
                  "Matching eclass via cvec analysis: {} (id {})",
                  exp, eclass.id
                );
              }
            }
          }
        };
      },
    );
  }

  /// Returns a vector of lemmas necessary to discharge this goal.
  pub fn find_lemmas_that_discharge(
    &self,
    lemmas_state: &LemmasState,
    lemma_rws: &Vec<Rw>,
  ) -> BTreeSet<usize> {
    let rewrites = self
      .global_search_state
      .reductions
      .iter()
      .chain(self.lemmas.values())
      .chain(lemmas_state.lemma_rewrites.values())
      .chain(lemma_rws.iter());
    let lhs_id = self.eq.lhs.id;
    let rhs_id = self.eq.rhs.id;
    let mut runner = Runner::default()
      .with_explanations_enabled()
      // We need to clone the egraph so as to not disturb it.
      .with_egraph(self.egraph.clone())
      .with_hook(move |runner| {
        // Stop iteration if we have proven lhs == rhs
        if runner.egraph.find(lhs_id) == runner.egraph.find(rhs_id) {
          Err("Goal proven".to_string())
        } else {
          Ok(())
        }
      })
      .run(rewrites);
    // None of these lemmas helped.
    if runner.egraph.find(lhs_id) != runner.egraph.find(rhs_id) {
      return BTreeSet::default();
    }
    let exp = runner
      .egraph
      .explain_equivalence(&self.eq.lhs.expr, &self.eq.rhs.expr);
    exp
      .explanation_trees
      .into_iter()
      .flat_map(|expl_tree| {
        expl_tree
          .backward_rule
          .or(expl_tree.forward_rule)
          .and_then(|rule| {
            let rule_str = rule.to_string();
            if let Some(rest) = rule_str.strip_prefix("lemma_") {
              rest
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .join("")
                .parse()
                .ok()
            } else {
              None
            }
          })
      })
      .collect()
  }
}

impl<'a> Display for Goal<'a> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if !self.premises.is_empty() {
      let premises_string = self
        .premises
        .iter()
        .map(|premise| format!("{}", premise))
        .collect::<Vec<String>>()
        .join(", ");
      write!(f, "{} ==> ", premises_string)?;
    }
    write!(f, "{}", self.eq)
  }
}

#[derive(Clone, Debug)]
pub enum ProofTerm {
  /// - Arg0: Name of the variable we split on
  /// - Arg1: List of cases we split on
  ///   * Arg0: Sexp we split to
  ///   * Arg1: Name of goal from this split
  ///
  /// Example:
  /// ```
  /// CaseSplit("x", [("(Z)", "goal_1"), ("(S x')","goal_2")])
  /// ```
  /// corresponds to the proof
  /// ```
  /// case x of
  ///   Z    -> goal_1
  ///   S x' -> goal_2
  /// ```
  CaseSplit(String, Vec<(String, String)>),
  /// The same thing as a case split, except instead of splitting on one of the
  /// symbolic variables, we split on an expression.
  ///
  /// - Arg0: A fresh variable introduced that is equal to the expression
  /// - Arg1: The expression we split on
  /// - Arg2: List of cases we split on (same as above).
  ///         There will always be two cases, corresponding to `True` and `False`.
  ///
  /// Example:
  /// ```
  /// ITESplit("g0", "(lt x y)", [("True", "goal_1"), ("False", "goal_2")])
  /// ```
  /// corresponds to the proof
  /// ```
  /// let g0 = lt x y in
  ///   case g0 of
  ///     True  -> goal_1
  ///     False -> goal_2
  /// ```
  ITESplit(String, String, Vec<(String, String)>),
}

pub enum ProofLeaf {
  /// Constructive equality shown: LHS = RHS
  Refl(Explanation<SymbolLang>),
  /// Proof by strong fertilization
  StrongFertilization(Option<Explanation<SymbolLang>>),
  // StrongFertilization(Explanation<SymbolLang>),
  /// Contradiction shown (e.g. True = False)
  Contradiction(Explanation<SymbolLang>),
  /// Unimplemented proof type (will crash on proof emission)
  Todo,
}

impl ProofLeaf {
  fn _name(&self) -> String {
    match &self {
      Self::Refl(_) => "Refl".to_string(),
      Self::StrongFertilization(_) => "Strong Fertilization".to_string(),
      Self::Contradiction(_) => "Contradiction".to_string(),
      Self::Todo => "TODO".to_string(),
    }
  }
}

impl std::fmt::Display for ProofLeaf {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      ProofLeaf::Refl(expl) => write!(f, "{}", expl.get_string()),
      ProofLeaf::StrongFertilization(expl) => {
        write!(
          f,
          "{}",
          expl
            .as_ref()
            .map(|expl| expl.get_string())
            .unwrap_or_else(|| panic!("Strong fertilization proof missing explanation"))
        )
      }
      ProofLeaf::Contradiction(expl) => write!(f, "{}", expl.get_string()),
      ProofLeaf::Todo => write!(f, "TODO: proof"),
    }
  }
}

#[derive(Default)]
pub struct LemmasState {
  pub proven_lemmas: MinElements<Prop>,
  pub invalid_lemmas: MaxElements<Prop>,
  pub lemma_rewrites: BTreeMap<String, Rw>,
  // FIXME: This is duplicated due to the type difference, in an ideal world we
  // wouldn't have this.
  pub lemma_rewrites_no_analysis: BTreeMap<String, Rewrite<SymbolLang, ()>>,
  // When we make a new state, this gets initialized to 0
  pub lemma_number: usize,
  /// (lemma number, proof depth)
  pub all_lemmas: HashMap<Prop, (usize, usize)>,
}

impl LemmasState {
  pub fn is_valid_new_prop(&self, prop: &Prop) -> bool {
    let is_proven = self.proven_lemmas.contains_leq(prop);
    let is_invalid = self.invalid_lemmas.contains_geq(prop);
    let is_too_big = CONFIG.max_lemma_size > 0
      && sexp_size(&prop.eq.lhs) + sexp_size(&prop.eq.rhs) > CONFIG.max_lemma_size;
    !is_proven && !is_invalid && !is_too_big
  }

  pub fn find_or_make_fresh_lemma(&mut self, prop: Prop, proof_depth: usize) -> usize {
    self
      .all_lemmas
      .entry(prop)
      .or_insert_with(|| {
        // This is duplicated from fresh_lemma but necessary for
        // the borrow checker.
        let number = self.lemma_number;
        self.lemma_number += 1;
        (number, proof_depth)
      })
      .0
  }

  pub fn fresh_lemma(&mut self) -> usize {
    let number = self.lemma_number;
    self.lemma_number += 1;
    number
  }

  pub fn add_lemmas<I: IntoIterator<Item = Prop>>(
    &mut self,
    iter: I,
    proof_depth: usize,
  ) -> Vec<(usize, Prop, bool)> {
    let mut new_lemmas = Vec::new();
    for lemma in iter.into_iter() {
      if self.is_valid_new_prop(&lemma) {
        let backup = lemma.clone();
        new_lemmas.push((
          self.find_or_make_fresh_lemma(lemma, proof_depth),
          backup,
          false,
        ));
      }
    }
    new_lemmas
  }
}

#[derive(Default)]
pub struct ProofInfo {
  pub solved_goal_proofs: BTreeMap<String, ProofLeaf>,
  pub proof: BTreeMap<String, ProofTerm>,
}

pub struct Timer {
  pub start_time: Instant,
}

impl Timer {
  fn new(start_time: Instant) -> Self {
    Self { start_time }
  }

  /// Has timeout been reached?
  pub fn timeout(&self) -> bool {
    CONFIG.timeout.map_or(false, |timeout| {
      self.start_time.elapsed() > Duration::new(timeout, 0)
    })
  }
}

pub struct ProofState<'a> {
  pub timer: Timer,
  pub lemmas_state: LemmasState,
  pub lemma_proofs: BTreeMap<usize, LemmaProofState<'a>>,
  pub global_search_state: GlobalSearchState<'a>,
}

pub struct LemmaProofState<'a> {
  pub prop: Prop,
  pub goals: VecDeque<Goal<'a>>,
  pub lemma_proof: ProofInfo,
  pub outcome: Option<Outcome>,
  pub proof_depth: usize,
  pub case_split_depth: usize,
  pub ih_lemma_number: usize,
  // NOTE: We are phasing this out at least for proving lemmas breadth-first
  pub theorized_lemmas: ChainSet<Prop>,
  // FIXME: Should not be an option - if we can't get any rewrites from a lemma
  // we shouldn't try to prove it
  pub rw: Option<LemmaRewrite<CycleggAnalysis>>,
  // FIXME: This is duplicated pretty much solely because
  // Goal::make_lemma_rewrite_unchecked requires access to the goal's local
  // context.
  pub rw_no_analysis: Option<LemmaRewrite<()>>,
}

fn get_lemma_name(lemma_id: usize) -> String {
  format!("lemma_{}", lemma_id)
}

impl<'a> LemmaProofState<'a> {
  pub fn new(
    lemma_number: usize,
    prop: Prop,
    premise: &Option<Equation>,
    global_search_state: GlobalSearchState<'a>,
    proof_depth: usize,
  ) -> Self {
    let lemma_name = get_lemma_name(lemma_number);
    let mut goal = Goal::top(&lemma_name, &prop, premise, global_search_state);
    let lemma_rw_opt =
      goal.make_lemma_rewrite_type_only(&goal.eq.lhs.expr, &goal.eq.rhs.expr, lemma_number, false);
    let lemma_rw_opt_no_analysis =
      goal.make_lemma_rewrite_unchecked(&goal.eq.lhs.expr, &goal.eq.rhs.expr, lemma_number, false);
    let mut outcome = goal.cvecs_valid().and_then(|is_valid| {
      // println!("{} cvec is valid = {}", lemma_name, is_valid);
      // FIXME: Handle premises in cvecs so that we can reject invalid props
      // with preconditions
      if premise.is_none() && !is_valid {
        Some(Outcome::Invalid)
      } else {
        None
      }
    });
    //* HACK: This means we don't have an IH. This lemma probably should not have
    //* been considered.

    if lemma_number == 0 {
      if outcome.is_none() {
        println!("Property accepted by cvec analysis");
      } else {
        println!("Property rejected by cvec analysis");
        print_cvec(
          &goal.egraph.analysis.cvec_analysis,
          &goal.egraph[goal.eq.lhs.id].data.cvec_data,
        );
        print_cvec(
          &goal.egraph.analysis.cvec_analysis,
          &goal.egraph[goal.eq.rhs.id].data.cvec_data,
        );
      }
    }

    if lemma_rw_opt.is_none() {
      outcome = Some(Outcome::Invalid);
    }
    Self {
      prop,
      goals: [goal].into(),
      lemma_proof: ProofInfo::default(),
      outcome,
      proof_depth,
      case_split_depth: 0,
      ih_lemma_number: lemma_number,
      theorized_lemmas: ChainSet::default(),
      rw: lemma_rw_opt,
      rw_no_analysis: lemma_rw_opt_no_analysis,
    }
  }

  pub fn add_to_theorized_lemmas<I: IntoIterator<Item = Prop>>(
    &mut self,
    iter: I,
    lemmas_state: &LemmasState,
  ) {
    self
      .theorized_lemmas
      .extend(iter.into_iter().filter(|lemma| {
        let is_valid = lemmas_state.is_valid_new_prop(lemma);
        if is_valid {
          // println!("adding lemma: forall {:?} {} == {}" , lemma.params, lemma.eq.lhs, lemma.eq.rhs);
        }
        is_valid
      }));
  }

  fn get_info_index(&mut self, info: &GoalInfo) -> usize {
    for (index, goal) in self.goals.iter().enumerate() {
      if goal.name == info.name {
        return index;
      }
    }
    panic!();
  }

  // three outputs
  // 1. if the goal is not proved, return it out
  // 2. the indices of all related lemmas
  // 3. the info of all related goals
  pub fn try_goal(
    &mut self,
    info: &GoalInfo,
    timer: &Timer,
    lemmas_state: &mut LemmasState,
  ) -> Option<(Vec<(usize, Prop, bool)>, Vec<GoalInfo>)> {
    let pos = self.get_info_index(info);
    let goal = self.goals.get_mut(pos).unwrap();

    // let temp_ih = goal.ih.clone();
    // if let Some(temp_ih) = temp_ih {
    //   println!("IH LHS: {}", temp_ih.0.searcher);
    //   println!("IH RHS: {}", temp_ih.1.searcher);
    // goal._print_lhs_rhs();
    // let goal_egraph_snapshot = goal.egraph.clone();
    // let goal_egraph_saturated = goal.saturate(&lemmas_state.lemma_rewrites);
    goal.egraph = goal.saturate(&lemmas_state.lemma_rewrites);

    // goal.egraph.analysis.cvec_analysis.saturate();
    // if let Some(true) = cvecs_equal(
    //   &goal.egraph.analysis.cvec_analysis,
    //   &goal.egraph[goal.egraph.find(goal.eq.lhs.id)].data.cvec_data,
    //   &goal.egraph[goal.egraph.find(goal.eq.lhs.id)].data.cvec_data,
    // ) {
    // } else {
    //   if CONFIG.verbose {
    //     println!("Cvecs disagree");
    //   }
    //   self.outcome = Some(Outcome::Invalid);
    //   return None;
    // }

    if CONFIG.save_graphs {
      goal.save_egraph();
    }
    // for _ in 0..2 {
    if let Some(proof_leaf) = goal.find_proof() {
      match proof_leaf {
        ProofLeaf::Todo => {
          if goal.premises.is_empty() {
            self.outcome = Some(Outcome::Invalid);
            // if CONFIG.verbose {
            //   println!("todo -> claimed invalid");
            // }
            return None;
          }
        }
        ProofLeaf::StrongFertilization(_) => {
          //* N.B. temporarily set the outcome of the _lemma_ this goal belongs to
          // We later reset the outcome to None
          self.outcome = Some(Outcome::StrongFertilization(info.clone()));
          return None;
        }
        _ => {
          self.process_goal_explanation(proof_leaf, &self.goals[pos].name.clone());
          // if CONFIG.verbose {
          //   println!("other -> claimed {:?}", self.outcome);
          // }
          return None;
        }
      }
    }
    // goal.egraph = goal_egraph_saturated.clone();
    // }
    if CONFIG.verbose {
      explain_goal_failure(goal);
    }

    /*let resolved_lhs_id = goal.egraph.find(goal.eq.lhs.id);
    let resolved_rhs_id = goal.egraph.find(goal.eq.rhs.id);
    let extractor = Extractor::new(&goal.egraph, AstSize);
    let (best_l_cost, best_l_prog) = extractor.find_best(resolved_lhs_id);
    let (best_r_cost, best_r_prog) = extractor.find_best(resolved_rhs_id);
    println!("cost pair {} {}", best_l_cost, best_l_prog);
    println!("cost pair {} {}", best_r_cost, best_r_prog);*/

    // TODO: Probably can be postponed as well
    goal.split_ite();

    let mut related_lemmas = Vec::new();

    // This ends up being really slow so we'll just take the lemma duplication for now
    // It's unclear that it lets us prove that much more anyway.
    // state.add_cyclic_lemmas(&goal);

    // goal.debug_search_for_patterns_in_egraph();

    // goal.egraph.rebuild();

    if CONFIG.ripple_mode {
      let mut goal = goal.clone();
      if let Some(ihs) = goal.ih.clone() {
        let lhs = goal.egraph.find(goal.eq.lhs.id);
        let rhs = goal.egraph.find(goal.eq.rhs.id);
        for (lhs_ih, rhs_ih) in ihs {
          let mut ih_replacements = vec![];
          let mut cache = HashMap::new();
          let mut weak_fert_lhs = goal.pattern_replace_in_eclass_with_analysis_help(
            &mut cache,
            &mut ih_replacements,
            lhs,
            &lhs_ih,
            &rhs_ih,
          );
          weak_fert_lhs.extend(goal.pattern_replace_in_eclass_with_analysis_help(
            &mut cache,
            &mut ih_replacements,
            lhs,
            &rhs_ih,
            &lhs_ih,
          ));
          weak_fert_lhs.retain(|&id| id != lhs);
          let mut weak_fert_rhs = goal.pattern_replace_in_eclass_with_analysis_help(
            &mut cache,
            &mut ih_replacements,
            rhs,
            &lhs_ih,
            &rhs_ih,
          );
          weak_fert_rhs.extend(goal.pattern_replace_in_eclass_with_analysis_help(
            &mut cache,
            &mut ih_replacements,
            rhs,
            &rhs_ih,
            &lhs_ih,
          ));
          weak_fert_rhs.retain(|&id| id != rhs);
          for (lhs, rhs) in weak_fert_lhs.iter().cartesian_product(&weak_fert_rhs) {
            let mut lemmas = vec![];
            let (_, rewrite_infos) = goal.make_lemma_rewrites_from_all_exprs(
              *lhs,
              *rhs,
              vec![],
              timer,
              lemmas_state,
              false,
              false,
              true,
            );
            let new_rewrite_eqs = rewrite_infos
              .into_iter()
              .map(|rw_info| (rw_info.lemma_prop, rw_info.renamed_params))
              .collect::<Vec<_>>();
            let fresh_name = format!("fresh_{}_{}", goal.name, goal.egraph.total_size());
            for (new_rewrite_eq, renamed_params) in &new_rewrite_eqs {
              let generalized_lemmas = find_generalizations_prop(
                new_rewrite_eq,
                goal.global_search_state.context,
                &goal.local_context,
                renamed_params,
                fresh_name.clone(),
              );
              lemmas.extend(generalized_lemmas);
            }
            lemmas.extend::<Vec<_>>(new_rewrite_eqs.into_iter().unzip::<_, _, _, Vec<_>>().0);
            // println!(
            //   "Weak fert lemmas: {:#?}",
            //   lemmas.iter().map(|l| l.to_string()).collect::<Vec<_>>()
            // );
            let lemma_indices = lemmas_state.add_lemmas(lemmas, self.proof_depth + 1);
            related_lemmas.extend(lemma_indices);
          }
        }
      }
    }

    let mut ripple_out_success = false;
    if CONFIG.ripple_mode {
      // goal.egraph = goal_egraph_snapshot;
      if let Some(new_goals) = goal.ripple_out() {
        ripple_out_success = true;
        for new_goal in new_goals {
          let mut lemmas = vec![];
          let (_, rewrite_infos) = new_goal.make_lemma_rewrites_from_all_exprs(
            new_goal.eq.lhs.id,
            new_goal.eq.rhs.id,
            vec![],
            timer,
            lemmas_state,
            false,
            false,
            true,
          );
          let new_rewrite_eqs = rewrite_infos
            .into_iter()
            .map(|rw_info| (rw_info.lemma_prop, rw_info.renamed_params))
            .collect::<Vec<_>>();
          let fresh_name = format!("fresh_{}_{}", new_goal.name, new_goal.egraph.total_size());
          for (new_rewrite_eq, renamed_params) in &new_rewrite_eqs {
            lemmas.extend(find_generalizations_prop(
              new_rewrite_eq,
              new_goal.global_search_state.context,
              &new_goal.local_context,
              renamed_params,
              fresh_name.clone(),
            ));
          }
          lemmas.extend::<Vec<_>>(new_rewrite_eqs.into_iter().unzip::<_, _, _, Vec<_>>().0);
          let lemma_indices = lemmas_state.add_lemmas(lemmas, self.proof_depth + 1);
          related_lemmas.extend(lemma_indices);
        }
      }
      // goal.egraph = goal_egraph_saturated;
    }

    if CONFIG.ripple_mode {
      match goal.syntactic_decomp() {
        // All anti-unified holes pass cvec analysis, syntactic decomposition succeeds
        Ok(new_goals) => {
          // for new_goal in &new_goals {
          //   let (_, rewrite_infos) = new_goal.make_lemma_rewrites_from_all_exprs(
          //     new_goal.eq.lhs.id,
          //     new_goal.eq.rhs.id,
          //     vec![],
          //     timer,
          //     lemmas_state,
          //     false,
          //     false,
          //     // TODO: false?
          //     true,
          //   );
          //   let new_rewrite_eqs = rewrite_infos
          //     .into_iter()
          //     .map(|rw_info| rw_info.lemma_prop.clone())
          //     .collect::<Vec<_>>();
          //   let lemma_indices = lemmas_state.add_lemmas(new_rewrite_eqs, self.proof_depth + 1);
          //   related_lemmas.extend(lemma_indices);
          // }

          let new_goal_infos = new_goals
            .iter()
            // Treat syntactically decomposed lemmas as if they are case-split goals
            .map(|(new_goal, depth)| GoalInfo::new_au(new_goal, info.lemma_id, *depth))
            .collect();
          self
            .goals
            .extend::<Vec<_>>(new_goals.into_iter().unzip::<_, _, _, Vec<_>>().0);
          // Don't bother case splitting when we've simplified the goal via syntactic decomposition
          return Some((related_lemmas, new_goal_infos));
        }
        // There is at least one lemma from anti-unification that fails cvec analysis, decomposition fails
        Err(_new_goals) => {
          if CONFIG.verbose {
            println!("Failed");
            println!("*** syntactic_decomp ***");
          }
        }
      }
    }

    if CONFIG.ripple_mode {
      if let Some(new_goals) = goal.semantic_decomp(timer) {
        for new_goal in new_goals {
          let mut lemmas = vec![];
          let (_rewrites, rewrite_infos) = new_goal.make_lemma_rewrites_from_all_exprs(
            new_goal.eq.lhs.id,
            new_goal.eq.rhs.id,
            new_goal.premises.clone(),
            timer,
            lemmas_state,
            false,
            false,
            true,
          );
          // TODO: Also generalize here?
          let new_rewrite_eqs = rewrite_infos
            .into_iter()
            .map(|rw_info| (rw_info.lemma_prop, rw_info.renamed_params))
            .collect::<Vec<_>>();
          let fresh_name = format!("fresh_{}_{}", new_goal.name, new_goal.egraph.total_size());
          for (new_rewrite_eq, renamed_params) in &new_rewrite_eqs {
            lemmas.extend(find_generalizations_prop(
              new_rewrite_eq,
              new_goal.global_search_state.context,
              &new_goal.local_context,
              renamed_params,
              fresh_name.clone(),
            ));
          }
          lemmas.extend::<Vec<_>>(new_rewrite_eqs.into_iter().unzip::<_, _, _, Vec<_>>().0);
          let lemma_indices = lemmas_state.add_lemmas(lemmas, self.proof_depth + 1);
          related_lemmas.extend(lemma_indices);
        }
      }
    }

    if goal.scrutinees.is_empty() {
      self.outcome = Some(Outcome::Invalid);
      // if CONFIG.verbose {
      //   println!("claimed invalid");
      // }
      return None;
    }
    let (blocking_vars, blocking_exprs) = if !CONFIG.blocking_vars_analysis {
      warn!("Blocking var analysis is disabled");
      (
        goal.scrutinees.iter().map(|s| s.name).collect(),
        BTreeSet::default(),
      )
    } else {
      let (blocking_vars, blocking_exprs) = goal.find_blocking(timer);
      if CONFIG.verbose {
        println!("blocking vars: {:?}", blocking_vars);
      }
      (blocking_vars, blocking_exprs)
    };

    if CONFIG.generalization {
      let lemma_indices = lemmas_state.add_lemmas(
        goal.find_generalized_goals(&blocking_exprs),
        self.proof_depth + 1,
      );
      // println!(
      //   "Generalized lemmas: {:?}",
      //   lemma_indices
      //     .iter()
      //     .map(|(_, prop, _)| prop.to_string())
      //     .collect::<Vec<_>>()
      // );
      related_lemmas.extend(lemma_indices);
    }

    // TODO
    if true && CONFIG.ripple_mode {
      let mut lemmas = vec![];
      let (_, rewrite_infos) = goal.make_lemma_rewrites_from_all_exprs(
        goal.eq.lhs.id,
        goal.eq.rhs.id,
        vec![],
        timer,
        lemmas_state,
        false,
        false,
        true,
      );
      let new_rewrite_eqs = rewrite_infos
        .into_iter()
        .map(|rw_info| (rw_info.lemma_prop, rw_info.renamed_params))
        .collect::<Vec<_>>();

      let fresh_name = format!("fresh_{}_{}", goal.name, goal.egraph.total_size());
      for (new_rewrite_eq, renamed_params) in &new_rewrite_eqs {
        lemmas.extend(find_generalizations_prop(
          new_rewrite_eq,
          goal.global_search_state.context,
          &goal.local_context,
          renamed_params,
          fresh_name.clone(),
        ));
      }
      if lemmas.len() > 1 {
        let lemma_indices = lemmas_state.add_lemmas(lemmas, self.proof_depth + 1);
        related_lemmas.extend(lemma_indices);
      }
    }

    if (!CONFIG.ripple_mode && CONFIG.cc_lemmas)
      || (CONFIG.ripple_mode && CONFIG.fallback_mode && !ripple_out_success)
    {
      let possible_lemmas = goal.search_for_cc_lemmas(timer, lemmas_state);
      let lemma_indices = lemmas_state.add_lemmas(possible_lemmas, self.proof_depth + 1);
      related_lemmas.extend(lemma_indices);
    }

    // println!("Goal scrutinees: {:?}", goal.scrutinees);
    if let Some(scrutinee) = goal.next_scrutinee(blocking_vars) {
      if CONFIG.verbose {
        println!(
          "{}: {}",
          "Case splitting and continuing".purple(),
          scrutinee.name.to_string().purple()
        );
      }
      let goal_name = goal.name.clone();
      let (proof_term, goals) =
        goal
          .clone()
          .case_split(scrutinee, timer, lemmas_state, self.ih_lemma_number);
      // This goal is now an internal node in the proof tree.
      self.lemma_proof.proof.insert(goal_name, proof_term);
      // Add the new goals to the back of the VecDeque.
      let goal_infos = goals
        .iter()
        //* Assign the same lemma ID to all branches of the case split
        .map(|new_goal| GoalInfo::new(new_goal, info.lemma_id))
        .collect();
      self.goals.extend(goals);
      Some((related_lemmas, goal_infos))
    } else {
      if CONFIG.verbose {
        println!(
          "{}",
          "Could not case split: no blocking variables found".red()
        );
        // for remaining_goal in &self.goals {
        //   println!("{} {}", "Remaining case".yellow(), remaining_goal.name);
        // }
      }
      // FIXME: Why?
      self.outcome = Some(Outcome::Invalid);
      None
    }
  }

  pub fn try_finish(&mut self, info: &GoalInfo, lemmas_state: &mut LemmasState) -> bool {
    let pos = self.get_info_index(info);
    let goal = self.goals.get_mut(pos).unwrap();
    goal.egraph = goal.saturate(&lemmas_state.lemma_rewrites);
    if let Some(leaf) = goal.find_proof() {
      let name = goal.name.clone();
      // UNCOMMENT
      // if pos != 0 {
      //   if CONFIG.verbose {
      //     println!("Process goal expl for {}", name);
      //   }
      self.process_goal_explanation(leaf, &name);
      // }
      true
    } else {
      false
    }
  }

  pub fn extract_lemmas(
    &mut self,
    info: &GoalInfo,
    timer: &Timer,
    lemmas_state: &mut LemmasState,
  ) -> Vec<(usize, Prop, bool)> {
    if !CONFIG.cc_lemmas {
      return vec![];
    }
    let pos = self.get_info_index(info);
    let goal = self.goals.get_mut(pos).unwrap();
    let possible_lemmas = goal.search_for_cc_lemmas(timer, lemmas_state);

    lemmas_state.add_lemmas(possible_lemmas, self.proof_depth + 1)
  }

  fn process_goal_explanation(&mut self, proof_leaf: ProofLeaf, goal_name: &str) {
    // This goal has been discharged, proceed to the next goal
    self
      .lemma_proof
      .solved_goal_proofs
      .insert(goal_name.to_string(), proof_leaf);
  }
}

impl<'a> ProofState<'a> {
  fn handle_lemma_outcome(lemmas_state: &mut LemmasState, lemma_proof_state: &mut LemmaProofState) {
    //println!("handle outcomes for {:?}", lemma_proof_state.outcome);
    //println!("  {:?}", lemmas_state.lemma_rewrites_no_analysis);
    if lemma_proof_state.outcome.is_none() {
      return;
    }
    match lemma_proof_state.outcome.as_ref().unwrap() {
      Outcome::Valid => {
        //println!("new lemma: {}", lemma_proof_state.prop);
        lemmas_state
          .proven_lemmas
          .insert(lemma_proof_state.prop.clone());
        if let Some(rw) = lemma_proof_state.rw.as_ref() {
          if CONFIG.verbose {
            if let Some(rw) = rw.lhs_to_rhs.as_ref() {
              println!("Adding rewrite rule: {}", rw.0);
            }
            if let Some(rw) = rw.rhs_to_lhs.as_ref() {
              println!("Adding rewrite rule: {}", rw.0);
            }
          }
          rw.add_to_rewrites(&mut lemmas_state.lemma_rewrites)
        }
        if let Some(rw) = lemma_proof_state.rw_no_analysis.as_ref() {
          rw.add_to_rewrites(&mut lemmas_state.lemma_rewrites_no_analysis)
        }
      }
      Outcome::Invalid => {
        lemmas_state
          .invalid_lemmas
          .insert(lemma_proof_state.prop.clone());
      }
      Outcome::StrongFertilization(_) => {
        unreachable!();
        // let pos = lemma_proof_state.get_info_index(&info);
        // let proven_goal = lemma_proof_state.goals.get_mut(pos).unwrap();
        // let proven_goal_name = info.name.clone();
        // proven_goal.saturate(&lemmas_state.lemma_rewrites);
        // let expl = proven_goal
        //   .egraph
        //   .explain_equivalence(&proven_goal.eq.lhs.expr, &proven_goal.eq.rhs.expr);
        // let proof_leaf = ProofLeaf::StrongFertilization(Some(expl));
        // println!("Process goal expl for {}", proven_goal_name);
        // lemma_proof_state.process_goal_explanation(proof_leaf, &proven_goal_name);
      }
      _ => {}
    }
  }

  fn prove_breadth_first(
    &mut self,
    top_level_lemma_number: usize,
    scheduler: &mut impl BreadthFirstScheduler,
  ) -> Outcome {
    if CONFIG.verbose {
      println!("=== prove_breadth_first ===");
    }
    let mut _i = 0;
    let mut visited_lemma = HashSet::new();
    loop {
      _i += 1;
      // Stop if the top-level lemma is proved.
      let top_level_lemma = self.lemma_proofs.get(&top_level_lemma_number).unwrap();
      // println!("top outcome {:?}", top_level_lemma.outcome);
      if top_level_lemma.outcome == Some(Outcome::Valid)
        || top_level_lemma.outcome == Some(Outcome::Invalid)
      {
        return top_level_lemma.outcome.as_ref().unwrap().clone();
      }
      let lemma_batch_res = scheduler.next_lemma_batch(top_level_lemma_number, self);
      match lemma_batch_res {
        Err(outcome) => {
          if CONFIG.verbose {
            println!("*** prove_breadth_first ***");
          }
          return outcome;
        }
        Ok(_) => {}
      }
      if CONFIG.verbose {
        println!("go over each lemma in batch");
      }
      let lemma_index = lemma_batch_res.unwrap();
      if self.timer.timeout() {
        if CONFIG.verbose {
          println!("*** prove_breadth_first ***");
        }
        return Outcome::Timeout;
      }
      let lemma_number = scheduler.get_lemma_number(&lemma_index);

      if let Some(lemma_proof_state) = self.lemma_proofs.get(&lemma_number) {
        //println!("info {} {:?}", lemma_proof_state.prop, lemma_proof_state.outcome);
        // This lemma has been declared valid/invalid
        if lemma_proof_state.outcome.is_some()
          && lemma_proof_state.outcome != Some(Outcome::Unknown)
        {
          panic!();
        }
        // Check that there isn't a valid/invalid lemma that subsumes/is
        // subsumed by this lemma.
        if !self.lemmas_state.is_valid_new_prop(&lemma_proof_state.prop)
          && !visited_lemma.contains(&lemma_number)
        {
          panic!();
        }
      }
      visited_lemma.insert(lemma_number);
      scheduler.handle_lemma(lemma_index, self);
      // Clean up after the scheduler handles the lemma.
      let lemma_proof_state = self.lemma_proofs.get_mut(&lemma_number).unwrap();
      ProofState::handle_lemma_outcome(&mut self.lemmas_state, lemma_proof_state);
      // Check for a definite result if this is the top level lemma.
      if lemma_number == top_level_lemma_number && lemma_proof_state.outcome.is_some() {
        match lemma_proof_state.outcome.as_ref().unwrap() {
          Outcome::Valid | Outcome::Invalid | Outcome::Timeout => {
            if CONFIG.verbose {
              println!("*** prove_breadth_first ***");
            }
            return lemma_proof_state.outcome.as_ref().unwrap().clone();
          }
          Outcome::StrongFertilization(_) => {
            unreachable!();
          }
          _ => {}
        }
      }
      match &lemma_proof_state.outcome {
        Some(Outcome::Valid) => {
          scheduler.on_proven_lemma(lemma_number, self);
        }
        _ => {}
      }
    }
  }
}

trait BreadthFirstScheduler {
  /// How you index lemmas. Typically this should just be their number (a
  /// usize).
  type LemmaIndex;

  /// Gives the next batch of lemmas to try and prove
  fn next_lemma_batch(
    &mut self,
    top_level_lemma_number: usize,
    proof_state: &mut ProofState<'_>,
  ) -> Result<Self::LemmaIndex, Outcome>;

  fn get_lemma_number(&self, lemma_index: &Self::LemmaIndex) -> usize;

  /// What we do when trying to prove each lemma.
  ///
  /// There is boilerplate that is taken care of before (checking the timer,
  /// skipping invalid or proven lemmas) and after (updating the lemma state
  /// based on the outcome).
  fn handle_lemma(&mut self, lemma_index: Self::LemmaIndex, proof_state: &mut ProofState<'_>);

  /// A hook that is called whenever a lemma is proven. Theoretically you could
  /// have this logic be in handle_lemma instead.
  fn on_proven_lemma(&mut self, lemma: usize, proof_state: &mut ProofState<'_>);
}

#[derive(Default)]
struct GoalLevelPriorityQueue {
  goal_graph: GoalGraph,
  next_goal: Option<GoalInfo>,
  prop_map: HashMap<usize, Prop>,
  progress_set: HashSet<usize>,
  is_found_new_lemma: bool,
}

impl GoalLevelPriorityQueue {
  fn add_lemmas_for(
    &mut self,
    info: &GoalInfo,
    lemmas: Vec<(usize, Prop, bool)>,
    proof_state: &mut ProofState<'_>,
  ) {
    if lemmas.is_empty() {
      return;
    }
    let mut related_lemma_root = Vec::new();
    //println!("found lemmas for {}", info.full_exp);
    for (index, prop, _from_rippling) in lemmas {
      //println!("  {}", prop);
      if proof_state.timer.timeout() {
        return;
      }

      /*let lemma_state = proof_state.lemma_proofs.entry(index).or_insert_with(|| {
        LemmaProofState::new(index, prop, &None, proof_state.global_search_state, 0)
      });
      if lemma_state.outcome.is_some() {
        assert_eq!(lemma_state.outcome, Some(Outcome::Invalid));
        continue;
      }
      related_lemma_root.push(GoalInfo::new(&lemma_state.goals[0], index))*/
      self.prop_map.entry(index).or_insert(prop.clone());
      let start_info = GoalInfo {
        name: get_lemma_name(index),
        lemma_id: index,
        full_exp: prop.eq.to_string(),
        size: prop.size(),
        au_depth: None,
      };
      related_lemma_root.push((start_info, prop));
    }
    self
      .goal_graph
      .record_related_lemmas(info, &related_lemma_root);
  }

  fn insert_waiting(
    &mut self,
    info: &GoalInfo,
    related_lemmas: Vec<(usize, Prop, bool)>,
    related_goals: Vec<GoalInfo>,
    proof_state: &mut ProofState<'_>,
  ) {
    self.add_lemmas_for(info, related_lemmas, proof_state);
    self.goal_graph.record_case_split(info, &related_goals);
  }

  fn update_subsumed_lemmas(&mut self, proof_state: &mut ProofState<'_>) {
    let active_lemmas = self.goal_graph.send_subsumed_check();
    let subsumed_lemmas = active_lemmas
      .into_iter()
      .filter({
        |lemma| {
          !proof_state
            .lemmas_state
            .is_valid_new_prop(&self.prop_map[lemma])
        }
      })
      .collect();
    self.goal_graph.receive_subsumed_check(subsumed_lemmas);
  }
}

impl BreadthFirstScheduler for GoalLevelPriorityQueue {
  /// This is just the lemma number
  type LemmaIndex = usize;

  fn next_lemma_batch(
    &mut self,
    _top_level_lemma_number: usize,
    proof_state: &mut ProofState<'_>,
  ) -> Result<Self::LemmaIndex, Outcome> {
    if CONFIG.verbose {
      println!("=== next_lemma_batch ===");
    }
    // No more lemmas to try and prove
    if self.is_found_new_lemma {
      self.update_subsumed_lemmas(proof_state);
    }
    self.is_found_new_lemma = true;
    let mut frontier = self.goal_graph.get_frontier_goals();
    if CONFIG.verbose {
      println!("\n\n================= current queue ==============");
      for info in frontier.iter() {
        println!("[{}] {}", info.size, info.full_exp);
        println!("  ({}) {}", info.lemma_id, self.prop_map[&info.lemma_id]);
      }
      println!("Progress set: {:?}", self.progress_set);
      println!("\n");
    }
    if frontier
      .iter()
      .any(|info| self.progress_set.contains(&info.lemma_id))
    {
      frontier.retain(|info| self.progress_set.contains(&info.lemma_id));
    }
    if let Some(optimal) = frontier.iter().min_by_key(|info| info.size) {
      if CONFIG.verbose {
        println!("select min-size goal: {optimal:#?}");
      }
      self.next_goal = Some(optimal.clone());
      if self.progress_set.contains(&optimal.lemma_id) {
        self.progress_set.remove(&optimal.lemma_id);
      }
      if CONFIG.verbose {
        println!("*** next_lemma_batch ***");
      }
      Ok(optimal.lemma_id)
    } else {
      if CONFIG.verbose {
        println!("report unknown because of an empty queue");
        println!("*** next_lemma_batch ***");
      }
      Err(Outcome::Unknown)
    }
  }

  fn get_lemma_number(&self, lemma_index: &Self::LemmaIndex) -> usize {
    *lemma_index
  }

  fn handle_lemma(&mut self, lemma_index: Self::LemmaIndex, proof_state: &mut ProofState<'_>) {
    if CONFIG.verbose {
      println!("=== handle_lemma ===");
    }
    assert!(self.next_goal.is_some());
    let info = self.next_goal.clone().unwrap();
    assert_eq!(info.lemma_id, lemma_index);

    if let std::collections::btree_map::Entry::Vacant(e) =
      proof_state.lemma_proofs.entry(lemma_index)
    {
      let prop = self.prop_map.get(&lemma_index).unwrap().clone();
      e.insert(LemmaProofState::new(
        lemma_index,
        prop,
        &None,
        proof_state.global_search_state,
        0,
      ));
    }

    if CONFIG.verbose {
      println!(
        "\nTry goal {} from {}",
        info.full_exp, self.prop_map[&info.lemma_id]
      );
    }
    let lemma_state = proof_state.lemma_proofs.get_mut(&lemma_index).unwrap();
    if lemma_state.outcome.is_some() {
      assert_eq!(lemma_state.outcome, Some(Outcome::Invalid));
      self
        .goal_graph
        .set_lemma_res(info.lemma_id, GraphProveStatus::Invalid);
      return;
    }

    let lemma_proof_state = proof_state.lemma_proofs.get_mut(&lemma_index).unwrap();
    // println!("\ntry goal {} from {} {}", info.full_exp, self.prop_map[&info.lemma_id], lemma_proof_state.case_split_depth);

    let step_res =
      lemma_proof_state.try_goal(&info, &proof_state.timer, &mut proof_state.lemmas_state);

    if let Some((raw_related_lemmas, related_goals)) = step_res {
      let mut related_lemmas = raw_related_lemmas;
      if CONFIG.exclude_bid_reachable {
        let _pre_size = related_lemmas.len();
        related_lemmas = self
          .goal_graph
          .exclude_bid_reachable_lemmas(&related_lemmas);
        /*if pre_size > related_lemmas.len() {
          println!("Reduce from {} to {}", pre_size, related_lemmas.len());
        }*/
      }

      if CONFIG.verbose {
        println!("\nFrom {}", info.full_exp);
        println!("(Lemma) {}", self.prop_map[&info.lemma_id]);
        println!(
          "  lemmas: {}, goals: {}",
          related_lemmas.len(),
          related_goals.len()
        );
        for (_, lemma, _) in related_lemmas.iter() {
          println!(
            "  [{}] {}",
            sexp_size(&lemma.eq.lhs) + sexp_size(&lemma.eq.rhs),
            lemma
          );
        }
      }
      self.insert_waiting(&info, related_lemmas, related_goals, proof_state);
    } else {
      if lemma_proof_state.outcome.is_none() {
        self
          .goal_graph
          .record_node_status(&info, GraphProveStatus::Valid);
        if !self.goal_graph.is_lemma_proved(info.lemma_id) {
          self.progress_set.insert(info.lemma_id);
        }
      } else if let Some(Outcome::StrongFertilization(_)) = lemma_proof_state.outcome {
        //* Reset lemma outcome to None
        lemma_proof_state.outcome = None;
        self
          .goal_graph
          .record_node_status(&info, GraphProveStatus::Valid);
        if !self.goal_graph.is_lemma_proved(info.lemma_id) {
          self.progress_set.insert(info.lemma_id);
        }
      } else if let Some(Outcome::Invalid) = lemma_proof_state.outcome {
        self
          .goal_graph
          .record_node_status(&info, GraphProveStatus::Invalid);
      }
      if self.goal_graph.is_lemma_proved(info.lemma_id)
        && (!CONFIG.reduce_proven_lemma || !self.goal_graph.is_root(&info) || info.lemma_id == 0)
      {
        let state = proof_state.lemma_proofs.get_mut(&info.lemma_id).unwrap();
        state.outcome = Some(Outcome::Valid);
        println!("proved lemma {} {}", state.prop, info.full_exp);
        // println!(
        //   "reason: {}",
        //   state
        //     .lemma_proof
        //     .solved_goal_proofs
        //     .get(&info.name)
        //     .unwrap()
        // );

        if CONFIG.exclude_bid_reachable {
          state.rw_no_analysis.clone().map(|rw| {
            if rw.lhs_to_rhs.is_some() && rw.rhs_to_lhs.is_some() {
              self
                .goal_graph
                .add_bid_rewrites(rw.lhs_to_rhs.unwrap().1, rw.rhs_to_lhs.unwrap().1);
            }
          });
        }
      }
      if let Some(outcome) = proof_state
        .lemma_proofs
        .get(&info.lemma_id)
        .unwrap()
        .outcome
        .clone()
      {
        self.is_found_new_lemma = true;
        match outcome {
          Outcome::Valid => {
            self
              .goal_graph
              .set_lemma_res(info.lemma_id, GraphProveStatus::Valid);
          }
          Outcome::Invalid => {
            self
              .goal_graph
              .set_lemma_res(info.lemma_id, GraphProveStatus::Invalid);
          }
          Outcome::StrongFertilization(_) => {
            unreachable!();
          }
          _ => {
            panic!();
          }
        }
      }
    }
    if CONFIG.verbose {
      println!("*** handle_lemma ***");
    }
  }

  fn on_proven_lemma(&mut self, lemma: usize, proof_state: &mut ProofState<'_>) {
    if CONFIG.verbose {
      println!("=== on_proven_lemma ===");
    }
    let mut new_lemma = HashSet::new();
    new_lemma.insert(lemma);

    while !new_lemma.is_empty() {
      // update those subsumed lemmas
      self.update_subsumed_lemmas(proof_state);

      let proved_goals: Vec<_> = self
        .goal_graph
        .get_waiting_goals(Some(&new_lemma))
        .into_iter()
        .filter(|info| {
          let state = proof_state.lemma_proofs.get_mut(&info.lemma_id).unwrap();
          if CONFIG.verbose {
            println!("try finish goal: {:#?}", info);
          }
          state.try_finish(info, &mut proof_state.lemmas_state)
        })
        .collect();

      new_lemma.clear();

      let mut directly_improved_lemmas = HashSet::new();
      if CONFIG.reduce_proven_lemma {
        for goal in proved_goals.iter() {
          if self.goal_graph.is_root(goal) && goal.lemma_id != 0 {
            directly_improved_lemmas.insert(goal.lemma_id);
          }
        }
      }

      for goal in proved_goals.into_iter() {
        // println!("  retry and prove ({}) {}", goal.lemma_id, goal.full_exp);
        self
          .goal_graph
          .record_node_status(&goal, GraphProveStatus::Valid);
        if !self.goal_graph.is_lemma_proved(goal.lemma_id) {
          self.progress_set.insert(goal.lemma_id);
        }
        if self.goal_graph.is_lemma_proved(goal.lemma_id)
          && !directly_improved_lemmas.contains(&goal.lemma_id)
        {
          let lemma_state = proof_state.lemma_proofs.get_mut(&goal.lemma_id).unwrap();
          if lemma_state.outcome.is_some() {
            continue;
          }
          lemma_state.outcome = Some(Outcome::Valid);
          self
            .goal_graph
            .set_lemma_res(goal.lemma_id, GraphProveStatus::Valid);

          println!("proved lemma {}", lemma_state.prop);
          new_lemma.insert(goal.lemma_id);

          if CONFIG.exclude_bid_reachable {
            lemma_state.rw_no_analysis.clone().map(|rw| {
              if rw.lhs_to_rhs.is_some() && rw.rhs_to_lhs.is_some() {
                self
                  .goal_graph
                  .add_bid_rewrites(rw.lhs_to_rhs.unwrap().1, rw.rhs_to_lhs.unwrap().1);
              }
            });
          }

          ProofState::handle_lemma_outcome(&mut proof_state.lemmas_state, lemma_state);
          if goal.lemma_id == 0 {
            if CONFIG.verbose {
              println!("*** on_proven_lemma ***");
            }
            return;
          }
        }
      }
    }

    // self.re_extract_lemmas(proof_state);
    if CONFIG.verbose {
      println!("*** on_proven_lemma ***");
    }
  }
}

/// Pretty-printed proof state
pub fn pretty_state(state: &LemmaProofState) -> String {
  format!(
    "[{}]",
    state
      .goals
      .iter()
      .map(|g| g.name.clone())
      .collect::<Vec<String>>()
      .join(", ")
  )
}

/// Outcome of a proof attempt
#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone)]
pub enum Outcome {
  Valid,
  StrongFertilization(GoalInfo),
  Invalid,
  Unknown,
  Timeout,
}

impl Outcome {
  pub fn is_strong_fertilization(&self) -> bool {
    match self {
      Outcome::StrongFertilization(_) => true,
      _ => false,
    }
  }
}

impl std::fmt::Display for Outcome {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match *self {
      Outcome::Valid => write!(f, "{}", "VALID".green()),
      Outcome::StrongFertilization(_) => write!(f, "{}", "STRONG FERTILIZATION".green()),
      Outcome::Invalid => write!(f, "{}", "INVALID".red()),
      Outcome::Unknown => write!(f, "{}", "UNKNOWN".yellow()),
      Outcome::Timeout => write!(f, "{}", "TIMEOUT".yellow()),
    }
  }
}

pub fn explain_goal_failure(goal: &Goal) {
  println!("{} {}", "Could not prove".red(), goal.name);
}

pub fn prove_top(
  goal_prop: Prop,
  goal_premise: Option<Equation>,
  global_search_state: GlobalSearchState<'_>,
) -> (Outcome, ProofState) {
  if CONFIG.verbose {
    println!("=== prove_top ===");
    println!("goal prop: {goal_prop}");
  }

  let mut proof_state = ProofState {
    timer: Timer::new(Instant::now()),
    lemmas_state: LemmasState::default(),
    lemma_proofs: BTreeMap::default(),
    global_search_state,
  };

  let top_goal_lemma_number = proof_state
    .lemmas_state
    .find_or_make_fresh_lemma(goal_prop.clone(), 0);
  let top_goal_lemma_proof = LemmaProofState::new(
    top_goal_lemma_number,
    goal_prop,
    &goal_premise,
    global_search_state,
    0,
  );

  if CONFIG.verbose {
    for goal in &top_goal_lemma_proof.goals {
      println!("proof goal: {}", goal.eq);
    }
  }
  let start_info = GoalInfo::new(&top_goal_lemma_proof.goals[0], top_goal_lemma_number);
  if CONFIG.verbose {
    println!("start info: {:#?}", start_info);
  }
  let mut scheduler = GoalLevelPriorityQueue::default();
  scheduler.goal_graph.new_lemma(&start_info, None);
  scheduler
    .prop_map
    .insert(top_goal_lemma_number, top_goal_lemma_proof.prop.clone());

  proof_state
    .lemma_proofs
    .insert(top_goal_lemma_number, top_goal_lemma_proof);

  // let outcome = proof_state.prove_lemma(top_goal_lemma_number);
  // let outcome = proof_state.prove_breadth_first(top_goal_lemma_number, &mut LemmaSizePriorityQueue::default());
  let outcome = proof_state.prove_breadth_first(top_goal_lemma_number, &mut scheduler);
  if CONFIG.verbose {
    println!("outcome: {outcome}");
    println!("*** prove_top ***");
  }
  (outcome, proof_state)
}
