use crate::config::CONFIG;
use crate::utils::{cartesian_product, dump_eclass_exprs};
use egg::*;
use std::collections::HashMap;
use std::{
  collections::{BTreeMap, BTreeSet},
  iter::zip,
};

/// Denotation of an egraph (or its subgraph)
/// is a map from eclass ids to sets of expressions
type Denotation<L> = BTreeMap<Id, Vec<RecExpr<L>>>;

/// Compute the denotation of all roots in egraph, ignoring cycles
pub fn get_all_expressions<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  roots: Vec<Id>,
) -> Denotation<L> {
  let mut memo = BTreeMap::new();
  for root in roots {
    collect_expressions(egraph, root, &mut memo);
  }
  memo
}

/// Compute the denotation of eclass ignoring cycles and store it in memo
fn collect_expressions<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  eclass: Id,
  memo: &mut Denotation<L>,
) {
  if memo.get(&eclass).is_some() {
    // Already visited
  } else {
    // Initialize the memo entry for this eclass with an empty denotation,
    // collect denotations in a separate vector and update the map only at the end;
    // this guarantees that we are not following cycles
    memo.insert(eclass, vec![]);
    let mut denotations: Vec<RecExpr<L>> = vec![];
    // Join denotations of every node in the eclass
    for node in egraph[eclass].iter() {
      if node.is_leaf() {
        // If this node is a leaf, its denotation is itself
        let expr = RecExpr::from(vec![node.clone()]);
        denotations.push(expr);
      } else {
        // Otherwise, recursively collect the denotations of its children
        // and create a new expression from each tuple of their cross product.
        // Each products[i] stores the product of denotation sizes of all nodes from i+1 onwards
        let mut products: BTreeMap<Id, usize> = BTreeMap::new();
        for (i, c) in node.children().iter().enumerate() {
          collect_expressions(egraph, *c, memo);
          products.insert(*c, 1);
          for j in 0..i {
            products
              .entry(node.children()[j])
              .and_modify(|p| *p *= memo[c].len());
          }
        }
        // Now create the new expressions
        let c0 = &node.children()[0];
        // First compute the size of the cross product; we almost have it in products[c0]; just the size of c0's denotation is missing
        let cross_product_size = products[c0] * memo[c0].len();
        for k in 0..cross_product_size {
          // For the k-th element of the cross product, which element from the denotation of id should we take?
          // The formula is: k / (the product of all following denotation sizes) % this denotation size
          let lookup_id = |id: Id| k / products[&id] % memo[&id].len();
          let expr = node.join_recexprs(|id| memo.get(&id).unwrap()[lookup_id(id)].clone());
          denotations.push(expr);
        }
      }
    }
    memo.insert(eclass, denotations);
  }
}

#[derive(Copy, Clone)]
pub struct ExtractInfo {
  size: usize,
  loop_num: usize,
}

fn merge_infos(infos: &Vec<ExtractInfo>) -> ExtractInfo {
  let mut size = 1usize;
  let mut loop_num = 0usize;
  for info in infos.iter() {
    size += info.size;
    loop_num += info.loop_num;
  }
  ExtractInfo { size, loop_num }
}

fn is_valid_info(info: &ExtractInfo) -> bool {
  info.size <= CONFIG.extraction_max_size // && info.loop_num <= CONFIG.extraction_loop_limit
}

pub fn collect_expressions_with_loops_aux<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  id: Id,
  loop_num: usize,
  depth: usize,
  trace: &mut HashMap<Id, usize>,
) -> Vec<(ExtractInfo, RecExpr<L>)> {
  let mut res = Vec::new();
  if depth > CONFIG.extraction_max_depth {
    return res;
  }
  let class = &egraph[id];
  if loop_num > CONFIG.extraction_loop_limit {
    if CONFIG.extraction_allow_end_loop {
      for node in class.nodes.iter() {
        if node.children().is_empty() {
          let expr = RecExpr::from(vec![node.clone()]);
          res.push((
            ExtractInfo {
              size: 1usize,
              loop_num: 0usize,
            },
            expr,
          ));
        }
      }
    }
    return res;
  }
  trace.entry(id).and_modify(|w| {
    *w += 1usize;
  });
  for node in class.nodes.iter() {
    let mut sub_exprs: Vec<Vec<(ExtractInfo, RecExpr<L>)>> = Vec::new();
    for child in node.children() {
      let next_id = egraph.find(*child);
      let extra_loop = if trace[&next_id] > 0 { 1usize } else { 0usize };
      let raw_res = collect_expressions_with_loops_aux(
        egraph,
        next_id,
        extra_loop + loop_num,
        depth + 1,
        trace,
      );
      sub_exprs.push(
        raw_res
          .into_iter()
          .map(|(info, expr)| {
            (
              ExtractInfo {
                loop_num: info.loop_num + extra_loop,
                size: info.size,
              },
              expr,
            )
          })
          .collect(),
      );
    }
    let mut local_node = node.clone();
    for (index, child) in local_node.children_mut().iter_mut().enumerate() {
      *child = Id::from(index);
    }
    if sub_exprs.iter().any(|exprs| exprs.is_empty()) {
      continue;
    }
    for sub_expr in cartesian_product(&sub_exprs) {
      let sub_infos: Vec<ExtractInfo> = sub_expr.iter().map(|x| x.0).collect();
      let new_info = merge_infos(&sub_infos);
      if !is_valid_info(&new_info) {
        continue;
      }
      let expr = local_node.join_recexprs(|id| sub_expr[usize::from(id)].1.clone());
      res.push((new_info, expr));
      if res.len() > CONFIG.extraction_max_num {
        println!("Reach the limit");
        return res;
      }
    }
  }
  trace.entry(id).and_modify(|w| {
    *w -= 1usize;
  });

  res
}

pub fn collect_expressions_with_loops<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  id: Id,
) -> Vec<RecExpr<L>> {
  let mut trace: HashMap<Id, _> = egraph.classes().map(|class| (class.id, 0usize)).collect();
  //println!("start collect");
  let res = collect_expressions_with_loops_aux(egraph, id, 0, 0, &mut trace);
  res.into_iter().map(|(_, expr)| expr).collect()
}

pub fn get_all_expressions_with_loop<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  roots: Vec<Id>,
) -> Denotation<L> {
  let mut memo = BTreeMap::new();
  for root in roots {
    memo.insert(root, collect_expressions_with_loops(egraph, root));
  }
  memo
}

pub fn eclasses_descended_from<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  root: Id,
) -> BTreeSet<Id> {
  let mut seen = BTreeSet::new();
  eclasses_descended_from_helper(egraph, root, &mut seen);
  seen
}

fn eclasses_descended_from_helper<L: Language, A: Analysis<L>>(
  egraph: &EGraph<L, A>,
  root: Id,
  seen: &mut BTreeSet<Id>,
) {
  if seen.contains(&root) {
    return;
  }
  seen.insert(root);
  for node in egraph[root].nodes.iter() {
    for child in node.children() {
      eclasses_descended_from_helper(egraph, *child, seen);
    }
  }
}

/// Remove node from egraph
pub fn remove_node<L: Language, A: Analysis<L>>(egraph: &mut EGraph<L, A>, node: &L) {
  for c in egraph.classes_mut() {
    c.nodes.retain(|n| n != node);
  }
}

pub fn rec_expr_to_pattern_ast<L: Clone>(rec_expr: RecExpr<L>) -> RecExpr<ENodeOrVar<L>> {
  let enode_or_vars: Vec<ENodeOrVar<L>> = rec_expr
    .as_ref()
    .iter()
    .cloned()
    .map(|node| ENodeOrVar::ENode(node))
    .collect();
  enode_or_vars.into()
}

/// A term whose root is a given enode and children are extracted by extractor
pub fn extract_with_node<L: Language, A: Analysis<L>, CF: CostFunction<L>>(
  enode: &L,
  extractor: &Extractor<CF, L, A>,
) -> RecExpr<L> {
  enode.join_recexprs(|id| extractor.find_best(id).1)
}

/// Variables of a pattern as a set
pub fn var_set<L: Language>(pattern: &Pattern<L>) -> BTreeSet<Var> {
  pattern.vars().iter().cloned().collect()
}

/// Like egg's Condition, but for searchers
pub trait SearchCondition<L, N>
where
  L: Language,
  N: Analysis<L>,
{
  fn check(&self, egraph: &EGraph<L, N>, eclass: Id, subst: &Subst) -> bool;
}

/// Conditional searcher
#[derive(Clone, Debug)]
pub struct ConditionalSearcher<C, S> {
  /// The searcher we apply first
  pub searcher: S,
  /// The condition we will check on each match found by the searcher
  pub condition: C,
}

impl<C, S, N, L> Searcher<L, N> for ConditionalSearcher<C, S>
where
  C: SearchCondition<L, N>,
  S: Searcher<L, N>,
  L: Language,
  N: Analysis<L>,
{
  fn search_eclass_with_limit(
    &self,
    egraph: &EGraph<L, N>,
    eclass: Id,
    limit: usize,
  ) -> Option<SearchMatches<L>> {
    // Use the underlying searcher first
    let matches = self
      .searcher
      .search_eclass_with_limit(egraph, eclass, limit)?;
    // Filter the matches using the condition
    let filtered_matches: Vec<Subst> = matches
      .substs
      .into_iter()
      .filter(|subst| self.condition.check(egraph, eclass, subst))
      .collect();
    if filtered_matches.is_empty() {
      // If all substitutions were filtered out,
      // it's as if this eclass hasn't matched at all
      None
    } else {
      Some(SearchMatches {
        eclass: matches.eclass,
        substs: filtered_matches,
        ast: matches.ast,
      })
    }
  }

  fn vars(&self) -> Vec<Var> {
    self.searcher.vars()
  }
}

/// When we apply the subst to pattern, does it exist in the e-graph?
pub fn lookup_pattern<L, N>(pattern: &Pattern<L>, egraph: &EGraph<L, N>, subst: &Subst) -> bool
where
  L: Language,
  N: Analysis<L>,
{
  let mut ids: Vec<Option<Id>> = vec![None; pattern.ast.as_ref().len()];
  for (i, enode_or_var) in pattern.ast.as_ref().iter().enumerate() {
    match enode_or_var {
      ENodeOrVar::Var(v) => {
        ids[i] = subst.get(*v).copied();
      }
      ENodeOrVar::ENode(e) => {
        let mut resolved_enode: L = e.clone();
        for child in resolved_enode.children_mut() {
          match ids[usize::from(*child)] {
            None => {
              return false;
            }
            Some(id) => {
              *child = id;
            }
          }
        }
        match egraph.lookup(resolved_enode) {
          None => {
            return false;
          }
          Some(id) => {
            ids[i] = Some(id);
          }
        }
      }
    }
  }
  true
}

impl<L, N> SearchCondition<L, N> for Pattern<L>
where
  L: Language,
  N: Analysis<L>,
{
  fn check(&self, egraph: &EGraph<L, N>, _eclass: Id, subst: &Subst) -> bool {
    lookup_pattern(self, egraph, subst)
  }
}

pub struct DestructiveApplier {
  searcher: Pattern<SymbolLang>,
  applier: Pattern<SymbolLang>,
}

impl DestructiveApplier {
  pub fn new(searcher: Pattern<SymbolLang>, applier: Pattern<SymbolLang>) -> Self {
    Self { searcher, applier }
  }
}

impl<N> Applier<SymbolLang, N> for DestructiveApplier
where
  N: Analysis<SymbolLang>,
{
  fn apply_one(
    &self,
    egraph: &mut egg::EGraph<SymbolLang, N>,
    eclass: Id,
    subst: &Subst,
    searcher_ast: Option<&PatternAst<SymbolLang>>,
    rule_name: Symbol,
  ) -> Vec<Id> {
    // let memo = (rule_name, subst.clone(), self.original_pattern.ast.clone());
    // if egraph[eclass].data.previous_rewrites.contains(&memo) {
    //     return vec!();
    // }
    // egraph[eclass].data.previous_rewrites.insert(memo);
    let mut ids = self
      .applier
      .apply_one(egraph, eclass, subst, searcher_ast, rule_name);
    if prune_enodes_matching(egraph, &self.searcher.ast, subst, &eclass) {
      ids.push(eclass);
    }
    ids
  }

  fn get_pattern_ast(&self) -> Option<&PatternAst<SymbolLang>> {
    egg::Applier::<SymbolLang, N>::get_pattern_ast(&self.applier)
  }

  fn vars(&self) -> Vec<Var> {
    egg::Applier::<SymbolLang, N>::vars(&self.applier)
  }
}

/// Removes enodes matching the rec_expr from the egraph.
///
/// I think that we could do slightly better than a HashMap by having a mutable
/// RecExpr and storing which Ids we've visited on the nodes, but the difference
/// between passing around clones of a HashMap/BTreeSet everywhere and using a
/// single mutable HashMap is minimal in my testing (0.2s for a test taking 9s -
/// although this was just a single test).
fn prune_enodes_matching<N>(
  egraph: &mut EGraph<SymbolLang, N>,
  rec_expr: &RecExpr<ENodeOrVar<SymbolLang>>,
  subst: &Subst,
  eclass: &Id,
) -> bool
where
  N: Analysis<SymbolLang>,
{
  let mut memo = BTreeMap::default();
  let rec_expr_id: Id = (rec_expr.as_ref().len() - 1).into();
  // Handles cycles - if we get back here then it matches.
  memo.insert((rec_expr_id, *eclass), true);
  let original_len = egraph[*eclass].nodes.len();

  if original_len == 1 {
    return false;
  }
  egraph[*eclass].nodes = egraph[*eclass]
    .nodes
    .to_owned()
    .into_iter()
    .filter(|node| {
      !match_enode(egraph, rec_expr, &rec_expr_id, subst, node, &mut memo)
      // if res {
      //     // println!("{} filtering node {:?}", rule_name, node)
      // }
      // !res
    })
    .collect();
  original_len > egraph[*eclass].nodes.len()
}

/// This function recursively traverses the rec_expr and enode in lock step. If
/// they have matching constants, then we can simply check their equality. Most
/// of the cases, however, come from recursively checking the contained rec_expr
/// nodes against contained eclasses.
fn match_enode<N>(
  egraph: &EGraph<SymbolLang, N>,
  rec_expr: &RecExpr<ENodeOrVar<SymbolLang>>,
  rec_expr_id: &Id,
  subst: &Subst,
  enode: &SymbolLang,
  memo: &mut BTreeMap<(Id, Id), bool>,
) -> bool
where
  N: Analysis<SymbolLang>,
{
  match &rec_expr[*rec_expr_id] {
    ENodeOrVar::ENode(n) => {
      let ops_match = n.op == enode.op;
      // The ops need to match
      if !ops_match {
        return false;
      }
      // As do the number of children (this should never be false)
      let children_lengths_match = n.children.len() == enode.children.len();
      if !children_lengths_match {
        return false;
      }
      // As do the children themselves
      zip(n.children(), enode.children()).all(|(n_child, enode_child)| {
        any_enode_in_eclass_matches(egraph, rec_expr, n_child, subst, enode_child, memo)
      })
    }
    // I think this is incomparable - an enode is not an eclass. Perhaps
    // they are equal if the enode is in the eclass? I kind of don't think
    // so.
    //
    // This should only occur if you have
    ENodeOrVar::Var(_) => false,
  }
}

/// In this case, we have a concrete AST node (ENodeOrVar::EnNode) or Var
/// (ENodeOrVar::Var) in the rec_expr that we want to compare to an entire
/// eclass. Comparing a Var to an eclass is a base case - we just check to see
/// if they're the same. Otherwise, we need to check if there is any enode in
/// the class that we can match with the concrete AST node.
fn any_enode_in_eclass_matches<N>(
  egraph: &EGraph<SymbolLang, N>,
  rec_expr: &RecExpr<ENodeOrVar<SymbolLang>>,
  rec_expr_id: &Id,
  subst: &Subst,
  eclass: &Id,
  memo: &mut BTreeMap<(Id, Id), bool>,
) -> bool
where
  N: Analysis<SymbolLang>,
{
  if let Some(res) = memo.get(&(*rec_expr_id, *eclass)) {
    return *res;
  }
  let res = {
    // This is the second and last base case (aside from cycles) where we can
    // conclude a pattern matches.
    if let ENodeOrVar::Var(v) = rec_expr[*rec_expr_id] {
      return subst[v] == *eclass;
    }
    // If we cycle back to this node, then the pattern matches.
    memo.insert((*rec_expr_id, *eclass), true);
    egraph[*eclass]
      .iter()
      .any(|node| match_enode(egraph, rec_expr, rec_expr_id, subst, node, memo))
  };
  // Update the memo since we only set it to 'true' temporarily to handle cycles.
  memo.insert((*rec_expr_id, *eclass), res);
  res
}

pub fn search_wave_fronts<N>(
  egraph: &mut EGraph<SymbolLang, N>,
  pat: &Pattern<SymbolLang>,
  eclass: Id,
) -> ()
where
  N: Analysis<SymbolLang>,
{
  let pat_ref = pat.ast.as_ref();
  let mut mismatches = BTreeSet::new();
  let mut cache = BTreeSet::new();
  let mut subst = HashMap::new();
  if CONFIG.verbose {
    println!("Before labeling:");
    dump_eclass_exprs(egraph, eclass);
  }
  label_eclass(
    egraph,
    pat_ref,
    pat_ref.len() - 1,
    eclass,
    &mut mismatches,
    false,
    &mut cache,
    &mut subst,
  );
  if CONFIG.verbose {
    println!("After labeling:");
    dump_eclass_exprs(egraph, eclass);
  }
  let mut cache = BTreeSet::new();
  let (rippled, _) = ripple_out_eclass(egraph, &mut cache, eclass);
  egraph.rebuild();
  if CONFIG.verbose {
    println!("After rippling:");
    dump_eclass_exprs(egraph, rippled);
  }
}

fn ripple_out_eclass<N>(
  egraph: &mut EGraph<SymbolLang, N>,
  cache: &mut BTreeSet<Id>,
  eclass: Id,
) -> (Id, Vec<Id>)
where
  N: Analysis<SymbolLang>,
{
  // println!("cur eclass:");
  // dump_eclass_exprs(egraph, eclass);
  let nodes = egraph[eclass].nodes.clone();
  let mut wave_holes = vec![];
  for (idx, enode) in nodes.iter().enumerate() {
    let rippled_enode = ripple_out_enode(egraph, cache, enode);
    if rippled_enode.op.to_string().starts_with("_wave_front_") {
      wave_holes.extend(rippled_enode.children.clone());
      // let eclass_idx = parent.children.iter().position(|&id| id == eclass).unwrap();
      // return wave_front_children[0];
      // (*parent).children = (*parent)
      //   .children
      //   .splice(
      //     eclass_idx..eclass_idx + 1,
      //     wave_front_children.iter().cloned(),
      //   )
      //   .collect();
      // *parent
      // let parent_id = egraph.add(parent.clone());
      // wave_front.children = vec![parent_id];
      // let _wave_front_id = egraph.add(wave_front.clone());
      // println!("ayo:");
      // dump_eclass_exprs(egraph, _wave_front_id);
      // **parent = wave_front;
      // egraph[eclass].nodes[idx] =
    } else {
      egraph[eclass].nodes[idx] = rippled_enode;
    }
  }
  // [(S [(_ih_root_plus [x_30] [(_wave_front_S [x_30])])])]
  // [(S [(_wave_front_S [(_ih_root_plus [x_30] [x_30])])])]

  (eclass, wave_holes)
}

fn ripple_out_enode<N>(
  egraph: &mut EGraph<SymbolLang, N>,
  cache: &mut BTreeSet<Id>,
  enode: &SymbolLang,
) -> SymbolLang
where
  N: Analysis<SymbolLang>,
{
  if enode.op.to_string().starts_with("_wave_front_") {
    return enode.clone();
  }
  let mut new_children = vec![];
  for &enode_child in &enode.children {
    let (_, wave_holes) = ripple_out_eclass(egraph, cache, enode_child);
    if wave_holes.is_empty() {
      new_children.push(enode_child);
    } else {
      new_children.extend(wave_holes);
    }
  }

  SymbolLang {
    op: enode.op.clone(),
    children: new_children,
  }
}

fn label_eclass<N>(
  egraph: &mut EGraph<SymbolLang, N>,
  pat: &[ENodeOrVar<SymbolLang>],
  pat_idx: usize,
  eclass: Id,
  mismatches: &mut BTreeSet<(usize, SymbolLang)>,
  found_head: bool,
  cache: &mut BTreeSet<(Id, usize)>,
  subst: &mut HashMap<Var, SymbolLang>,
) -> Id
where
  N: Analysis<SymbolLang>,
{
  let state = (eclass, pat_idx);
  if cache.contains(&state) {
    return eclass;
  }
  cache.insert(state);

  let nodes = egraph[eclass].nodes.clone();
  for (i, enode) in nodes.iter().enumerate() {
    let ripple = label_enode(
      egraph, pat, pat_idx, enode, mismatches, found_head, cache, subst,
    );
    egraph[eclass].nodes[i] = ripple;
  }

  eclass
}

fn label_enode<N>(
  egraph: &mut EGraph<SymbolLang, N>,
  pat: &[ENodeOrVar<SymbolLang>],
  pat_idx: usize,
  enode: &SymbolLang,
  mismatches: &mut BTreeSet<(usize, SymbolLang)>,
  mut found_head: bool,
  cache: &mut BTreeSet<(Id, usize)>,
  subst: &mut HashMap<Var, SymbolLang>,
) -> SymbolLang
where
  N: Analysis<SymbolLang>,
{
  match &pat[pat_idx] {
    ENodeOrVar::Var(v) => match subst.get(v) {
      None => {
        subst.insert(*v, enode.clone());
        enode.clone()
      }
      Some(enode_seen) => {
        if enode_seen != enode {
          mismatches.insert((pat_idx, enode.clone()));
          let mut new_enode = enode.clone();
          new_enode.op = format!("_wave_front_{}", enode.op)
            .parse::<Symbol>()
            .unwrap();
          new_enode
        } else {
          enode.clone()
        }
      }
    },
    ENodeOrVar::ENode(e) => {
      if e.matches(enode) {
        let mut new_op = enode.op.clone();
        if !found_head {
          found_head = true;
          new_op = format!("_ih_root_{}", enode.op).parse::<Symbol>().unwrap();
        }
        SymbolLang::new(
          new_op,
          e.children
            .iter()
            .zip(&enode.children)
            .map(|(&child_idx, &enode_child)| {
              label_eclass(
                egraph,
                pat,
                usize::from(child_idx),
                enode_child,
                mismatches,
                found_head,
                cache,
                subst,
              )
            })
            .collect(),
        )
      } else {
        if !found_head {
          SymbolLang::new(
            enode.op,
            enode
              .children
              .iter()
              .map(|&enode_child| {
                label_eclass(
                  egraph,
                  pat,
                  pat_idx,
                  enode_child,
                  mismatches,
                  found_head,
                  cache,
                  subst,
                )
              })
              .collect(),
          )
        } else {
          if CONFIG.verbose {
            println!("no match");
          }
          mismatches.insert((pat_idx, enode.clone()));
          let mut new_enode = enode.clone();
          new_enode.op = format!("_wave_front_{}", enode.op)
            .parse::<Symbol>()
            .unwrap();
          new_enode
        }
      }
    }
  }
}
