# PottsMPNN energy & conditional — what the code computes and why

This documents `calc_potts_eners`, `potts_candidate_energies`,
`potts_mutation_delta`, `_directed_pair_edges`, `_incoming_adjacency`, and the
conditional used inside `potts_gibbs_optimize` (all in `pottsmpnn.py`).

---

## 1. What the Potts head outputs

The encoder runs message passing on a **directed k‑NN graph**. Each residue `i`
has `K` neighbour slots `E_idx[i, :]`:
- slot `0` = residue `i` itself (the "self" slot),
- slots `1..K-1` = its `K-1` nearest other residues by distance.

The graph is **directed and asymmetric**: `j` being a neighbour of `i` does not
imply `i` is a neighbour of `j` (`E_idx` is a plain `topk`, no symmetrisation).
Measured on 9NNF: ~16% of edges are non‑reciprocal.

For each directed edge `(i, k)` the head produces a `V×V` matrix
`etab[i, k, :, :]` from that edge's embedding `h_E[i,k]`:

```
etab[i, k, a, b]  =  energy if residue i (the SOURCE) has identity a
                     and its slot-k neighbour j = E_idx[i,k] (the TARGET) has identity b
```

Slot 0 is masked to its diagonal, so `etab[i,0,a,a]` is the single‑site field
`h_i(a)`. Shapes: `etab_out` is `[B, L, K, V, V]`, `E_idx` is `[B, L, K]`.

---

## 2. The Hamiltonian — the number we score and optimise (`calc_potts_eners`)

```
H(s) = Σ_i Σ_k  etab[i, k, s_i, s_{E_idx[i,k]}]
```

Walk over every residue `i` and every slot `k`; look up the matrix entry for the
identities actually present (source `s_i`, target `s_neighbour`); add them all.
Each directed edge contributes exactly **one** scalar. This is *the definition*
of the model's energy — everything else must be consistent with it.

Consequences of this definition (stated plainly, not hidden):
- a **reciprocal** pair (`i→j` and `j→i` both present) contributes **two** terms
  → reciprocal pairs are *double‑counted*;
- a **non‑reciprocal** pair contributes **one** term;
- the matrices are directed and are **not** required to be symmetric.

These are properties of the model's chosen energy, kept as‑is on purpose.

---

## 3. The conditional energy — what Gibbs / placement scans need

**Goal:** when we change one residue `p` (everyone else fixed), how does `H`
change? Define the conditional `E_p(a)` = the part of `H` that depends on `s_p`.
Then, exactly:

```
H(flip p→a) − H(s)  =  E_p(a) − E_p(s_p)
```

**Which terms of `H` contain `s_p`?** Scan `H = Σ etab[i,k,s_i,s_neighbour]`.
`s_p` appears in a term in exactly two ways, plus the self term:

| role of p in the term | term | name |
|---|---|---|
| `p` is the **source** (`i = p`) | `etab[p, k, s_p, s_j]` | outgoing |
| `p` is the **target** (`E_idx[i,k] = p`) | `etab[i, k, s_i, s_p]` | incoming |
| self slot | `etab[p, 0, s_p, s_p]` | self |

So:

```
E_p(a) =  etab[p, 0, a, a]                            (self)
        + Σ_{k≥1}            etab[p, k, a, s_{E_idx[p,k]}]   (outgoing: p is source)
        + Σ_{(i,k): E_idx[i,k]=p}  etab[i, k, s_i, a]        (incoming: p is target)
```

This is exact and **forced** — it is literally every term of `H` containing
`s_p`. Nothing is invented and nothing optional is added.

---

## 4. The "directed edge" worry (read this carefully)

> "The table came from a directed edge `i→j`. Using it for node `j` feels like
> computing energy in the opposite direction."

It isn't. Separate two ideas:

- **Provenance** — *who computed the table.* `etab[i,k]` was produced from node
  `i`'s edge embedding (directed message passing). True, but irrelevant to the
  bookkeeping.
- **Dependence** — *what the table's value depends on.* `etab[i,k,a,b]` is a
  function of **both** `a = s_i` and `b = s_j`. It is a `V×V` matrix; both
  indices are genuine inputs.

The energy of edge `i→j` is **one number**, `etab[i,k,s_i,s_j]`, that depends on
both endpoints. "Reading it along the `s_j` axis" (fix `s_i`, vary `s_j`) is just
asking how that number changes with `s_j` — a partial difference of a
two‑variable function. It does **not** re‑run the head backwards and it does
**not** build a reverse table.

Analogy: a function `f(x, y)` may have been *derived* in a context centred on
`x`, but `∂f/∂y` is still perfectly well defined — `y` is one of its inputs.
Reading `f`'s dependence on `y` is not "using `f` backwards."

Concrete (asymmetric table, so it's unambiguous):

```
etab[i, k]          to optimise i (vary s_i, fix s_j=1):  read COLUMN s_j=1
     s_j=0 s_j=1    to optimise j (vary s_j, fix s_i=0):  read ROW    s_i=0
s_i=0[ 2    5 ]
s_i=1[ 1    3 ]     same matrix, same number 5 shared, two different questions.
```

For node `j` we read the **row** `[2, 5]` — `etab[i,k,s_i,:]` in code. We never
form `etab[j, k', s_j, s_i]` (a table at `j`); that table does not exist for a
non‑reciprocal edge, and we never reference it.

---

## 5. Two different "transposes" — do not conflate

1. **Graph transpose** (what the fix uses): invert the *adjacency* to answer
   "which edges point AT residue `p`?" Built once by `_incoming_adjacency`. This
   reindexes the **edge list**; it transposes no matrices.
2. **Matrix transpose** (used only in the *merge*, for reciprocal pairs): when
   both `i→j` and `j→i` exist, `compute_potts_context` replaces both with the
   average `½(etab[i→j] + etab[j→i]ᵀ)` so the pair has one consistent symmetric
   coupling. The matrix transpose only lines up the two matrices' axes before
   averaging. This happens *before* energy evaluation and only for reciprocal
   pairs. The energy/conditional code never transposes a matrix — it indexes.

---

## 6. Reciprocal vs non‑reciprocal, and the honest caveat

- **Reciprocal pair:** the head produced **two** matrices (one per direction);
  they disagree substantially (measured ~0.62 relative); the merge **averages**
  them into one symmetric coupling. Both endpoints' conditionals read that single
  merged coupling along their respective axes. The 0.62 disagreement is averaged
  out, so it does not affect the energy.
- **Non‑reciprocal edge:** the head produced **one** matrix (from the source's
  embedding). There is no second view to average. Both endpoints' conditionals
  use that single matrix. This is a genuine model limitation on ~16% of edges
  (a one‑sided estimate) and is part of the motivation for retraining with the
  corrected loss. It does **not** change the bookkeeping: that one matrix is a
  real term in `H` and depends on both endpoints, so both conditionals must read
  it.

---

## 7. The functions

| function | role |
|---|---|
| `calc_potts_eners(etab_out, E_idx, seqs)` | the Hamiltonian `H(s)`. Unchanged by the fix. |
| `_directed_pair_edges(E_idx)` | flatten non‑self edges → `(src, slot, tgt)`. |
| `_incoming_adjacency(E_idx)` | graph transpose: per residue, the `(source, slot)` of edges pointing at it (padded + mask). |
| `potts_candidate_energies(etab_out, E_idx, seqs)` | `E_p(a)` for all `p, a` = self + outgoing + incoming → `[N, L, V]`. |
| `potts_mutation_delta(...)` | `candidate − current` = exact `ΔH`. |
| `potts_gibbs_optimize(...)` | per‑position conditional now = self + outgoing + incoming (via `_incoming_adjacency`); accepted moves cannot raise `H`. |

---

## 8. The bug this fixed

The old conditional summed only **self + outgoing** (residue `p`'s own edges) and
dropped the **incoming** terms (edges where `p` is the target). On this
asymmetric graph that is wrong:

| conditional | mean error vs true `ΔH` (9NNF) | max error |
|---|---|---|
| outgoing‑only (old) | ~50% of mean\|ΔH\| | 24.2 |
| reverse_k‑only | ~5.5% | 5.4 (on the 16% non‑reciprocal edges) |
| full transpose (this fix) | ~0% | 0.0003 |

Verified on tiny random tables (`tests/test_potts_energy.py`) and on the real
9NNF checkpoint (Gibbs descends `H` monotonically).
