# common-cause-fleet

Measurement code for **"One Brain, Many Workers: Common-Cause Failure and the Limits of
Redundancy in Agent Fleets."**

The repository fits reliability-engineering common-cause-failure (CCF) parameters to
same-model and cross-model LLM agent panels on synthetic enterprise process tasks, and
converts the fitted parameters into redundancy design rules.

Everything here runs against a **local** Ollama instance. There are no API keys, no paid
endpoints, and no network calls other than `http://localhost:11434`.

---

## PRE-REGISTRATION (written before any panel was executed)

**Timestamp of this section:** 2026-10-27, written before `run_panels.py` was first
executed. Commit order in this repository reflects this: `README.md` precedes `results/`.

### Primary outcome

Per case, per agent slot: **binary correctness**, all-or-nothing. A response counts as
correct only if the parsed answer exactly equals the generator's ground truth after the
documented normalisation (label match for classification tasks; all five fields exact for
extraction; `VERDICT:CLAUSE` for policy tasks). No partial credit. No answer-format
leniency beyond the documented parser.

### Difficulty strata are fixed by construction

Every case carries a `difficulty` label (`easy` / `medium` / `hard`) assigned by the
**generator parameters** at creation time (tolerance margins, number of injected
distractors, number of simultaneously failing policy conditions, presence of superseding
clauses). Labels are never derived from observed model performance and are never revised
after seeing outcomes. Realised per-stratum error rates are reported as a *manipulation
check*, not used to relabel cases.

### Estimators (fixed before execution)

- Pairwise failure correlation: phi coefficient on the 2x2 table of pass/fail.
- Difficulty-adjusted pooled correlation: Mantel-Haenszel weighting of stratum-specific
  phi, weights `w_k = n_1k * n_0k / n_k`.
- Difficulty confound quantification: Monte-Carlo panel of conditionally independent
  agents that reproduces the observed per-stratum marginal failure rates and the observed
  stratum mix; the pooled phi this synthetic independent fleet produces is
  `phi_indep_mix`, and `1 - (phi_pooled - phi_indep_mix)/(1 - phi_indep_mix)` ... reported
  as the share of pooled correlation attributable to the difficulty mix.
- Common-cause share: beta-factor inversion `beta = (p11 - p^2) / (p - p^2)` for
  equal-marginal pairs; normalised excess joint failure for unequal-marginal pairs.
- Effective redundancy: `k_eff` = the panel size an *independent* panel would need to
  reach the observed realised error at nominal size `k`; efficiency `k_eff / k`.
- Checker catch rate: `P(corrected | author wrong)`.

### Uncertainty

2000 case-level bootstrap resamples (percentile intervals) for every reported estimate;
5000 label permutations for the null `phi = 0`. Alpha = 0.05, one-sided where the
hypothesis is directional and stated as such.

### Pre-registered success bars

| # | Bar | Operationalisation |
|---|-----|--------------------|
| i | Same-model failure correlation is substantial and real | difficulty-adjusted `phi >= 0.30` **and** permutation `p < 0.05` against `phi = 0` |
| ii | Same-model panels buy far less than independence predicts | realised majority-vote error at `k = 5` is `<= 50%` of the independence prediction |
| iii | Cross-model panels largely restore the predicted gain | gain recovery at `k = 5` is `>= 70%`, where recovery `= (e_1 - e_5^real) / (e_1 - e_5^indep)` |
| iv | Checker value declines with author-checker relatedness | catch rate strictly ordered `same-same < mixed pairs < cross-cross`, with one-sided paired bootstrap tests |

**Misses are results.** If a bar is missed it is reported as missed. No post-hoc changes
to the task battery, the panel compositions, or the estimators after outcomes are seen.
Any deviation from this document is recorded as a dated amendment at the bottom of this
file.

### What is *not* pre-registered

Exploratory: the per-task-type breakdown, the four-cell checker design (the pre-registered
bar concerns only the ordering), and any post-hoc decomposition of the residual. These are
labelled exploratory in the paper.

---

## Layout

```
tasks/       task battery generators + materialised battery (tasks/battery.jsonl)
panels/      panel runner + Ollama backend
stats/       estimators: phi, beta, redundancy curve, catch rate, bootstrap
run_panels.py
make_figs.py
results/     raw JSONL, derived CSVs, figures
```

## Backends

`panels/ollama_backend.py` talks to `http://localhost:11434`. Model pool detected on the
host at run time (see `results/model_manifest.json`):

| role | model | family (Ollama) | params |
|------|-------|-----------------|--------|
| M1 primary | `qwen2.5:14b-instruct` | qwen2 | 14.8B |
| M2 same-family, different size | `qwen2.5-coder:32b` | qwen2 | 32.8B |
| M3 cross-family | `llama3.1:8b` | llama | 8.0B |

Relatedness ladder used throughout:

- **L1 same-model** — M1 vs M1; differs only in sampling seed and prompt paraphrase.
- **L2 same-family / different size** — M1 vs M2 (shared family, different parameter
  count and tuning).
- **L3 cross-family** — M1 or M2 vs M3 (different pretraining lineage).

## Reproducing

```bash
python3 tasks/generate_tasks.py            # writes tasks/battery.jsonl (deterministic)
python3 run_panels.py --pilot 6            # smoke test, writes results/pilot/
python3 run_panels.py --full               # full run, appends results/grid.jsonl + checkers.jsonl
python3 -m stats.analyse --reps 2000       # writes results/*.csv + summary.json
python3 -m stats.design_rules              # writes results/design_rules.json
python3 make_figs.py                       # writes figures/*.pdf and *.png
```

`stats/analyse.py` and `stats/design_rules.py` use package-relative imports and must be
run as modules (`-m`), not as scripts.

Requires `numpy` for the analysis and `matplotlib` for the figures. The panel runner is
pure standard library.

## Run record

The results in `results/` come from one full run on a 32 GB Apple-silicon host with
Ollama serving the three models locally:

| | |
|---|---|
| Cases | 408 (4 task types × 3 strata × 34) |
| Panel-member calls | 4,896 (M1 × 5 slots, M3 × 5 slots, M2 × 2 slots, × 408 cases) |
| Checker calls | 1,632 (4 author–checker cells × 408 cases) |
| Transport errors | 0 |
| Unparseable answers | 0 |
| Wall clock | 3 h 59 min |

**Model residency matters.** The three models total 9.0 + 19.9 + 4.9 GB and do not fit in
memory together. An early attempt ran with all three resident and thrashed swap so badly
that throughput fell by an order of magnitude. `panels/runner.py` now evicts every other
model at each phase boundary (`ollama_backend.unload`), which is why the runner batches
strictly by model rather than interleaving.

## Notes on the measurement

- Panel members answer in a single pass with `num_predict = 72`; they typically emit only
  the answer line. This measures the reliability of the raw decision, which is the object
  a panel votes on. It is not a chain-of-thought evaluation.
- Slots fix a (seed, paraphrase) pair, so slot `i` means the same seed and the same
  paraphrase in every panel composition. Only the model occupying the slot changes. This
  is what makes the cross-composition comparison clean.
- M2 occupies only slots 1 and 3. That is enough for a matched same-family contrast at
  every panel size we report (k = 1, 3, 5) and it halves the cost of the 32 B model, which
  is the slowest member of the pool.


---

## AMENDMENT 1 — 2026-10-27, before the main run

**Change.** In task type A (exception classification), the `MISSING_PO` condition was
rendered as `PURCHASE ORDER (none referenced): quantity N, unit price P`, i.e. the record
simultaneously asserted that no purchase order was referenced and displayed that purchase
order's quantity and price. A smoke test on 6 cases showed this is genuinely ambiguous
rather than merely difficult. The renderer now prints the ERP purchase-order record
unconditionally and adds a separate line, `PO REFERENCE PRINTED ON INVOICE: <ref>|none`,
which is the field the rule actually tests. Ground truth is unchanged: `MISSING_PO` is
still emitted iff the invoice cites no reference.

**Scope.** This is a rendering-validity fix, not a difficulty change. No tolerance,
margin, distractor count or stratum assignment was touched. The battery is regenerated
from the same seed, so the fix applies uniformly to all 408 cases.

**Data.** No outcome data from the smoke test enters any reported estimate. The 6-case
pilot in `results/pilot/` was a parser/latency check and is discarded.

**Unchanged.** All four pre-registered success bars, all estimators, the panel
compositions and the case count (408 cases = 4 task types x 3 strata x 34).
