# CLAIMS.md

Every claim in `paper.tex`, mapped to the estimate that supports it and to the file that
contains that estimate. No number in the paper is typed by hand: `make_paper.py` emits
`paper.tex` from `results/summary.json`, `results/design_rules.json` and
`results/*.csv`, so the mapping below is mechanical rather than reconstructive.

**Provenance chain.** `tasks/battery.jsonl` (cases + ground truth) → `results/grid.jsonl`
and `results/checkers.jsonl` (raw outcomes, one JSON object per call) → `stats/analyse.py`
→ `results/summary.json` + CSVs → `stats/design_rules.py` → `results/design_rules.json`
→ `make_paper.py` → `paper.tex`. Every raw record carries model, seed, paraphrase index,
task id, difficulty stratum, parsed answer and pass/fail.

---

## 1. Descriptive facts about the experiment

| # | Claim | Estimate | Source |
|---|-------|----------|--------|
| D1 | The battery has 408 cases, 34 per (task type × difficulty) cell, 136 per stratum | 408; 136/136/136 | `summary.json: n_cases, n_cases_by_stratum` |
| D2 | 4,896 panel-member observations, 1,632 checker observations, no errors | 4896, 1632 | `summary.json: n_observations`; `results/grid.jsonl`, `results/checkers.jsonl` |
| D3 | Three models occupy three rungs of a relatedness ladder | M1 qwen2/14.8B, M2 qwen2/32.8B, M3 llama/8.0B | `results/model_manifest.json: models` |
| D4 | Error rises monotonically with the generator's difficulty label (manipulation check) | easy 72% correct, medium 59%, hard 43% | `results/model_marginals.csv` |
| D5 | The three models are not equally accurate | M1 31%, M2 30%, M3 61% error | `results/model_marginals.csv` |

## 2. Failure correlation and the relatedness gradient

| # | Claim | Estimate | Source |
|---|-------|----------|--------|
| P1 | Two copies of the same model fail together far more than independence predicts | φ = 0.818 [0.774, 0.860] | `summary.json: phi.same_model.phi_pooled(_ci)` |
| P2 | The same-model correlation survives difficulty adjustment | φ_MH = 0.803 [0.756, 0.850] | `phi.same_model.phi_mh(_ci)` |
| P3 | The association is not chance: conditional independence is rejected | p = 0.00033 (1/3000 floor) | `phi.same_model.perm_p_one_sided` |
| P4 | The common-cause share equals the correlation under equal marginals | β = 0.818 = φ | `phi.same_model.beta_equal`; identity proved in Appendix A |
| P5 | Same-family pairs are less correlated than same-model pairs | φ = 0.487 [0.404, 0.564]; φ_MH = 0.437 | `phi.same_family.*` |
| P6 | Cross-family pairs are less correlated still | φ = 0.272 [0.207, 0.341]; φ_MH = 0.226 | `phi.cross_family.*` |
| P7 | The gradient is monotone in the pooled and difficulty-adjusted estimates | 0.818 > 0.487 > 0.272; MH 0.803 > 0.437 > 0.226 | `phi.*.phi_pooled`, `phi.*.phi_mh` |
| P8 | The same-model effect replicates on a second model (not an M1 artefact) | M3–M3 φ = 0.585 [0.535, 0.635] | `phi.same_model_M3.phi_pooled` |
| P9 | The ordering does **not** hold inside the easy stratum | easy: cross 0.515 > family 0.475 | `phi.*.by_stratum.easy`; reported as a deviation in §5.2 |

## 3. The difficulty confound

| # | Claim | Estimate | Source |
|---|-------|----------|--------|
| Q1 | Difficulty alone explains only a small share of the same-model correlation | φ_indep = 0.077, i.e. 9.4% of 0.818 | `phi.same_model.phi_indep_mix`, `.frac_pooled_explained_by_difficulty` |
| Q2 | The residual attributed to the fleet is large | 0.818 − 0.077 = 0.741 | `phi.same_model.excess_over_difficulty` |
| Q3 | Across levels the difficulty-only share is 9–21% | 9.4%, 20.8%, 16.2% | `phi.*.frac_pooled_explained_by_difficulty` |
| Q4 | The residual is positive at every level | 0.741, 0.386, 0.228 | `phi.*.excess_over_difficulty` |

## 4. Redundancy that does not buy redundancy

| # | Claim | Estimate | Source |
|---|-------|----------|--------|
| R1 | A single M1 agent errs on 31% of cases | 0.306 | `panels.same_model.per_k.1.error` |
| R2 | Independence predicts a 5-member same-model panel at 22% | 0.215 | `panels.same_model.per_k.5.indep_stratified` |
| R3 | The same-model panel actually errs on 32% — worse than one agent | 0.321 | `panels.same_model.per_k.5.error` |
| R4 | It is 1.49× its own independence prediction | 1.492 [1.287, 1.708] | `panels.same_model.ratio_k5(_ci)` |
| R5 | Its effective redundancy is 1.0 against a nominal 5 | k_eff = 1.000 | `panels.same_model.k_eff` |
| R6 | The same-family panel is barely better | ratio 1.406, k_eff = 1.072 | `panels.same_family.ratio_k5`, `.k_eff` |
| R7 | The cross-family panel matches its own independence prediction | ratio 1.025 [0.889, 1.162], k_eff = 3.941 | `panels.cross_family.ratio_k5(_ci)`, `.k_eff` |
| R8 | But it does not beat the same-model panel in absolute error | 0.350 vs 0.321 | `panels.cross_family.per_k.5.error` vs `panels.same_model.per_k.5.error` |
| R9 | The fitted beta model reproduces the panels approximately | predicted 25.9% vs observed 32.1% | `design_rules.json: model_check_k5` |

## 5. Checker economics

| # | Claim | Estimate | Source |
|---|-------|----------|--------|
| K1 | Catch rates by cell: same-same, same-cross, cross-same, cross-cross | 0.320, 0.096, 0.510, 0.207 | `summary.json: checkers.*.catch` |
| K2 | 95% intervals for each cell | [0.24,0.40], [0.05,0.15], [0.45,0.58], [0.16,0.26] | `checkers.*.catch_ci` |
| K3 | The ordering is **not** monotone in relatedness (bar iv missed) | P(ss<mixed)=0.342; P(mixed<cc)=0.000 | `checker_ordering.p_same_lt_mixed`, `.p_mixed_lt_cross` |
| K4 | Checker competence dominates: M1 checker 42% vs M3 checker 15% | (0.320+0.510)/2 = 0.415; (0.096+0.207)/2 = 0.152 | derived from `checkers.*.catch` |
| K5 | The weaker checker overrides far more correct answers | false alarm 0.618, 0.629 vs 0.438, 0.365 | `checkers.*.false_alarm` |
| K6 | Net accuracy after review, per cell | 0.792, 0.723, 0.711, 0.532 | `checkers.*.net` |

## 6. Design rules

| # | Claim | Estimate | Source |
|---|-------|----------|--------|
| G1 | The same-model error floor is β_s·p | 0.818 × 0.313 = 0.256 | `design_rules.json: fitted.floor_beta_times_p` |
| G2 | The maximum useful same-model panel is 3 at ε=0.05, 5 at ε=0.01 | k_max = 3, 5 | `fitted.k_max_eps05`, `.k_max_eps01` |
| G3 | At n=5 no composition reaches below 18% | 0.181 | `budget[n=5, target=0.20].all_diverse_error` |
| G4 | A 20% target at n=5 needs 4 of 5 cross-family (d*=0.80) | 0.181 achieved | `budget[n=5, target=0.20]` |
| G5 | A 10% target at n=11 needs 10 of 11 cross-family (d*=0.91) | 0.094 achieved | `budget[n=11, target=0.10]` |
| G6 | d > 1/2 is necessary for any target below β_s·p | analytic | Eq. (11) in §6.2; verified in `budget.*.block_is_majority` |

## 7. Pre-registered bars (outcome)

| Bar | Criterion | Estimate | Verdict | Source |
|-----|-----------|----------|---------|--------|
| (i) | φ_MH ≥ 0.30 and φ > 0 | 0.803, p = 0.00033 | **met** | `phi.same_model.phi_mh`, `.perm_p_one_sided` |
| (ii) | same-model gain recovery ≤ 0.50 | −0.161 | **met** | `panels.same_model.gain_recovery_k5` |
| (iii) | cross-family gain recovery ≥ 0.70 | undefined (denominator negative) | **missed / not evaluable as written** | `panels.cross_family.gain_recovery_k5`; substitute ratio 1.025 |
| (iv) | catch: same-same < mixed < cross-cross | 0.320 / 0.303 / 0.207; p = 0.342, 0.000 | **missed** | `checkers.*.catch`, `checker_ordering.*` |

Bar (iii) is reported as missed rather than reinterpreted. The pre-registered ratio is
undefined because the cross-family composition's independence-predicted gain relative to
the anchor is negative: its members are individually less accurate than M1, so the
composition is predicted to be worse than a single M1 agent. The interpretable substitute
(realised ÷ predicted error = 1.025) is labelled as a substitute in the paper.

---

## What is *not* claimed

- No claim about frontier models. The pool is three open-weight models that fit in 32 GB.
- No claim that synthetic enterprise tasks are production work.
- No claim that prompt paraphrase is equivalent to retraining; it is explicitly the
  weakest form of version diversity and is discussed as part of the finding.
- No claim that cross-family redundancy is a solution: the cross-family panel matches its
  independence prediction but does not beat the same-model panel in absolute error.
- No claim that relatedness does not matter for checkers. The claim is that relatedness
  and capability are confounded in a three-model pool, so this experiment cannot separate
  them; the pre-registered ordering is reported as missed.

## Pre-registered vs exploratory

**Pre-registered** (README, before the run): the four success bars, the estimators
(φ, Mantel–Haenszel pooling, the independent-fleet decomposition, the redundancy curve,
effective redundancy, catch rate), the panel compositions, 2000 bootstrap reps, 3000
permutation shuffles, α = 0.05, and the rule that difficulty labels come only from the
generator.

**Exploratory** (labelled as such where they appear): the per-task-type breakdown, the
four-cell checker design (the bar concerns only the ordering), and the model-check
comparison of predicted against observed panel error in §6.3.

**Amendments**: one, dated 2026-10-27 and recorded at the end of `common-cause-fleet/README.md`.
It fixed a rendering ambiguity in the exception-classification task (a missing-PO case
still displayed the purchase order's quantity and price) before the main run. No
tolerance, margin, distractor count or stratum label changed; the battery was regenerated
from the same seed, so the fix applies uniformly to all 408 cases; no smoke-test outcome
enters any estimate.
