"""Turn raw JSONL outcomes into the tables the paper reports.

Writes results/*.csv and results/summary.json.  Every number in the paper comes
from these files; nothing is typed by hand.

  python3 stats/analyse.py [--reps 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import estimators as E

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TAGS = {"qwen2.5:14b-instruct": "M1", "qwen2.5-coder:32b": "M2", "llama3.1:8b": "M3"}
LEVELS = ["same_model", "same_family", "cross_family", "same_model_M3"]


# ---------------------------------------------------------------------------
def load_grid(path: Path):
    cells, meta = {}, {}
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            tag = TAGS.get(r["model"])
            if tag is None:
                continue
            cells[(r["case_id"], tag, r["slot"])] = 0.0 if r["correct"] else 1.0
            meta[r["case_id"]] = dict(task_type=r["task_type"],
                                      difficulty=r["difficulty"])
    return cells, meta


EXPECTED_KEYS = ([("M1", s) for s in range(5)] + [("M3", s) for s in range(5)]
                 + [("M2", s) for s in (1, 3)])


def build_matrices(cells, case_ids):
    """MAT[(tag, slot)] -> (N,) failure indicator, NaN where unobserved.

    All expected slots are allocated up front so that a partially finished run
    still analyses, with the missing arms contributing NaN rather than a crash.
    """
    keys = set(EXPECTED_KEYS) | set((t, s) for (_, t, s) in cells)
    MAT = {}
    for (tag, slot) in keys:
        v = np.full(len(case_ids), np.nan)
        for i, cid in enumerate(case_ids):
            x = cells.get((cid, tag, slot))
            if x is not None:
                v[i] = x
        MAT[(tag, slot)] = v
    return MAT


def level_arrays(MAT, pairs):
    A = np.stack([MAT[(ta, sa)] for ta, sa, _, _ in pairs], axis=1)
    B = np.stack([MAT[(tb, sb)] for _, _, tb, sb in pairs], axis=1)
    return A, B


def pooled_phi(A, B, idx, strata=None, stratum=None):
    a = A[idx]
    b = B[idx]
    if stratum is not None:
        m = (strata[idx] == stratum)
        a, b = a[m], b[m]
    a = a.ravel(); b = b.ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 5:
        return float("nan"), 0
    n11, n10, n01, n00 = E.table(a[ok], b[ok])
    return float(E.phi_from_counts(n11, n10, n01, n00)), int(ok.sum())


def mh_phi(A, B, idx, strata):
    phis, ws = [], []
    for st in range(3):
        m = strata[idx] == st
        if m.sum() == 0:
            continue
        a = A[idx][m].ravel(); b = B[idx][m].ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 5:
            continue
        n11, n10, n01, n00 = E.table(a[ok], b[ok])
        ph = float(E.phi_from_counts(n11, n10, n01, n00))
        w = float(E.mh_weights(n11 + n10, n01 + n00, n11 + n10 + n01 + n00))
        phis.append(ph); ws.append(w)
    phis = np.array(phis); ws = np.array(ws)
    ok = ~np.isnan(phis) & (ws > 0)
    return float(np.sum(phis[ok] * ws[ok]) / np.sum(ws[ok])) if ok.any() else float("nan")


def strata_marginals(A, B, idx, strata):
    pa, pb, ns = [], [], []
    for st in range(3):
        m = strata[idx] == st
        if m.sum() == 0:
            pa.append(0.0); pb.append(0.0); ns.append(0)
            continue
        a = A[idx][m].ravel(); b = B[idx][m].ravel()
        ok = ~np.isnan(a) & ~np.isnan(b)
        pa.append(float(np.nanmean(a[ok])) if ok.any() else 0.0)
        pb.append(float(np.nanmean(b[ok])) if ok.any() else 0.0)
        ns.append(int(ok.sum()))
    return pa, pb, ns


def beta_from(A, B, idx):
    a = A[idx].ravel(); b = B[idx].ravel()
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 5:
        return float("nan"), float("nan")
    pa, pb = float(np.mean(a[ok])), float(np.mean(b[ok]))
    p11 = float(np.mean((a[ok] == 1.0) & (b[ok] == 1.0)))
    return E.beta_equal(p11, (pa + pb) / 2), E.beta_norm(p11, pa, pb)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(20251027)

    cells, meta = load_grid(RESULTS / "grid.jsonl")
    case_ids = sorted(meta)
    N = len(case_ids)
    strata = np.array([E.STRATA.index(meta[c]["difficulty"]) for c in case_ids])
    MAT = build_matrices(cells, case_ids)

    summary = dict(
        n_cases=N, reps=args.reps,
        n_cases_by_stratum={E.STRATA[s]: int((strata == s).sum()) for s in range(3)},
        n_observations=len(cells),
    )

    # ---------------- per-model marginals (manipulation check) ----------------
    acc_rows = []
    for tag in ("M1", "M2", "M3"):
        slots = sorted({s for (_, t, s) in cells if t == tag})
        for slot in slots:
            row = dict(model=tag, slot=slot)
            for st in range(3):
                v = [cells[(c, tag, slot)] for c in case_ids
                     if (c, tag, slot) in cells and E.STRATA.index(meta[c]["difficulty"]) == st]
                row[f"err_{E.STRATA[st]}"] = float(np.mean(v)) if v else float("nan")
                row[f"n_{E.STRATA[st]}"] = len(v)
            acc_rows.append(row)
    write_csv(RESULTS / "model_marginals.csv", acc_rows)

    # ---------------- pairwise phi -------------------------------------------
    pair_rows, level_rows = [], []
    summary["phi"] = {}
    boot = defaultdict(lambda: defaultdict(list))   # level -> metric -> [vals]

    for level in LEVELS:
        pairs = E.PAIRS[level]
        A, B = level_arrays(MAT, pairs)
        for pi, (ta, sa, tb, sb) in enumerate(pairs):
            for st in list(range(3)) + [None]:
                idx = np.arange(N)
                ph, n = pooled_phi(A[:, [pi]], B[:, [pi]], idx, strata, st)
                if n < 10:
                    continue
                a = A[:, pi]; b = B[:, pi]
                m = np.ones(N, bool) if st is None else (strata == st)
                ok = ~np.isnan(a) & ~np.isnan(b) & m
                pair_rows.append(dict(
                    level=level, pair=f"{ta}s{sa}-{tb}s{sb}",
                    stratum=E.STRATA[st] if st is not None else "all",
                    n=int(ok.sum()), phi=ph,
                    p_a=float(np.nanmean(a[ok])), p_b=float(np.nanmean(b[ok]))))

        idx = np.arange(N)
        raw, n_obs = pooled_phi(A, B, idx)
        mh = mh_phi(A, B, idx, strata)
        pa_s, pb_s, ns = strata_marginals(A, B, idx, strata)
        ind_mean, ind_lo, ind_hi = E.phi_indep_mix(pa_s, pb_s, ns, rng, reps=args.reps)
        if n_obs >= 40:
            p_perm, _ = E.permutation_p(A.ravel(), B.ravel(),
                                        np.repeat(strata, A.shape[1]), rng, reps=3000)
        else:
            p_perm = float("nan")
        b_eq, b_norm = beta_from(A, B, idx)
        _a, _b = A.ravel(), B.ravel()
        _ok = ~np.isnan(_a) & ~np.isnan(_b)
        p_a = float(np.mean(_a[_ok])); p_b = float(np.mean(_b[_ok]))
        p_11 = float(np.mean((_a[_ok] == 1.0) & (_b[_ok] == 1.0)))

        strat = {}
        for st in range(3):
            ph, n = pooled_phi(A, B, idx, strata, st)
            strat[E.STRATA[st]] = dict(phi=ph, n=int(n))
            level_rows.append(dict(level=level, stratum=E.STRATA[st], phi=ph, n=int(n)))
        level_rows.append(dict(level=level, stratum="all", phi=raw, n=int(n_obs)))

        # bootstrap
        for _ in range(args.reps):
            bi = rng.integers(0, N, N)
            boot[level]["raw"].append(pooled_phi(A, B, bi)[0])
            boot[level]["mh"].append(mh_phi(A, B, bi, strata))
            for st in range(3):
                boot[level][f"strat_{E.STRATA[st]}"].append(
                    pooled_phi(A, B, bi, strata, st)[0])
            pa2, pb2, ns2 = strata_marginals(A, B, bi, strata)
            boot[level]["indep"].append(E.phi_indep_mix(pa2, pb2, ns2, rng, reps=1)[0])
            boot[level]["beta"].append(beta_from(A, B, bi)[0])

        summary["phi"][level] = dict(
            phi_pooled=raw, phi_pooled_ci=list(E.ci(boot[level]["raw"])),
            phi_mh=mh, phi_mh_ci=list(E.ci(boot[level]["mh"])),
            phi_indep_mix=ind_mean, phi_indep_mix_ci=[ind_lo, ind_hi],
            phi_indep_mix_boot_ci=list(E.ci(boot[level]["indep"])),
            excess_over_difficulty=raw - ind_mean,
            frac_pooled_explained_by_difficulty=(ind_mean / raw if raw > 0 else None),
            perm_p_one_sided=p_perm,
            beta_equal=b_eq, beta_equal_ci=list(E.ci(boot[level]["beta"])),
            beta_norm=b_norm, p_a=p_a, p_b=p_b, p11=p_11, n_pairs=len(pairs),
            n_pair_observations=int(n_obs), by_stratum=strat,
            strata_ci={E.STRATA[st]: list(E.ci(boot[level][f"strat_{E.STRATA[st]}"]))
                       for st in range(3)},
        )
        print(f"  {level:14s} pooled={raw:+.3f} mh={mh:+.3f} "
              f"indep={ind_mean:+.3f} p={p_perm:.4f} beta={b_eq:+.3f}", flush=True)

    write_csv(RESULTS / "phi_pairs.csv", pair_rows)
    write_csv(RESULTS / "phi_levels.csv", level_rows)

    # ---------------- redundancy curves --------------------------------------
    summary["panels"] = {}
    red_rows = []
    F = {}
    for config, members in E.PANELS.items():
        for k in E.PANEL_K:
            F[(config, k)] = np.stack([MAT[(members[j], j)] for j in range(k)], axis=1)

    def panel_stats(Fk):
        complete = ~np.isnan(Fk).any(axis=1)
        pf = E.majority_fail(np.nan_to_num(Fk, nan=0.0))
        return complete, pf

    def indep_pred(Fk, complete, per_stratum=True):
        k = Fk.shape[1]
        if not per_stratum:
            p = [float(np.nanmean(Fk[complete, j])) for j in range(k)]
            return E.indep_majority_fail(p)
        tot, e = 0, 0.0
        for st in range(3):
            m = complete & (strata == st)
            if m.sum() == 0:
                continue
            p = [float(np.nanmean(Fk[m, j])) for j in range(k)]
            e += m.sum() * E.indep_majority_fail(p)
            tot += m.sum()
        return e / tot if tot > 0 else float("nan")

    for config in E.PANELS:
        per_k = {}
        for k in E.PANEL_K:
            Fk = F[(config, k)]
            complete, pf = panel_stats(Fk)
            e_real = float(np.mean(pf[complete]))
            per_k[k] = dict(
                error=e_real, n=int(complete.sum()),
                indep_stratified=indep_pred(Fk, complete, True),
                indep_global=indep_pred(Fk, complete, False),
                member_marginals=[float(np.nanmean(Fk[complete, j])) for j in range(k)],
            )
            red_rows.append(dict(config=config, k=k, realised_error=e_real,
                                 indep_stratified=per_k[k]["indep_stratified"],
                                 indep_global=per_k[k]["indep_global"],
                                 realised_gain=per_k[1]["error"] - e_real,
                                 nominal_gain=per_k[1]["error"] - per_k[k]["indep_stratified"],
                                 n=int(complete.sum())))
        e1, e5 = per_k[1]["error"], per_k[5]["error"]
        i5 = per_k[5]["indep_stratified"]
        p_mean = float(np.mean(per_k[5]["member_marginals"]))
        keff = E.k_effective(p_mean, e5)
        summary["panels"][config] = dict(
            per_k={str(k): v for k, v in per_k.items()},
            ratio_k5=(e5 / i5 if i5 > 0 else None),
            gain_recovery_k5=((e1 - e5) / (e1 - i5) if e1 > i5 else None),
            k_eff=keff, k_eff_frac=keff / 5, p_mean_k5=p_mean,
        )
        def _f(x):
            return "nan" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"
        print(f"  {config:13s} e1={_f(e1)} e3={_f(per_k[3]['error'])} e5={_f(e5)} "
              f"indep5={_f(i5)} ratio={_f(summary['panels'][config]['ratio_k5'])} "
              f"rec={_f(summary['panels'][config]['gain_recovery_k5'])} "
              f"k_eff={_f(keff)}", flush=True)

    # joint bootstrap for panel quantities
    for _ in range(args.reps):
        bi = rng.integers(0, N, N)
        for config in E.PANELS:
            for k in E.PANEL_K:
                Fk = F[(config, k)][bi]
                complete, pf = panel_stats(Fk)
                if complete.sum() == 0:
                    continue
                boot["panel"][f"{config}_{k}"].append(float(np.mean(pf[complete])))
    for config in E.PANELS:
        for k in E.PANEL_K:
            v = boot["panel"][f"{config}_{k}"]
            summary["panels"][config]["per_k"][str(k)]["error_ci"] = list(E.ci(v))
        # recovery CI
        recs = []
        for r_i in range(len(boot["panel"][f"{config}_5"])):
            e1 = boot["panel"][f"{config}_1"][r_i]
            e5 = boot["panel"][f"{config}_5"][r_i]
            i5 = summary["panels"][config]["per_k"]["5"]["indep_stratified"]
            if e1 > i5:
                recs.append((e1 - e5) / (e1 - i5))
        summary["panels"][config]["gain_recovery_k5_ci"] = list(E.ci(recs))
        ratios = [boot["panel"][f"{config}_5"][r_i] / summary["panels"][config]["per_k"]["5"]["indep_stratified"]
                  for r_i in range(len(boot["panel"][f"{config}_5"]))]
        summary["panels"][config]["ratio_k5_ci"] = list(E.ci(ratios))
    write_csv(RESULTS / "redundancy.csv", red_rows)

    # ---------------- checker economics --------------------------------------
    chk_path = RESULTS / "checkers.jsonl"
    chk = [json.loads(l) for l in chk_path.open()] if chk_path.exists() else []
    chk = [r for r in chk if not r.get("error")]
    by_cond = defaultdict(list)
    for r in chk:
        by_cond[r["condition"]].append(r)
    conds = ["same-same", "same-cross", "cross-same", "cross-cross"]
    summary["checkers"] = {}
    chk_rows = []
    for cond in conds:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        wrong = [r for r in rs if not r["author_correct"]]
        right = [r for r in rs if r["author_correct"]]

        def _final_correct(r):
            # Realistic system: on "accept"/unparseable the author's answer stands;
            # on "correct" the checker's revised answer replaces it.
            if r.get("decision") == "correct":
                return bool(r["checker_correct"])
            return bool(r["author_correct"])

        base = float(np.mean([bool(r["author_correct"]) for r in rs]))
        catch = float(np.mean([r["checker_correct"] for r in wrong])) if wrong else float("nan")
        summary["checkers"][cond] = dict(
            n=len(rs), n_wrong=len(wrong),
            author_accuracy=base,
            catch=catch,
            changed=float(np.mean([r["decision"] == "correct" for r in wrong])) if wrong else float("nan"),
            false_alarm=float(np.mean([r["decision"] == "correct" and not r["checker_correct"]
                                       for r in right])) if right else float("nan"),
            # realistic post-checker accuracy of the final answer
            net=float(np.mean([_final_correct(r) for r in rs])),
            # legacy union bound P(author correct OR checker correct), reported only
            # as a diagnostic; it overstates the system because it credits a case as
            # correct whenever EITHER agent happens to be individually right.
            net_union=float(np.mean([bool(r["author_correct"]) or bool(r["checker_correct"]) for r in rs])),
            author_error=len(wrong) / len(rs),
        )
        chk_rows.append(dict(condition=cond, n=len(rs), n_author_wrong=len(wrong),
                             author_accuracy=base,
                             author_error_rate=len(wrong) / len(rs),
                             catch_rate=summary["checkers"][cond]["catch"],
                             changed_rate=summary["checkers"][cond]["changed"],
                             false_alarm_rate=summary["checkers"][cond]["false_alarm"],
                             net_accuracy=summary["checkers"][cond]["net"],
                             net_accuracy_union=summary["checkers"][cond]["net_union"]))
    write_csv(RESULTS / "checkers.csv", chk_rows)

    # bootstrap CIs + ordering tests
    if all(c in by_cond for c in conds):
        lookup = {c: {r["case_id"]: r for r in by_cond[c]} for c in conds}
        common = sorted(set.intersection(*[set(lookup[c]) for c in conds]))
        catch_boot = {c: [] for c in conds}
        d1, d2, mono = [], [], []
        for _ in range(args.reps):
            sample = [common[i] for i in rng.integers(0, len(common), len(common))]
            rates = {}
            for c in conds:
                w = [lookup[c][cid] for cid in sample if not lookup[c][cid]["author_correct"]]
                rates[c] = float(np.mean([r["checker_correct"] for r in w])) if w else np.nan
                catch_boot[c].append(rates[c])
            mixed = np.nanmean([rates["same-cross"], rates["cross-same"]])
            d1.append(mixed - rates["same-same"])
            d2.append(rates["cross-cross"] - mixed)
            mono.append((rates["same-same"] < mixed) and (mixed < rates["cross-cross"]))
        for c in conds:
            summary["checkers"][c]["catch_ci"] = list(E.ci(catch_boot[c]))
        # Relatedness contrast.  NOTE on the pre-registered bar (iv): its stated
        # ordering "same-same < mixed < cross-cross" is mis-specified, because
        # cross-cross (M3 checking M3) is a SELF-REVIEW cell and therefore a
        # RELATED cell, not an unrelated one.  The related cells are
        # {same-same, cross-cross} (author and checker share model family) and the
        # unrelated cells are {same-cross, cross-same} (mixed family).  The honest
        # contrast is related minus unrelated, computed per bootstrap replicate.
        rel_arr = (np.array(catch_boot["same-same"]) + np.array(catch_boot["cross-cross"])) / 2
        unrel_arr = (np.array(catch_boot["same-cross"]) + np.array(catch_boot["cross-same"])) / 2
        rel_minus_unrel = rel_arr - unrel_arr
        summary["checker_ordering"] = dict(
            diff_ss_to_mixed=list(E.ci(d1)), diff_mixed_to_cc=list(E.ci(d2)),
            p_same_lt_mixed=float(np.mean(np.array(d1) > 0)),
            p_mixed_lt_cross=float(np.mean(np.array(d2) > 0)),
            p_monotone=float(np.mean(mono)),
            related_avg_ci=list(E.ci(rel_arr)), unrelated_avg_ci=list(E.ci(unrel_arr)),
            related_minus_unrelated_ci=list(E.ci(rel_minus_unrel)),
            p_related_lt_unrelated=float(np.mean(rel_minus_unrel < 0)),
        )

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nwrote results/{model_marginals,phi_pairs,phi_levels,redundancy,checkers}.csv + summary.json")


if __name__ == "__main__":
    main()
