"""Estimators for common-cause failure in agent fleets.

Notation
--------
For a pair of agents (a, b) observed on the same N cases, the 2x2 table of
pass/fail is (n11, n10, n01, n00) with 1 = failure.  The phi coefficient

    phi = (n11 n00 - n10 n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))

is the pairwise failure correlation.  When the two agents have equal marginal
failure probabilities p, the beta-factor inversion

    beta = (p11 - p^2) / (p - p^2)

is algebraically identical to phi, because p11 - p^2 = phi * p(1-p) and
p - p^2 = p(1-p).  The beta factor is therefore not a new quantity bolted onto
the reliability literature: under equal marginals it *is* the correlation, and
that identity is what licenses reading phi as a common-cause share.  For
unequal marginals we use the normalised excess joint failure

    beta_norm = (p11 - p_a p_b) / (min(p_a, p_b) - p_a p_b)

which reduces to phi when p_a = p_b.

Difficulty handling
-------------------
Hard cases fail together under *any* design, including a genuinely independent
one.  Raw pooled phi therefore overstates common cause.  We report
(i) stratum-specific phi, (ii) a Mantel-Haenszel weighted pooled phi, and
(iii) `phi_indep_mix`, the pooled phi that a conditionally independent fleet
with the same per-stratum marginals and the same stratum mix would produce.
The difference between the raw pooled value and `phi_indep_mix` is the part of
the correlation that difficulty alone cannot explain.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

STRATA = ["easy", "medium", "hard"]

# anchor = M1; partner varies in relatedness
PAIRS = {
    # same model, only seed and prompt paraphrase differ
    "same_model": [("M1", i, "M1", j) for i, j in itertools.combinations(range(5), 2)],
    # same family, different size and tuning
    "same_family": [("M1", s, "M2", s) for s in (1, 3)],
    # different pretraining family
    "cross_family": [("M1", s, "M3", s) for s in range(5)],
    # replication of the same-model level on the second model
    "same_model_M3": [("M3", i, "M3", j) for i, j in itertools.combinations(range(5), 2)],
}

PANELS = {
    "same_model": ["M1", "M1", "M1", "M1", "M1"],
    "same_family": ["M1", "M2", "M1", "M2", "M1"],
    "cross_family": ["M1", "M3", "M1", "M3", "M1"],
}
PANEL_K = [1, 3, 5]


# ---------------------------------------------------------------------------
# basic 2x2 machinery
# ---------------------------------------------------------------------------
def phi_from_counts(n11, n10, n01, n00):
    """Vectorised phi coefficient; NaN where a margin is degenerate."""
    n11 = np.asarray(n11, dtype=float)
    n10 = np.asarray(n10, dtype=float)
    n01 = np.asarray(n01, dtype=float)
    n00 = np.asarray(n00, dtype=float)
    num = n11 * n00 - n10 * n01
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


def table(a, b):
    """Counts from two indicator arrays (1.0 = failure, 0.0 = pass, NaN = unobserved).

    NaN is neither a failure nor a pass, so callers must filter it out; here it is
    simply treated as "not 1" and "not 0" would be wrong, hence the explicit tests.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    af = a == 1.0
    bf = b == 1.0
    ap = a == 0.0
    bp = b == 0.0
    return (int(np.sum(af & bf)), int(np.sum(af & bp)),
            int(np.sum(ap & bf)), int(np.sum(ap & bp)))


def mh_weights(n1, n0, n):
    """Mantel-Haenszel weights for a set of strata."""
    n1 = np.asarray(n1, float)
    n0 = np.asarray(n0, float)
    n = np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(n > 0, n1 * n0 / n, 0.0)
    return w


def pooled_phi_by_stratum(a, b, strata):
    """phi within each stratum, plus the MH-weighted combination.

    a, b : (N,) boolean failure indicators, aligned on cases
    strata : (N,) integer stratum index
    """
    out = {}
    phis, ws = [], []
    for s in np.unique(strata):
        m = strata == s
        n11, n10, n01, n00 = table(a[m], b[m])
        phis.append(float(phi_from_counts(n11, n10, n01, n00)))
        ws.append(float(mh_weights(n11 + n10, n01 + n00, n11 + n10 + n01 + n00)))
        out[int(s)] = dict(phi=phis[-1], n=int(m.sum()),
                           n11=int(n11), n10=int(n10), n01=int(n01), n00=int(n00))
    phis = np.array(phis, float)
    ws = np.array(ws, float)
    ok = ~np.isnan(phis) & (ws > 0)
    mh = float(np.sum(phis[ok] * ws[ok]) / np.sum(ws[ok])) if ok.any() else float("nan")
    n11, n10, n01, n00 = table(a, b)
    out["pooled"] = dict(phi=float(phi_from_counts(n11, n10, n01, n00)),
                         n=int(a.size), n11=int(n11), n10=int(n10),
                         n01=int(n01), n00=int(n00))
    out["mh"] = mh
    return out


def beta_equal(p11, p):
    """Beta-factor inversion for equal marginals: (p11 - p^2) / (p - p^2)."""
    den = p - p * p
    return (p11 - p * p) / den if den > 0 else float("nan")


def beta_norm(p11, pa, pb):
    """Normalised excess joint failure for unequal marginals."""
    den = min(pa, pb) - pa * pb
    return (p11 - pa * pb) / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# independence counterfactual
# ---------------------------------------------------------------------------
def phi_indep_mix(pa_s, pb_s, n_s, rng, reps=2000):
    """Pooled phi produced by a conditionally independent pair.

    Draws independent Bernoulli failures at the observed per-stratum marginal
    rates, with the observed stratum sizes, and reports the pooled phi.  This is
    the correlation that difficulty alone manufactures.
    """
    vals = []
    keep = [(pa, pb, n) for pa, pb, n in zip(pa_s, pb_s, n_s)
            if n > 0 and np.isfinite(pa) and np.isfinite(pb)]
    if not keep:
        return float("nan"), float("nan"), float("nan")
    for _ in range(reps):
        n11 = n10 = n01 = n00 = 0.0
        for pa, pb, n in keep:
            a = rng.random(n) < pa
            b = rng.random(n) < pb
            c = table(a, b)
            n11 += c[0]; n10 += c[1]; n01 += c[2]; n00 += c[3]
        vals.append(phi_from_counts(n11, n10, n01, n00))
    v = np.array(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _phi_pool(a, b):
    """Pooled phi from two indicator arrays, ignoring strata."""
    ok = np.isfinite(a) & np.isfinite(b)
    n11, n10, n01, n00 = table(a[ok], b[ok])
    return float(phi_from_counts(n11, n10, n01, n00))


def permutation_p(a, b, strata, rng, reps=5000):
    """One-sided p for phi > 0 under conditional independence.

    Shuffles b within each stratum, which preserves the difficulty structure of
    both agents and destroys only their case-level association.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    obs = _phi_pool(a, b)
    ok = np.isfinite(a) & np.isfinite(b)
    blocks = [np.where(ok & (strata == s))[0] for s in np.unique(strata)]
    blocks = [m for m in blocks if m.size > 1]
    ge = 0
    for _ in range(reps):
        bp = b.copy()
        for m in blocks:
            bp[m] = b[rng.permutation(m)]
        v = _phi_pool(a, bp)
        if np.isfinite(v) and v >= obs:
            ge += 1
    return (ge + 1) / (reps + 1), obs


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def majority_fail(outcomes):
    """outcomes: (N, k) boolean failures -> panel failure (k odd)."""
    k = outcomes.shape[1]
    return outcomes.sum(axis=1) > (k // 2)


def indep_majority_fail(p_members):
    """Exact P(majority failure) under independence with member marginals p.

    Poisson-binomial DP over the number of failing members; O(k^2), exact, and
    valid for unequal marginals.  (Enumerating subsets is 2^k and unusable once
    k_effective probes large panel sizes.)
    """
    p = np.asarray(p_members, dtype=float)
    if p.size == 0 or not np.all(np.isfinite(p)):
        return float("nan")
    dist = np.array([1.0])
    for pi in p:
        nxt = np.zeros(dist.size + 1)
        nxt[:-1] += dist * (1.0 - pi)
        nxt[1:] += dist * pi
        dist = nxt
    k = p.size
    thr = k // 2 + 1
    return float(dist[thr:].sum())


def k_effective(p_mean, target_error, max_k=41):
    """Smallest odd panel size at which an independent homogeneous panel would
    reach `target_error`; linearly interpolated between bracketing odd sizes.

    Returns NaN if the target is unreachable within max_k, and 1.0 if the
    observed panel is no better than a single agent.
    """
    if not (np.isfinite(p_mean) and np.isfinite(target_error)):
        return float("nan")
    if target_error >= p_mean:      # panel no better than one independent agent
        return 1.0
    prev_k, prev_e = 1, p_mean
    for k in range(3, max_k + 1, 2):
        e = indep_majority_fail([p_mean] * k)
        if e <= target_error:
            if prev_e <= e:
                return float(k)
            frac = (prev_e - target_error) / (prev_e - e) if prev_e > e else 1.0
            return float(max(1.0, prev_k + frac * (k - prev_k)))
        prev_k, prev_e = k, e
    return float(max_k)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_idx(n, rng, reps):
    for _ in range(reps):
        yield rng.integers(0, n, n)


def ci(vals, lo=2.5, hi=97.5):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))],
                   float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
