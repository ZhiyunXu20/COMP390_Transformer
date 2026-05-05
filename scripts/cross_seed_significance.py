#!/usr/bin/env python3
"""
Cross-seed Welch t-tests: compare per-seed test metrics between two experiments
(e.g. fast_dot vs fast_add from results/ablation_per_seed.csv).

Complements scripts/significance_test.py (sentence-level paired bootstrap on one seed).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

_HAS_SCIPY = False
scipy_stats = None
try:
    from scipy import stats as scipy_stats  # type: ignore

    _HAS_SCIPY = True
except Exception:
    scipy_stats = None  # type: ignore[misc, assignment]
    _HAS_SCIPY = False


def _t_cdf_mpmath(t: float, df: float) -> float:
    """Student t CDF at t (df > 0), via mpmath regularized incomplete beta."""
    import mpmath as mp

    if df <= 0 or not math.isfinite(t):
        return float("nan")
    abs_t = abs(t)
    x = df / (abs_t * abs_t + df)
    ib = float(mp.betainc(df / 2, mp.mpf(0.5), 0, x, regularized=True))
    inner = 1.0 - 0.5 * ib
    if t >= 0:
        return inner
    return 1.0 - inner


def _t_pvalue_two_sided_any(t_stat: float, df: float) -> float | None:
    if not math.isfinite(t_stat) or not math.isfinite(df) or df <= 0:
        return None
    if _HAS_SCIPY and scipy_stats is not None:
        return float(2.0 * scipy_stats.t.sf(abs(t_stat), df))
    cdf = _t_cdf_mpmath(abs(t_stat), df)
    return float(2.0 * (1.0 - cdf))


def _t_ppf_any(df: float, cdf_target: float) -> float | None:
    """Inverse Student-t CDF: P(T <= t) = cdf_target."""
    if not math.isfinite(df) or df <= 0 or not (0 < cdf_target < 1):
        return None
    if _HAS_SCIPY and scipy_stats is not None:
        return float(scipy_stats.t.ppf(cdf_target, df))
    lo, hi = -1e4, 1e4
    flo, fhi = _t_cdf_mpmath(lo, df), _t_cdf_mpmath(hi, df)
    if not (flo < cdf_target < fhi):
        return None
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if _t_cdf_mpmath(mid, df) < cdf_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def stats_backend_name() -> str:
    return "scipy" if _HAS_SCIPY else "mpmath"


def welch_t_two_sample(
    a: list[float],
    b: list[float],
) -> tuple[float, float, float | None]:
    """Return (t_statistic, df, p_two_sided). Numerator: mean_A - mean_B."""
    na, nb = len(a), len(b)
    if na < 1 or nb < 1:
        raise ValueError("need at least one value per group")
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a) if na > 1 else 0.0
    var_b = statistics.variance(b) if nb > 1 else 0.0
    se_a2 = var_a / na
    se_b2 = var_b / nb
    se = math.sqrt(se_a2 + se_b2)
    if se == 0.0:
        return float("nan"), float("nan"), None
    t = (mean_a - mean_b) / se
    num = (se_a2 + se_b2) ** 2
    den = 0.0
    if na > 1:
        den += (se_a2**2) / (na - 1)
    if nb > 1:
        den += (se_b2**2) / (nb - 1)
    df = num / den if den > 0 else float("nan")

    p = _t_pvalue_two_sided_any(t, df) if math.isfinite(df) and df > 0 else None

    return t, df, p


def ci_mean_difference(
    a: list[float],
    b: list[float],
    df: float,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """95% CI for (mean_A - mean_B)."""
    na, nb = len(a), len(b)
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    var_a = statistics.variance(a) if na > 1 else 0.0
    var_b = statistics.variance(b) if nb > 1 else 0.0
    se = math.sqrt(var_a / na + var_b / nb)
    if not math.isfinite(df) or df <= 0:
        return None, None
    t_crit = _t_ppf_any(df, 1.0 - alpha / 2.0)
    if t_crit is None:
        return None, None
    delta = mean_a - mean_b
    return delta - t_crit * se, delta + t_crit * se


def conclusion_text(
    delta: float,
    p: float | None,
    experiment_a: str,
    experiment_b: str,
    alpha: float = 0.05,
) -> str:
    if p is None:
        return (
            f"exact p-value not computed; compare Welch t and df to a Student-t table at α={alpha} (two-sided)."
        )
    if p >= alpha:
        return (
            f"{experiment_a} is not significantly different from {experiment_b} at α={alpha} (two-sided)."
        )
    if delta > 0:
        return (
            f"{experiment_a} is significantly better than {experiment_b} at α={alpha} (two-sided)."
        )
    return (
        f"{experiment_a} is significantly worse than {experiment_b} at α={alpha} (two-sided)."
    )


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_metric_values(
    rows: list[dict], experiment: str, metric: str
) -> list[float]:
    keyed: list[tuple[int, float]] = []
    for row in rows:
        if row.get("experiment") != experiment:
            continue
        raw = row.get(metric)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            s = int(row["seed"])
        except (KeyError, ValueError):
            s = 0
        keyed.append((s, float(raw)))
    keyed.sort(key=lambda x: x[0])
    return [v for _, v in keyed]


def analyze_metric(
    metric: str,
    vals_a: list[float],
    vals_b: list[float],
    experiment_a: str,
    experiment_b: str,
) -> dict[str, Any]:
    na, nb = len(vals_a), len(vals_b)
    mean_a = statistics.mean(vals_a) if vals_a else float("nan")
    mean_b = statistics.mean(vals_b) if vals_b else float("nan")
    std_a = statistics.stdev(vals_a) if len(vals_a) > 1 else 0.0
    std_b = statistics.stdev(vals_b) if len(vals_b) > 1 else 0.0
    delta = mean_a - mean_b
    t, df, p = welch_t_two_sample(vals_a, vals_b)
    lo, hi = ci_mean_difference(vals_a, vals_b, df)

    return {
        "metric": metric,
        experiment_a: {
            "mean": mean_a,
            "std": std_a,
            "n": na,
            "values": vals_a,
        },
        experiment_b: {
            "mean": mean_b,
            "std": std_b,
            "n": nb,
            "values": vals_b,
        },
        "mean_difference_a_minus_b": delta,
        "welch_t": t,
        "welch_df": df,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "p_value_two_sided": p,
        "conclusion": conclusion_text(delta, p, experiment_a, experiment_b),
    }


def build_markdown(
    experiment_a: str,
    experiment_b: str,
    results: list[dict],
    scipy_note: str,
) -> str:
    lines = [
        f"# Cross-seed significance: {experiment_a} vs {experiment_b}",
        "",
        "## Methodology note",
        "",
        "This is cross-**SEED** evidence using per-seed **final test** BLEU / chrF++ / COMET as "
        "**independent samples** (one value per training seed). It **complements** but does "
        "**NOT** replace the sentence-level paired bootstrap in `scripts/significance_test.py`, "
        "which measures **within–prediction-set** resampling variation on a **single** trained "
        "model. **Both** perspectives should be reported when discussing dot vs additive.",
        "",
    ]
    for r in results:
        m = r["metric"]
        lines.append(f"## {m}: results")
        lines.append("")
        ea = r[experiment_a]
        eb = r[experiment_b]
        lines.append(
            f"- **{experiment_a}**: mean={ea['mean']:.6g}, std={ea['std']:.6g}, n={ea['n']}, "
            f"values={ea['values']}"
        )
        lines.append(
            f"- **{experiment_b}**: mean={eb['mean']:.6g}, std={eb['std']:.6g}, n={eb['n']}, "
            f"values={eb['values']}"
        )
        lines.append(
            f"- mean difference: {r['mean_difference_a_minus_b']:.6g} ({experiment_a} minus {experiment_b})"
        )
        lines.append(f"- Welch t: {r['welch_t']:.6g}, df ≈ {r['welch_df']:.6g}")
        if r["ci_95_low"] is not None and r["ci_95_high"] is not None:
            lines.append(
                f"- 95% CI for difference: [{r['ci_95_low']:.6g}, {r['ci_95_high']:.6g}]"
            )
        else:
            lines.append("- 95% CI for difference: (unavailable without scipy `t.ppf`)")
        if r["p_value_two_sided"] is not None:
            lines.append(f"- p-value (two-sided): {r['p_value_two_sided']:.6g}")
        else:
            lines.append(
                "- p-value: **exact p-value not computed; use t and df with a t-distribution table**"
            )
        lines.append(f"- Conclusion: {r['conclusion']}")
        lines.append("")
    lines.append("## All metrics summary")
    lines.append("")
    lines.append("| Metric | Δ (A-B) | 95% CI | t | df | p |")
    lines.append("|--------|---------|--------|---|---|---|")
    for r in results:
        d = r["mean_difference_a_minus_b"]
        lo, hi = r["ci_95_low"], r["ci_95_high"]
        ci = f"[{lo:.4g}, {hi:.4g}]" if lo is not None and hi is not None else "—"
        pv = f"{r['p_value_two_sided']:.4g}" if r["p_value_two_sided"] is not None else "—"
        lines.append(
            f"| {r['metric']} | {d:.6g} | {ci} | {r['welch_t']:.6g} | {r['welch_df']:.6g} | {pv} |"
        )
    lines.append("")
    lines.append(scipy_note)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=str, default="results/ablation_per_seed.csv")
    ap.add_argument("--experiment-a", type=str, default="fast_dot")
    ap.add_argument("--experiment-b", type=str, default="fast_add")
    ap.add_argument("--metrics", type=str, default="BLEU chrF++ COMET")
    ap.add_argument("--output-md", type=str, default="results/cross_seed_significance.md")
    ap.add_argument("--output-json", type=str, default="results/cross_seed_significance.json")
    args = ap.parse_args()

    repo = REPO_ROOT
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo / csv_path
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    metrics = args.metrics.split()
    rows = load_csv_rows(csv_path)

    experiment_a = args.experiment_a
    experiment_b = args.experiment_b

    results: list[dict] = []
    for metric in metrics:
        va = collect_metric_values(rows, experiment_a, metric)
        vb = collect_metric_values(rows, experiment_b, metric)
        if not va or not vb:
            print(f"warning: insufficient data for metric={metric!r}", file=sys.stderr)
            continue
        results.append(analyze_metric(metric, va, vb, experiment_a, experiment_b))

    scipy_note = (
        "Two-sided p-values and 95% CIs: **SciPy** `scipy.stats.t`."
        if _HAS_SCIPY
        else (
            "SciPy unavailable (import failed); using **mpmath** "
            "Student-t CDF (regularized incomplete beta) and bisection for `ppf`."
        )
    )

    payload: dict[str, Any] = {
        "experiment_a": experiment_a,
        "experiment_b": experiment_b,
        "csv": str(csv_path.resolve()),
        "scipy_available": _HAS_SCIPY,
        "stats_backend": stats_backend_name(),
        "metrics": {r["metric"]: r for r in results},
    }

    out_md = Path(args.output_md)
    out_json = Path(args.output_json)
    if not out_md.is_absolute():
        out_md = repo / out_md
    if not out_json.is_absolute():
        out_json = repo / out_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        build_markdown(experiment_a, experiment_b, results, scipy_note),
        encoding="utf-8",
    )
    print(f"Wrote {out_json}", file=sys.stderr)
    print(f"Wrote {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
