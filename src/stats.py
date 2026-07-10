"""Shared statistics helpers.

Project rigor rule: no rate is ever reported without its raw n and a 95%
confidence interval — fmt_rate() is the only sanctioned way to print one.
"""

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def fmt_rate(k: int, n: int) -> str:
    """'k/n = p% (95% CI lo–hi%)' — the only way a rate gets printed."""
    if n == 0:
        return f"{k}/0 = n/a (no samples)"
    lo, hi = wilson_ci(k, n)
    return f"{k:,}/{n:,} = {k / n:.2%} (95% CI {lo:.2%}–{hi:.2%})"


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters over the same items."""
    if len(a) != len(b) or not a:
        raise ValueError("raters must cover the same non-empty item set")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_yes = sum(a) / n
    pb_yes = sum(b) / n
    expected = pa_yes * pb_yes + (1 - pa_yes) * (1 - pb_yes)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(counts: list[tuple[int, int]]) -> float:
    """Fleiss' kappa for binary ratings.

    counts: per item, (n_yes, n_no). Every item must have the same total
    number of raters.
    """
    if not counts:
        raise ValueError("no items")
    n_raters = counts[0][0] + counts[0][1]
    if n_raters < 2:
        raise ValueError("need at least 2 raters")
    if any(yes + no != n_raters for yes, no in counts):
        raise ValueError("all items must have the same number of raters")

    n_items = len(counts)
    p_yes = sum(yes for yes, _ in counts) / (n_items * n_raters)
    p_no = 1 - p_yes

    p_i = [
        (yes * (yes - 1) + no * (no - 1)) / (n_raters * (n_raters - 1))
        for yes, no in counts
    ]
    p_bar = sum(p_i) / n_items
    p_e = p_yes * p_yes + p_no * p_no
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)
