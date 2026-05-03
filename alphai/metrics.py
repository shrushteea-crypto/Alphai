"""
metrics.py — Evaluation helpers for interval forecasts.
"""


def winkler_score(
    low:    float,
    high:   float,
    actual: float,
    alpha:  float = 0.05,
) -> float:
    """
    Winkler Interval Score — penalises both wide intervals and misses.

    Lower score = better (narrower *and* accurate).

    Formula
    -------
    score = (high - low)
          + (2/alpha) * max(0, low  - actual)   # penalty for under-predicting
          + (2/alpha) * max(0, actual - high)    # penalty for over-predicting

    Parameters
    ----------
    low, high : float
        Predicted interval bounds.
    actual : float
        Realised value.
    alpha : float
        Significance level matching (1 - confidence).  Default 0.05 ↔ 95 % CI.

    Returns
    -------
    float
        Winkler score (non-negative).
    """
    width   = high - low
    penalty = (2.0 / alpha) * (max(0.0, low - actual) + max(0.0, actual - high))
    return width + penalty


def coverage_rate(results: list[dict]) -> float:
    """
    Fraction of backtest rows where ``covered`` is True.

    Parameters
    ----------
    results : list[dict]
        Each dict must have a boolean ``"covered"`` key.

    Returns
    -------
    float  in [0, 1]
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["covered"]) / len(results)
