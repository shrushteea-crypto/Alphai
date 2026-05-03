"""
model.py — GBM + Student-t forecast engine.

Core statistical model used by both the live dashboard and the backtest.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ForecastResult:
    """Output of a single GBM + Student-t price range prediction."""
    low:           float
    high:          float
    mean:          float
    sigma_ann:     float   # annualised volatility (for display)
    current_price: float   # input price to compute relative metrics

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def expected_move_usd(self) -> float:
        # Maximum expected absolute deviation from current price based on range
        return max(abs(self.high - self.current_price), abs(self.current_price - self.low))

    @property
    def expected_move_pct(self) -> float:
        return (self.expected_move_usd / self.current_price) * 100.0


def predict_range(
    window_returns: np.ndarray,
    current_price:  float,
    confidence:     float = 0.95,
    n_sims:         int   = 20_000,
    horizon:        int   = 1,
) -> ForecastResult:
    """
    Forecast the next-step price range using Geometric Brownian Motion (GBM)
    with a Student-t shock distribution.

    Why GBM?
    --------
    GBM models price as a random walk where *percentage* changes are random,
    so prices stay positive and compound naturally.  It is the backbone of
    Black-Scholes and standard in quantitative finance.

    Why Student-t instead of Normal?
    ---------------------------------
    BTC (and most financial assets) exhibit fat tails — extreme moves occur
    far more often than a Gaussian predicts.  The Student-t distribution has
    heavier tails controlled by its degrees-of-freedom parameter (df).
    ``scipy.stats.t.fit()`` learns df directly from recent returns.

    Simulation flow
    ---------------
    1. Fit Student-t to `window_returns`  →  (df, loc, scale)
    2. Draw `n_sims` shocks from that distribution
    3. One GBM step:  S(t+h) = S(t) * exp((mu − 0.5σ²)*h + shock)
    4. Return the confidence-interval percentiles of the simulated paths

    Parameters
    ----------
    window_returns : np.ndarray
        Recent log-returns used to fit the distribution.
    current_price : float
        Most recent close price.
    confidence : float
        Confidence level, e.g. 0.95 for 95 % CI.
    n_sims : int
        Number of Monte Carlo paths.
    horizon : int
        Number of steps ahead (default 1 = next hour).

    Returns
    -------
    ForecastResult
    """
    # Fit Student-t distribution to the window of log-returns
    df_t, loc_t, scale_t = stats.t.fit(window_returns, floc=0)

    mu    = float(np.mean(window_returns))
    sigma = float(np.std(window_returns, ddof=1))

    shocks = stats.t.rvs(df=df_t, loc=loc_t, scale=scale_t, size=n_sims)

    # GBM one-step price paths (Itô correction keeps the mean unbiased)
    future_prices = current_price * np.exp(
        (mu - 0.5 * sigma ** 2) * horizon + shocks
    )

    alpha = 1.0 - confidence
    low  = float(np.percentile(future_prices, 100 * alpha / 2))
    high = float(np.percentile(future_prices, 100 * (1 - alpha / 2)))
    mean = float(np.mean(future_prices))

    # Annualised volatility: hourly σ × √8760
    sigma_ann = sigma * math.sqrt(8760)

    return ForecastResult(low=low, high=high, mean=mean, sigma_ann=sigma_ann, current_price=current_price)
