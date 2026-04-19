import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


class PortfolioOptimizer:
    """Modern Portfolio Theory optimizer with efficient frontier computation."""

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        risk_free_rate: float = 0.02,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.prices: pd.DataFrame | None = None
        self.returns: pd.DataFrame | None = None
        self.mean_returns: pd.Series | None = None
        self.cov_matrix: pd.DataFrame | None = None
        self.num_assets = len(tickers)

    def fetch_data(self) -> pd.DataFrame:
        """Download historical adjusted close prices from Yahoo Finance."""
        self.prices = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
        )["Close"]
        return self.prices

    def calculate_returns(self) -> pd.DataFrame:
        """Compute daily and annualized return statistics."""
        if self.prices is None:
            raise ValueError("Call fetch_data() first")
        self.returns = self.prices.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        return self.returns

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Annualized expected portfolio return."""
        return float(np.dot(weights, self.mean_returns))

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Annualized portfolio volatility (standard deviation)."""
        return float(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))))

    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Sharpe ratio of the portfolio."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol

    def efficient_frontier(
        self, num_portfolios: int = 5000, random_state: int | None = None
    ) -> pd.DataFrame:
        """Generate random portfolios to approximate the efficient frontier."""
        if self.mean_returns is None:
            raise ValueError("Call calculate_returns() first")

        rng = np.random.default_rng(random_state)
        results = []
        for _ in range(num_portfolios):
            weights = rng.dirichlet(np.ones(self.num_assets))
            ret = self.portfolio_return(weights)
            vol = self.portfolio_volatility(weights)
            sharpe = (ret - self.risk_free_rate) / vol
            results.append({
                "return": ret,
                "volatility": vol,
                "sharpe": sharpe,
                **{f"w_{t}": w for t, w in zip(self.tickers, weights)},
            })

        return pd.DataFrame(results)

    def optimize_sharpe(self) -> PortfolioResult:
        """Find the portfolio that maximizes the Sharpe ratio."""
        if self.mean_returns is None:
            raise ValueError("Call calculate_returns() first")

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1 / self.num_assets] * self.num_assets)

        result = minimize(
            fun=lambda w: -self.portfolio_sharpe(w),
            x0=initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            raise ValueError(f"Sharpe optimization failed: {result.message}")
        weights = result.x
        return PortfolioResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.portfolio_sharpe(weights),
        )

    def optimize_min_volatility(self) -> PortfolioResult:
        """Find the minimum volatility portfolio."""
        if self.mean_returns is None:
            raise ValueError("Call calculate_returns() first")

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial = np.array([1 / self.num_assets] * self.num_assets)

        result = minimize(
            fun=self.portfolio_volatility,
            x0=initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            raise ValueError(f"Min-volatility optimization failed: {result.message}")
        weights = result.x
        return PortfolioResult(
            weights=weights,
            expected_return=self.portfolio_return(weights),
            volatility=self.portfolio_volatility(weights),
            sharpe_ratio=self.portfolio_sharpe(weights),
        )
