from src.optimizer import PortfolioOptimizer
from src.monte_carlo import MonteCarloSimulator
from src.visualization import (
    plot_efficient_frontier,
    plot_correlation_matrix,
    plot_portfolio_weights,
    plot_cumulative_returns,
)


def main() -> None:
    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "JPM", "GS"]

    print("=" * 60)
    print("Portfolio Optimization Engine")
    print("=" * 60)

    # Initialize and fetch data
    optimizer = PortfolioOptimizer(
        tickers=tickers,
        start_date="2020-01-01",
        end_date="2024-01-01",
        risk_free_rate=0.02,
    )
    print(f"\nFetching data for {', '.join(tickers)}...")
    optimizer.fetch_data()
    optimizer.calculate_returns()

    # Generate efficient frontier
    print("Generating efficient frontier (5000 portfolios)...")
    frontier = optimizer.efficient_frontier(num_portfolios=5000)

    # Optimize
    print("Optimizing portfolios...")
    max_sharpe = optimizer.optimize_sharpe()
    min_vol = optimizer.optimize_min_volatility()

    # Print results
    print("\n" + "-" * 60)
    print("MAX SHARPE RATIO PORTFOLIO")
    print("-" * 60)
    print(f"  Expected Return:  {max_sharpe.expected_return:.2%}")
    print(f"  Volatility:       {max_sharpe.volatility:.2%}")
    print(f"  Sharpe Ratio:     {max_sharpe.sharpe_ratio:.2f}")
    print("  Weights:")
    for ticker, weight in zip(tickers, max_sharpe.weights):
        if weight > 0.01:
            print(f"    {ticker:6s}: {weight:.2%}")

    print("\n" + "-" * 60)
    print("MINIMUM VOLATILITY PORTFOLIO")
    print("-" * 60)
    print(f"  Expected Return:  {min_vol.expected_return:.2%}")
    print(f"  Volatility:       {min_vol.volatility:.2%}")
    print(f"  Sharpe Ratio:     {min_vol.sharpe_ratio:.2f}")
    print("  Weights:")
    for ticker, weight in zip(tickers, min_vol.weights):
        if weight > 0.01:
            print(f"    {ticker:6s}: {weight:.2%}")

    # Monte Carlo simulation on optimal portfolio
    print("\nRunning Monte Carlo simulation (10,000 paths)...")
    mc = MonteCarloSimulator(
        expected_return=max_sharpe.expected_return,
        volatility=max_sharpe.volatility,
        initial_value=100_000,
    )
    mc.simulate(num_simulations=10_000, num_days=252)
    var_95 = mc.calculate_var(0.95)
    cvar_95 = mc.calculate_cvar(0.95)

    print(f"\n  1-Year VaR (95%):  ${var_95:,.0f}")
    print(f"  1-Year CVaR (95%): ${cvar_95:,.0f}")

    # Visualizations
    print("\nGenerating visualizations...")
    plot_efficient_frontier(frontier, max_sharpe, min_vol)
    plot_correlation_matrix(optimizer.returns)
    plot_portfolio_weights(max_sharpe, tickers)
    plot_cumulative_returns(optimizer.returns)
    mc.plot_simulations()

    print("\nDone.")


if __name__ == "__main__":
    main()
