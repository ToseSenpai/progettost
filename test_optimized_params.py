"""
Test Optimized Parameters
Tests each symbol with its optimized parameters and compares results
"""

import asyncio
import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.config import config


async def test_symbol_with_params(symbol, params):
    """
    Test a symbol with specific optimized parameters

    Args:
        symbol: Symbol to test
        params: Optimized parameters dictionary

    Returns:
        Dictionary with test results
    """
    print(f"\n[*] Testing {symbol} with optimized parameters...")

    # Backup original config
    original_config = {
        'ema_fast': config.strategy.ema_fast,
        'ema_slow': config.strategy.ema_slow,
        'rsi_period': config.strategy.rsi_period,
        'rsi_overbought': config.strategy.rsi_overbought,
        'rsi_oversold': config.strategy.rsi_oversold,
        'atr_period': config.strategy.atr_period,
        'atr_stop_multiplier': config.strategy.atr_stop_multiplier,
        'atr_target_multiplier': config.strategy.atr_target_multiplier,
        'min_atr_threshold': config.strategy.min_atr_threshold,
        'symbols': config.instruments.symbols.copy()
    }

    try:
        # Apply optimized parameters
        config.strategy.ema_fast = params['ema_fast']
        config.strategy.ema_slow = params['ema_slow']
        config.strategy.rsi_period = params['rsi_period']
        config.strategy.rsi_overbought = params['rsi_overbought']
        config.strategy.rsi_oversold = params['rsi_oversold']
        config.strategy.atr_period = params['atr_period']
        config.strategy.atr_stop_multiplier = params['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = params['atr_target_multiplier']
        config.strategy.min_atr_threshold = params['min_atr_threshold']
        config.instruments.symbols = [symbol]

        # Run backtest
        backtester = Backtester()
        results = await backtester.run_backtest()

        # Extract statistics
        stats = results.statistics

        return {
            'symbol': symbol,
            'total_return_pct': stats['total_return_pct'],
            'sharpe_ratio': stats['sharpe_ratio'],
            'profit_factor': stats['profit_factor'],
            'win_rate_pct': stats['win_rate_pct'],
            'total_trades': stats['total_trades'],
            'expectancy': stats['expectancy'],
            'max_drawdown_pct': stats['max_drawdown_pct'],
            'total_pnl': stats['total_pnl'],
            'final_equity': stats['final_equity'],
            'params': params
        }

    except Exception as e:
        print(f"    >> Error: {e}")
        return None

    finally:
        # Restore original config
        config.strategy.ema_fast = original_config['ema_fast']
        config.strategy.ema_slow = original_config['ema_slow']
        config.strategy.rsi_period = original_config['rsi_period']
        config.strategy.rsi_overbought = original_config['rsi_overbought']
        config.strategy.rsi_oversold = original_config['rsi_oversold']
        config.strategy.atr_period = original_config['atr_period']
        config.strategy.atr_stop_multiplier = original_config['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = original_config['atr_target_multiplier']
        config.strategy.min_atr_threshold = original_config['min_atr_threshold']
        config.instruments.symbols = original_config['symbols'].copy()


async def main():
    """Main testing function"""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZED PARAMETERS")
    print("=" * 60)

    # Load comparison results
    comparison_file = Path('optimization_results/per_symbol/comparison_20251112_193005.json')

    if not comparison_file.exists():
        print("\n[ERROR] Comparison file not found!")
        print(f"Expected: {comparison_file}")
        return

    with open(comparison_file, 'r') as f:
        optimization_results = json.load(f)

    print(f"\nLoaded optimization results for {len(optimization_results)} symbols")
    print("\nOptimization Scores:")
    for symbol, data in sorted(optimization_results.items(), key=lambda x: x[1]['best_value'], reverse=True):
        print(f"  {symbol}: {data['best_value']:.2f}")

    print("\n" + "=" * 60)
    print("Running backtests with optimized parameters...")
    print("=" * 60)

    # Test each symbol
    results = []
    for symbol, opt_data in optimization_results.items():
        result = await test_symbol_with_params(symbol, opt_data['best_params'])
        if result:
            result['optimization_score'] = opt_data['best_value']
            results.append(result)

    if not results:
        print("\n[ERROR] No results generated!")
        return

    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('total_return_pct', ascending=False)

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS WITH OPTIMIZED PARAMETERS")
    print("=" * 60)

    print("\n[*] Performance Ranking:")
    print(f"\n{'Rank':<6} {'Symbol':<8} {'Return %':<12} {'Sharpe':<10} {'P.Factor':<10} {'Win %':<10} {'Trades':<8} {'Opt.Score':<10}")
    print("-" * 90)

    for i, row in enumerate(df.itertuples(), 1):
        print(f"{i:<6} {row.symbol:<8} {row.total_return_pct:>10.2f}%  "
              f"{row.sharpe_ratio:>8.2f}  {row.profit_factor:>8.2f}  "
              f"{row.win_rate_pct:>8.2f}%  {row.total_trades:>6}  "
              f"{row.optimization_score:>8.2f}")

    # Top 3 performers
    print("\n" + "=" * 60)
    print("TOP 3 BEST PERFORMERS")
    print("=" * 60)

    for i, row in enumerate(df.head(3).itertuples(), 1):
        print(f"\n#{i} - {row.symbol}")
        print(f"  Total Return: {row.total_return_pct:.2f}%")
        print(f"  Sharpe Ratio: {row.sharpe_ratio:.2f}")
        print(f"  Profit Factor: {row.profit_factor:.2f}")
        print(f"  Win Rate: {row.win_rate_pct:.2f}%")
        print(f"  Total Trades: {row.total_trades}")
        print(f"  Expectancy: ${row.expectancy:.2f}")
        print(f"  Max Drawdown: {row.max_drawdown_pct:.2f}%")
        print(f"  Final Equity: ${row.final_equity:,.2f}")

    # Get best symbol
    best_symbol = df.iloc[0]['symbol']
    best_params = df.iloc[0]['params']

    print("\n" + "=" * 60)
    print(f"RECOMMENDED: Use {best_symbol}")
    print("=" * 60)

    print(f"\n[*] Best Parameters for {best_symbol}:")
    print(f"\n# EMA Settings")
    print(f"ema_fast: int = {best_params['ema_fast']}")
    print(f"ema_slow: int = {best_params['ema_slow']}")

    print(f"\n# RSI Settings")
    print(f"rsi_period: int = {best_params['rsi_period']}")
    print(f"rsi_overbought: float = {best_params['rsi_overbought']:.4f}")
    print(f"rsi_oversold: float = {best_params['rsi_oversold']:.4f}")

    print(f"\n# ATR Settings")
    print(f"atr_period: int = {best_params['atr_period']}")
    print(f"atr_stop_multiplier: float = {best_params['atr_stop_multiplier']:.4f}")
    print(f"atr_target_multiplier: float = {best_params['atr_target_multiplier']:.4f}")
    print(f"min_atr_threshold: float = {best_params['min_atr_threshold']:.4f}")

    # Save results
    results_file = Path('optimization_results/per_symbol/backtest_comparison.csv')
    df.to_csv(results_file, index=False)
    print(f"\n[OK] Results saved to {results_file}")

    return best_symbol, best_params


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("\n[ERROR] .env file not found!")
        sys.exit(1)

    # Run tests
    asyncio.run(main())
