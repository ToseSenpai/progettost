"""
Multi-Timeframe Backtest Analysis

Tests the volumetric strategy across M1, M2, M3, M4, M5, M10 timeframes
to identify which timeframe performs best with the optimized parameters.

Uses baseline parameters with RR 2.5:1 (already optimized for 500t bars).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.volumetric_tick_strategy import VolumetricTickStrategy
from datetime import datetime


def load_minute_data(timeframe: int = 1, symbol: str = "MNQ"):
    """Load minute bar data from CSV."""
    filepath = Path(f"tick_data/databento/minute_bars/{symbol}_M{timeframe}_bars.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"    [OK] Loaded {len(df)} M{timeframe} bars")
    return df


def run_backtest(df, params, timeframe, initial_capital=2000, commission=0.75):
    """
    Run backtest on minute timeframe data.

    Args:
        df: DataFrame with minute bar data
        params: Dictionary of strategy parameters
        timeframe: Timeframe label (e.g., "M1", "M5")
        initial_capital: Starting capital ($2,000)
        commission: Commission per trade ($0.75)

    Returns:
        Dictionary with performance metrics
    """
    strategy = VolumetricTickStrategy()

    # Apply parameters (baseline + RR 2.5:1)
    strategy.min_volume_imbalance = params['min_volume_imbalance']
    strategy.strong_imbalance = params.get('strong_imbalance', 0.34)
    strategy.delta_slope_period = params['delta_slope_period']
    strategy.min_stacked_bars = params.get('min_stacked_bars', 3)
    strategy.stacked_imbalance_ratio = params.get('stacked_imbalance_ratio', 3.0)
    strategy.buy_sell_long_threshold = params['buy_sell_long_threshold']
    strategy.buy_sell_short_threshold = params['buy_sell_short_threshold']
    strategy.buy_sell_exit_long = params['buy_sell_exit_long']
    strategy.buy_sell_exit_short = params['buy_sell_exit_short']
    strategy.volume_ratio_threshold = params.get('volume_ratio_threshold', 2.0)
    strategy.min_confluence = params['min_confluence']
    strategy.stop_multiplier = params['stop_multiplier']
    strategy.target_multiplier = params.get('target_multiplier', strategy.stop_multiplier * 2.5)
    strategy.exit_confluence_threshold = params['exit_confluence_threshold']

    # Calculate indicators
    df_calc = strategy.calculate_indicators(df.copy())

    # Run simulation
    capital = initial_capital
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    trades = []
    equity = [capital]

    for i in range(len(df_calc)):
        if i < 250:
            equity.append(capital)
            continue

        row = df_calc.iloc[:i+1]
        signal = strategy.generate_signal(row, position)
        current_price = row.iloc[-1]['close']
        current_high = row.iloc[-1]['high']
        current_low = row.iloc[-1]['low']

        # Exit logic
        if position is not None:
            # Stop loss check
            if (position == 'LONG' and current_low <= stop_loss) or \
               (position == 'SHORT' and current_high >= stop_loss):
                exit_price = stop_loss
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl
                trades.append({'pnl': pnl, 'exit_reason': 'STOP_LOSS'})
                position = None

            # Take profit check
            elif (position == 'LONG' and current_high >= take_profit) or \
                 (position == 'SHORT' and current_low <= take_profit):
                exit_price = take_profit
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl
                trades.append({'pnl': pnl, 'exit_reason': 'TAKE_PROFIT'})
                position = None

            # Strategy exit signal
            elif signal.signal in ['EXIT_LONG', 'EXIT_SHORT']:
                exit_price = current_price
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl
                trades.append({'pnl': pnl, 'exit_reason': signal.reason})
                position = None

        # Entry logic
        if position is None:
            if signal.signal == 'LONG':
                position = 'LONG'
                entry_price = current_price
                entry_time = row.index[-1]
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

            elif signal.signal == 'SHORT':
                position = 'SHORT'
                entry_price = current_price
                entry_time = row.index[-1]
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

        equity.append(capital)

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

    # Calculate drawdown
    equity_series = pd.Series(equity)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Calculate Sharpe Ratio
    returns = equity_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # Calculate commission impact
    total_commission = commission * len(trades)

    return {
        'timeframe': timeframe,
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'final_capital': capital,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
        'total_commission': total_commission,
    }


def main():
    print("=" * 80)
    print("MULTI-TIMEFRAME BACKTEST ANALYSIS")
    print("=" * 80)
    print("\nTesting strategy across multiple timeframes: M1, M2, M3, M4, M5, M10")
    print("Parameters: Baseline (optimized for 500t) + RR 2.5:1")
    print("=" * 80)

    # Load baseline parameters (RR 2.5:1)
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_173532.json")

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    baseline_params = params_data['best_params'].copy()

    # Add RR 2.5:1 (target_multiplier = stop_multiplier * 2.5)
    baseline_params['target_multiplier'] = baseline_params['stop_multiplier'] * 2.5

    print("\n[PARAMETERS]")
    print(f"  RR Ratio:                  2.5:1")
    print(f"  Stop Multiplier:           {baseline_params['stop_multiplier']:.4f}")
    print(f"  Target Multiplier:         {baseline_params['target_multiplier']:.4f}")
    print(f"  Min Volume Imbalance:      {baseline_params['min_volume_imbalance']:.4f}")
    print(f"  Delta Slope Period:        {baseline_params['delta_slope_period']}")
    print(f"  Min Confluence:            {baseline_params['min_confluence']:.4f}")

    # Timeframes to test
    timeframes = [1, 2, 3, 4, 5, 10]

    # Store results
    all_results = []

    # Run backtests for each timeframe
    print("\n" + "=" * 80)
    print("RUNNING BACKTESTS")
    print("=" * 80)

    for tf in timeframes:
        print(f"\n[*] Testing M{tf} timeframe...")

        # Load data
        df = load_minute_data(timeframe=tf, symbol="MNQ")

        # Run backtest
        result = run_backtest(df, baseline_params, f"M{tf}", initial_capital=2000, commission=0.75)

        if result is None:
            print(f"    [WARNING] NO TRADES GENERATED on M{tf}")
            continue

        all_results.append(result)

        print(f"    Return: {result['return_pct']:+.2f}%, Trades: {result['total_trades']}, "
              f"WR: {result['win_rate']:.1f}%, PF: {result['profit_factor']:.2f}")

    # Create comparison DataFrame
    if not all_results:
        print("\n[ERROR] NO VALID RESULTS! Strategy didn't generate trades on any timeframe.")
        return

    results_df = pd.DataFrame(all_results)

    # Display comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON - ALL TIMEFRAMES")
    print("=" * 80)

    print("\n{:<12} {:>12} {:>12} {:>10} {:>10} {:>12} {:>10}".format(
        "Timeframe", "Return %", "Final $", "Trades", "Win Rate", "Profit Fct", "Sharpe"))
    print("-" * 80)

    for _, row in results_df.iterrows():
        print("{:<12} {:>+11.2f}% {:>11,.0f} {:>10} {:>9.1f}% {:>12.2f} {:>10.4f}".format(
            row['timeframe'],
            row['return_pct'],
            row['final_capital'],
            row['total_trades'],
            row['win_rate'],
            row['profit_factor'],
            row['sharpe_ratio']
        ))

    # Find best performers
    print("\n" + "=" * 80)
    print("BEST PERFORMERS BY METRIC")
    print("=" * 80)

    best_return = results_df.loc[results_df['return_pct'].idxmax()]
    best_pf = results_df.loc[results_df['profit_factor'].idxmax()]
    best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    best_wr = results_df.loc[results_df['win_rate'].idxmax()]

    print(f"\n  Best Return:        {best_return['timeframe']} ({best_return['return_pct']:+.2f}%)")
    print(f"  Best Profit Factor: {best_pf['timeframe']} ({best_pf['profit_factor']:.2f})")
    print(f"  Best Sharpe Ratio:  {best_sharpe['timeframe']} ({best_sharpe['sharpe_ratio']:.4f})")
    print(f"  Best Win Rate:      {best_wr['timeframe']} ({best_wr['win_rate']:.1f}%)")

    # Overall recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Calculate composite score: Return 40% + PF 30% + Sharpe 30%
    results_df['composite_score'] = (
        (results_df['return_pct'] / results_df['return_pct'].max() * 0.40) +
        (results_df['profit_factor'] / results_df['profit_factor'].max() * 0.30) +
        (results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max() * 0.30)
    )

    best_overall = results_df.loc[results_df['composite_score'].idxmax()]

    print(f"\n  BEST OVERALL TIMEFRAME: {best_overall['timeframe']}")
    print(f"  - Return:          {best_overall['return_pct']:+.2f}%")
    print(f"  - Final Capital:   ${best_overall['final_capital']:,.2f}")
    print(f"  - Profit Factor:   {best_overall['profit_factor']:.2f}")
    print(f"  - Win Rate:        {best_overall['win_rate']:.1f}%")
    print(f"  - Sharpe Ratio:    {best_overall['sharpe_ratio']:.4f}")
    print(f"  - Total Trades:    {best_overall['total_trades']}")
    print(f"  - Max Drawdown:    {best_overall['max_drawdown_pct']:.2f}%")

    # Comparison with 500t baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH 500T BASELINE")
    print("=" * 80)

    baseline_500t = {
        'return_pct': 313.88,
        'profit_factor': 1.4005,
        'sharpe_ratio': 0.0872,
        'win_rate': 35.12,
        'max_drawdown_pct': -40.53,
        'final_capital': 8278,
        'total_trades': 373
    }

    print(f"\n  500T Baseline:  Return {baseline_500t['return_pct']:.2f}%, "
          f"PF {baseline_500t['profit_factor']:.2f}, "
          f"WR {baseline_500t['win_rate']:.1f}%, "
          f"Trades {baseline_500t['total_trades']}")

    print(f"\n  Best Timeframe: Return {best_overall['return_pct']:.2f}%, "
          f"PF {best_overall['profit_factor']:.2f}, "
          f"WR {best_overall['win_rate']:.1f}%, "
          f"Trades {best_overall['total_trades']}")

    improvement = best_overall['return_pct'] - baseline_500t['return_pct']
    print(f"\n  Improvement:    {improvement:+.2f}% return")

    if best_overall['profit_factor'] > baseline_500t['profit_factor']:
        print(f"  [OK] Profit Factor improved: {best_overall['profit_factor']:.2f} vs {baseline_500t['profit_factor']:.2f}")
    else:
        print(f"  [NEUTRAL] Profit Factor: {best_overall['profit_factor']:.2f} vs {baseline_500t['profit_factor']:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"optimization_results/multi_timeframe_comparison_{timestamp}.json")

    output_data = {
        'comparison_timestamp': timestamp,
        'baseline_params_file': str(params_file),
        'results': results_df.to_dict('records'),
        'best_overall': {
            'timeframe': best_overall['timeframe'],
            'metrics': best_overall.to_dict()
        },
        'baseline_500t': baseline_500t
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[OK] Results saved to: {output_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
