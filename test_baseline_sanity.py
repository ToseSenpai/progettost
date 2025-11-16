"""
Test Baseline Parameters - Sanity Check

Tests if the baseline parameters that gave +40.80% return still work.
If they don't work, there's a BUG in the code.
If they work, the optimizer parameter ranges are wrong.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.volumetric_tick_strategy import VolumetricTickStrategy


def load_tick_data(tick_interval: int = 500, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars")
    return df


def run_backtest(df, params, initial_capital=10000, commission=2.50):
    """
    Run backtest with given parameters.

    CRITICAL: This is the CURRENT backtest function from optimize_pure_volumetric.py
    It should match what the optimizer is using!
    """
    strategy = VolumetricTickStrategy()

    # Apply parameters
    strategy.min_volume_imbalance = params['min_volume_imbalance']
    strategy.strong_imbalance = params['strong_imbalance']
    strategy.delta_slope_period = params['delta_slope_period']
    strategy.min_stacked_bars = params['min_stacked_bars']
    strategy.stacked_imbalance_ratio = params['stacked_imbalance_ratio']
    strategy.buy_sell_long_threshold = params['buy_sell_long_threshold']
    strategy.buy_sell_short_threshold = params['buy_sell_short_threshold']
    strategy.buy_sell_exit_long = params['buy_sell_exit_long']
    strategy.buy_sell_exit_short = params['buy_sell_exit_short']
    strategy.volume_ratio_threshold = params['volume_ratio_threshold']
    strategy.min_confluence = params['min_confluence']
    strategy.stop_multiplier = params['stop_multiplier']
    strategy.target_multiplier = params['target_multiplier']
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

    return {
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'final_capital': capital,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
    }


def main():
    print("=" * 80)
    print("BASELINE PARAMETERS SANITY CHECK")
    print("=" * 80)
    print("\nTesting if baseline parameters (+40.80%) still work...")
    print("If YES -> optimizer parameter ranges are wrong")
    print("If NO  -> BUG in code modifications\n")

    # Load baseline parameters
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_173532.json")

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    baseline_params = params_data['best_params'].copy()

    # Add derived parameters (same as optimizer)
    baseline_params['strong_imbalance'] = baseline_params['min_volume_imbalance'] * 2.0
    baseline_params['stacked_imbalance_ratio'] = 3.0
    baseline_params['min_stacked_bars'] = 3
    baseline_params['volume_ratio_threshold'] = 2.0
    baseline_params['target_multiplier'] = baseline_params['stop_multiplier'] * 2.5  # RR 2.5:1

    print("[BASELINE PARAMETERS]")
    print(f"  min_volume_imbalance:      {baseline_params['min_volume_imbalance']:.4f}")
    print(f"  delta_slope_period:        {baseline_params['delta_slope_period']}")
    print(f"  buy_sell_long_threshold:   {baseline_params['buy_sell_long_threshold']:.4f}")
    print(f"  buy_sell_short_threshold:  {baseline_params['buy_sell_short_threshold']:.4f}")
    print(f"  min_confluence:            {baseline_params['min_confluence']:.4f}")
    print(f"  stop_multiplier:           {baseline_params['stop_multiplier']:.4f}")
    print(f"  target_multiplier:         {baseline_params['target_multiplier']:.4f} (RR 2.5:1)")
    print(f"  exit_confluence_threshold: {baseline_params['exit_confluence_threshold']:.4f}")

    print("\n[EXPECTED RESULTS FROM BASELINE]")
    expected = params_data['metrics']
    print(f"  Return:        {expected['return_pct']:.2f}%")
    print(f"  Trades:        {expected['total_trades']}")
    print(f"  Win Rate:      {expected['win_rate']:.2f}%")
    print(f"  Profit Factor: {expected['profit_factor']:.4f}")
    print(f"  Max Drawdown:  {expected['max_drawdown_pct']:.2f}%")

    # Load data
    print("\n[*] Loading tick data...")
    df = load_tick_data(tick_interval=500, symbol="MNQ")

    # Run backtest with baseline parameters
    print("[*] Running backtest with baseline parameters...")
    result = run_backtest(df, baseline_params)

    if result is None:
        print("\n" + "=" * 80)
        print("CRITICAL ERROR: NO TRADES GENERATED!")
        print("=" * 80)
        print("\nBUG CONFIRMED - Baseline parameters should generate 1141 trades!")
        return

    print("\n" + "=" * 80)
    print("ACTUAL RESULTS")
    print("=" * 80)
    print(f"\n  Return:        {result['return_pct']:.2f}%")
    print(f"  Trades:        {result['total_trades']}")
    print(f"  Win Rate:      {result['win_rate']:.2f}%")
    print(f"  Profit Factor: {result['profit_factor']:.4f}")
    print(f"  Max Drawdown:  {result['max_drawdown_pct']:.2f}%")

    # Compare results
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    return_diff = abs(result['return_pct'] - expected['return_pct'])
    trades_diff = abs(result['total_trades'] - expected['total_trades'])
    pf_diff = abs(result['profit_factor'] - expected['profit_factor'])

    if return_diff < 1.0 and trades_diff < 50 and pf_diff < 0.05:
        print("\n[OK] BASELINE PARAMETERS STILL WORK!")
        print("\nCONCLUSION: Code is OK, optimizer parameter ranges are WRONG")
        print("\nRECOMMENDATION:")
        print("  1. Center parameter ranges around baseline values")
        print("  2. Use narrower search ranges (+/- 20% of baseline)")
        print("  3. Remove or soften PF hard filter")
    else:
        print("\n[ERROR] BASELINE PARAMETERS DON'T MATCH EXPECTED!")
        print("\nCONCLUSION: BUG exists in code modifications")
        print("\nDIFFERENCES:")
        print(f"  Return difference:  {return_diff:.2f}%")
        print(f"  Trade difference:   {trades_diff}")
        print(f"  PF difference:      {pf_diff:.4f}")

        if result['profit_factor'] < 1.0:
            print("\nCRITICAL: Profit Factor < 1.0 (losing strategy)")
            print("This explains why optimizer found ALL trials with PF < 1.0!")
            print("\nLIKELY CAUSE: Recent code modifications broke the backtest")
            print("CHECK: Break-even stops, time filter, or other recent changes")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
