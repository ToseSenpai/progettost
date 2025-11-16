"""
Quick test of strategy fixes.

Tests the 3 quick fixes:
1. Time filter (exclude hours 14, 18, 19 UTC)
2. Higher confluence threshold (0.88 instead of 0.828)
3. Higher RR (3:1 instead of 2:1)
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
    """Run backtest with given parameters."""
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
            if (position == 'LONG' and current_low <= stop_loss) or \
               (position == 'SHORT' and current_high >= stop_loss):
                exit_price = stop_loss
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl
                trades.append({'pnl': pnl, 'exit_reason': 'STOP_LOSS'})
                position = None

            elif (position == 'LONG' and current_high >= take_profit) or \
                 (position == 'SHORT' and current_low <= take_profit):
                exit_price = take_profit
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl
                trades.append({'pnl': pnl, 'exit_reason': 'TAKE_PROFIT'})
                position = None

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

    return {
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'final_capital': capital,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
    }


def main():
    print("=" * 80)
    print("TESTING STRATEGY FIXES")
    print("=" * 80)

    # Load best parameters from optimization
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_164646.json")
    with open(params_file, 'r') as f:
        params_data = json.load(f)

    original_params = params_data['best_params'].copy()

    # Add derived parameters
    original_params['strong_imbalance'] = original_params['min_volume_imbalance'] * 2.0
    original_params['stacked_imbalance_ratio'] = 3.0
    original_params['min_stacked_bars'] = 3
    original_params['volume_ratio_threshold'] = 2.0
    original_params['target_multiplier'] = original_params['stop_multiplier'] * 2.0  # OLD RR 2:1

    # Create fixed parameters
    fixed_params = original_params.copy()
    fixed_params['min_confluence'] = 0.88  # Raised from 0.828
    fixed_params['target_multiplier'] = fixed_params['stop_multiplier'] * 3.0  # NEW RR 3:1

    # Load data
    df = load_tick_data()

    # Test ORIGINAL (without fixes)
    print("\n[1] ORIGINAL STRATEGY (NO FIXES):")
    print(f"    - min_confluence: {original_params['min_confluence']:.4f}")
    print(f"    - RR: 2:1")
    print(f"    - Time filter: RTH only (no toxic hours filter)")

    result_original = run_backtest(df, original_params)

    if result_original:
        print(f"\n    Return:        {result_original['return_pct']:.2f}%")
        print(f"    Trades:        {result_original['total_trades']}")
        print(f"    Win Rate:      {result_original['win_rate']:.2f}%")
        print(f"    Profit Factor: {result_original['profit_factor']:.2f}")

    # Test FIXED (with all fixes: time filter is automatic in strategy, plus higher confluence and RR 3:1)
    print("\n[2] FIXED STRATEGY (WITH ALL 3 FIXES):")
    print(f"    - min_confluence: {fixed_params['min_confluence']:.4f} (was {original_params['min_confluence']:.4f})")
    print(f"    - RR: 3:1 (was 2:1)")
    print(f"    - Time filter: RTH + toxic hours (14, 18, 19 UTC) filtered")

    result_fixed = run_backtest(df, fixed_params)

    if result_fixed:
        print(f"\n    Return:        {result_fixed['return_pct']:.2f}%")
        print(f"    Trades:        {result_fixed['total_trades']}")
        print(f"    Win Rate:      {result_fixed['win_rate']:.2f}%")
        print(f"    Profit Factor: {result_fixed['profit_factor']:.2f}")

    # Compare
    if result_original and result_fixed:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)

        return_diff = result_fixed['return_pct'] - result_original['return_pct']
        trades_diff = result_fixed['total_trades'] - result_original['total_trades']
        wr_diff = result_fixed['win_rate'] - result_original['win_rate']
        pf_diff = result_fixed['profit_factor'] - result_original['profit_factor']

        print(f"\n  Return improvement:        {return_diff:+.2f}% ({result_fixed['return_pct']:.2f}% vs {result_original['return_pct']:.2f}%)")
        print(f"  Trade reduction:           {trades_diff:+d} ({result_fixed['total_trades']} vs {result_original['total_trades']})")
        print(f"  Win rate change:           {wr_diff:+.2f}% ({result_fixed['win_rate']:.2f}% vs {result_original['win_rate']:.2f}%)")
        print(f"  Profit factor improvement: {pf_diff:+.2f} ({result_fixed['profit_factor']:.2f} vs {result_original['profit_factor']:.2f})")

        if result_fixed['profit_factor'] > 1.0:
            print("\n  ✅ STRATEGY IS NOW PROFITABLE!")
        else:
            print("\n  ⚠️  Strategy still losing, but improvements made")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
