"""
Backtest 100-Tick Bars with Baseline Parameters

Tests the baseline optimized parameters (RR 2.5:1) on 100-tick bars
to see if smaller tick intervals improve performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.volumetric_tick_strategy import VolumetricTickStrategy


def load_tick_data(tick_interval: int = 100, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} {tick_interval}-tick bars")
    print(f"    Date range: {df.index[0]} to {df.index[-1]}")
    return df


def run_backtest(df, params, initial_capital=2000, commission=0.75):
    """
    Run backtest with baseline parameters.

    Args:
        df: DataFrame with tick bar data
        params: Dictionary of strategy parameters
        initial_capital: Starting capital ($2,000)
        commission: Commission per trade ($0.75)

    Returns:
        Dictionary with performance metrics
    """
    strategy = VolumetricTickStrategy()

    # Apply parameters
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
    print("BACKTEST: 100-TICK BARS")
    print("=" * 80)
    print("\nTesting baseline parameters (RR 2.5:1) on 100-tick bars")
    print("Goal: Determine if smaller tick intervals improve performance")
    print("=" * 80)

    # Load baseline parameters (optimized for 500t, RR 2.5:1)
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_173532.json")

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    baseline_params = params_data['best_params'].copy()

    # Add RR 2.5:1
    baseline_params['target_multiplier'] = baseline_params['stop_multiplier'] * 2.5

    print("\n[PARAMETERS]")
    print(f"  Tick Interval:             100t")
    print(f"  RR Ratio:                  2.5:1")
    print(f"  Stop Multiplier:           {baseline_params['stop_multiplier']:.4f}")
    print(f"  Target Multiplier:         {baseline_params['target_multiplier']:.4f}")
    print(f"  Min Volume Imbalance:      {baseline_params['min_volume_imbalance']:.4f}")
    print(f"  Delta Slope Period:        {baseline_params['delta_slope_period']}")

    # Load 100t data
    print("\n[*] Loading 100-tick bar data...")
    df = load_tick_data(tick_interval=100, symbol="MNQ")

    # Run backtest
    print("[*] Running backtest on 100t bars...")
    result = run_backtest(df, baseline_params, initial_capital=2000, commission=0.75)

    if result is None:
        print("\n[ERROR] NO TRADES GENERATED!")
        return

    print("\n" + "=" * 80)
    print("RESULTS: 100-TICK BARS")
    print("=" * 80)
    print(f"\n  Return:              {result['return_pct']:+.2f}%")
    print(f"  Profit (absolute):   ${result['total_return']:+,.2f}")
    print(f"  Final Capital:       ${result['final_capital']:,.2f}")
    print(f"  Initial Capital:     $2,000.00")
    print(f"\n  Total Trades:        {result['total_trades']}")
    print(f"  Win Rate:            {result['win_rate']:.2f}%")
    print(f"  Profit Factor:       {result['profit_factor']:.4f}")
    print(f"  Sharpe Ratio:        {result['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:        {result['max_drawdown_pct']:.2f}%")
    print(f"\n  Avg Win:             ${result['avg_win']:+.2f}")
    print(f"  Avg Loss:            ${result['avg_loss']:+.2f}")
    print(f"  Total Commission:    ${result['total_commission']:,.2f}")

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

    print(f"\n  500T Baseline:  Return {baseline_500t['return_pct']:+.2f}%, "
          f"PF {baseline_500t['profit_factor']:.2f}, "
          f"WR {baseline_500t['win_rate']:.1f}%, "
          f"Trades {baseline_500t['total_trades']}")

    print(f"\n  100T Result:    Return {result['return_pct']:+.2f}%, "
          f"PF {result['profit_factor']:.2f}, "
          f"WR {result['win_rate']:.1f}%, "
          f"Trades {result['total_trades']}")

    improvement = result['return_pct'] - baseline_500t['return_pct']
    print(f"\n  Difference:     {improvement:+.2f}% return")

    if improvement > 0:
        print(f"  [OK] 100T performs BETTER than 500T by {improvement:.2f}%!")
    elif improvement < -50:
        print(f"  [WARNING] 100T performs significantly WORSE than 500T")
    else:
        print(f"  [NEUTRAL] Similar performance to 500T")

    # Status
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if result['profit_factor'] > 1.3 and result['win_rate'] > 33:
        print("\n  [EXCELLENT] 100T bars work very well!")
        print(f"  - Strong profit factor: {result['profit_factor']:.2f}")
        print(f"  - Good win rate: {result['win_rate']:.1f}%")
        if improvement > 0:
            print(f"  - BETTER than 500T baseline!")
    elif result['profit_factor'] > 1.0:
        print("\n  [GOOD] 100T bars are profitable")
        print(f"  - Profit factor: {result['profit_factor']:.2f}")
    else:
        print("\n  [WARNING] 100T bars underperform")
        print(f"  - Profit factor < 1.0: {result['profit_factor']:.2f}")
        print(f"  - Strategy works better with larger tick intervals")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
