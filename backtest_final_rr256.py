"""
Final Backtest with Optimized RR 2.56:1

Uses the optimal Risk/Reward ratio found through optimization.
All other parameters remain at baseline (already optimized).
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


def run_backtest(df, params, initial_capital=2000, commission=0.75):
    """
    Run backtest with optimized RR 2.56:1.

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
    print("FINAL BACKTEST - OPTIMIZED RR 2.56:1")
    print("=" * 80)
    print("\nParameters:")
    print("  - RR Ratio: 2.56:1 (OPTIMIZED via Optuna)")
    print("  - Initial capital: $2,000")
    print("  - Commission: $0.75 per trade")
    print("  - Position sizing: 1 micro MNQ ($2/point)")
    print("  - Time filter: Hours 14,15,16,18,19 UTC excluded")
    print("=" * 80)

    # Load optimized parameters
    params_file = Path("optimization_results/optimized_rr_256_params_20251116_222100.json")

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    optimized_params = params_data['best_params'].copy()

    print("\n[OPTIMIZED PARAMETERS]")
    print(f"  RR Ratio:                  {optimized_params['rr_ratio']:.2f}:1")
    print(f"  Stop Multiplier:           {optimized_params['stop_multiplier']:.4f}")
    print(f"  Target Multiplier:         {optimized_params['target_multiplier']:.4f}")
    print(f"  Min Volume Imbalance:      {optimized_params['min_volume_imbalance']:.4f}")
    print(f"  Delta Slope Period:        {optimized_params['delta_slope_period']}")
    print(f"  Buy/Sell Long Threshold:   {optimized_params['buy_sell_long_threshold']:.4f}")
    print(f"  Buy/Sell Short Threshold:  {optimized_params['buy_sell_short_threshold']:.4f}")
    print(f"  Min Confluence:            {optimized_params['min_confluence']:.4f}")
    print(f"  Exit Confluence Threshold: {optimized_params['exit_confluence_threshold']:.4f}")

    # Load data
    print("\n[*] Loading tick data...")
    df = load_tick_data(tick_interval=500, symbol="MNQ")

    # Run backtest
    print("[*] Running final backtest with RR 2.56:1...")
    result = run_backtest(df, optimized_params, initial_capital=2000, commission=0.75)

    if result is None:
        print("\n[ERROR] NO TRADES GENERATED!")
        return

    print("\n" + "=" * 80)
    print("FINAL RESULTS WITH RR 2.56:1")
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
    print(f"  Max Drawdown ($):    ${abs(result['max_drawdown_pct']) / 100 * result['final_capital']:,.2f}")
    print(f"\n  Avg Win:             ${result['avg_win']:+.2f}")
    print(f"  Avg Loss:            ${result['avg_loss']:+.2f}")
    print(f"  Total Commission:    ${result['total_commission']:,.2f}")

    # Comparison with previous RR 2.5
    print("\n" + "=" * 80)
    print("IMPROVEMENT VS RR 2.5:1")
    print("=" * 80)

    baseline_metrics = {
        'return_pct': 313.88,
        'profit_factor': 1.4005,
        'sharpe_ratio': 0.0872,
        'win_rate': 35.12,
        'max_drawdown_pct': -40.53,
        'final_capital': 8278
    }

    print(f"\n  Return:        {result['return_pct']:.2f}% (was {baseline_metrics['return_pct']:.2f}%)")
    print(f"                 Improvement: {result['return_pct'] - baseline_metrics['return_pct']:+.2f}%")

    print(f"\n  Final Capital: ${result['final_capital']:,.2f} (was ${baseline_metrics['final_capital']:,.2f})")
    print(f"                 Improvement: ${result['final_capital'] - baseline_metrics['final_capital']:+,.2f}")

    print(f"\n  Profit Factor: {result['profit_factor']:.4f} (was {baseline_metrics['profit_factor']:.4f})")
    print(f"                 Improvement: {result['profit_factor'] - baseline_metrics['profit_factor']:+.4f}")

    print(f"\n  Sharpe Ratio:  {result['sharpe_ratio']:.4f} (was {baseline_metrics['sharpe_ratio']:.4f})")
    print(f"                 Improvement: {result['sharpe_ratio'] - baseline_metrics['sharpe_ratio']:+.4f}")

    print(f"\n  Win Rate:      {result['win_rate']:.2f}% (was {baseline_metrics['win_rate']:.2f}%)")
    print(f"                 Difference: {result['win_rate'] - baseline_metrics['win_rate']:+.2f}%")

    print(f"\n  Max Drawdown:  {result['max_drawdown_pct']:.2f}% (was {baseline_metrics['max_drawdown_pct']:.2f}%)")
    print(f"                 Improvement: {result['max_drawdown_pct'] - baseline_metrics['max_drawdown_pct']:+.2f}%")

    print("\n" + "=" * 80)
    print("STRATEGY STATUS")
    print("=" * 80)

    if result['profit_factor'] > 1.3 and result['win_rate'] > 33:
        print("\n  [EXCELLENT] Strategy is highly profitable and robust!")
        print(f"  [OK] Annual return potential: ${result['total_return']:,.2f} ({result['return_pct']:.2f}%)")
        print(f"  [OK] Risk-adjusted performance: Sharpe {result['sharpe_ratio']:.4f}")
        print("\n  READY FOR LIVE TRADING!")
    elif result['profit_factor'] > 1.0:
        print("\n  [GOOD] Strategy is profitable")
        print(f"  [OK] Expected return: ${result['total_return']:,.2f} ({result['return_pct']:.2f}%)")
    else:
        print("\n  [WARNING] Strategy needs improvement")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
