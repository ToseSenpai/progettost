"""
Test CLEAN volumetric strategy across all tick intervals.
Quick comparison: 250t vs 500t vs 1000t
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.volumetric_tick_strategy import VolumetricTickStrategy


def load_tick_data(tick_interval: int, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return None

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} bars from {filepath}")
    return df


def run_quick_backtest(df, strategy, initial_capital=2000, commission=0.75):
    """Quick backtest with real broker conditions."""

    # Calculate indicators
    df = strategy.calculate_indicators(df)

    capital = initial_capital
    position = None
    trades = []
    equity = [capital]

    for i in range(len(df)):
        if i < 100:
            equity.append(capital)
            continue

        row = df.iloc[:i+1]
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
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
            elif signal.signal == 'SHORT':
                position = 'SHORT'
                entry_price = current_price
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

        equity.append(capital)

    # Stats
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    winning = trades_df[trades_df['pnl'] > 0]
    losing = trades_df[trades_df['pnl'] < 0]

    win_rate = (len(winning) / len(trades) * 100) if trades else 0
    profit_factor = abs(winning['pnl'].sum() / losing['pnl'].sum()) if len(losing) > 0 and losing['pnl'].sum() != 0 else 0

    equity_series = pd.Series(equity)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

    return {
        'return_pct': (total_pnl / initial_capital * 100),
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
    }


def main():
    print("=" * 80)
    print("CLEAN VOLUMETRIC STRATEGY - TICK INTERVAL COMPARISON")
    print("=" * 80)
    print()

    strategy = VolumetricTickStrategy()
    intervals = [100, 250, 500, 1000]
    results = {}

    for interval in intervals:
        print(f"\n{'='*80}")
        print(f"TESTING {interval}-TICK BARS")
        print(f"{'='*80}")

        df = load_tick_data(interval)
        if df is None:
            continue

        try:
            stats = run_quick_backtest(df, strategy)
            if stats:
                results[f"{interval}t"] = stats

                print(f"\n[RESULTS]")
                print(f"  Return: {stats['return_pct']:.2f}%")
                print(f"  Trades: {stats['total_trades']}")
                print(f"  Win Rate: {stats['win_rate']:.2f}%")
                print(f"  Profit Factor: {stats['profit_factor']:.2f}")
                print(f"  Max DD: {stats['max_drawdown_pct']:.2f}%")
                print(f"  Sharpe: {stats['sharpe_ratio']:.4f}")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()

    # Comparison
    if results:
        print(f"\n\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")

        print(f"{'Metric':<25} " + " ".join(f"{k:>18}" for k in results.keys()))
        print("-" * 80)

        metrics = [
            ('Return %', 'return_pct'),
            ('Total Trades', 'total_trades'),
            ('Win Rate %', 'win_rate'),
            ('Profit Factor', 'profit_factor'),
            ('Max Drawdown %', 'max_drawdown_pct'),
            ('Sharpe Ratio', 'sharpe_ratio'),
        ]

        for metric_name, metric_key in metrics:
            values = [results[k][metric_key] for k in results.keys()]
            print(f"{metric_name:<25} " + " ".join(f"{v:>18.2f}" for v in values))

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
