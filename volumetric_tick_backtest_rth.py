"""
Volumetric Strategy Backtest on Tick Data - RTH Only

Tests the enhanced VolumetricTickStrategy using optimized parameters
and trading ONLY during Regular Trading Hours (RTH) session.

RTH Session for MNQ:
- 9:30 AM - 4:00 PM Eastern Time (ET)
- No trades opened or held outside this session
- All positions closed at session end
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time
import json
from src.volumetric_tick_strategy import VolumetricTickStrategy
import pytz


def load_best_params(filepath: str = None):
    """Load best parameters from optimization results."""
    if filepath is None:
        # Find most recent optimization results
        results_dir = Path("optimization_results")
        files = list(results_dir.glob("volumetric_1000t_best_params_*.json"))

        if not files:
            print("[WARNING] No optimization results found. Using default parameters.")
            return None

        # Get most recent
        filepath = max(files, key=lambda p: p.stat().st_mtime)

    print(f"[OK] Loading best parameters from: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"[INFO] Optimization metrics:")
    print(f"  Return: {data['metrics']['return_pct']:.2f}%")
    print(f"  Sharpe: {data['best_sharpe']:.4f}")
    print(f"  Win Rate: {data['metrics']['win_rate']:.2f}%")
    print(f"  Trades: {data['metrics']['total_trades']}")
    print(f"  Max DD: {data['metrics']['max_drawdown_pct']:.2f}%")
    print()

    return data['best_params']


def is_rth_session(timestamp):
    """
    Check if timestamp is during Regular Trading Hours (RTH).

    RTH for MNQ: 9:30 AM - 4:00 PM Eastern Time

    Args:
        timestamp: pandas Timestamp

    Returns:
        bool: True if during RTH, False otherwise
    """
    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')

    # Handle timezone-aware and naive timestamps
    if timestamp.tzinfo is None:
        # Assume UTC if no timezone
        utc = pytz.UTC
        timestamp = utc.localize(timestamp)

    timestamp_et = timestamp.astimezone(eastern)

    # RTH: 9:30 AM - 4:00 PM ET
    rth_start = time(9, 30)
    rth_end = time(16, 0)

    current_time = timestamp_et.time()

    # Check if weekday (Monday=0, Sunday=6)
    if timestamp_et.weekday() >= 5:  # Saturday or Sunday
        return False

    return rth_start <= current_time < rth_end


def load_tick_data(tick_interval: int, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")

    if not filepath.exists():
        print(f"[ERROR] File not found: {filepath}")
        return None

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars from {filepath}")
    print(f"[INFO] Date range: {df.index[0]} to {df.index[-1]}")

    # Verify tick data columns
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'sell_volume', 'volume_delta']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return None

    print(f"[OK] Tick data columns verified: buy_volume, sell_volume, volume_delta")

    return df


def run_backtest(df, strategy, params=None, rth_only=False, initial_capital=10000, commission=2.50):
    """
    Run backtest with volumetric strategy.

    Args:
        df: DataFrame with tick data
        strategy: VolumetricTickStrategy instance
        params: Optional parameter overrides
        rth_only: If True, only trade during RTH session
        initial_capital: Starting capital
        commission: Commission per trade

    Returns:
        tuple: (trades, equity)
    """
    print(f"[*] Running volumetric backtest (RTH only: {rth_only})...")

    # Apply parameter overrides if provided
    if params:
        strategy.mfi_overbought = params['mfi_overbought']
        strategy.mfi_oversold = params['mfi_oversold']
        strategy.st_multiplier = params['st_multiplier']
        strategy.min_volume_imbalance = params['min_volume_imbalance']
        strategy.strong_imbalance = params['strong_imbalance']
        strategy.stop_multiplier = params['stop_multiplier']
        strategy.target_multiplier = params['target_multiplier']
        min_confluence = params['min_confluence']
    else:
        min_confluence = 0.75

    # Calculate indicators
    print("[*] Calculating volumetric indicators...")
    df = strategy.calculate_indicators(df)

    capital = initial_capital
    position = None  # None, 'LONG', or 'SHORT'
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    entry_confidence = 0
    trades = []
    equity = [capital]

    rth_bars = 0
    non_rth_bars = 0

    for i in range(len(df)):
        if i < 100:  # Skip first 100 bars for indicator warmup
            equity.append(capital)
            continue

        row = df.iloc[:i+1]  # Pass all data up to current bar
        current_timestamp = row.index[-1]

        # Check if in RTH session
        in_rth = is_rth_session(current_timestamp)

        if in_rth:
            rth_bars += 1
        else:
            non_rth_bars += 1

        # If RTH only mode and not in RTH, close any open positions
        if rth_only and not in_rth and position is not None:
            # Force close at end of session
            current_price = row.iloc[-1]['close']
            pnl = (current_price - entry_price) if position == 'LONG' else (entry_price - current_price)
            pnl = pnl * 4 - commission
            capital += pnl

            trades.append({
                'entry_time': entry_time,
                'exit_time': current_timestamp,
                'side': position,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'exit_reason': 'SESSION_END',
                'confidence': entry_confidence
            })
            position = None

        # Generate signal
        signal = strategy.generate_signal(row, position)

        # Override confluence threshold if using optimized params
        if params and signal.signal in ['LONG', 'SHORT']:
            if signal.confidence < min_confluence:
                signal.signal = 'HOLD'

        current_price = row.iloc[-1]['close']
        current_high = row.iloc[-1]['high']
        current_low = row.iloc[-1]['low']

        # Check for exit
        if position is not None:
            # Check stop loss
            if (position == 'LONG' and current_low <= stop_loss) or \
               (position == 'SHORT' and current_high >= stop_loss):
                exit_price = stop_loss
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_timestamp,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'STOP_LOSS',
                    'confidence': entry_confidence
                })
                position = None

            # Check take profit
            elif (position == 'LONG' and current_high >= take_profit) or \
                 (position == 'SHORT' and current_low <= take_profit):
                exit_price = take_profit
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_timestamp,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'TAKE_PROFIT',
                    'confidence': entry_confidence
                })
                position = None

            # Check strategy exit signal
            elif signal.signal in ['EXIT_LONG', 'EXIT_SHORT']:
                exit_price = current_price
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_timestamp,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': signal.reason,
                    'confidence': entry_confidence
                })
                position = None

        # Check for entry (only if in RTH when rth_only mode)
        if position is None and (not rth_only or in_rth):
            if signal.signal == 'LONG':
                position = 'LONG'
                entry_price = current_price
                entry_time = current_timestamp
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_confidence = signal.confidence

            elif signal.signal == 'SHORT':
                position = 'SHORT'
                entry_price = current_price
                entry_time = current_timestamp
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_confidence = signal.confidence

        equity.append(capital)

    print(f"[OK] Backtest complete - {len(trades)} trades executed")

    if rth_only:
        total_bars = rth_bars + non_rth_bars
        rth_pct = (rth_bars / total_bars * 100) if total_bars > 0 else 0
        print(f"[INFO] RTH bars: {rth_bars}/{total_bars} ({rth_pct:.1f}%)")

    return trades, equity


def calculate_stats(trades, equity, initial_capital):
    """Calculate performance statistics."""
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    total_pnl = trades_df['pnl'].sum()
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0

    # Drawdown
    equity_series = pd.Series(equity)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

    # Average confidence
    avg_confidence = trades_df['confidence'].mean()

    # Exit reasons breakdown
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

    return {
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'avg_confidence': avg_confidence,
        'exit_reasons': exit_reasons,
    }


def main():
    """Main function."""

    print("=" * 80)
    print("VOLUMETRIC TICK STRATEGY BACKTEST - RTH SESSION ONLY")
    print("=" * 80)
    print("\nTrading ONLY during Regular Trading Hours (9:30 AM - 4:00 PM ET)")
    print("All positions closed at session end")
    print()

    # Load best parameters
    best_params = load_best_params()

    if best_params is None:
        print("[ERROR] Could not load optimization results")
        return

    # Initialize strategy
    strategy = VolumetricTickStrategy()
    print(f"[OK] Strategy: {strategy.name}")
    print(f"[INFO] Using optimized parameters\n")

    # Load data
    print("[*] Loading 1000-tick data...")
    df = load_tick_data(tick_interval=1000)
    if df is None:
        return

    # Run backtest WITH RTH filter
    print("\n" + "=" * 80)
    print("BACKTEST 1: RTH ONLY (9:30 AM - 4:00 PM ET)")
    print("=" * 80)
    trades_rth, equity_rth = run_backtest(df, strategy, params=best_params, rth_only=True)
    stats_rth = calculate_stats(trades_rth, equity_rth, 10000)

    # Run backtest WITHOUT RTH filter (24/7)
    print("\n" + "=" * 80)
    print("BACKTEST 2: 24/7 (ALL HOURS)")
    print("=" * 80)
    trades_24h, equity_24h = run_backtest(df, strategy, params=best_params, rth_only=False)
    stats_24h = calculate_stats(trades_24h, equity_24h, 10000)

    # Print comparison
    print("\n\n" + "=" * 80)
    print("COMPARISON: RTH ONLY vs 24/7")
    print("=" * 80)

    if stats_rth and stats_24h:
        print(f"\n{'Metric':<30} {'RTH Only':>20} {'24/7':>20} {'Difference':>15}")
        print("-" * 90)

        metrics = [
            ('Return %', 'return_pct', '%'),
            ('Total Trades', 'total_trades', ''),
            ('Win Rate %', 'win_rate', '%'),
            ('Profit Factor', 'profit_factor', ''),
            ('Avg Win', 'avg_win', '$'),
            ('Avg Loss', 'avg_loss', '$'),
            ('Max Drawdown %', 'max_drawdown_pct', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Avg Confidence', 'avg_confidence', ''),
        ]

        for metric_name, metric_key, suffix in metrics:
            rth_val = stats_rth[metric_key]
            h24_val = stats_24h[metric_key]

            if metric_key == 'total_trades':
                diff = rth_val - h24_val
                diff_str = f"{diff:+.0f}"
            else:
                diff = ((rth_val - h24_val) / abs(h24_val) * 100) if h24_val != 0 else 0
                diff_str = f"{diff:+.1f}%"

            print(f"{metric_name:<30} {rth_val:>20.2f} {h24_val:>20.2f} {diff_str:>15}")

        print("\n" + "=" * 80)
        print("EXIT REASONS BREAKDOWN")
        print("=" * 80)

        print("\nRTH Only:")
        for reason, count in stats_rth['exit_reasons'].items():
            pct = (count / stats_rth['total_trades'] * 100)
            print(f"  {reason:<20} {count:>5} ({pct:>5.1f}%)")

        print("\n24/7:")
        for reason, count in stats_24h['exit_reasons'].items():
            pct = (count / stats_24h['total_trades'] * 100)
            print(f"  {reason:<20} {count:>5} ({pct:>5.1f}%)")

        # Recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)

        rth_sharpe = stats_rth['sharpe_ratio']
        h24_sharpe = stats_24h['sharpe_ratio']

        if rth_sharpe > h24_sharpe:
            improvement = ((rth_sharpe - h24_sharpe) / abs(h24_sharpe) * 100)
            print(f"\n[WINNER] RTH ONLY - {improvement:+.1f}% better Sharpe ratio")
            print(f"  Trading during RTH improves risk-adjusted returns")
            print(f"  Return: {stats_rth['return_pct']:.2f}% vs {stats_24h['return_pct']:.2f}%")
            print(f"  Sharpe: {rth_sharpe:.4f} vs {h24_sharpe:.4f}")
        else:
            decline = ((rth_sharpe - h24_sharpe) / abs(h24_sharpe) * 100)
            print(f"\n[WINNER] 24/7 - RTH filter reduces Sharpe by {decline:.1f}%")
            print(f"  Trading overnight provides better opportunities")
            print(f"  Return: {stats_24h['return_pct']:.2f}% vs {stats_rth['return_pct']:.2f}%")
            print(f"  Sharpe: {h24_sharpe:.4f} vs {rth_sharpe:.4f}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
