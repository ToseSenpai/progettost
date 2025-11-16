"""
Test backtest with Trial 967 parameters from optimization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.volumetric_tick_strategy import VolumetricTickStrategy

# Trial 967 parameters (BEST from old optimization) + NEW IMPROVEMENTS
PARAMS = {
    'min_volume_imbalance': 0.11468882605842083,
    'strong_imbalance': 0.4369672464664113,
    'delta_slope_period': 206,
    'min_stacked_bars': 2,
    'stacked_imbalance_ratio': 3.3852159419956305,
    'buy_sell_long_threshold': 1.4227042427115058,
    'buy_sell_short_threshold': 0.5665066171443081,
    'buy_sell_exit_long': 0.8392751619128284,
    'buy_sell_exit_short': 1.2662341118001372,
    'volume_ratio_threshold': 2.2715987229071026,
    'min_confluence': 0.8974880436379368,
    'stop_multiplier': 1.2753049872679252,
    'target_multiplier': 2.4046785715968984,
    'exit_confluence_threshold': 0.70  # NEW! Conservative exits (70%)
}

def load_tick_data(tick_interval: int = 500, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars from {filepath}")
    print(f"[INFO] Date range: {df.index[0]} to {df.index[-1]}")

    return df


def run_backtest(df, params, initial_capital=10000, commission=2.50):
    """Run backtest with given parameters."""

    # Create strategy with custom parameters
    strategy = VolumetricTickStrategy()

    # Apply parameter overrides
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
    print("\n[*] Calculating indicators...")
    df_calc = strategy.calculate_indicators(df.copy())

    # Run simulation
    print("[*] Running backtest simulation...")
    capital = initial_capital
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    entry_confidence = 0
    trades = []
    equity = [capital]

    for i in range(len(df_calc)):
        if i < 250:  # Skip warmup
            equity.append(capital)
            continue

        row = df_calc.iloc[:i+1]

        # Generate signal
        signal = strategy.generate_signal(row, position)

        current_price = row.iloc[-1]['close']
        current_high = row.iloc[-1]['high']
        current_low = row.iloc[-1]['low']

        # Exit logic
        if position is not None:
            # Stop loss
            if (position == 'LONG' and current_low <= stop_loss) or \
               (position == 'SHORT' and current_high >= stop_loss):
                exit_price = stop_loss
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission  # MNQ = $0.50/tick, 4 ticks per point
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row.index[-1],
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'STOP_LOSS',
                    'confidence': entry_confidence
                })
                position = None

            # Take profit
            elif (position == 'LONG' and current_high >= take_profit) or \
                 (position == 'SHORT' and current_low <= take_profit):
                exit_price = take_profit
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row.index[-1],
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'TAKE_PROFIT',
                    'confidence': entry_confidence
                })
                position = None

            # Strategy exit
            elif signal.signal in ['EXIT_LONG', 'EXIT_SHORT']:
                exit_price = current_price
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row.index[-1],
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': signal.reason,
                    'confidence': entry_confidence
                })
                position = None

        # Entry logic
        if position is None:
            if signal.signal == 'LONG':
                position = 'LONG'
                entry_price = current_price
                entry_time = row.index[-1]
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_confidence = signal.confidence

            elif signal.signal == 'SHORT':
                position = 'SHORT'
                entry_price = current_price
                entry_time = row.index[-1]
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_confidence = signal.confidence

        equity.append(capital)

    # Calculate metrics
    if not trades:
        print("\n[ERROR] No trades generated!")
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

    # Drawdown
    equity_series = pd.Series(equity)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

    # Average trade metrics
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

    return {
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'final_capital': capital,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades_df': trades_df,
        'equity_curve': equity_series
    }


def main():
    print("=" * 80)
    print("BACKTEST - IMPROVED Strategy with ALL Research Features (500t)")
    print("=" * 80)
    print("\nNEW FEATURES:")
    print("  - Volume Profile (POC, VAH, VAL) integration")
    print("  - CVD Divergence detection")
    print("  - VWAP Mean Reversion (+/-2 std dev)")
    print("  - Delta Trend as WEIGHT not FILTER")
    print("  - Exit confluence threshold: 70% (was 60%)")
    print("  - Rebalanced confluence weights")
    print("=" * 80)

    # Display parameters
    print("\n[*] Parameters:")
    for key, value in PARAMS.items():
        print(f"    {key:<30} = {value:.4f}" if isinstance(value, float) else f"    {key:<30} = {value}")

    # Load data
    print("\n[*] Loading tick data...")
    df = load_tick_data(tick_interval=500, symbol="MNQ")

    # Run backtest
    result = run_backtest(df, PARAMS, initial_capital=10000, commission=2.50)

    if result is None:
        return

    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n[PERFORMANCE]")
    print(f"  Total Return:         ${result['total_return']:,.2f}")
    print(f"  Return %:             {result['return_pct']:.2f}%")
    print(f"  Final Capital:        ${result['final_capital']:,.2f}")
    print(f"  Sharpe Ratio:         {result['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:         {result['max_drawdown_pct']:.2f}%")

    print(f"\n[TRADES]")
    print(f"  Total Trades:         {result['total_trades']}")
    print(f"  Winning Trades:       {result['winning_trades']}")
    print(f"  Losing Trades:        {result['losing_trades']}")
    print(f"  Win Rate:             {result['win_rate']:.2f}%")
    print(f"  Profit Factor:        {result['profit_factor']:.2f}")
    print(f"  Avg Win:              ${result['avg_win']:.2f}")
    print(f"  Avg Loss:             ${result['avg_loss']:.2f}")

    # Show last 10 trades
    print(f"\n[LAST 10 TRADES]")
    trades_df = result['trades_df']
    print(trades_df.tail(10)[['entry_time', 'side', 'entry_price', 'exit_price', 'pnl', 'exit_reason']].to_string())

    # Exit reason breakdown
    print(f"\n[EXIT REASON BREAKDOWN]")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = (count / len(trades_df) * 100)
        print(f"  {reason:<30} {count:>4} ({pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
