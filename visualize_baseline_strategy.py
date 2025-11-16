"""
Visualizzazioni Baseline Strategy

Genera grafici dettagliati per analizzare:
1. Equity curve nel tempo
2. Prezzo con entry/exit points
3. Analisi trade (P&L distribution, win rate, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.volumetric_tick_strategy import VolumetricTickStrategy

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_tick_data(tick_interval: int = 500, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars")
    return df


def run_backtest_detailed(df, params, initial_capital=2000, commission=0.75):
    """
    Run backtest with DETAILED trade logging for visualization.

    Returns detailed trades list and equity curve.
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

    # Run simulation with DETAILED logging
    capital = initial_capital
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    entry_index = 0
    trades = []
    equity = []
    equity_timestamps = []

    for i in range(len(df_calc)):
        if i < 250:
            equity.append(capital)
            equity_timestamps.append(df_calc.index[i])
            continue

        row = df_calc.iloc[:i+1]
        signal = strategy.generate_signal(row, position)
        current_time = row.index[-1]
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

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_index': entry_index,
                    'exit_index': i,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'STOP_LOSS',
                    'is_win': pnl > 0
                })
                position = None

            # Take profit check
            elif (position == 'LONG' and current_high >= take_profit) or \
                 (position == 'SHORT' and current_low <= take_profit):
                exit_price = take_profit
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_index': entry_index,
                    'exit_index': i,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'TAKE_PROFIT',
                    'is_win': pnl > 0
                })
                position = None

            # Strategy exit signal
            elif signal.signal in ['EXIT_LONG', 'EXIT_SHORT']:
                exit_price = current_price
                pnl = (exit_price - entry_price) if position == 'LONG' else (entry_price - exit_price)
                pnl = pnl * 4 - commission
                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_index': entry_index,
                    'exit_index': i,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': signal.reason,
                    'is_win': pnl > 0
                })
                position = None

        # Entry logic
        if position is None:
            if signal.signal == 'LONG':
                position = 'LONG'
                entry_price = current_price
                entry_time = current_time
                entry_index = i
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

            elif signal.signal == 'SHORT':
                position = 'SHORT'
                entry_price = current_price
                entry_time = current_time
                entry_index = i
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit

        equity.append(capital)
        equity_timestamps.append(current_time)

    return {
        'trades': pd.DataFrame(trades),
        'equity': pd.Series(equity, index=equity_timestamps),
        'price_data': df_calc
    }


def plot_equity_curve(equity, initial_capital, save_path):
    """Plot equity curve with drawdown overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Equity curve
    ax1.plot(equity.index, equity.values, linewidth=2, color='#2E86AB', label='Equity')
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label=f'Starting Capital (${initial_capital:,})')

    # Fill areas
    ax1.fill_between(equity.index, initial_capital, equity.values,
                     where=equity.values >= initial_capital,
                     alpha=0.3, color='green', label='Profit Zone')
    ax1.fill_between(equity.index, initial_capital, equity.values,
                     where=equity.values < initial_capital,
                     alpha=0.3, color='red', label='Loss Zone')

    # Mark peak
    peak_value = equity.max()
    peak_time = equity.idxmax()
    ax1.scatter(peak_time, peak_value, color='gold', s=200, marker='*',
               label=f'Peak: ${peak_value:,.2f}', zorder=5)

    ax1.set_ylabel('Capital ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Equity Curve Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Drawdown
    running_peak = equity.expanding().max()
    drawdown = (equity - running_peak) / running_peak * 100

    ax2.fill_between(drawdown.index, 0, drawdown.values, alpha=0.5, color='red')
    ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)

    # Mark max drawdown
    max_dd_value = drawdown.min()
    max_dd_time = drawdown.idxmin()
    ax2.scatter(max_dd_time, max_dd_value, color='darkred', s=200, marker='v',
               label=f'Max DD: {max_dd_value:.2f}%', zorder=5)

    ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('Drawdown from Peak', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_price_with_trades(price_data, trades, save_path, max_samples=5000):
    """Plot price chart with entry/exit markers."""
    # Sample data if too large (keep every Nth bar)
    if len(price_data) > max_samples:
        step = len(price_data) // max_samples
        price_sample = price_data.iloc[::step].copy()
    else:
        price_sample = price_data.copy()

    fig, ax = plt.subplots(figsize=(18, 10))

    # Plot price
    ax.plot(price_sample.index, price_sample['close'], linewidth=1,
           color='black', alpha=0.6, label='Price')

    # Plot entry points
    long_entries = trades[trades['side'] == 'LONG']
    short_entries = trades[trades['side'] == 'SHORT']

    # LONG entries (green up arrow)
    ax.scatter(long_entries['entry_time'], long_entries['entry_price'],
              marker='^', s=100, color='green', alpha=0.8,
              label=f'LONG Entry (n={len(long_entries)})', zorder=5)

    # SHORT entries (red down arrow)
    ax.scatter(short_entries['entry_time'], short_entries['entry_price'],
              marker='v', s=100, color='red', alpha=0.8,
              label=f'SHORT Entry (n={len(short_entries)})', zorder=5)

    # Plot exits (colored by win/loss)
    winners = trades[trades['is_win'] == True]
    losers = trades[trades['is_win'] == False]

    # Winning exits (green circle)
    ax.scatter(winners['exit_time'], winners['exit_price'],
              marker='o', s=80, color='lime', edgecolors='darkgreen', linewidths=1.5,
              alpha=0.7, label=f'Exit (Win) (n={len(winners)})', zorder=4)

    # Losing exits (red X)
    ax.scatter(losers['exit_time'], losers['exit_price'],
              marker='x', s=100, color='darkred', linewidths=2,
              alpha=0.8, label=f'Exit (Loss) (n={len(losers)})', zorder=4)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title('MNQ Price with Entry/Exit Points (Baseline Strategy)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def plot_trade_analysis(trades, save_path):
    """Plot comprehensive trade analysis."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. P&L Distribution
    ax1 = fig.add_subplot(gs[0, :2])
    winners = trades[trades['pnl'] > 0]['pnl']
    losers = trades[trades['pnl'] <= 0]['pnl']

    ax1.hist(winners, bins=50, alpha=0.6, color='green', label=f'Wins (n={len(winners)})', edgecolor='black')
    ax1.hist(losers, bins=50, alpha=0.6, color='red', label=f'Losses (n={len(losers)})', edgecolor='black')
    ax1.axvline(winners.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Avg Win: ${winners.mean():.2f}')
    ax1.axvline(losers.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Avg Loss: ${losers.mean():.2f}')
    ax1.set_xlabel('P&L ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('P&L Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Win Rate Over Time (rolling 50 trades)
    ax2 = fig.add_subplot(gs[0, 2])
    trades['is_win_int'] = trades['is_win'].astype(int)
    rolling_wr = trades['is_win_int'].rolling(50, min_periods=1).mean() * 100
    ax2.plot(range(len(rolling_wr)), rolling_wr, linewidth=2, color='#2E86AB')
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% Breakeven')
    ax2.axhline(trades['is_win'].mean() * 100, color='red', linestyle='-', alpha=0.7,
               label=f'Overall: {trades["is_win"].mean() * 100:.1f}%')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Rolling Win Rate (50 trades)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative P&L
    ax3 = fig.add_subplot(gs[1, :])
    cumulative_pnl = trades['pnl'].cumsum()
    ax3.plot(range(len(cumulative_pnl)), cumulative_pnl.values, linewidth=2, color='#2E86AB')
    ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                     where=cumulative_pnl.values >= 0, alpha=0.3, color='green')
    ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                     where=cumulative_pnl.values < 0, alpha=0.3, color='red')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('Cumulative P&L ($)')
    ax3.set_title('Cumulative P&L Over Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Exit Reasons
    ax4 = fig.add_subplot(gs[2, 0])
    exit_counts = trades['exit_reason'].value_counts()
    colors = plt.cm.Set3(range(len(exit_counts)))
    ax4.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax4.set_title('Exit Reason Distribution', fontweight='bold')

    # 5. Hourly Performance
    ax5 = fig.add_subplot(gs[2, 1])
    trades['entry_hour'] = pd.to_datetime(trades['entry_time']).dt.hour
    hourly_pnl = trades.groupby('entry_hour')['pnl'].sum()
    colors_hourly = ['green' if x > 0 else 'red' for x in hourly_pnl.values]
    ax5.bar(hourly_pnl.index, hourly_pnl.values, color=colors_hourly, alpha=0.7)
    ax5.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax5.set_xlabel('Hour (UTC)')
    ax5.set_ylabel('Total P&L ($)')
    ax5.set_title('P&L by Hour of Day', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Trade Duration
    ax6 = fig.add_subplot(gs[2, 2])
    trades['duration_minutes'] = (pd.to_datetime(trades['exit_time']) -
                                  pd.to_datetime(trades['entry_time'])).dt.total_seconds() / 60
    winners_duration = trades[trades['is_win']]['duration_minutes']
    losers_duration = trades[~trades['is_win']]['duration_minutes']

    bp = ax6.boxplot([winners_duration.dropna(), losers_duration.dropna()],
                     labels=['Winners', 'Losers'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax6.set_ylabel('Duration (minutes)')
    ax6.set_title('Trade Duration: Winners vs Losers', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Comprehensive Trade Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("VISUALIZZAZIONI BASELINE STRATEGY")
    print("=" * 80)

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Load baseline parameters
    print("\n[*] Loading baseline parameters...")
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_173532.json")

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    baseline_params = params_data['best_params'].copy()

    # Add derived parameters
    baseline_params['strong_imbalance'] = baseline_params['min_volume_imbalance'] * 2.0
    baseline_params['stacked_imbalance_ratio'] = 3.0
    baseline_params['min_stacked_bars'] = 3
    baseline_params['volume_ratio_threshold'] = 2.0
    baseline_params['target_multiplier'] = baseline_params['stop_multiplier'] * 2.5

    # Load data
    print("[*] Loading tick data...")
    df = load_tick_data(tick_interval=500, symbol="MNQ")

    # Run detailed backtest
    print("[*] Running detailed backtest for visualization...")
    result = run_backtest_detailed(df, baseline_params, initial_capital=2000, commission=0.75)

    trades = result['trades']
    equity = result['equity']
    price_data = result['price_data']

    print(f"\n[OK] Backtest complete:")
    print(f"  - Total trades: {len(trades)}")
    print(f"  - Final equity: ${equity.iloc[-1]:,.2f}")
    print(f"  - Return: {((equity.iloc[-1] - 2000) / 2000 * 100):.2f}%")

    # Generate visualizations
    print("\n[*] Generating visualizations...")

    # 1. Equity Curve
    print("  [1/3] Creating equity curve...")
    plot_equity_curve(equity, 2000, reports_dir / "equity_curve.png")

    # 2. Price with Trades
    print("  [2/3] Creating price chart with entry/exit points...")
    plot_price_with_trades(price_data, trades, reports_dir / "price_with_trades.png")

    # 3. Trade Analysis
    print("  [3/3] Creating trade analysis charts...")
    plot_trade_analysis(trades, reports_dir / "trade_analysis.png")

    # Save detailed trade log
    trade_log_path = reports_dir / "trade_log.csv"
    trades.to_csv(trade_log_path, index=False)
    print(f"\n[SAVED] {trade_log_path}")

    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\nFiles saved in: {reports_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. equity_curve.png       - Equity growth + drawdown")
    print("  2. price_with_trades.png  - Price chart with all entry/exit points")
    print("  3. trade_analysis.png     - Comprehensive trade statistics")
    print("  4. trade_log.csv          - Detailed trade data for analysis")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
