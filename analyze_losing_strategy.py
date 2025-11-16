"""
Comprehensive Analysis of Losing Trading Strategy

Diagnoses why the optimized volumetric tick strategy is losing money.
Provides detailed analysis of trades, patterns, and actionable recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.volumetric_tick_strategy import VolumetricTickStrategy
from src.trade_analyzer import TradeAnalyzer
from visualizations.trade_analysis_plots import TradeAnalysisPlots


def load_tick_data(tick_interval: int = 500, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars from {filepath}")
    print(f"[INFO] Date range: {df.index[0]} to {df.index[-1]}")

    return df


def run_detailed_backtest(df, params, initial_capital=10000, commission=2.50):
    """
    Run backtest with enhanced trade logging.

    Args:
        df: DataFrame with tick bar data
        params: Dictionary of strategy parameters
        initial_capital: Starting capital
        commission: Commission per trade

    Returns:
        Dictionary with performance metrics and detailed trade data
    """
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

    # Run simulation with enhanced logging
    print("[*] Running backtest simulation...")
    capital = initial_capital
    position = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_time = None
    entry_confidence = 0
    entry_reason = ""
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
                pnl = pnl * 4 - commission

                capital += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': row.index[-1],
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'STOP_LOSS',
                    'confidence': entry_confidence,
                    'entry_reason': entry_reason,
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
                    'confidence': entry_confidence,
                    'entry_reason': entry_reason,
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
                    'confidence': entry_confidence,
                    'entry_reason': entry_reason,
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
                entry_reason = signal.reason  # Capture entry reason

            elif signal.signal == 'SHORT':
                position = 'SHORT'
                entry_price = current_price
                entry_time = row.index[-1]
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                entry_confidence = signal.confidence
                entry_reason = signal.reason

        equity.append(capital)

    # Convert to DataFrame
    if not trades:
        print("\n[ERROR] No trades generated!")
        return None

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(equity)

    # Calculate basic metrics
    total_pnl = trades_df['pnl'].sum()
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

    # Drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    return {
        'trades_df': trades_df,
        'equity_curve': equity_series,
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'final_capital': capital,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
    }


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def main():
    print_section_header("COMPREHENSIVE LOSING STRATEGY ANALYSIS")

    # 1. Load optimized parameters
    print("\n[*] Loading optimized parameters...")
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_164646.json")

    if not params_file.exists():
        print(f"[ERROR] Parameters file not found: {params_file}")
        return

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    params = params_data['best_params']

    # Add derived parameters
    params['strong_imbalance'] = params['min_volume_imbalance'] * 2.0
    params['stacked_imbalance_ratio'] = 3.0
    params['min_stacked_bars'] = 3
    params['volume_ratio_threshold'] = 2.0
    params['target_multiplier'] = params['stop_multiplier'] * 2.0

    print(f"[OK] Loaded parameters from {params_file.name}")
    print(f"[INFO] Optimization objective: {params_data['objective_function']}")
    print(f"[INFO] Objective score: {params_data['best_objective_score']:.4f}")

    # Display key parameters
    print("\n[KEY PARAMETERS]")
    print(f"  min_confluence:            {params['min_confluence']:.4f}")
    print(f"  stop_multiplier:           {params['stop_multiplier']:.4f}")
    print(f"  exit_confluence_threshold: {params['exit_confluence_threshold']:.4f}")

    # 2. Load tick data
    print("\n[*] Loading tick data...")
    df = load_tick_data(tick_interval=500, symbol="MNQ")

    # 3. Run detailed backtest
    result = run_detailed_backtest(df, params, initial_capital=10000, commission=2.50)

    if result is None:
        return

    # 4. Display basic performance
    print_section_header("BASIC PERFORMANCE METRICS")
    print(f"\n  Total Return:       ${result['total_return']:,.2f} ({result['return_pct']:.2f}%)")
    print(f"  Final Capital:      ${result['final_capital']:,.2f}")
    print(f"  Total Trades:       {result['total_trades']}")
    print(f"  Win Rate:           {result['win_rate']:.2f}%")
    print(f"  Profit Factor:      {result['profit_factor']:.2f}")
    print(f"  Max Drawdown:       {result['max_drawdown_pct']:.2f}%")
    print(f"  Avg Win:            ${result['avg_win']:.2f}")
    print(f"  Avg Loss:           ${result['avg_loss']:.2f}")

    # 5. Initialize analyzer
    analyzer = TradeAnalyzer(result['trades_df'], result['equity_curve'], df)

    # 6. Exit reason analysis
    print_section_header("EXIT REASON ANALYSIS")
    exit_analysis = analyzer.analyze_exit_reasons()
    print(f"\n{exit_analysis.to_string(index=False)}")

    # 7. Hourly performance
    print_section_header("HOURLY PERFORMANCE ANALYSIS")
    hourly_analysis = analyzer.analyze_by_hour()

    # Show worst hours
    worst_hours = hourly_analysis.nsmallest(5, 'total_pnl')
    print("\nWORST PERFORMING HOURS:")
    print(worst_hours[['hour', 'count', 'win_rate', 'total_pnl']].to_string(index=False))

    # Show best hours
    best_hours = hourly_analysis.nlargest(5, 'total_pnl')
    print("\nBEST PERFORMING HOURS:")
    print(best_hours[['hour', 'count', 'win_rate', 'total_pnl']].to_string(index=False))

    # 8. Trade duration
    print_section_header("TRADE DURATION ANALYSIS")
    duration_stats = analyzer.analyze_trade_duration()
    print(f"\n  Avg Duration:         {duration_stats['avg_duration_minutes']:.1f} minutes")
    print(f"  Median Duration:      {duration_stats['median_duration_minutes']:.1f} minutes")
    print(f"  Avg Winner Duration:  {duration_stats['avg_winner_duration']:.1f} minutes")
    print(f"  Avg Loser Duration:   {duration_stats['avg_loser_duration']:.1f} minutes")

    # 9. Streaks
    print_section_header("STREAK ANALYSIS")
    streak_stats = analyzer.analyze_streaks()
    print(f"\n  Longest Win Streak:   {streak_stats['longest_win_streak']}")
    print(f"  Longest Loss Streak:  {streak_stats['longest_loss_streak']}")
    print(f"  Avg Win Streak:       {streak_stats['avg_win_streak']:.1f}")
    print(f"  Avg Loss Streak:      {streak_stats['avg_loss_streak']:.1f}")

    # 10. Best/Worst trades
    print_section_header("EXTREME TRADES ANALYSIS")

    print("\nTOP 10 BEST TRADES:")
    best_trades = analyzer.get_best_trades(10)
    print(best_trades[['entry_time', 'side', 'pnl', 'exit_reason', 'entry_reason']].to_string(index=False))

    print("\nTOP 10 WORST TRADES:")
    worst_trades = analyzer.get_worst_trades(10)
    print(worst_trades[['entry_time', 'side', 'pnl', 'exit_reason', 'entry_reason']].to_string(index=False))

    # 11. Generate insights
    print_section_header("KEY INSIGHTS & RECOMMENDATIONS")
    insights = analyzer.generate_insights()

    print("\nKEY FINDINGS:")
    for i, finding in enumerate(insights['key_findings'], 1):
        print(f"  {i}. {finding}")

    print("\nRECOMMENDATIONS:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"  {i}. {rec}")

    # 12. Generate visualizations
    print_section_header("GENERATING VISUALIZATIONS")

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    plotter = TradeAnalysisPlots(result['trades_df'], result['equity_curve'])
    plotter.create_all_plots(str(reports_dir))

    # 13. Save detailed trade log
    trade_log_path = reports_dir / "trade_details.csv"
    result['trades_df'].to_csv(trade_log_path, index=False)
    print(f"\n[SAVED] Detailed trade log: {trade_log_path}")

    # 14. Save text report
    report_path = reports_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LOSING STRATEGY ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("PERFORMANCE SUMMARY:\n")
        f.write(f"  Total Return:    ${result['total_return']:,.2f} ({result['return_pct']:.2f}%)\n")
        f.write(f"  Win Rate:        {result['win_rate']:.2f}%\n")
        f.write(f"  Profit Factor:   {result['profit_factor']:.2f}\n")
        f.write(f"  Total Trades:    {result['total_trades']}\n")
        f.write(f"  Max Drawdown:    {result['max_drawdown_pct']:.2f}%\n\n")

        f.write("KEY FINDINGS:\n")
        for finding in insights['key_findings']:
            f.write(f"  - {finding}\n")

        f.write("\nRECOMMENDATIONS:\n")
        for rec in insights['recommendations']:
            f.write(f"  - {rec}\n")

    print(f"[SAVED] Text report: {report_path}")

    # Final summary
    print_section_header("ANALYSIS COMPLETE")
    print(f"\n[OK] All reports saved to: {reports_dir}/")
    print("\nFILES GENERATED:")
    print("  - equity_curve.png")
    print("  - pnl_distribution.png")
    print("  - exit_reasons.png")
    print("  - hourly_performance.png")
    print("  - trade_duration.png")
    print("  - rolling_win_rate.png")
    print("  - cumulative_pnl.png")
    print("  - trade_details.csv")
    print("  - analysis_report.txt")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
