"""
Optimize Risk/Reward Ratio using Optuna.

Keeps all baseline parameters FIXED and optimizes ONLY the RR ratio.
This is a fast, targeted optimization to find the optimal risk/reward for the strategy.

Objective: Return (50%) + Profit Factor (30%) + Sharpe Ratio (20%)
Range: 1.5:1 to 4.5:1
Trials: 50
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
import time
from src.volumetric_tick_strategy import VolumetricTickStrategy

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def load_tick_data(tick_interval: int = 500, symbol: str = "MNQ"):
    """Load tick bar data from CSV."""
    filepath = Path(f"tick_data/databento/bars/{symbol}_{tick_interval}t_bars.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars")
    return df


def run_backtest_with_rr(df, baseline_params, rr_ratio, initial_capital=2000, commission=0.75):
    """
    Run backtest with specific RR ratio.

    Args:
        df: DataFrame with tick bar data
        baseline_params: Baseline parameters (all fixed except RR)
        rr_ratio: Risk/Reward ratio to test (e.g., 2.5 for 2.5:1)
        initial_capital: Starting capital
        commission: Commission per trade

    Returns:
        Dictionary with performance metrics including Sharpe ratio
    """
    strategy = VolumetricTickStrategy()

    # Apply baseline parameters
    params = baseline_params.copy()

    # Override target_multiplier based on RR ratio
    params['target_multiplier'] = params['stop_multiplier'] * rr_ratio

    # Set strategy parameters
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

    # Calculate Sharpe Ratio (annualized)
    # Daily returns from equity curve
    returns = equity_series.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0

    return {
        'rr_ratio': rr_ratio,
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
    }


def objective(trial, df, baseline_params):
    """
    Optuna objective function.

    Optimizes: Return (50%) + Profit Factor (30%) + Sharpe Ratio (20%)
    """
    # Suggest RR ratio between 1.5:1 and 4.5:1
    rr_ratio = trial.suggest_float('rr_ratio', 1.5, 4.5)

    # Run backtest with this RR
    result = run_backtest_with_rr(df, baseline_params, rr_ratio)

    if result is None:
        return -999.0  # No trades generated

    # Extract metrics
    return_pct = result['return_pct']
    profit_factor = result['profit_factor']
    sharpe_ratio = result['sharpe_ratio']

    # Handle infinite profit factor
    if profit_factor == float('inf'):
        profit_factor = 10.0  # Cap at 10 for scoring

    # Normalize metrics for combination
    # Return: normalize to 0-1 scale (assume max ~500%)
    return_score = min(return_pct / 500.0, 1.0)

    # Profit Factor: normalize to 0-1 scale (assume max ~3.0)
    pf_score = min(profit_factor / 3.0, 1.0)

    # Sharpe Ratio: normalize to 0-1 scale (assume max ~3.0)
    sharpe_score = min(max(sharpe_ratio, 0) / 3.0, 1.0)

    # Combined objective: Return 50% + PF 30% + Sharpe 20%
    objective_value = (
        (return_score * 0.50) +
        (pf_score * 0.30) +
        (sharpe_score * 0.20)
    )

    # Store metrics
    trial.set_user_attr('return_pct', result['return_pct'])
    trial.set_user_attr('profit_factor', result['profit_factor'])
    trial.set_user_attr('sharpe_ratio', result['sharpe_ratio'])
    trial.set_user_attr('win_rate', result['win_rate'])
    trial.set_user_attr('max_drawdown_pct', result['max_drawdown_pct'])
    trial.set_user_attr('total_trades', result['total_trades'])

    return objective_value


def worker_optimize(args):
    """Worker function for parallel optimization."""
    worker_id, study_name, storage_path, n_trials, df_path, baseline_params = args

    print(f"[Worker {worker_id}] Starting {n_trials} trials...")

    # Load data in worker
    df = pd.read_csv(df_path, index_col=0, parse_dates=True)

    # Load study from storage
    study = optuna.load_study(study_name=study_name, storage=storage_path)

    # Run optimization
    study.optimize(lambda trial: objective(trial, df, baseline_params), n_trials=n_trials)

    print(f"[Worker {worker_id}] Completed {n_trials} trials")

    return worker_id


def plot_rr_analysis(study, save_path):
    """Generate comprehensive RR analysis plots."""

    # Extract data from trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df['value'] > -999]  # Remove failed trials

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. RR vs Return
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(trials_df['params_rr_ratio'], trials_df['user_attrs_return_pct'],
                          c=trials_df['value'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax1.set_xlabel('Risk/Reward Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return %', fontsize=12, fontweight='bold')
    ax1.set_title('RR vs Return %', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Objective Score')

    # 2. RR vs Profit Factor
    ax2 = fig.add_subplot(gs[0, 1])
    # Cap PF at 5 for visualization
    pf_capped = trials_df['user_attrs_profit_factor'].clip(upper=5)
    scatter2 = ax2.scatter(trials_df['params_rr_ratio'], pf_capped,
                          c=trials_df['value'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax2.set_xlabel('Risk/Reward Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Profit Factor', fontsize=12, fontweight='bold')
    ax2.set_title('RR vs Profit Factor', fontsize=14, fontweight='bold')
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='PF = 1.0 (Break-even)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Objective Score')

    # 3. RR vs Sharpe Ratio
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(trials_df['params_rr_ratio'], trials_df['user_attrs_sharpe_ratio'],
                          c=trials_df['value'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax3.set_xlabel('Risk/Reward Ratio', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('RR vs Sharpe Ratio', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Objective Score')

    # 4. RR vs Win Rate
    ax4 = fig.add_subplot(gs[1, 0])
    scatter4 = ax4.scatter(trials_df['params_rr_ratio'], trials_df['user_attrs_win_rate'],
                          c=trials_df['value'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax4.set_xlabel('Risk/Reward Ratio', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Win Rate %', fontsize=12, fontweight='bold')
    ax4.set_title('RR vs Win Rate', fontsize=14, fontweight='bold')
    ax4.axhline(35, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='35% Target')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Objective Score')

    # 5. RR vs Max Drawdown
    ax5 = fig.add_subplot(gs[1, 1])
    scatter5 = ax5.scatter(trials_df['params_rr_ratio'], trials_df['user_attrs_max_drawdown_pct'],
                          c=trials_df['value'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax5.set_xlabel('Risk/Reward Ratio', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Max Drawdown %', fontsize=12, fontweight='bold')
    ax5.set_title('RR vs Max Drawdown', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter5, ax=ax5, label='Objective Score')

    # 6. RR vs Total Trades
    ax6 = fig.add_subplot(gs[1, 2])
    scatter6 = ax6.scatter(trials_df['params_rr_ratio'], trials_df['user_attrs_total_trades'],
                          c=trials_df['value'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    ax6.set_xlabel('Risk/Reward Ratio', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Total Trades', fontsize=12, fontweight='bold')
    ax6.set_title('RR vs Total Trades', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter6, ax=ax6, label='Objective Score')

    # 7. Objective Score Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(trials_df['value'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax7.axvline(study.best_value, color='red', linestyle='--', linewidth=2, label=f'Best: {study.best_value:.4f}')
    ax7.set_xlabel('Objective Score', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax7.set_title('Objective Score Distribution', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Top 10 RR Ratios
    ax8 = fig.add_subplot(gs[2, 1:])
    top_trials = trials_df.nlargest(10, 'value')
    x_pos = range(len(top_trials))
    bars = ax8.barh(x_pos, top_trials['value'], color='green', alpha=0.6, edgecolor='black')
    ax8.set_yticks(x_pos)
    ax8.set_yticklabels([f"RR {rr:.2f}:1 (Return: {ret:.1f}%)"
                         for rr, ret in zip(top_trials['params_rr_ratio'], top_trials['user_attrs_return_pct'])])
    ax8.set_xlabel('Objective Score', fontsize=12, fontweight='bold')
    ax8.set_title('Top 10 Risk/Reward Ratios', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Risk/Reward Ratio Optimization Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("RISK/REWARD RATIO OPTIMIZATION - MULTIPROCESSING")
    print("=" * 80)
    print("\nOptimizing ONLY RR ratio (all other parameters FIXED at baseline)")
    print("\nObjective: Return (50%) + Profit Factor (30%) + Sharpe Ratio (20%)")
    print("Range: 1.5:1 to 4.5:1")
    print("Trials: 50 (parallel workers)")
    print("=" * 80)

    # Configuration for multiprocessing
    n_workers = 5  # 5 workers for 50 trials total
    trials_per_worker = 10  # 10 trials per worker
    total_trials = n_workers * trials_per_worker

    # Load baseline parameters
    print("\n[*] Loading baseline parameters...")
    params_file = Path("optimization_results/fixed_optimizer_500t_best_params_20251116_173532.json")

    with open(params_file, 'r') as f:
        params_data = json.load(f)

    baseline_params = params_data['best_params'].copy()

    # Add derived parameters (same as optimizer)
    baseline_params['strong_imbalance'] = baseline_params['min_volume_imbalance'] * 2.0
    baseline_params['stacked_imbalance_ratio'] = 3.0
    baseline_params['min_stacked_bars'] = 3
    baseline_params['volume_ratio_threshold'] = 2.0
    # target_multiplier will be set dynamically based on RR ratio

    print(f"[OK] Baseline parameters loaded")
    print(f"     Current RR: 2.5:1")
    print(f"     Stop Multiplier: {baseline_params['stop_multiplier']:.4f}")

    # Setup SQLite storage for multi-process sharing
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    timestamp_db = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_file = results_dir / f'optuna_rr_optimization_{timestamp_db}.db'
    storage_path = f"sqlite:///{db_file}"
    study_name = f"rr_optimization_{timestamp_db}"

    print(f"\n[*] Multiprocessing Configuration:")
    print(f"    - CPU Cores Available: {cpu_count()}")
    print(f"    - Workers: {n_workers} parallel processes")
    print(f"    - Trials per worker: {trials_per_worker}")
    print(f"    - Total trials: {total_trials}")
    print(f"    - Storage: SQLite RDBStorage")
    print(f"    - Database: {db_file.name}")

    # Create Optuna study
    print(f"\n[*] Creating Optuna study: {study_name}")
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage_path,
        load_if_exists=False,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Data file path
    df_path = Path("tick_data/databento/bars/MNQ_500t_bars.csv")

    print("\n[*] Starting parallel optimization...")
    print(f"    - Testing RR ratios from 1.5:1 to 4.5:1")
    print(f"    - {total_trials} trials with {n_workers} parallel workers")
    print(f"    - Estimated time: ~2-3 minutes\n")

    # Prepare worker arguments
    worker_args = [
        (i, study_name, storage_path, trials_per_worker, str(df_path), baseline_params)
        for i in range(n_workers)
    ]

    # Run optimization with multiprocessing
    start_time = datetime.now()

    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_optimize, worker_args)

    duration = (datetime.now() - start_time).total_seconds() / 60

    # Reload study to get all results
    study = optuna.load_study(study_name=study_name, storage=storage_path)

    # Get best trial
    best_trial = study.best_trial
    best_rr = best_trial.params['rr_ratio']

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)

    print(f"\n[*] Duration: {duration:.2f} minutes")
    print(f"[*] Trials completed: {len(study.trials)}")
    print(f"[*] Best objective score: {study.best_value:.4f}")

    print("\n" + "=" * 80)
    print("BEST RISK/REWARD RATIO")
    print("=" * 80)

    print(f"\n  Optimal RR:        {best_rr:.2f}:1")
    print(f"  Current RR:        2.50:1")
    print(f"  Improvement:       {((best_rr - 2.5) / 2.5 * 100):+.1f}%")

    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS (with optimal RR)")
    print("=" * 80)

    print(f"\n  Return:            {best_trial.user_attrs['return_pct']:.2f}%")
    print(f"  Profit Factor:     {best_trial.user_attrs['profit_factor']:.4f}")
    print(f"  Sharpe Ratio:      {best_trial.user_attrs['sharpe_ratio']:.4f}")
    print(f"  Win Rate:          {best_trial.user_attrs['win_rate']:.2f}%")
    print(f"  Max Drawdown:      {best_trial.user_attrs['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades:      {best_trial.user_attrs['total_trades']}")

    # Compare with current RR 2.5:1
    print("\n" + "=" * 80)
    print("COMPARISON WITH CURRENT RR 2.5:1")
    print("=" * 80)

    # Load data for comparison
    df = load_tick_data(tick_interval=500, symbol="MNQ")
    current_result = run_backtest_with_rr(df, baseline_params, 2.5)

    print("\nCURRENT (RR 2.5:1):")
    print(f"  Return:            {current_result['return_pct']:.2f}%")
    print(f"  Profit Factor:     {current_result['profit_factor']:.4f}")
    print(f"  Sharpe Ratio:      {current_result['sharpe_ratio']:.4f}")
    print(f"  Win Rate:          {current_result['win_rate']:.2f}%")
    print(f"  Max Drawdown:      {current_result['max_drawdown_pct']:.2f}%")

    print(f"\nOPTIMAL (RR {best_rr:.2f}:1):")
    print(f"  Return:            {best_trial.user_attrs['return_pct']:.2f}%")
    print(f"  Profit Factor:     {best_trial.user_attrs['profit_factor']:.4f}")
    print(f"  Sharpe Ratio:      {best_trial.user_attrs['sharpe_ratio']:.4f}")
    print(f"  Win Rate:          {best_trial.user_attrs['win_rate']:.2f}%")
    print(f"  Max Drawdown:      {best_trial.user_attrs['max_drawdown_pct']:.2f}%")

    print("\nIMPROVEMENT:")
    print(f"  Return:            {(best_trial.user_attrs['return_pct'] - current_result['return_pct']):+.2f}%")
    print(f"  Profit Factor:     {(best_trial.user_attrs['profit_factor'] - current_result['profit_factor']):+.4f}")
    print(f"  Sharpe Ratio:      {(best_trial.user_attrs['sharpe_ratio'] - current_result['sharpe_ratio']):+.4f}")
    print(f"  Win Rate:          {(best_trial.user_attrs['win_rate'] - current_result['win_rate']):+.2f}%")
    print(f"  Max Drawdown:      {(best_trial.user_attrs['max_drawdown_pct'] - current_result['max_drawdown_pct']):+.2f}%")

    # Save results
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to JSON
    results_file = results_dir / f"rr_optimization_{timestamp}.json"
    results_data = {
        'best_rr_ratio': best_rr,
        'current_rr_ratio': 2.5,
        'best_objective_score': study.best_value,
        'objective_function': 'Return (50%) + Profit Factor (30%) + Sharpe Ratio (20%)',
        'optimal_metrics': {
            'return_pct': best_trial.user_attrs['return_pct'],
            'profit_factor': best_trial.user_attrs['profit_factor'],
            'sharpe_ratio': best_trial.user_attrs['sharpe_ratio'],
            'win_rate': best_trial.user_attrs['win_rate'],
            'max_drawdown_pct': best_trial.user_attrs['max_drawdown_pct'],
            'total_trades': best_trial.user_attrs['total_trades'],
        },
        'current_metrics': {
            'return_pct': current_result['return_pct'],
            'profit_factor': current_result['profit_factor'],
            'sharpe_ratio': current_result['sharpe_ratio'],
            'win_rate': current_result['win_rate'],
            'max_drawdown_pct': current_result['max_drawdown_pct'],
            'total_trades': current_result['total_trades'],
        },
        'optimization_info': {
            'n_trials': 50,
            'actual_trials': len(study.trials),
            'duration_minutes': duration,
            'timestamp': timestamp,
            'rr_range': [1.5, 4.5],
            'baseline_params_file': str(params_file),
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[SAVED] Results: {results_file}")

    # Generate visualization
    print("\n[*] Generating analysis plots...")
    plot_path = results_dir / f"rr_optimization_analysis_{timestamp}.png"
    plot_rr_analysis(study, plot_path)

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    return_improvement = best_trial.user_attrs['return_pct'] - current_result['return_pct']
    pf_improvement = best_trial.user_attrs['profit_factor'] - current_result['profit_factor']

    if return_improvement > 10 and pf_improvement > 0.05:
        print(f"\n[STRONG RECOMMENDATION] Switch to RR {best_rr:.2f}:1")
        print(f"  - Significant return improvement: +{return_improvement:.2f}%")
        print(f"  - Better profit factor: +{pf_improvement:.4f}")
        print(f"  - Update target_multiplier to: {baseline_params['stop_multiplier'] * best_rr:.4f}")
    elif return_improvement > 5:
        print(f"\n[MODERATE RECOMMENDATION] Consider switching to RR {best_rr:.2f}:1")
        print(f"  - Modest return improvement: +{return_improvement:.2f}%")
        print(f"  - Profit factor change: {pf_improvement:+.4f}")
    else:
        print(f"\n[KEEP CURRENT] RR 2.5:1 is near-optimal")
        print(f"  - Minimal improvement with RR {best_rr:.2f}:1: +{return_improvement:.2f}%")
        print(f"  - Current RR is well-calibrated for this strategy")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Required for Windows multiprocessing
    from multiprocessing import freeze_support
    freeze_support()
    main()
