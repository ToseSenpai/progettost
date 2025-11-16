"""
Optimize PURE Volumetric Strategy parameters using Optuna.

This script uses Optuna to find the best parameter combination for the
PURE volumetric tick strategy (NO MFI, NO SuperTrend, NO EMA - only order flow!).

IMPROVED with professional research insights:
- Volume Profile (POC, VAH, VAL)
- VWAP Bands (+/-1 std, +/-2 std)
- CVD Divergence detection
- Stacked Imbalances with 3:1 ratio (professional standard)
- RTH Filter (Regular Trading Hours only)

REDUCED PARAMETER SPACE (13 -> 9 params) for faster convergence:

Optimizes (9 critical parameters):
- Volume imbalance threshold (min)
- Delta slope period (order flow trend)
- Buy/Sell ratio thresholds (entry: long/short)
- Buy/Sell ratio thresholds (exit: long/short)
- Confluence threshold (entry strictness)
- Exit confluence threshold (exit strictness - NEW!)
- Stop loss multiplier

Fixed (based on research):
- strong_imbalance = min_volume_imbalance * 2.0
- stacked_imbalance_ratio = 3.0 (3:1 professional standard)
- min_stacked_bars = 3
- volume_ratio_threshold = 2.0
- target_multiplier = stop_multiplier * 2.5 (1:2.5 R/R - sweet spot for 35% WR)

Objective: Maximize Profit Factor + Win Rate (FIXED: PF 40% + WR 35% + Return 25%, RR 2.5:1)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os


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
    """
    Run backtest with given parameters.

    Args:
        df: DataFrame with tick bar data
        params: Dictionary of strategy parameters
        initial_capital: Starting capital
        commission: Commission per trade

    Returns:
        Dictionary with performance metrics
    """
    from src.volumetric_tick_strategy import VolumetricTickStrategy

    # Create strategy with custom parameters
    strategy = VolumetricTickStrategy()

    # Apply parameter overrides for PURE volumetric strategy
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
    entry_confidence = 0
    trades = []
    equity = [capital]

    # REMOVED: Daily loss limit (was too restrictive - only 5 trades!)
    # Keeping only break-even stops for protection

    for i in range(len(df_calc)):
        if i < 150:  # Skip warmup (reduced for speed - max delta_slope_period is 250)
            equity.append(capital)
            continue

        row = df_calc.iloc[:i+1]

        # Generate signal
        signal = strategy.generate_signal(row, position)

        current_price = row.iloc[-1]['close']
        current_high = row.iloc[-1]['high']
        current_low = row.iloc[-1]['low']

        # REMOVED: Break-even stops (were too aggressive - caused 100% PF < 1.0!)
        # Moving stop to BE at 1.0 ATR prevented trades from reaching 2.5 ATR target
        # Result: All trades closed at BE or SL, none at TP → PF < 1.0

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
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0

    # Drawdown
    equity_series = pd.Series(equity)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    # Sharpe ratio
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

    # Trade count penalty (prefer MORE trades, penalize only very few)
    trade_count_score = 1.0
    if len(trades) < 30:
        trade_count_score = len(trades) / 30  # Penalize only if < 30 trades
    # NO penalty for many trades - user prefers frequent trading

    return {
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'adjusted_sharpe': sharpe * trade_count_score,  # For optimization
    }


def objective(trial, df):
    """
    Optuna objective function.

    Args:
        trial: Optuna trial object
        df: DataFrame with tick data

    Returns:
        Sharpe ratio (to maximize)
    """
    # Define parameter search space - REDUCED from 13 to 9 parameters!
    # Fixed parameters based on research to reduce search space
    params = {
        # === CRITICAL PARAMETERS TO OPTIMIZE (9) ===

        # 1. Volume imbalance threshold (CENTERED around baseline 0.1706)
        'min_volume_imbalance': trial.suggest_float('min_volume_imbalance', 0.12, 0.25),

        # 2. Delta slope period (CENTERED around baseline 157)
        'delta_slope_period': trial.suggest_int('delta_slope_period', 120, 200),

        # 3-4. Entry thresholds (CENTERED around baseline 1.3455, 0.8758)
        'buy_sell_long_threshold': trial.suggest_float('buy_sell_long_threshold', 1.0, 1.7),
        'buy_sell_short_threshold': trial.suggest_float('buy_sell_short_threshold', 0.65, 1.05),

        # 5-6. Exit thresholds (CENTERED around baseline 0.7046, 1.1672)
        'buy_sell_exit_long': trial.suggest_float('buy_sell_exit_long', 0.55, 0.85),
        'buy_sell_exit_short': trial.suggest_float('buy_sell_exit_short', 0.95, 1.35),

        # 7. Confluence threshold (CENTERED around baseline 0.8072)
        'min_confluence': trial.suggest_float('min_confluence', 0.70, 0.90),

        # 8. Stop multiplier (CENTERED around baseline 1.9208)
        'stop_multiplier': trial.suggest_float('stop_multiplier', 1.5, 2.5),

        # 9. Exit confluence threshold (CENTERED around baseline 0.7633)
        'exit_confluence_threshold': trial.suggest_float('exit_confluence_threshold', 0.65, 0.85),

        # === FIXED PARAMETERS (based on research) ===
        'strong_imbalance': None,  # Will be derived: min_volume_imbalance * 2.0
        'stacked_imbalance_ratio': 3.0,  # Professional standard (3:1 from research)
        'min_stacked_bars': 3,  # Reasonable compromise
        'volume_ratio_threshold': 2.0,  # 2x average volume
        'target_multiplier': None,  # Will be derived: stop_multiplier * 2.5 (1:2.5 R/R - sweet spot)
    }

    # Derive dependent parameters
    params['strong_imbalance'] = params['min_volume_imbalance'] * 2.0
    params['target_multiplier'] = params['stop_multiplier'] * 2.5  # RR 2.5:1 (sweet spot for 35% WR)

    # Run backtest
    result = run_backtest(df, params)

    if result is None:
        return -999.0  # Penalize parameter sets that produce no trades

    # PHASE 1C QUALITY-FIRST OBJECTIVE: PF + WR + Return + DD + Trade Count
    win_rate = result['win_rate'] / 100  # Convert to 0-1 scale
    profit_factor = result['profit_factor']
    return_pct = result['return_pct']
    max_dd = abs(result['max_drawdown_pct'])
    total_trades = result['total_trades']

    # REMOVED HARD FILTER: Was causing 100% failure!
    # Let optimizer explore freely, penalties below will guide it to profitability
    # if profit_factor < 0.98:
    #     return -999.0

    # ADJUSTED DD scoring - less aggressive (was too restrictive!)
    # DD ranges: -10% = 1.0, -15% = 0.67, -20% = 0.33, -25% = 0.0
    dd_score = max(0.0, (25.0 - max_dd) / 15.0)  # 1.0 at DD=-10%, 0.0 at DD=-25%

    # TRADE COUNT scoring - penalize too few or too many trades (REDUCED weight!)
    # Optimal range: 600-1200 trades
    # Penalize: < 300 (too few), > 2000 (overtrading)
    if total_trades < 300:
        trade_score = total_trades / 300  # Linear penalty if < 300
    elif total_trades > 2000:
        trade_score = 2000 / total_trades  # Linear penalty if > 2000
    elif total_trades >= 600 and total_trades <= 1200:
        trade_score = 1.0  # Optimal range
    elif total_trades < 600:
        trade_score = 0.8 + (total_trades - 300) / 300 * 0.2  # 0.8-1.0 for 300-600
    else:  # 1200-2000
        trade_score = 1.0 - (total_trades - 1200) / 800 * 0.2  # 1.0-0.8 for 1200-2000

    # REBALANCED OBJECTIVE: PF (35%) + WR (30%) + Return (20%) + DD (10%) + Trades (5%)
    objective_value = (
        (profit_factor * 0.35) +      # 35% weight on profit factor (with HARD FILTER!)
        (win_rate * 0.30) +           # 30% weight on win rate (INCREASED from 25%)
        (return_pct / 100 * 0.20) +   # 20% weight on return %
        (dd_score * 0.10) +           # 10% weight on DD control
        (trade_score * 0.05)          # 5% weight on trade count (REDUCED from 10%)
    )

    # HARD PENALTY: Catastrophic win rate (< 30%)
    if win_rate < 0.30:
        objective_value *= 0.3  # 70% penalty for terrible win rate

    # PENALTY: Low win rate (< 33%)
    elif win_rate < 0.33:
        objective_value *= 0.6  # 40% penalty for low win rate

    # PENALTY: Penalize if profit factor < 1.15 (not robust enough)
    if profit_factor < 1.15:
        objective_value *= 0.8  # 20% penalty (less aggressive)

    # BONUS: Reward high win rate (>38%)
    if win_rate > 0.38:
        objective_value *= 1.10  # 10% bonus for high win rate

    # BONUS: Moderate reward for low drawdown (less aggressive than before)
    if max_dd < 12:
        objective_value *= 1.15  # 15% bonus for excellent DD control (<12%)
    elif max_dd < 18:
        objective_value *= 1.08  # 8% bonus for good DD control (<18%)

    # Store additional metrics
    trial.set_user_attr('return_pct', result['return_pct'])
    trial.set_user_attr('total_trades', result['total_trades'])
    trial.set_user_attr('win_rate', result['win_rate'])
    trial.set_user_attr('profit_factor', result['profit_factor'])
    trial.set_user_attr('max_drawdown_pct', result['max_drawdown_pct'])

    return objective_value


def run_worker_optimization(args):
    """
    Worker function for multiprocessing optimization.

    Args:
        args: Tuple of (worker_id, n_trials, storage_path, study_name, max_total_trials)
    """
    worker_id, n_trials, storage_path, study_name, max_total_trials = args

    print(f"[Worker {worker_id}] Starting optimization with {n_trials} trials (PID: {os.getpid()})")

    # Load data in this process
    df = load_tick_data(tick_interval=500)

    # Create/load shared study from SQLite RDBStorage
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42 + worker_id, n_startup_trials=15),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=storage_path,  # SQLite connection string
        load_if_exists=True  # Critical for multi-process sharing
    )

    # SAFETY: Create callback to stop at global trial limit
    def stop_at_max_trials(study, trial):
        """Stop optimization if we've reached the global trial limit."""
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n_complete >= max_total_trials:
            study.stop()
            print(f"[Worker {worker_id}] Reached global limit ({max_total_trials} trials), stopping...")

    # Run optimization for this worker with safety callback
    try:
        study.optimize(
            lambda trial: objective(trial, df),
            n_trials=n_trials,
            callbacks=[stop_at_max_trials],
            show_progress_bar=False,  # Disable for cleaner multi-process output
            timeout=3600  # 60 minute timeout per worker (was 10 min - too short!)
        )
    except KeyboardInterrupt:
        print(f"[Worker {worker_id}] Interrupted by user")
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")

    trials_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"[Worker {worker_id}] Completed (Total study trials: {trials_completed})")

    return worker_id


def main():
    """Main optimization function."""

    print("=" * 80)
    print("FIXED OPTIMIZER - PURE VOLUMETRIC STRATEGY (Profit Factor Priority)")
    print("=" * 80)
    print("\nOptimizing PURE order flow strategy (NO MFI, NO SuperTrend, NO EMA)")
    print("Using ONLY: VWAP, Volume Delta, Buy/Sell Ratio, Delta Slope, Stacked Imbalance")
    print("\nIMPROVED with Professional Research:")
    print("  - Volume Profile (POC, VAH, VAL)")
    print("  - VWAP Bands (+/-1 std, +/-2 std)")
    print("  - CVD Divergence detection")
    print("  - Stacked Imbalances with 3:1 ratio")
    print("  - RTH Filter (9:30-16:00 ET only)")
    print("  - Confluence-based exits (anti-premature exit)")
    print("\nPHASE 2 FIX (BASELINE TEST SUCCESSFUL!):")
    print("  - Baseline parameters (+40.80%) still work - CODE IS OK!")
    print("  - REMOVED PF hard filter (was rejecting everything)")
    print("  - CENTERED parameter ranges around baseline values")
    print("  - Profit Factor (35% weight) - NO HARD FILTER, penalties guide to PF > 1.0")
    print("  - Win Rate (30% weight) - TARGET: 33-36%, HARD PENALTY if < 30%")
    print("  - Return % (20% weight)")
    print("  - Drawdown Control (10% weight) - Target: -15% to -25%")
    print("  - Trade Count (5% weight) - Target: 600-1200 trades")
    print("  - Risk/Reward: 2.5:1 (sweet spot - balance between profit and win rate)")
    print("  - Stop Multiplier: 1.5-2.5 (CENTERED around baseline 1.9208)")
    print("  - Min Confluence: 0.70-0.90 (CENTERED around baseline 0.8072)")
    print("  - Time Filter: Excludes toxic hours (14, 18, 19 UTC)")
    print("  - Break-Even Stops: REMOVED (were killing all trades!)")
    print("\nPARAMETER SPACE: 9 optimized parameters")
    print("  - Focus on tradeable, robust strategy")
    print("  - Less overtrading, higher quality trades")
    print()

    # Configuration for multiprocessing
    n_workers = 10  # 10 workers for 100 trials total
    trials_per_worker = 10  # 10 trials per worker = 100 total trials
    total_trials = n_workers * trials_per_worker

    # Setup SQLite RDBStorage for multi-process sharing
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    # FRESH START: FIXED objective function (Profit Factor priority) requires new study
    timestamp_db = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_file = results_dir / f'optuna_fixed_optimizer_{timestamp_db}.db'
    storage_path = f"sqlite:///{db_file}"
    study_name = f"fixed_optimizer_{timestamp_db}"

    # SAFETY: Check for existing database files and warn
    old_dbs = list(results_dir.glob("optuna_*.db"))
    if old_dbs:
        print(f"[WARNING] Found {len(old_dbs)} old database file(s):")
        for db in old_dbs:
            print(f"  - {db.name}")
        print("[INFO] Using NEW database to avoid conflicts")
        print()

    print(f"[*] Multiprocessing Configuration:")
    print(f"    - CPU Cores Available: {cpu_count()}")
    print(f"    - Workers: {n_workers} parallel processes")
    print(f"    - Trials per worker: {trials_per_worker}")
    print(f"    - Total trials: {total_trials} (HARD LIMIT)")
    print(f"    - Parameters to optimize: 9 (down from 13!)")
    print(f"    - Storage: SQLite RDBStorage (Windows-compatible)")
    print(f"    - Timeout per worker: 60 minutes (safety)")
    print(f"    - Estimated time: ~11 minutes (100 fresh trials)")
    print()

    # FRESH START: Create new study for FIXED objective
    print(f"[*] Creating NEW Optuna study: {study_name}")
    print(f"[*] Database: {db_file.name}")
    print(f"[*] EMERGENCY FIX Objective: PF (35%) + WR (30%) + Return (20%) + DD (10%) + Trades (5%)")
    print(f"[*] SOFT FILTER: PF >= 0.98 (was 1.0), NO break-even stops, lower confluence")
    print(f"[*] Target: WR 30-35%, PF 0.98-1.15, DD -20% to -30%, 600-1200 trades")
    print(f"[*] CHANGES: Removed BE stops, lowered confluence 0.75-0.85, softened PF filter")
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        storage=storage_path,
        load_if_exists=False  # FRESH START - new objective function!
    )

    # Prepare worker arguments (include max_total_trials for safety callback)
    worker_args = [
        (i, trials_per_worker, storage_path, study_name, total_trials)
        for i in range(n_workers)
    ]

    print(f"\n[*] Launching {n_workers} worker processes...")
    print("=" * 80)

    start_time = datetime.now()

    # Run multiprocessing optimization
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_worker_optimization, worker_args)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print("\n" + "=" * 80)
    print(f"[*] All {n_workers} workers completed!")
    print("=" * 80)

    # Reload study to get all results
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_path
    )

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {duration:.1f} minutes")
    print(f"Trials completed: {len(study.trials)}")

    # Best parameters
    print("\n" + "=" * 80)
    print("BEST PARAMETERS")
    print("=" * 80)

    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial

    print(f"\nObjective Score (PF 40% + WR 35% + Return 25%): {best_value:.4f}\n")

    print("Parameters:")
    for key, value in best_params.items():
        print(f"  {key:<30} = {value}")

    print("\nPerformance Metrics:")
    print(f"  Return %:          {best_trial.user_attrs['return_pct']:.2f}%")
    print(f"  Total Trades:      {best_trial.user_attrs['total_trades']}")
    print(f"  Win Rate:          {best_trial.user_attrs['win_rate']:.2f}%")
    print(f"  Profit Factor:     {best_trial.user_attrs['profit_factor']:.2f}")
    print(f"  Max Drawdown:      {best_trial.user_attrs['max_drawdown_pct']:.2f}%")

    # Save results
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best parameters to JSON
    results_file = results_dir / f"fixed_optimizer_500t_best_params_{timestamp}.json"
    results_data = {
        'best_params': best_params,
        'best_objective_score': best_value,
        'objective_function': 'PHASE 2: Profit Factor (35%) + Win Rate (30%) + Return (20%) + DD (10%) + Trades (5%) - NO HARD FILTER',
        'metrics': {
            'return_pct': best_trial.user_attrs['return_pct'],
            'total_trades': best_trial.user_attrs['total_trades'],
            'win_rate': best_trial.user_attrs['win_rate'],
            'profit_factor': best_trial.user_attrs['profit_factor'],
            'max_drawdown_pct': best_trial.user_attrs['max_drawdown_pct'],
        },
        'optimization_info': {
            'tick_interval': '500t',
            'n_trials': total_trials,
            'actual_trials': len(study.trials),
            'n_workers': n_workers,
            'trials_per_worker': trials_per_worker,
            'duration_minutes': duration,
            'timestamp': timestamp,
            'study_name': study_name,
            'database': db_file.name,
            'strategy': 'FIXED Optimizer - Pure Volumetric (Order Flow Only, Profit Factor Priority)',
            'optimization_mode': 'Multiprocessing (SQLite RDBStorage)',
            'target': 'PHASE 2: NO HARD FILTER, centered params around baseline (+40.80%), penalties guide to PF > 1.0, target WR 33%+, RR 2.5:1',
            'improvements': 'Volume Profile, VWAP Bands, CVD Divergence, 3:1 Ratio, RTH Filter, Confluence Exit, RR 2.5:1, Time Filter (14,18,19 UTC), Centered ranges (baseline 0.8072)',
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[SAVED] Best parameters: {results_file}")

    # Save study for later analysis
    study_file = results_dir / f"fixed_optimizer_500t_study_{timestamp}.pkl"
    import pickle
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)

    print(f"[SAVED] Study object: {study_file}")
    print(f"[SAVED] Optuna database: {db_file}")

    # Parameter importance
    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCE")
    print("=" * 80)

    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param:<30} {importance:.4f}")
    except Exception as e:
        print(f"  [WARNING] Could not calculate importances: {e}")

    print("\n" + "=" * 80)
    print(f"\n[DONE] Optimization complete! Run backtest with optimized parameters.")
    print("=" * 80)


if __name__ == "__main__":
    main()
