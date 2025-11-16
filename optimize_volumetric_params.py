"""
Optimize VolumetricTickStrategy parameters using Optuna.

This script uses Optuna to find the best parameter combination for the
volumetric tick strategy on 1000-tick bars.

Optimizes:
- MFI thresholds (overbought/oversold)
- Volume imbalance thresholds
- SuperTrend multiplier
- Stop loss / Take profit multipliers
- Confluence threshold

Objective: Maximize Sharpe ratio
"""

import pandas as pd
import numpy as np
from pathlib import Path
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
from datetime import datetime


def load_tick_data(tick_interval: int = 1000, symbol: str = "MNQ"):
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

    # Apply parameter overrides
    strategy.mfi_overbought = params['mfi_overbought']
    strategy.mfi_oversold = params['mfi_oversold']
    strategy.st_multiplier = params['st_multiplier']
    strategy.min_volume_imbalance = params['min_volume_imbalance']
    strategy.strong_imbalance = params['strong_imbalance']
    strategy.stop_multiplier = params['stop_multiplier']
    strategy.target_multiplier = params['target_multiplier']

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

    min_confluence = params['min_confluence']

    for i in range(len(df_calc)):
        if i < 100:  # Skip warmup
            equity.append(capital)
            continue

        row = df_calc.iloc[:i+1]

        # Generate signal
        signal = strategy.generate_signal(row, position)

        # Override confluence threshold
        if signal.signal in ['LONG', 'SHORT']:
            if signal.confidence < min_confluence:
                signal.signal = 'HOLD'

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

    # Combined metric: Sharpe with penalty for low trade count
    trade_count_penalty = min(len(trades) / 100, 1.0)  # Penalize if < 100 trades

    return {
        'total_return': total_pnl,
        'return_pct': (total_pnl / initial_capital * 100),
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'adjusted_sharpe': sharpe * trade_count_penalty,  # For optimization
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
    # Define parameter search space
    params = {
        'mfi_overbought': trial.suggest_int('mfi_overbought', 70, 90),
        'mfi_oversold': trial.suggest_int('mfi_oversold', 10, 30),
        'st_multiplier': trial.suggest_float('st_multiplier', 2.0, 4.0),
        'min_volume_imbalance': trial.suggest_float('min_volume_imbalance', 0.10, 0.25),
        'strong_imbalance': trial.suggest_float('strong_imbalance', 0.25, 0.40),
        'stop_multiplier': trial.suggest_float('stop_multiplier', 1.0, 2.5),
        'target_multiplier': trial.suggest_float('target_multiplier', 3.0, 6.0),
        'min_confluence': trial.suggest_float('min_confluence', 0.70, 0.85),
    }

    # Run backtest
    result = run_backtest(df, params)

    if result is None:
        return -999.0  # Penalize parameter sets that produce no trades

    # Primary objective: Sharpe ratio
    # Secondary: Penalize excessive drawdown
    sharpe = result['sharpe_ratio']
    max_dd = abs(result['max_drawdown_pct'])

    # Combined objective: Sharpe with drawdown penalty
    if max_dd > 25:  # Penalize drawdown > 25%
        drawdown_penalty = (max_dd - 25) / 100
        objective_value = sharpe - drawdown_penalty
    else:
        objective_value = sharpe

    # Store additional metrics
    trial.set_user_attr('return_pct', result['return_pct'])
    trial.set_user_attr('total_trades', result['total_trades'])
    trial.set_user_attr('win_rate', result['win_rate'])
    trial.set_user_attr('profit_factor', result['profit_factor'])
    trial.set_user_attr('max_drawdown_pct', result['max_drawdown_pct'])

    return objective_value


def main():
    """Main optimization function."""

    print("=" * 80)
    print("VOLUMETRIC TICK STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\nUsing Optuna to find optimal parameters for 1000-tick bars")
    print()

    # Load data
    print("[*] Loading tick data...")
    df = load_tick_data(tick_interval=1000)

    # Create Optuna study
    print("\n[*] Creating Optuna study...")
    study = optuna.create_study(
        direction='maximize',
        study_name='volumetric_tick_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization
    n_trials = 150
    print(f"\n[*] Running optimization ({n_trials} trials)...")
    print("    This may take 10-15 minutes...")
    print("    Progress will be shown every 10 trials\n")

    start_time = datetime.now()

    study.optimize(
        lambda trial: objective(trial, df),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Single threaded for stability
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

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

    print(f"\nObjective Value (Sharpe): {best_value:.4f}\n")

    print("Parameters:")
    for key, value in best_params.items():
        print(f"  {key:<25} = {value}")

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
    results_file = results_dir / f"volumetric_1000t_best_params_{timestamp}.json"
    results_data = {
        'best_params': best_params,
        'best_sharpe': best_value,
        'metrics': {
            'return_pct': best_trial.user_attrs['return_pct'],
            'total_trades': best_trial.user_attrs['total_trades'],
            'win_rate': best_trial.user_attrs['win_rate'],
            'profit_factor': best_trial.user_attrs['profit_factor'],
            'max_drawdown_pct': best_trial.user_attrs['max_drawdown_pct'],
        },
        'optimization_info': {
            'n_trials': n_trials,
            'duration_minutes': duration,
            'timestamp': timestamp,
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[SAVED] Best parameters: {results_file}")

    # Save study for later analysis
    study_file = results_dir / f"volumetric_1000t_study_{timestamp}.pkl"
    import pickle
    with open(study_file, 'wb') as f:
        pickle.dump(study, f)

    print(f"[SAVED] Study object: {study_file}")

    # Parameter importance
    print("\n" + "=" * 80)
    print("PARAMETER IMPORTANCE")
    print("=" * 80)

    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param:<25} {importance:.4f}")
    except Exception as e:
        print(f"  [WARNING] Could not calculate importances: {e}")

    # Compare with default parameters
    print("\n" + "=" * 80)
    print("COMPARISON: OPTIMIZED vs DEFAULT")
    print("=" * 80)

    default_params = {
        'mfi_overbought': 80,
        'mfi_oversold': 20,
        'st_multiplier': 3.0,
        'min_volume_imbalance': 0.15,
        'strong_imbalance': 0.30,
        'stop_multiplier': 1.65,
        'target_multiplier': 4.46,
        'min_confluence': 0.75,
    }

    print("\n[*] Running backtest with DEFAULT parameters...")
    default_result = run_backtest(df, default_params)

    print("\n[*] Running backtest with OPTIMIZED parameters...")
    optimized_result = run_backtest(df, best_params)

    print("\n" + "-" * 80)
    print(f"{'Metric':<30} {'Default':>20} {'Optimized':>20} {'Change':>15}")
    print("-" * 80)

    metrics = [
        ('Return %', 'return_pct', '%'),
        ('Total Trades', 'total_trades', ''),
        ('Win Rate %', 'win_rate', '%'),
        ('Profit Factor', 'profit_factor', ''),
        ('Max Drawdown %', 'max_drawdown_pct', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
    ]

    for metric_name, metric_key, suffix in metrics:
        default_val = default_result[metric_key]
        optimized_val = optimized_result[metric_key]

        if metric_key in ['total_trades']:
            change = optimized_val - default_val
            change_str = f"{change:+.0f}"
        else:
            change = ((optimized_val - default_val) / abs(default_val) * 100) if default_val != 0 else 0
            change_str = f"{change:+.1f}%"

        print(f"{metric_name:<30} {default_val:>20.2f} {optimized_val:>20.2f} {change_str:>15}")

    print("=" * 80)

    # Final recommendation
    improvement = ((optimized_result['sharpe_ratio'] - default_result['sharpe_ratio']) / abs(default_result['sharpe_ratio']) * 100) if default_result['sharpe_ratio'] != 0 else 0

    print(f"\n[RESULT] Optimized parameters improved Sharpe ratio by {improvement:+.1f}%")

    if improvement > 5:
        print("[RECOMMENDATION] Use optimized parameters - significant improvement!")
    elif improvement > 0:
        print("[RECOMMENDATION] Use optimized parameters - modest improvement")
    else:
        print("[RECOMMENDATION] Default parameters are competitive")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
