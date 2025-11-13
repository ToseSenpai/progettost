"""
Strategy Optimizer using Optuna
Optimizes strategy parameters using Bayesian optimization
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.config import config
import copy


class StrategyOptimizer:
    """Optimize trading strategy parameters using Optuna"""

    def __init__(self, n_trials=100, optimization_metric='sharpe_ratio'):
        """
        Initialize optimizer

        Args:
            n_trials: Number of optimization trials
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor')
        """
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.best_params = None
        self.best_value = None

    async def objective(self, trial):
        """
        Objective function for Optuna

        Args:
            trial: Optuna trial object

        Returns:
            Optimization metric value
        """
        # Suggest parameter values
        params = {
            # EMA parameters
            'ema_fast': trial.suggest_int('ema_fast', 10, 30),
            'ema_slow': trial.suggest_int('ema_slow', 40, 100),

            # RSI parameters
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'rsi_overbought': trial.suggest_float('rsi_overbought', 65, 80),
            'rsi_oversold': trial.suggest_float('rsi_oversold', 20, 35),

            # ATR parameters
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 3.0),
            'atr_target_multiplier': trial.suggest_float('atr_target_multiplier', 3.0, 6.0),
            'min_atr_threshold': trial.suggest_float('min_atr_threshold', 0.3, 1.0),
        }

        # Ensure ema_fast < ema_slow
        if params['ema_fast'] >= params['ema_slow']:
            return float('-inf')

        # Ensure target > stop (positive R/R)
        if params['atr_target_multiplier'] <= params['atr_stop_multiplier']:
            return float('-inf')

        try:
            # Update config with trial parameters
            original_config = self._backup_config()
            self._apply_params(params)

            # Run backtest
            backtester = Backtester()
            results = await backtester.run_backtest()

            # Restore original config
            self._restore_config(original_config)

            # Get optimization metric
            stats = results.statistics

            if stats['total_trades'] < 10:
                # Not enough trades, penalize
                return float('-inf')

            if self.optimization_metric == 'sharpe_ratio':
                return stats['sharpe_ratio']
            elif self.optimization_metric == 'total_return':
                return stats['total_return_pct']
            elif self.optimization_metric == 'profit_factor':
                return stats['profit_factor']
            elif self.optimization_metric == 'expectancy':
                return stats['expectancy']
            elif self.optimization_metric == 'composite':
                # Composite score: balance multiple metrics
                return (
                    stats['sharpe_ratio'] * 0.3 +
                    stats['total_return_pct'] * 0.3 +
                    (stats['profit_factor'] - 1) * 10 * 0.2 +
                    stats['win_rate_pct'] * 0.2
                )
            else:
                return stats['total_return_pct']

        except Exception as e:
            print(f"Trial failed: {e}")
            return float('-inf')

    def _backup_config(self):
        """Backup current config"""
        return {
            'ema_fast': config.strategy.ema_fast,
            'ema_slow': config.strategy.ema_slow,
            'rsi_period': config.strategy.rsi_period,
            'rsi_overbought': config.strategy.rsi_overbought,
            'rsi_oversold': config.strategy.rsi_oversold,
            'atr_period': config.strategy.atr_period,
            'atr_stop_multiplier': config.strategy.atr_stop_multiplier,
            'atr_target_multiplier': config.strategy.atr_target_multiplier,
            'min_atr_threshold': config.strategy.min_atr_threshold,
        }

    def _apply_params(self, params):
        """Apply parameters to config"""
        config.strategy.ema_fast = params['ema_fast']
        config.strategy.ema_slow = params['ema_slow']
        config.strategy.rsi_period = params['rsi_period']
        config.strategy.rsi_overbought = params['rsi_overbought']
        config.strategy.rsi_oversold = params['rsi_oversold']
        config.strategy.atr_period = params['atr_period']
        config.strategy.atr_stop_multiplier = params['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = params['atr_target_multiplier']
        config.strategy.min_atr_threshold = params['min_atr_threshold']

    def _restore_config(self, original):
        """Restore original config"""
        config.strategy.ema_fast = original['ema_fast']
        config.strategy.ema_slow = original['ema_slow']
        config.strategy.rsi_period = original['rsi_period']
        config.strategy.rsi_overbought = original['rsi_overbought']
        config.strategy.rsi_oversold = original['rsi_oversold']
        config.strategy.atr_period = original['atr_period']
        config.strategy.atr_stop_multiplier = original['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = original['atr_target_multiplier']
        config.strategy.min_atr_threshold = original['min_atr_threshold']

    def optimize(self):
        """Run optimization"""
        print("\n" + "=" * 60)
        print("     STRATEGY PARAMETER OPTIMIZATION")
        print("=" * 60)
        print(f"\nOptimization Settings:")
        print(f"  Trials: {self.n_trials}")
        print(f"  Metric: {self.optimization_metric}")
        print(f"  Lookback: {config.backtest.lookback_days} days")
        print(f"  Symbols: {', '.join(config.instruments.symbols)}")

        print("\n[*] Parameter Search Spaces:")
        print("  EMA Fast: 10-30")
        print("  EMA Slow: 40-100")
        print("  RSI Period: 10-20")
        print("  RSI Overbought: 65-80")
        print("  RSI Oversold: 20-35")
        print("  ATR Period: 10-20")
        print("  ATR Stop: 1.5-3.0x")
        print("  ATR Target: 3.0-6.0x")
        print("  Min ATR Threshold: 0.3-1.0")

        print("\n" + "=" * 60)
        print("[*] Starting optimization...")
        print("=" * 60)

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner()
        )

        # Run optimization with async wrapper
        def objective_wrapper(trial):
            # Create new event loop for each trial
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.objective(trial))
                return result
            finally:
                loop.close()

        study.optimize(
            objective_wrapper,
            n_trials=self.n_trials,
            show_progress_bar=True,
            callbacks=[self._trial_callback]
        )

        # Store best results
        self.best_params = study.best_params
        self.best_value = study.best_value

        # Print results
        self._print_results(study)

        # Save results
        self._save_results(study)

        return self.best_params, self.best_value

    def _trial_callback(self, study, trial):
        """Callback after each trial"""
        if trial.number % 10 == 0:
            print(f"\n[Trial {trial.number}] Current best {self.optimization_metric}: {study.best_value:.4f}")

    def _print_results(self, study):
        """Print optimization results"""
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"\n[*] Best {self.optimization_metric}: {study.best_value:.4f}")

        print(f"\n[*] Best Parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")

        print(f"\n[*] Optimization Statistics:")
        print(f"  Completed Trials: {len(study.trials)}")
        print(f"  Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Failed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

        # Show parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n[*] Parameter Importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {param}: {imp:.4f}")
        except:
            pass

        print("\n" + "=" * 60)

    def _save_results(self, study):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create optimization results directory
        results_dir = Path('optimization_results')
        results_dir.mkdir(exist_ok=True)

        # Save best parameters
        params_file = results_dir / f'best_params_{timestamp}.txt'
        with open(params_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BEST PARAMETERS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Optimization Metric: {self.optimization_metric}\n")
            f.write(f"Best Value: {study.best_value:.4f}\n\n")
            f.write("Parameters:\n")
            for param, value in study.best_params.items():
                if isinstance(value, float):
                    f.write(f"  {param} = {value:.4f}\n")
                else:
                    f.write(f"  {param} = {value}\n")

        print(f"\n[OK] Best parameters saved to {params_file}")

        # Save optimization history
        history_file = results_dir / f'optimization_history_{timestamp}.csv'
        df = study.trials_dataframe()
        df.to_csv(history_file, index=False)
        print(f"[OK] Optimization history saved to {history_file}")


def main():
    """Main optimization function"""
    print("\n")
    print("=" * 60)
    print("     PROJECTX TRADING BOT - OPTIMIZATION")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    # Get optimization settings from user
    print("Optimization Metrics:")
    print("  1. Sharpe Ratio (risk-adjusted returns)")
    print("  2. Total Return (% profit)")
    print("  3. Profit Factor (gross profit / gross loss)")
    print("  4. Expectancy (average $ per trade)")
    print("  5. Composite (balanced multi-metric)")

    choice = input("\nSelect metric (1-5, default=5): ").strip() or '5'

    metrics = {
        '1': 'sharpe_ratio',
        '2': 'total_return',
        '3': 'profit_factor',
        '4': 'expectancy',
        '5': 'composite'
    }
    metric = metrics.get(choice, 'composite')

    n_trials = input("\nNumber of trials (default=50): ").strip() or '50'
    try:
        n_trials = int(n_trials)
    except:
        n_trials = 50

    # Create optimizer
    optimizer = StrategyOptimizer(n_trials=n_trials, optimization_metric=metric)

    try:
        # Run optimization
        best_params, best_value = optimizer.optimize()

        print("\n" + "=" * 60)
        print("[SUCCESS] Optimization completed!")
        print("=" * 60)

        print("\n[*] To use these parameters, update src/config.py:")
        print("\n# Strategy Configuration")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"{param}: float = {value:.4f}")
            else:
                print(f"{param}: int = {value}")

        print("\n[*] Or run a backtest with optimized parameters to verify:")
        print("    python run_backtest.py")

    except KeyboardInterrupt:
        print("\n\n[STOP] Optimization interrupted by user")

    except Exception as e:
        print(f"\n[ERROR] Error during optimization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("\n[ERROR] .env file not found!")
        print("\nPlease create a .env file with your ProjectX credentials")
        sys.exit(1)

    # Run optimization
    main()
