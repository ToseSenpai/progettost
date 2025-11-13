"""
Per-Symbol Optimization Script
Optimizes strategy parameters separately for each symbol using Optuna
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.config import config


class PerSymbolOptimizer:
    """Optimize strategy parameters separately for each symbol"""

    def __init__(self, symbols, n_trials=50, optimization_metric='composite'):
        """
        Initialize optimizer

        Args:
            symbols: List of symbols to optimize
            n_trials: Number of optimization trials per symbol
            optimization_metric: Metric to optimize
        """
        self.symbols = symbols
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.results = {}

    async def objective(self, trial, symbol):
        """
        Objective function for Optuna

        Args:
            trial: Optuna trial object
            symbol: Symbol to optimize

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
            'min_atr_threshold': trial.suggest_float('min_atr_threshold', 0.1, 1.5),
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

            # Set symbol
            config.instruments.symbols = [symbol]

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
            print(f"    Trial failed: {e}")
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
            'symbols': config.instruments.symbols.copy()
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
        config.instruments.symbols = original['symbols'].copy()

    def optimize_symbol(self, symbol):
        """
        Optimize parameters for a single symbol

        Args:
            symbol: Symbol to optimize

        Returns:
            Dictionary with best parameters and value
        """
        print(f"\n{'=' * 60}")
        print(f"OPTIMIZING {symbol}")
        print(f"{'=' * 60}")
        print(f"Trials: {self.n_trials}")
        print(f"Metric: {self.optimization_metric}")
        print(f"{'=' * 60}\n")

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
                result = loop.run_until_complete(self.objective(trial, symbol))
                return result
            finally:
                loop.close()

        # Optimize
        study.optimize(
            objective_wrapper,
            n_trials=self.n_trials,
            show_progress_bar=True,
            callbacks=[lambda study, trial: self._trial_callback(study, trial, symbol)]
        )

        # Store results
        result = {
            'symbol': symbol,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }

        self.results[symbol] = result

        # Print summary
        print(f"\n[*] {symbol} Optimization Complete!")
        print(f"    Best {self.optimization_metric}: {study.best_value:.4f}")
        print(f"    Best Parameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"      {param}: {value:.4f}")
            else:
                print(f"      {param}: {value}")

        return result

    def _trial_callback(self, study, trial, symbol):
        """Callback after each trial"""
        if trial.number % 10 == 0 and trial.number > 0:
            print(f"[{symbol}] Trial {trial.number}/{self.n_trials} - Best: {study.best_value:.4f}")

    def optimize_all(self):
        """Optimize all symbols"""
        print("\n" + "=" * 60)
        print("PER-SYMBOL OPTIMIZATION")
        print("=" * 60)
        print(f"\nSymbols to optimize: {', '.join(self.symbols)}")
        print(f"Trials per symbol: {self.n_trials}")
        print(f"Total trials: {len(self.symbols) * self.n_trials}")
        print(f"Optimization metric: {self.optimization_metric}")
        print("\n" + "=" * 60)

        # Optimize each symbol
        for symbol in self.symbols:
            try:
                self.optimize_symbol(symbol)
            except Exception as e:
                print(f"\n[ERROR] Failed to optimize {symbol}: {e}")
                import traceback
                traceback.print_exc()

        # Generate comparison report
        self.print_comparison()
        self.save_results()

    def print_comparison(self):
        """Print comparison of optimized results"""
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPARISON")
        print("=" * 60)

        if not self.results:
            print("\nNo results to compare")
            return

        # Create comparison DataFrame
        comparison_data = []
        for symbol, result in self.results.items():
            comparison_data.append({
                'symbol': symbol,
                'best_value': result['best_value'],
                'ema_fast': result['best_params']['ema_fast'],
                'ema_slow': result['best_params']['ema_slow'],
                'rsi_period': result['best_params']['rsi_period'],
                'atr_period': result['best_params']['atr_period'],
                'atr_stop': result['best_params']['atr_stop_multiplier'],
                'atr_target': result['best_params']['atr_target_multiplier'],
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('best_value', ascending=False)

        print("\n[*] Best Values by Symbol:")
        print(f"\n{'Symbol':<8} {self.optimization_metric.title():<15} EMA      RSI   ATR   Stop  Target")
        print("-" * 70)
        for row in df.itertuples():
            print(f"{row.symbol:<8} {row.best_value:>12.2f}    "
                  f"{row.ema_fast:2d}/{row.ema_slow:<3d} "
                  f"{row.rsi_period:2d}    "
                  f"{row.atr_period:2d}    "
                  f"{row.atr_stop:.2f}  "
                  f"{row.atr_target:.2f}")

        print("\n" + "=" * 60)

    def save_results(self):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create results directory
        results_dir = Path('optimization_results/per_symbol')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save individual symbol results
        for symbol, result in self.results.items():
            # Save parameters
            params_file = results_dir / f'{symbol}_params_{timestamp}.txt'
            with open(params_file, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write(f"OPTIMIZED PARAMETERS FOR {symbol}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Optimization Metric: {self.optimization_metric}\n")
                f.write(f"Best Value: {result['best_value']:.4f}\n\n")
                f.write("Parameters:\n")
                for param, value in result['best_params'].items():
                    if isinstance(value, float):
                        f.write(f"  {param} = {value:.4f}\n")
                    else:
                        f.write(f"  {param} = {value}\n")

            # Save optimization history
            history_file = results_dir / f'{symbol}_history_{timestamp}.csv'
            df = result['study'].trials_dataframe()
            df.to_csv(history_file, index=False)

        # Save comparison
        comparison_file = results_dir / f'comparison_{timestamp}.json'
        comparison_data = {}
        for symbol, result in self.results.items():
            comparison_data[symbol] = {
                'best_value': result['best_value'],
                'best_params': result['best_params'],
                'n_trials': result['n_trials']
            }

        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"\n[OK] Results saved to {results_dir}")
        print(f"[OK] Comparison saved to {comparison_file}")


def main():
    """Main optimization function"""
    print("\n")
    print("=" * 60)
    print("PROJECTX TRADING BOT - PER-SYMBOL OPTIMIZATION")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    # Symbols to optimize
    symbols = ['MNQ', 'MET', 'MES', 'MYM', 'MGC', 'MBT']

    # Get optimization settings
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

    n_trials = input("\nTrials per symbol (default=30): ").strip() or '30'
    try:
        n_trials = int(n_trials)
    except:
        n_trials = 30

    print(f"\n[*] Will optimize {len(symbols)} symbols with {n_trials} trials each")
    print(f"[*] Total trials: {len(symbols) * n_trials}")
    print(f"[*] This may take a while...")

    response = input("\nContinue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    # Create optimizer
    optimizer = PerSymbolOptimizer(
        symbols=symbols,
        n_trials=n_trials,
        optimization_metric=metric
    )

    try:
        # Run optimization
        optimizer.optimize_all()

        print("\n" + "=" * 60)
        print("[SUCCESS] Per-symbol optimization completed!")
        print("=" * 60)

        print("\n[*] Next steps:")
        print("    1. Review optimization_results/per_symbol/comparison_*.json")
        print("    2. Choose best symbol(s) based on results")
        print("    3. Update src/config.py with optimal parameters")

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
