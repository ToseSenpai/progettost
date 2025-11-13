"""
5-Minute Optimization for MET
Optimizes strategy parameters specifically for 5-minute bars on MET
Focus: Maximum return with drawdown < 3%
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


class Optimizer5MinMET:
    """Optimize strategy parameters for 5-minute MET trading"""

    def __init__(self, n_trials=300):
        """
        Initialize optimizer

        Args:
            n_trials: Number of optimization trials
        """
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = None

    async def objective(self, trial):
        """
        Objective function for Optuna - optimized for 5min MET

        Args:
            trial: Optuna trial object

        Returns:
            Composite optimization score
        """
        # Suggest parameter values - WIDER RANGE for 5min
        params = {
            # EMA parameters - explore wide range
            'ema_fast': trial.suggest_int('ema_fast', 8, 20),
            'ema_slow': trial.suggest_int('ema_slow', 50, 120),

            # RSI parameters
            'rsi_period': trial.suggest_int('rsi_period', 12, 20),
            'rsi_overbought': trial.suggest_float('rsi_overbought', 65, 80),
            'rsi_oversold': trial.suggest_float('rsi_oversold', 20, 35),

            # ATR parameters - balanced for 5min
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 2.5),
            'atr_target_multiplier': trial.suggest_float('atr_target_multiplier', 3.0, 5.0),
            'min_atr_threshold': trial.suggest_float('min_atr_threshold', 0.1, 0.8),
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

            # Set symbol to MET only
            config.instruments.symbols = ['MET']

            # Use 5min timeframe
            config.strategy.timeframe = '5min'

            # Run backtest
            backtester = Backtester()
            results = await backtester.run_backtest()

            # Restore original config
            self._restore_config(original_config)

            # Get optimization metric
            stats = results.statistics

            # Need at least 10 trades
            if stats['total_trades'] < 10:
                return float('-inf')

            # CRITICAL: Reject if drawdown > 3%
            if abs(stats['max_drawdown_pct']) > 3.0:
                return float('-inf')

            # Composite score optimized for 5min trading
            # Priority: Return, Win Rate, Profit Factor, Low Drawdown
            score = (
                stats['total_return_pct'] * 0.4 +  # Total return most important
                stats['win_rate_pct'] * 0.3 +  # Win rate important
                (stats['profit_factor'] - 1) * 10 * 0.2 +  # Profit factor
                (3.0 - abs(stats['max_drawdown_pct'])) * 5 * 0.1  # Reward low drawdown
            )

            return score

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
            'symbols': config.instruments.symbols.copy(),
            'timeframe': config.strategy.timeframe
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
        config.strategy.timeframe = original['timeframe']

    def optimize(self):
        """Run optimization"""
        print("\n" + "=" * 60)
        print("     5-MINUTE OPTIMIZATION FOR MET")
        print("=" * 60)
        print(f"\nOptimization Settings:")
        print(f"  Symbol: MET (Micro Ether)")
        print(f"  Timeframe: 5min")
        print(f"  Trials: {self.n_trials}")
        print(f"  Lookback: {config.backtest.lookback_days} days")
        print(f"  Max Drawdown: 3.0% (HARD LIMIT)")
        print(f"  Min Trades: 10")

        print("\n[*] Parameter Search Spaces:")
        print("  EMA Fast: 8-20")
        print("  EMA Slow: 50-120")
        print("  RSI Period: 12-20")
        print("  RSI Overbought: 65-80")
        print("  RSI Oversold: 20-35")
        print("  ATR Period: 10-20")
        print("  ATR Stop: 1.5-2.5x")
        print("  ATR Target: 3.0-5.0x")
        print("  Min ATR Threshold: 0.1-0.8")

        print("\n[*] Optimization Goals:")
        print("  - Maximize total return")
        print("  - High win rate")
        print("  - Strong profit factor")
        print("  - Drawdown < 3% (REQUIRED)")

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
        if trial.number % 30 == 0 and trial.number > 0:
            print(f"\n[Trial {trial.number}/{self.n_trials}] Current best score: {study.best_value:.4f}")

    def _print_results(self, study):
        """Print optimization results"""
        print("\n" + "=" * 60)
        print("5-MINUTE OPTIMIZATION RESULTS FOR MET")
        print("=" * 60)

        print(f"\n[*] Best Score: {study.best_value:.4f}")

        print(f"\n[*] Best Parameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")

        # Calculate R/R ratio
        stop = study.best_params['atr_stop_multiplier']
        target = study.best_params['atr_target_multiplier']
        rr_ratio = target / stop
        print(f"\n[*] Risk/Reward Ratio: 1:{rr_ratio:.2f}")

        print(f"\n[*] Optimization Statistics:")
        print(f"  Completed Trials: {len(study.trials)}")
        print(f"  Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  Failed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

        # Show parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n[*] Most Important Parameters:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {param}: {imp:.4f}")
        except:
            pass

        print("\n" + "=" * 60)

    def _save_results(self, study):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create optimization results directory
        results_dir = Path('optimization_results/5min')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        params_file = results_dir / f'met_5min_params_{timestamp}.txt'
        with open(params_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("OPTIMIZED PARAMETERS FOR MET (5min)\n")
            f.write("=" * 60 + "\n\n")
            f.write("Timeframe: 5min\n")
            f.write(f"Best Score: {study.best_value:.4f}\n")
            f.write(f"Max Drawdown Limit: 3.0%\n\n")
            f.write("Parameters:\n")
            for param, value in study.best_params.items():
                if isinstance(value, float):
                    f.write(f"  {param} = {value:.4f}\n")
                else:
                    f.write(f"  {param} = {value}\n")

            stop = study.best_params['atr_stop_multiplier']
            target = study.best_params['atr_target_multiplier']
            f.write(f"\nRisk/Reward Ratio: 1:{target/stop:.2f}\n")

        print(f"\n[OK] Best parameters saved to {params_file}")

        # Save optimization history
        history_file = results_dir / f'met_5min_history_{timestamp}.csv'
        df = study.trials_dataframe()
        df.to_csv(history_file, index=False)
        print(f"[OK] Optimization history saved to {history_file}")


def main():
    """Main optimization function"""
    print("\n")
    print("=" * 60)
    print("     MET 5-MINUTE OPTIMIZATION")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    print("[*] This will optimize MET for 5-minute trading")
    print("[*] Focus: Maximum return with drawdown < 3%")
    print(f"[*] Trials: 300")
    print(f"[*] Estimated time: 45-60 minutes")

    response = input("\nStart optimization? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    # Create optimizer
    optimizer = Optimizer5MinMET(n_trials=300)

    try:
        # Run optimization
        best_params, best_value = optimizer.optimize()

        print("\n" + "=" * 60)
        print("[SUCCESS] Optimization completed!")
        print("=" * 60)

        print("\n[*] To use these parameters, update src/config.py:")
        print("\n# Optimized 5-Minute Strategy for MET")
        print("timeframe: str = '5min'")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"{param}: float = {value:.4f}")
            else:
                print(f"{param}: int = {value}")

        print("\n[*] Next steps:")
        print("    1. Update src/config.py with these parameters")
        print("    2. Run backtest to verify performance")
        print("    3. Test on demo account")
        print("    4. Monitor drawdown carefully")

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
