"""
Scalping Optimization for MET
Optimizes strategy parameters specifically for scalping on MET using 1-minute bars
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


class ScalpingOptimizer:
    """Optimize scalping strategy parameters for MET"""

    def __init__(self, n_trials=100):
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
        Objective function for Optuna - optimized for scalping

        Args:
            trial: Optuna trial object

        Returns:
            Composite optimization score
        """
        # Suggest parameter values - SCALPING FOCUSED
        params = {
            # EMA parameters - FASTER for scalping
            'ema_fast': trial.suggest_int('ema_fast', 5, 15),
            'ema_slow': trial.suggest_int('ema_slow', 20, 50),

            # RSI parameters - standard
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'rsi_overbought': trial.suggest_float('rsi_overbought', 65, 80),
            'rsi_oversold': trial.suggest_float('rsi_oversold', 20, 35),

            # ATR parameters - TIGHTER stops/targets for scalping
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 0.8, 2.0),  # Tighter stops
            'atr_target_multiplier': trial.suggest_float('atr_target_multiplier', 1.5, 3.5),  # Smaller targets
            'min_atr_threshold': trial.suggest_float('min_atr_threshold', 0.1, 1.0),
        }

        # Ensure ema_fast < ema_slow
        if params['ema_fast'] >= params['ema_slow']:
            return float('-inf')

        # Ensure target > stop (positive R/R) - but allow tighter for scalping
        if params['atr_target_multiplier'] <= params['atr_stop_multiplier']:
            return float('-inf')

        try:
            # Update config with trial parameters
            original_config = self._backup_config()
            self._apply_params(params)

            # Set symbol to MET only
            config.instruments.symbols = ['MET']

            # Use 3min timeframe for fast scalping (more signals than 1min)
            config.strategy.timeframe = '3min'

            # Run backtest
            backtester = Backtester()
            results = await backtester.run_backtest()

            # Restore original config
            self._restore_config(original_config)

            # Get optimization metric
            stats = results.statistics

            # For scalping, we need MORE trades (at least 10)
            if stats['total_trades'] < 10:
                return float('-inf')

            # Scalping composite score: prioritize win rate, profit factor, and # of trades
            # We want: high frequency + good win rate + good profit factor
            score = (
                stats['total_trades'] * 0.15 +  # More trades is good for scalping
                stats['win_rate_pct'] * 0.35 +  # Win rate very important
                (stats['profit_factor'] - 1) * 15 * 0.25 +  # Profit factor
                stats['total_return_pct'] * 0.15 +  # Total return
                (100 - abs(stats['max_drawdown_pct'])) * 0.10  # Low drawdown bonus
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
        print("     SCALPING OPTIMIZATION FOR MET")
        print("=" * 60)
        print(f"\nOptimization Settings:")
        print(f"  Symbol: MET (Micro Ether)")
        print(f"  Timeframe: 3min (fast scalping)")
        print(f"  Trials: {self.n_trials}")
        print(f"  Lookback: {config.backtest.lookback_days} days")
        print(f"  Min trades required: 10")

        print("\n[*] Scalping Parameter Search Spaces:")
        print("  EMA Fast: 5-15 (faster for scalping)")
        print("  EMA Slow: 20-50 (shorter period)")
        print("  RSI Period: 10-20")
        print("  RSI Overbought: 65-80")
        print("  RSI Oversold: 20-35")
        print("  ATR Period: 10-20")
        print("  ATR Stop: 0.8-2.0x (tighter stops)")
        print("  ATR Target: 1.5-3.5x (smaller targets)")
        print("  Min ATR Threshold: 0.1-1.0")

        print("\n[*] Optimization Focus:")
        print("  - High trade frequency")
        print("  - Good win rate")
        print("  - Tight risk management")
        print("  - Quick entries/exits")

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
        if trial.number % 10 == 0 and trial.number > 0:
            print(f"\n[Trial {trial.number}/{self.n_trials}] Current best score: {study.best_value:.4f}")

    def _print_results(self, study):
        """Print optimization results"""
        print("\n" + "=" * 60)
        print("SCALPING OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"\n[*] Best Scalping Score: {study.best_value:.4f}")

        print(f"\n[*] Best Scalping Parameters for MET:")
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
            print(f"\n[*] Most Important Parameters for Scalping:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {param}: {imp:.4f}")
        except:
            pass

        print("\n" + "=" * 60)

    def _save_results(self, study):
        """Save optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create optimization results directory
        results_dir = Path('optimization_results/scalping')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        params_file = results_dir / f'met_scalping_params_{timestamp}.txt'
        with open(params_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FAST SCALPING PARAMETERS FOR MET\n")
            f.write("=" * 60 + "\n\n")
            f.write("Timeframe: 3min\n")
            f.write(f"Best Score: {study.best_value:.4f}\n\n")
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
        history_file = results_dir / f'met_scalping_history_{timestamp}.csv'
        df = study.trials_dataframe()
        df.to_csv(history_file, index=False)
        print(f"[OK] Optimization history saved to {history_file}")


def main():
    """Main optimization function"""
    print("\n")
    print("=" * 60)
    print("     MET SCALPING OPTIMIZATION")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    print("[*] This will optimize MET for fast scalping on 3-minute bars")
    print("[*] Focus: High frequency, tight stops, quick profits")
    print(f"[*] Trials: 100")
    print(f"[*] Min trades: 10 (reduced from 20 for more results)")
    print(f"[*] Estimated time: 15-20 minutes")

    response = input("\nStart optimization? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    # Create optimizer
    optimizer = ScalpingOptimizer(n_trials=100)

    try:
        # Run optimization
        best_params, best_value = optimizer.optimize()

        print("\n" + "=" * 60)
        print("[SUCCESS] Scalping optimization completed!")
        print("=" * 60)

        print("\n[*] To use these parameters for scalping, update src/config.py:")
        print("\n# Fast Scalping Strategy Configuration for MET")
        print("timeframe: str = '3min'  # Fast scalping timeframe")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"{param}: float = {value:.4f}")
            else:
                print(f"{param}: int = {value}")

        print("\n[*] Next steps:")
        print("    1. Update src/config.py with these parameters")
        print("    2. Set timeframe to '3min'")
        print("    3. Run backtest to verify performance")
        print("    4. Test on demo account before going live")

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
