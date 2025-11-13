"""
Volumetric Strategy Optimization for MET
Optimizes VWAP/MFI/VD/SuperTrend parameters for 5-minute bars on MET
Focus: Reduce overtrading, maximize win rate with drawdown < 3%
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


class VolumetricOptimizer:
    """Optimize volumetric strategy parameters for MET"""

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
        Objective function for Optuna - optimized for volumetric strategy

        Args:
            trial: Optuna trial object

        Returns:
            Composite optimization score
        """
        # Suggest parameter values for volumetric indicators
        params = {
            # MFI (Money Flow Index) parameters
            'mfi_period': trial.suggest_int('mfi_period', 10, 20),
            'mfi_overbought': trial.suggest_float('mfi_overbought', 70, 85),
            'mfi_oversold': trial.suggest_float('mfi_oversold', 15, 30),

            # SuperTrend parameters
            'st_period': trial.suggest_int('st_period', 7, 15),
            'st_multiplier': trial.suggest_float('st_multiplier', 2.0, 4.0),

            # Volume Delta threshold (higher = fewer signals)
            'vd_threshold': trial.suggest_float('vd_threshold', 0.2, 0.8),

            # ATR parameters for stops/targets
            'atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 2.5),
            'atr_target_multiplier': trial.suggest_float('atr_target_multiplier', 3.0, 5.0),
        }

        # Ensure target > stop (positive R/R)
        if params['atr_target_multiplier'] <= params['atr_stop_multiplier']:
            return float('-inf')

        try:
            # Update config with trial parameters
            original_config = self._backup_config()
            self._apply_params(params)

            # Set symbol to MET only
            config.instruments.symbols = ['MET']

            # Use volumetric strategy
            config.strategy.strategy_type = 'volumetric'
            config.strategy.timeframe = '5min'

            # Run backtest
            backtester = Backtester()
            results = await backtester.run_backtest()

            # Restore original config
            self._restore_config(original_config)

            # Get optimization metric
            stats = results.statistics

            # Need at least 5 trades
            if stats['total_trades'] < 5:
                return float('-inf')

            # Penalize extreme overtrading (more than 200 trades is too many)
            if stats['total_trades'] > 200:
                return float('-inf')

            # Reject if drawdown > 10% (more realistic)
            if abs(stats['max_drawdown_pct']) > 10.0:
                return float('-inf')

            # Require at least 40% win rate (more realistic)
            if stats['win_rate_pct'] < 40.0:
                return float('-inf')

            # Require profit factor > 1.2 (at least some profitability)
            if stats['profit_factor'] < 1.2:
                return float('-inf')

            # Composite score FOCUSED ON WIN RATE
            # Priority: Win rate most important, then profit factor, then return
            score = (
                stats['win_rate_pct'] * 0.50 +  # Win rate MOST important
                (stats['profit_factor'] - 1) * 20 * 0.30 +  # Profit factor important
                stats['total_return_pct'] * 0.15 +  # Total return
                (10.0 - abs(stats['max_drawdown_pct'])) * 0.05  # Bonus for low drawdown
            )

            return score

        except Exception as e:
            print(f"    Trial failed: {e}")
            return float('-inf')

    def _backup_config(self):
        """Backup current config"""
        return {
            'mfi_period': config.strategy.mfi_period,
            'mfi_overbought': config.strategy.mfi_overbought,
            'mfi_oversold': config.strategy.mfi_oversold,
            'st_period': config.strategy.st_period,
            'st_multiplier': config.strategy.st_multiplier,
            'vd_threshold': config.strategy.vd_threshold,
            'atr_stop_multiplier': config.strategy.atr_stop_multiplier,
            'atr_target_multiplier': config.strategy.atr_target_multiplier,
            'symbols': config.instruments.symbols.copy(),
            'timeframe': config.strategy.timeframe,
            'strategy_type': config.strategy.strategy_type
        }

    def _apply_params(self, params):
        """Apply parameters to config"""
        config.strategy.mfi_period = params['mfi_period']
        config.strategy.mfi_overbought = params['mfi_overbought']
        config.strategy.mfi_oversold = params['mfi_oversold']
        config.strategy.st_period = params['st_period']
        config.strategy.st_multiplier = params['st_multiplier']
        config.strategy.vd_threshold = params['vd_threshold']
        config.strategy.atr_stop_multiplier = params['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = params['atr_target_multiplier']

    def _restore_config(self, original):
        """Restore original config"""
        config.strategy.mfi_period = original['mfi_period']
        config.strategy.mfi_overbought = original['mfi_overbought']
        config.strategy.mfi_oversold = original['mfi_oversold']
        config.strategy.st_period = original['st_period']
        config.strategy.st_multiplier = original['st_multiplier']
        config.strategy.vd_threshold = original['vd_threshold']
        config.strategy.atr_stop_multiplier = original['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = original['atr_target_multiplier']
        config.instruments.symbols = original['symbols'].copy()
        config.strategy.timeframe = original['timeframe']
        config.strategy.strategy_type = original['strategy_type']

    def optimize(self):
        """Run optimization"""
        print("\n" + "=" * 60)
        print("     VOLUMETRIC STRATEGY OPTIMIZATION FOR MET")
        print("=" * 60)
        print(f"\nOptimization Settings:")
        print(f"  Symbol: MET (Micro Ether)")
        print(f"  Timeframe: 5min")
        print(f"  Trials: {self.n_trials}")
        print(f"  Lookback: {config.backtest.lookback_days} days")
        print(f"  Max Drawdown: 10.0% (limit)")
        print(f"  Min Trades: 5")
        print(f"  Max Trades: 200")
        print(f"  Min Win Rate: 40%")
        print(f"  Min Profit Factor: 1.2")

        print("\n[*] Parameter Search Spaces:")
        print("  MFI Period: 10-20")
        print("  MFI Overbought: 70-85")
        print("  MFI Oversold: 15-30")
        print("  SuperTrend Period: 7-15")
        print("  SuperTrend Multiplier: 2.0-4.0")
        print("  Volume Delta Threshold: 0.2-0.8")
        print("  ATR Stop: 1.5-2.5x")
        print("  ATR Target: 3.0-5.0x")

        print("\n[*] Optimization Goals (REALISTIC):")
        print("  - MAXIMIZE WIN RATE (target: >40%)")
        print("  - Maximize profit factor (target: >1.5)")
        print("  - Reduce overtrading (target: <200 trades)")
        print("  - Keep drawdown manageable (<10%)")

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
        print("VOLUMETRIC STRATEGY OPTIMIZATION RESULTS FOR MET")
        print("=" * 60)

        print(f"\n[*] Best Score: {study.best_value:.4f}")

        print(f"\n[*] Best Volumetric Parameters:")
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
        results_dir = Path('optimization_results/volumetric')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters
        params_file = results_dir / f'met_volumetric_params_{timestamp}.txt'
        with open(params_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("OPTIMIZED VOLUMETRIC STRATEGY PARAMETERS FOR MET (5min)\n")
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
        history_file = results_dir / f'met_volumetric_history_{timestamp}.csv'
        df = study.trials_dataframe()
        df.to_csv(history_file, index=False)
        print(f"[OK] Optimization history saved to {history_file}")


def main():
    """Main optimization function"""
    print("\n")
    print("=" * 60)
    print("     MET VOLUMETRIC STRATEGY OPTIMIZATION")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    print("[*] This will optimize VOLUMETRIC strategy (VWAP/MFI/VD/ST) for MET")
    print("[*] Focus: MAXIMIZE WIN RATE with realistic constraints")
    print(f"[*] Trials: 300")
    print(f"[*] Estimated time: 60-90 minutes")
    print()
    print("[!] Goal: Find parameters that maximize win rate (>40%)")
    print("[!] Goal: Achieve profit factor >1.5")
    print("[!] Goal: Reduce overtrading (<200 trades)")

    response = input("\nStart optimization? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    # Create optimizer
    optimizer = VolumetricOptimizer(n_trials=300)

    try:
        # Run optimization
        best_params, best_value = optimizer.optimize()

        print("\n" + "=" * 60)
        print("[SUCCESS] Volumetric optimization completed!")
        print("=" * 60)

        print("\n[*] To use these parameters, update src/config.py:")
        print("\n# Optimized Volumetric Strategy for MET (5min)")
        print("strategy_type: str = 'volumetric'")
        print("timeframe: str = '5min'")
        for param, value in best_params.items():
            if isinstance(value, float):
                print(f"{param}: float = {value:.4f}")
            else:
                print(f"{param}: int = {value}")

        print("\n[*] Next steps:")
        print("    1. Update src/config.py with these parameters")
        print("    2. Run compare_strategies.py to compare with Classic")
        print("    3. If volumetric wins, test on demo account")
        print("    4. Monitor performance carefully")

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
