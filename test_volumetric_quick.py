"""
Quick Volumetric Test - 10 trials per symbol
Test 3 different symbols to see which works best with volumetric strategy
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


class QuickVolumetricTest:
    """Quick test of volumetric strategy on multiple symbols"""

    def __init__(self, symbols, n_trials=10):
        self.symbols = symbols
        self.n_trials = n_trials
        self.results = {}

    async def objective(self, trial, symbol):
        """Objective function for optimization"""
        # Suggest parameter values
        params = {
            'mfi_period': trial.suggest_int('mfi_period', 10, 20),
            'mfi_overbought': trial.suggest_float('mfi_overbought', 70, 85),
            'mfi_oversold': trial.suggest_float('mfi_oversold', 15, 30),
            'st_period': trial.suggest_int('st_period', 7, 15),
            'st_multiplier': trial.suggest_float('st_multiplier', 2.0, 4.0),
            'vd_threshold': trial.suggest_float('vd_threshold', 0.2, 0.8),
            'atr_stop_multiplier': trial.suggest_float('atr_stop_multiplier', 1.5, 2.5),
            'atr_target_multiplier': trial.suggest_float('atr_target_multiplier', 3.0, 5.0),
        }

        if params['atr_target_multiplier'] <= params['atr_stop_multiplier']:
            return float('-inf')

        try:
            original_config = self._backup_config()
            self._apply_params(params)

            config.instruments.symbols = [symbol]
            config.strategy.strategy_type = 'volumetric'
            config.strategy.timeframe = '5min'

            backtester = Backtester()
            results = await backtester.run_backtest()

            self._restore_config(original_config)

            stats = results.statistics

            # Realistic filters
            if stats['total_trades'] < 5:
                return float('-inf')
            if stats['total_trades'] > 200:
                return float('-inf')
            if abs(stats['max_drawdown_pct']) > 10.0:
                return float('-inf')
            if stats['win_rate_pct'] < 40.0:
                return float('-inf')
            if stats['profit_factor'] < 1.2:
                return float('-inf')

            # Score focused on win rate
            score = (
                stats['win_rate_pct'] * 0.50 +
                (stats['profit_factor'] - 1) * 20 * 0.30 +
                stats['total_return_pct'] * 0.15 +
                (10.0 - abs(stats['max_drawdown_pct'])) * 0.05
            )

            return score

        except Exception as e:
            print(f"    Trial failed: {e}")
            return float('-inf')

    def _backup_config(self):
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
        config.strategy.mfi_period = params['mfi_period']
        config.strategy.mfi_overbought = params['mfi_overbought']
        config.strategy.mfi_oversold = params['mfi_oversold']
        config.strategy.st_period = params['st_period']
        config.strategy.st_multiplier = params['st_multiplier']
        config.strategy.vd_threshold = params['vd_threshold']
        config.strategy.atr_stop_multiplier = params['atr_stop_multiplier']
        config.strategy.atr_target_multiplier = params['atr_target_multiplier']

    def _restore_config(self, original):
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

    def run(self):
        """Run quick test on all symbols"""
        print("\n" + "=" * 60)
        print("QUICK VOLUMETRIC TEST - MULTI-SYMBOL")
        print("=" * 60)
        print(f"\nTesting {len(self.symbols)} symbols with {self.n_trials} trials each")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Total trials: {len(self.symbols) * self.n_trials}")
        print(f"Estimated time: {len(self.symbols) * self.n_trials * 4} minutes")
        print("\n" + "=" * 60)

        for symbol in self.symbols:
            print(f"\n[*] Testing {symbol}...")
            print(f"    Trials: {self.n_trials}")

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )

            def objective_wrapper(trial):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.objective(trial, symbol))
                    return result
                finally:
                    loop.close()

            study.optimize(
                objective_wrapper,
                n_trials=self.n_trials,
                show_progress_bar=False
            )

            if study.best_value != float('-inf'):
                self.results[symbol] = {
                    'best_score': study.best_value,
                    'best_params': study.best_params,
                    'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                    'failed_trials': len([t for t in study.trials if t.value == float('-inf')])
                }
                print(f"    [OK] Best score: {study.best_value:.2f}")
            else:
                self.results[symbol] = None
                print(f"    [FAIL] No valid configuration found")

        self._print_results()

    def _print_results(self):
        """Print comparison results"""
        print("\n" + "=" * 60)
        print("QUICK TEST RESULTS")
        print("=" * 60)

        # Filter successful results
        successful = {k: v for k, v in self.results.items() if v is not None}

        if not successful:
            print("\n[!] No symbol found valid parameters with current constraints")
            print("\n[*] Constraints:")
            print("    - Win Rate > 40%")
            print("    - Profit Factor > 1.2")
            print("    - Drawdown < 10%")
            print("    - Trades: 5-200")
            print("\n[*] Try relaxing constraints or increasing trials")
            return

        # Sort by best score
        sorted_results = sorted(successful.items(), key=lambda x: x[1]['best_score'], reverse=True)

        print(f"\n[*] Found {len(sorted_results)} symbol(s) with valid configurations:\n")

        for rank, (symbol, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {symbol}")
            print(f"   Score: {result['best_score']:.2f}")
            print(f"   Trials: {result['completed_trials']}/{self.n_trials} completed")
            print(f"   Failed: {result['failed_trials']}/{self.n_trials}")
            print(f"   Best params:")
            for param, value in result['best_params'].items():
                if isinstance(value, float):
                    print(f"     {param}: {value:.4f}")
                else:
                    print(f"     {param}: {value}")
            print()

        # Best symbol
        best_symbol, best_result = sorted_results[0]
        print("=" * 60)
        print(f"[WINNER] Best symbol: {best_symbol}")
        print(f"         Score: {best_result['best_score']:.2f}")
        print("=" * 60)

        # Save best result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path('optimization_results/volumetric')
        results_dir.mkdir(parents=True, exist_ok=True)

        params_file = results_dir / f'quick_test_{best_symbol}_{timestamp}.txt'
        with open(params_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"QUICK VOLUMETRIC TEST - BEST RESULT: {best_symbol}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Score: {best_result['best_score']:.2f}\n")
            f.write(f"Trials: {self.n_trials}\n\n")
            f.write("Parameters:\n")
            for param, value in best_result['best_params'].items():
                if isinstance(value, float):
                    f.write(f"  {param} = {value:.4f}\n")
                else:
                    f.write(f"  {param} = {value}\n")

        print(f"\n[OK] Best parameters saved to {params_file}")


def main():
    """Main function"""
    print("\n")
    print("=" * 60)
    print("     QUICK VOLUMETRIC MULTI-SYMBOL TEST")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        return

    # Test 3 different symbols
    symbols = ['MET', 'MNQ', 'MES']  # Micro Ether, Micro Nasdaq, Micro S&P

    print(f"[*] Testing volumetric strategy on {len(symbols)} symbols")
    print(f"[*] Symbols: {', '.join(symbols)}")
    print(f"[*] Trials per symbol: 10")
    print(f"[*] Total trials: {len(symbols) * 10}")
    print(f"[*] Estimated time: ~{len(symbols) * 10 * 4} minutes")

    response = input("\nStart quick test? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    tester = QuickVolumetricTest(symbols, n_trials=10)

    try:
        tester.run()

        print("\n[*] Next steps:")
        print("    1. If a symbol shows promise, run full 300-trial optimization")
        print("    2. Use: python optimize_volumetric_met.py")
        print("    3. Update config.py with best symbol and parameters")

    except KeyboardInterrupt:
        print("\n\n[STOP] Test interrupted by user")

    except Exception as e:
        print(f"\n[ERROR] Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    env_file = Path('.env')
    if not env_file.exists():
        print("\n[ERROR] .env file not found!")
        sys.exit(1)

    main()
