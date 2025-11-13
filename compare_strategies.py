"""
Compare Classic vs Volumetric Strategy
Runs both strategies and compares performance
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.config import config


async def run_comparison():
    """Run both strategies and compare results"""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON: CLASSIC vs VOLUMETRIC")
    print("=" * 60)

    results = {}

    # Test 1: Classic Strategy (EMA/RSI/ATR)
    print("\n[1/2] Testing CLASSIC strategy (EMA/RSI/ATR)...")
    config.strategy.strategy_type = 'classic'

    backtester_classic = Backtester()
    try:
        results_classic = await backtester_classic.run_backtest()
        results['classic'] = results_classic.statistics
        print("[OK] Classic strategy completed")
    except Exception as e:
        print(f"[ERROR] Classic strategy failed: {e}")
        results['classic'] = None

    # Test 2: Volumetric Strategy (VWAP/MFI/VD/ST)
    print("\n[2/2] Testing VOLUMETRIC strategy (VWAP/MFI/VD/ST)...")
    config.strategy.strategy_type = 'volumetric'

    backtester_vol = Backtester()
    try:
        results_vol = await backtester_vol.run_backtest()
        results['volumetric'] = results_vol.statistics
        print("[OK] Volumetric strategy completed")
    except Exception as e:
        print(f"[ERROR] Volumetric strategy failed: {e}")
        results['volumetric'] = None

    # Print comparison
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 60)

    if results['classic'] and results['volumetric']:
        classic = results['classic']
        volumetric = results['volumetric']

        print(f"\n{'Metric':<25} {'Classic':>15} {'Volumetric':>15} {'Winner':>12}")
        print("-" * 70)

        metrics = [
            ('Total Return %', 'total_return_pct'),
            ('Total Trades', 'total_trades'),
            ('Win Rate %', 'win_rate_pct'),
            ('Profit Factor', 'profit_factor'),
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Max Drawdown %', 'max_drawdown_pct'),
            ('Expectancy $', 'expectancy'),
            ('Avg Win $', 'avg_win'),
            ('Avg Loss $', 'avg_loss'),
        ]

        for label, key in metrics:
            c_val = classic.get(key, 0)
            v_val = volumetric.get(key, 0)

            # Determine winner (lower is better for drawdown)
            if 'Drawdown' in label:
                winner = 'Classic' if abs(c_val) < abs(v_val) else 'Volumetric'
            elif 'Loss' in label:
                winner = 'Classic' if abs(c_val) < abs(v_val) else 'Volumetric'
            else:
                winner = 'Classic' if c_val > v_val else 'Volumetric'

            # Format values
            if '%' in label:
                c_str = f"{c_val:>14.2f}%"
                v_str = f"{v_val:>14.2f}%"
            elif '$' in label:
                c_str = f"${c_val:>13.2f}"
                v_str = f"${v_val:>13.2f}"
            elif key == 'total_trades':
                c_str = f"{int(c_val):>15}"
                v_str = f"{int(v_val):>15}"
            else:
                c_str = f"{c_val:>15.2f}"
                v_str = f"{v_val:>15.2f}"

            print(f"{label:<25} {c_str} {v_str} {winner:>12}")

        print("-" * 70)

        # Overall winner
        score_classic = 0
        score_volumetric = 0

        # Weight different metrics
        if classic['total_return_pct'] > volumetric['total_return_pct']:
            score_classic += 3  # Return is most important
        else:
            score_volumetric += 3

        if classic['win_rate_pct'] > volumetric['win_rate_pct']:
            score_classic += 2
        else:
            score_volumetric += 2

        if classic['profit_factor'] > volumetric['profit_factor']:
            score_classic += 2
        else:
            score_volumetric += 2

        if abs(classic['max_drawdown_pct']) < abs(volumetric['max_drawdown_pct']):
            score_classic += 2
        else:
            score_volumetric += 2

        if classic['sharpe_ratio'] > volumetric['sharpe_ratio']:
            score_classic += 1
        else:
            score_volumetric += 1

        print(f"\n[*] OVERALL SCORE:")
        print(f"  Classic:    {score_classic}/10")
        print(f"  Volumetric: {score_volumetric}/10")

        if score_volumetric > score_classic:
            print(f"\n[WINNER] Volumetric Strategy wins by {score_volumetric - score_classic} points!")
            print(f"  Improvement: {((volumetric['total_return_pct'] / classic['total_return_pct']) - 1) * 100:.1f}% better return")
        elif score_classic > score_volumetric:
            print(f"\n[WINNER] Classic Strategy wins by {score_classic - score_volumetric} points!")
        else:
            print(f"\n[TIE] Both strategies tied!")

    print("\n" + "=" * 60)


def main():
    """Main function"""
    print("\n")
    print("=" * 60)
    print("     STRATEGY COMPARISON TEST")
    print("=" * 60)
    print()

    print("[*] This will compare:")
    print("    1. Classic Strategy (EMA/RSI/ATR)")
    print("    2. Volumetric Strategy (VWAP/MFI/VD/ST)")
    print()
    print(f"[*] Symbol: MET")
    print(f"[*] Timeframe: {config.strategy.timeframe}")
    print(f"[*] Lookback: {config.backtest.lookback_days} days")
    print()

    response = input("Run comparison? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    # Run comparison
    asyncio.run(run_comparison())


if __name__ == "__main__":
    main()
