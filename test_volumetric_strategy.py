"""
Test Volumetric Strategy
Quick test to verify the new volumetric strategy works correctly
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_manager import DataManager
from src.volumetric_strategy import VolumetricStrategy
from src.config import config
import pandas as pd


async def test_indicators():
    """Test indicator calculations"""
    print("\n" + "=" * 60)
    print("TESTING VOLUMETRIC STRATEGY - INDICATORS")
    print("=" * 60)

    # Initialize data manager
    print("\n[*] Initializing data manager...")
    data_manager = DataManager()
    await data_manager.initialize()

    try:
        # Download data
        print(f"\n[*] Downloading 30 days of {config.strategy.timeframe} data for MET...")
        df = await data_manager.download_historical_data(
            symbol='MET',
            days=30,
            interval=config.strategy.timeframe
        )

        if df is None or len(df) == 0:
            print("[ERROR] No data retrieved!")
            return

        print(f"  [OK] Downloaded {len(df)} bars")
        print(f"  [*] Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        # Initialize strategy
        print("\n[*] Initializing Volumetric Strategy...")
        strategy = VolumetricStrategy()
        print(f"  [OK] Strategy: {strategy}")
        print(f"  [*] VWAP Session: {strategy.vwap_session}")
        print(f"  [*] MFI Period: {strategy.mfi_period}")
        print(f"  [*] SuperTrend: Period={strategy.st_period}, Mult={strategy.st_multiplier}")

        # Calculate indicators
        print("\n[*] Calculating indicators...")
        df_with_indicators = strategy.calculate_indicators(df)

        # Check for NaN values
        indicator_cols = ['vwap', 'mfi', 'volume_delta', 'cumulative_delta', 'supertrend', 'st_direction', 'atr']
        nan_counts = {}
        for col in indicator_cols:
            if col in df_with_indicators.columns:
                nan_count = df_with_indicators[col].isna().sum()
                nan_counts[col] = nan_count

        print("\n[*] Indicator NaN counts:")
        for col, count in nan_counts.items():
            print(f"  {col}: {count} NaN values")

        # Show sample data
        print("\n[*] Sample indicator values (last 5 bars):")
        print(df_with_indicators[['timestamp', 'close', 'vwap', 'mfi', 'volume_delta',
                                   'supertrend', 'st_direction', 'atr']].tail(5).to_string())

        # Test signal generation
        print("\n[*] Testing signal generation...")
        signal = strategy.generate_signal(df_with_indicators, current_position=None)

        print(f"\n[*] Generated Signal:")
        print(f"  Signal: {signal.signal}")
        print(f"  Entry Price: ${signal.entry_price:.2f}")
        if signal.stop_loss:
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        if signal.take_profit:
            print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Reason: {signal.reason}")

        # Generate multiple signals to test different scenarios
        print("\n[*] Generating signals for last 10 bars:")
        for i in range(-10, 0):
            test_df = df_with_indicators.iloc[:i]
            if len(test_df) < 50:
                continue

            sig = strategy.generate_signal(test_df, current_position=None)
            if sig.signal != 'HOLD':
                print(f"  Bar {i}: {sig.signal} @ ${sig.entry_price:.2f} (conf: {sig.confidence:.2f}) - {sig.reason[:80]}")

        print("\n" + "=" * 60)
        print("[SUCCESS] Indicator tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n[*] Cleaning up...")
        await data_manager.cleanup()
        print("  [OK] Data manager cleaned up")


async def test_backtest():
    """Run a quick backtest with volumetric strategy"""
    print("\n" + "=" * 60)
    print("TESTING VOLUMETRIC STRATEGY - BACKTEST")
    print("=" * 60)

    # Import backtester
    from src.backtester import Backtester

    # Temporarily set strategy type to volumetric
    original_strategy_type = config.strategy.strategy_type
    config.strategy.strategy_type = 'volumetric'

    try:
        print("\n[*] Running backtest with Volumetric Strategy...")
        print(f"  Symbol: MET")
        print(f"  Timeframe: {config.strategy.timeframe}")
        print(f"  Lookback: 30 days")

        # Temporarily reduce lookback for faster testing
        original_lookback = config.backtest.lookback_days
        config.backtest.lookback_days = 30

        # Run backtest
        backtester = Backtester()
        results = await backtester.run_backtest()

        # Restore lookback
        config.backtest.lookback_days = original_lookback

        # Print results
        stats = results.statistics

        print("\n" + "=" * 60)
        print("VOLUMETRIC STRATEGY BACKTEST RESULTS")
        print("=" * 60)

        print(f"\n[*] Performance:")
        print(f"  Total Return:        {stats['total_return_pct']:>10.2f}%")
        print(f"  Total Trades:        {stats['total_trades']:>10}")
        print(f"  Win Rate:            {stats['win_rate_pct']:>10.2f}%")
        print(f"  Profit Factor:       {stats['profit_factor']:>10.2f}")
        print(f"  Sharpe Ratio:        {stats['sharpe_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {stats['max_drawdown_pct']:>10.2f}%")
        print(f"  Expectancy:          ${stats['expectancy']:>9.2f}")

        print("\n" + "=" * 60)
        print("[SUCCESS] Backtest completed!")
        print("=" * 60)

        # Restore strategy type
        config.strategy.strategy_type = original_strategy_type

        return stats

    except Exception as e:
        print(f"\n[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()

        # Restore settings
        config.strategy.strategy_type = original_strategy_type
        return None


def main():
    """Main test function"""
    print("\n")
    print("=" * 60)
    print("     VOLUMETRIC STRATEGY TEST SUITE")
    print("=" * 60)
    print()

    print("[*] This will test the new Volumetric Strategy:")
    print("    - VWAP (Volume Weighted Average Price)")
    print("    - MFI (Money Flow Index)")
    print("    - Volume Delta (Buy/Sell pressure)")
    print("    - SuperTrend (ATR-based trend filter)")
    print()

    response = input("Run tests? (y/n): ").lower()
    if response != 'y':
        print("Cancelled")
        return

    # Run tests
    asyncio.run(test_indicators())

    print("\n")
    response = input("Run full backtest? (y/n): ").lower()
    if response == 'y':
        asyncio.run(test_backtest())


if __name__ == "__main__":
    main()
