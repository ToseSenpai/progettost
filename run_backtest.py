"""
Backtest Runner Script
Run backtests on historical data to test strategy performance
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.config import config


async def main():
    """Main function to run backtest"""
    print("\n")
    print("=" * 60)
    print("     PROJECTX TRADING BOT - BACKTEST")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    # Create backtester
    backtester = Backtester()

    try:
        # Run backtest
        results = await backtester.run_backtest()

        # Generate report
        if config.backtest.save_results:
            print("\n" + "=" * 60)
            backtester.generate_report()

        # Ask user if they want to see trade details
        print("\n" + "=" * 60)
        response = input("\nShow individual trades? (y/n): ").lower()

        if response == 'y' and results.trades:
            print("\n" + "=" * 60)
            print("TRADE DETAILS")
            print("=" * 60)

            for i, trade in enumerate(results.trades, 1):
                print(f"\nTrade #{i}:")
                print(f"  Symbol: {trade.symbol}")
                print(f"  Direction: {trade.direction}")
                print(f"  Size: {trade.size}")
                print(f"  Entry: ${trade.entry_price:.2f} @ {trade.entry_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"  Exit: ${trade.exit_price:.2f} @ {trade.exit_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"  P&L: ${trade.pnl:.2f}")
                print(f"  Commission: ${trade.commission:.2f}")
                print(f"  Net P&L: ${trade.net_pnl:.2f}")
                print(f"  Exit Reason: {trade.exit_reason}")
                print(f"  Result: {'[WIN]' if trade.is_winner else '[LOSS]'}")

        print("\n" + "=" * 60)
        print("[SUCCESS] Backtest completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n[STOP] Backtest interrupted by user")

    except Exception as e:
        print(f"\n[ERROR] Error running backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("\n[ERROR] .env file not found!")
        print("\nPlease create a .env file with your ProjectX credentials:")
        print("  1. Copy .env.template to .env")
        print("  2. Fill in your PROJECT_X_USERNAME and PROJECT_X_API_KEY")
        print("\nExample:")
        print("  PROJECT_X_USERNAME=your_username")
        print("  PROJECT_X_API_KEY=your_api_key")
        print("  TRADING_MODE=demo")
        print("  PROJECT_X_ENVIRONMENT=demo")
        sys.exit(1)

    # Run backtest
    asyncio.run(main())
