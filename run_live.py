"""
Live Trading Runner Script
Run the trading bot in live/demo mode
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.trading_engine import TradingEngine
from src.config import config


async def main():
    """Main function to run live trading"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 8 + "PROJECTX TRADING BOT - LIVE TRADING" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    # Show trading mode warning
    if config.api.trading_mode == 'live':
        print("\n" + "⚠️ " * 20)
        print("WARNING: YOU ARE RUNNING IN LIVE MODE!")
        print("⚠️ " * 20)
        print("\nThis bot will trade with REAL MONEY on your TopStepX account.")
        print("All trades are executed automatically based on the strategy.")
        print("\nIMPORTANT REMINDERS:")
        print("  • You are 100% responsible for all trades")
        print("  • Trades made via API are NOT eligible for review or reversal")
        print("  • Must run from your PERSONAL DEVICE (no VPS/VPN allowed)")
        print("  • TopStepX will terminate your account if you violate device policy")
        print("\n" + "=" * 60)

        response = input("\nType 'I UNDERSTAND' to continue in LIVE mode: ")

        if response != "I UNDERSTAND":
            print("\n❌ Live trading cancelled")
            return

    else:
        print("\n✅ Running in DEMO mode")
        print("   Safe to test - no real money involved")

    # Final confirmation
    print("\n" + "=" * 60)
    print("TRADING CONFIGURATION")
    print("=" * 60)
    print(f"Mode: {config.api.trading_mode.upper()}")
    print(f"Environment: {config.api.environment.upper()}")
    print(f"Instruments: {', '.join(config.instruments.symbols)}")
    print(f"Strategy: {config.strategy.timeframe} timeframe")
    print(f"Max Daily Loss: ${config.risk.max_daily_loss}")
    print(f"Max Positions: {config.risk.max_total_positions}")
    print("=" * 60)

    response = input("\nStart trading? (y/n): ").lower()

    if response != 'y':
        print("\n❌ Trading cancelled")
        return

    # Create trading engine
    engine = TradingEngine()

    try:
        # Initialize
        await engine.initialize()

        # Run trading
        await engine.run()

    except KeyboardInterrupt:
        print("\n\n⏹️  Received stop signal")
        await engine.stop()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        await engine.emergency_stop()


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("\n❌ ERROR: .env file not found!")
        print("\nPlease create a .env file with your ProjectX credentials:")
        print("  1. Copy .env.template to .env")
        print("  2. Fill in your PROJECT_X_USERNAME and PROJECT_X_API_KEY")
        print("\nExample:")
        print("  PROJECT_X_USERNAME=your_username")
        print("  PROJECT_X_API_KEY=your_api_key")
        print("  TRADING_MODE=demo  # or 'live' for real trading")
        print("  PROJECT_X_ENVIRONMENT=demo  # or 'topstepx' for live")
        print("\n⚠️  IMPORTANT: Always test on DEMO first before going live!")
        sys.exit(1)

    # Run live trading
    asyncio.run(main())
