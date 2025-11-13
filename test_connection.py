"""
Test Connection Script
Verifica la connessione all'API ProjectX e mostra informazioni account
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from project_x_py import ProjectX
from src.config import config


async def test_connection():
    """Test API connection and display account info"""

    print("\n")
    print("=" * 60)
    print("         PROJECTX CONNECTION TEST")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        print("[1/5] Validating configuration...")
        config.validate()
        print("      >> Configuration valid")
    except Exception as e:
        print(f"      >> Configuration error: {e}")
        print("\n[!] Please check your .env file!")
        return False

    # Display configuration
    print("\n[2/5] Configuration:")
    print(f"      Username: {config.api.username}")
    print(f"      API Key: {'*' * 20}{config.api.api_key[-4:] if len(config.api.api_key) > 4 else '****'}")
    print(f"      Trading Mode: {config.api.trading_mode.upper()}")
    print(f"      Environment: {config.api.environment.upper()}")

    # Test connection
    print("\n[3/5] Testing API connection...")

    try:
        # Create client
        async with ProjectX.from_env() as client:
            print("      >> Client created")

            # Authenticate
            print("      >> Authenticating...")
            await client.authenticate()
            print("      >> Authentication successful!")

            # Get account info
            print("\n[4/5] Fetching account information...")
            try:
                accounts = await client.search_for_account()

                if accounts and len(accounts) > 0:
                    print(f"      >> Found {len(accounts)} account(s)")

                    for i, account in enumerate(accounts, 1):
                        print(f"\n      Account #{i}:")
                        print(f"         ID: {account.id if hasattr(account, 'id') else 'N/A'}")
                        print(f"         Name: {account.name if hasattr(account, 'name') else 'N/A'}")

                        if hasattr(account, 'balance'):
                            print(f"         Balance: ${account.balance:,.2f}")

                        if hasattr(account, 'realizedPnl'):
                            print(f"         Realized P&L: ${account.realizedPnl:,.2f}")

                        if hasattr(account, 'unrealizedPnl'):
                            print(f"         Unrealized P&L: ${account.unrealizedPnl:,.2f}")
                else:
                    print("      >> No accounts found")

            except Exception as e:
                print(f"      >> Could not fetch account info: {e}")

            # Test contract search
            print("\n[5/5] Testing contract search...")
            try:
                # Search for MNQ
                contracts = await client.search_for_contracts(
                    searchText="MNQ",
                    live=True
                )

                if contracts and len(contracts) > 0:
                    print(f"      >> Found {len(contracts)} MNQ contract(s)")

                    # Show first contract
                    contract = contracts[0]
                    print(f"\n      Sample Contract:")
                    print(f"         ID: {contract.id if hasattr(contract, 'id') else 'N/A'}")
                    print(f"         Name: {contract.name if hasattr(contract, 'name') else 'N/A'}")
                    print(f"         Description: {contract.description if hasattr(contract, 'description') else 'N/A'}")

                    if hasattr(contract, 'tickSize'):
                        print(f"         Tick Size: {contract.tickSize}")
                    if hasattr(contract, 'tickValue'):
                        print(f"         Tick Value: ${contract.tickValue}")
                else:
                    print("      >> No contracts found")

            except Exception as e:
                print(f"      >> Could not search contracts: {e}")

            # Test historical data
            print("\n[BONUS] Testing historical data download...")
            try:
                bars = await client.get_bars(
                    symbol="MNQ",
                    days=1,  # Just 1 day for quick test
                    interval=5
                )

                if bars and len(bars) > 0:
                    print(f"      >> Downloaded {len(bars)} bars")

                    # Show latest bar
                    latest = bars[-1]
                    print(f"\n      Latest 5min Bar:")
                    print(f"         Time: {latest.t if hasattr(latest, 't') else 'N/A'}")
                    print(f"         Open: ${latest.o:.2f}" if hasattr(latest, 'o') else "         Open: N/A")
                    print(f"         High: ${latest.h:.2f}" if hasattr(latest, 'h') else "         High: N/A")
                    print(f"         Low: ${latest.l:.2f}" if hasattr(latest, 'l') else "         Low: N/A")
                    print(f"         Close: ${latest.c:.2f}" if hasattr(latest, 'c') else "         Close: N/A")
                    print(f"         Volume: {latest.v}" if hasattr(latest, 'v') else "         Volume: N/A")
                else:
                    print("      >> No historical data available")

            except Exception as e:
                print(f"      >> Could not download historical data: {e}")

        print("\n" + "=" * 60)
        print("SUCCESS! CONNECTION TEST PASSED!")
        print("=" * 60)
        print("\n[*] Your ProjectX connection is working correctly!")
        print("\n[*] Next steps:")
        print("    1. Run backtest: python run_backtest.py")
        print("    2. Test live trading: python run_live.py")
        print("\n")

        return True

    except Exception as e:
        print(f"\n" + "=" * 60)
        print("ERROR! CONNECTION TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\n[*] Troubleshooting:")
        print("    1. Check your .env file exists and has correct credentials")
        print("    2. Verify your PROJECT_X_USERNAME and PROJECT_X_API_KEY")
        print("    3. Make sure your API subscription is active")
        print("    4. Check internet connection")
        print(f"    5. Verify TRADING_MODE and PROJECT_X_ENVIRONMENT are correct")
        print("\n[*] For demo testing, use:")
        print("    TRADING_MODE=demo")
        print("    PROJECT_X_ENVIRONMENT=demo")
        print("\n[*] For TopStepX live, use:")
        print("    TRADING_MODE=live")
        print("    PROJECT_X_ENVIRONMENT=topstepx")
        print("\n")

        return False


async def quick_test():
    """Quick connection test"""
    print("\n>> Starting quick connection test...\n")

    success = await test_connection()

    return success


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("\n" + "!" * 60)
        print("ERROR: .env file not found!")
        print("!" * 60)
        print("\n[*] Please create a .env file:")
        print("    1. Copy .env.template to .env")
        print("    2. Edit .env and add your credentials:")
        print("\n       PROJECT_X_USERNAME=your_username")
        print("       PROJECT_X_API_KEY=your_api_key")
        print("       TRADING_MODE=demo")
        print("       PROJECT_X_ENVIRONMENT=demo")
        print("\n[*] Get your API key from: https://dashboard.projectx.com")
        print()
        sys.exit(1)

    # Run test
    success = asyncio.run(quick_test())

    sys.exit(0 if success else 1)
