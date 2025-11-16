"""
Download historical tick data from Databento.

This script downloads tick-by-tick trade data for futures contracts from Databento
and prepares it for backtesting with our tick aggregator.

Databento provides high-quality historical market data including:
- All trades (every single transaction)
- Bid/ask quotes
- Market depth
- And more

Usage:
    python download_databento_ticks.py

Requirements:
    - Databento API key in .env file
    - Standard account or higher (for historical tick data)
"""

import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from decimal import Decimal
import pandas as pd
from dotenv import load_dotenv
import databento as db


def load_databento_client():
    """Load Databento client with API key from environment."""
    load_dotenv()

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key or api_key == "your_databento_api_key_here":
        raise ValueError(
            "DATABENTO_API_KEY not found in .env file.\n"
            "Please add your Databento API key to the .env file:\n"
            "DATABENTO_API_KEY=your_actual_key_here\n"
            "Get your key from: https://databento.com/portal/keys"
        )

    print(f"[OK] Loaded Databento API key: {api_key[:8]}...")
    return db.Historical(api_key)


def get_mnq_symbol_for_date(date: datetime) -> str:
    """
    Get the correct MNQ futures contract symbol for a given date.

    Micro E-mini NASDAQ-100 (MNQ) quarterly contracts:
    - March (H): expires mid-March
    - June (M): expires mid-June
    - September (U): expires mid-September
    - December (Z): expires mid-December

    Args:
        date: The date to get the contract for

    Returns:
        Symbol like 'MNQH4' for March 2024, 'MNQM4' for June 2024, etc.
    """
    year_code = str(date.year)[-1]  # Last digit of year (2024 -> 4)

    # Contract month codes and their expiry months
    contracts = [
        ('H', 3),   # March
        ('M', 6),   # June
        ('U', 9),   # September
        ('Z', 12),  # December
    ]

    # Find the active contract for this date
    # Usually the contract for the next quarter
    for i, (code, month) in enumerate(contracts):
        if date.month <= month:
            return f"MNQ{code}{year_code}"

    # If we're past December, use next year's March contract
    next_year_code = str(date.year + 1)[-1]
    return f"MNQH{next_year_code}"


async def download_tick_data(
    symbol: str = "MNQ",
    start_date: str = None,
    end_date: str = None,
    days: int = 30,
    dataset: str = "GLBX.MDP3",  # CME Globex data
):
    """
    Download tick data from Databento.

    Args:
        symbol: Base symbol (e.g., 'MNQ')
        start_date: Start date in YYYY-MM-DD format (default: 30 days ago)
        end_date: End date in YYYY-MM-DD format (default: today)
        days: Number of days to download if start_date not specified
        dataset: Databento dataset name

    Returns:
        DataFrame with tick data
    """

    print("=" * 80)
    print("Databento Historical Tick Data Download")
    print("=" * 80)

    # Load Databento client
    client = load_databento_client()

    # Set date range
    if end_date is None:
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        start = end - timedelta(days=days)
    else:
        start = datetime.strptime(start_date, "%Y-%m-%d")

    print(f"\n[CONFIG]")
    print(f"  Symbol: {symbol}")
    print(f"  Dataset: {dataset}")
    print(f"  Start: {start.strftime('%Y-%m-%d')}")
    print(f"  End: {end.strftime('%Y-%m-%d')}")
    print(f"  Duration: {(end - start).days} days")

    # Use Databento continuous contract format: MNQ.c.0
    # Format: [ROOT].[ROLL_RULE].[RANK]
    # c = calendar roll (most common)
    # 0 = front month contract
    contract_symbol = f"{symbol}.c.0"
    print(f"  Contract: {contract_symbol} (continuous front month)")

    print(f"\n[*] Downloading tick data...")
    print(f"    This may take several minutes for large date ranges...")
    print(f"    Expected: ~{(end - start).days * 1.5:.0f}M ticks for {(end - start).days} days")

    # Start download
    download_start = datetime.now()
    print(f"\n[{download_start.strftime('%H:%M:%S')}] Initiating download request...")

    try:
        # Download trades (tick data)
        # Schema: trades = individual transaction records
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling Databento API...")
        print(f"    Downloading from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")

        data = client.timeseries.get_range(
            dataset=dataset,
            symbols=[contract_symbol],
            schema="trades",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            stype_in="continuous",  # Use continuous contracts
        )

        download_end = datetime.now()
        download_duration = (download_end - download_start).total_seconds()
        print(f"[{download_end.strftime('%H:%M:%S')}] Download completed in {download_duration:.1f}s!")

        # Convert to DataFrame
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Converting to DataFrame...")
        df = data.to_df()

        conversion_end = datetime.now()
        conversion_duration = (conversion_end - download_end).total_seconds()
        print(f"[{conversion_end.strftime('%H:%M:%S')}] Conversion completed in {conversion_duration:.1f}s!")

        if df.empty:
            print(f"\n[WARNING] No data returned!")
            print(f"[INFO] This could mean:")
            print(f"  1. The contract {contract_symbol} doesn't exist for these dates")
            print(f"  2. No trading occurred during this period")
            print(f"  3. Your Databento account doesn't have access to this data")
            print(f"\n[SUGGESTION] Try using the continuous contract or check available symbols")
            return None

        print(f"\n[SUCCESS] Downloaded {len(df):,} ticks")
        print(f"\n[DATA INFO]")
        print(f"  First tick: {df.index[0]}")
        print(f"  Last tick: {df.index[-1]}")
        print(f"  Columns: {list(df.columns)}")

        # Display sample
        print(f"\n[SAMPLE - First 5 ticks]")
        print(df.head())

        # Save raw data
        tick_dir = Path("tick_data/databento")
        tick_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = tick_dir / f"{symbol}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{timestamp}.parquet"

        df.to_parquet(filename)
        print(f"\n[SAVED] Raw tick data: {filename}")
        print(f"  File size: {filename.stat().st_size / 1024 / 1024:.2f} MB")

        return df

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"\n[DEBUG INFO]")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

        # Provide helpful suggestions
        print(f"\n[TROUBLESHOOTING]")
        print(f"  1. Check your Databento account has access to {dataset}")
        print(f"  2. Verify the symbol {contract_symbol} is correct")
        print(f"  3. Check your account tier (Standard required for historical data)")
        print(f"  4. Review available datasets: https://databento.com/datasets")

        return None


def convert_to_tick_format(df: pd.DataFrame) -> list:
    """
    Convert Databento DataFrame to our tick format for aggregation.
    Uses vectorized operations for much faster conversion.

    Args:
        df: Databento DataFrame with tick data

    Returns:
        List of tick dictionaries
    """

    print(f"\n[*] Converting {len(df):,} ticks to aggregator format (vectorized)...")

    # Map sides using vectorized operation
    side_map = {'B': 'buy', 'A': 'sell', 'N': 'unknown'}
    df['side_mapped'] = df['side'].map(side_map).fillna('unknown')

    # Convert to list of dicts using vectorized operations
    ticks = [
        {
            "timestamp": ts.isoformat(),
            "price": price,
            "size": size,
            "side": side,
            "symbol": symbol,
        }
        for ts, price, size, side, symbol in zip(
            df.index,
            df['price'],
            df['size'],
            df['side_mapped'],
            df['symbol']
        )
    ]

    print(f"[OK] Converted {len(ticks):,} ticks")
    print(f"\n[SAMPLE TICK]")
    print(f"  {ticks[0]}")

    return ticks


async def main():
    """Main function."""

    import sys

    print("""
    ================================================================
             Databento Historical Tick Data Downloader
    ================================================================

    This script will download tick-by-tick trade data from Databento
    for backtesting with tick-based strategies.

    Requirements:
    - Databento API key in .env file
    - Standard account or higher

    """)

    # Configuration
    symbol = "MNQ"  # Micro E-mini NASDAQ-100

    # Allow days parameter from command line
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    else:
        days = 30  # Default: 1 month

    months = days / 30
    print(f"Configuration:")
    print(f"  - Symbol: {symbol} (Micro E-mini NASDAQ-100)")
    print(f"  - Period: Last {days} days (~{months:.1f} months)")
    print(f"  - Dataset: GLBX.MDP3 (CME Globex)")
    print(f"  - Output directory: tick_data/databento/")
    print()

    # Track total time
    total_start = datetime.now()
    print(f"\n[{total_start.strftime('%H:%M:%S')}] Starting full download and aggregation process...")

    # Download data
    df = await download_tick_data(symbol=symbol, days=days)

    if df is not None and not df.empty:
        # Convert to our format
        conversion_start = datetime.now()
        print(f"\n[{conversion_start.strftime('%H:%M:%S')}] Converting to tick format...")
        ticks = convert_to_tick_format(df)

        # Save in our format for backtesting
        from src.tick_aggregator import TickAggregator, Tick

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting aggregation into tick bars...")

        # Test multiple tick intervals
        for interval in [100, 250, 500, 1000]:
            interval_start = datetime.now()
            print(f"\n[{interval_start.strftime('%H:%M:%S')}] Creating {interval}-tick bars...")

            # Convert to Tick objects
            print(f"    Converting {len(ticks):,} ticks to Tick objects...")
            tick_objects = []
            for i, t in enumerate(ticks):
                if i % 500000 == 0 and i > 0:
                    print(f"    Progress: {i:,}/{len(ticks):,} ticks ({i/len(ticks)*100:.1f}%)")

                tick_obj = Tick(
                    instrument=t["symbol"],
                    price=Decimal(str(t["price"])),
                    size=t["size"],
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                    side=t["side"]
                )
                tick_objects.append(tick_obj)

            # Aggregate
            print(f"    Aggregating to {interval}-tick bars...")
            agg_start = datetime.now()
            bars_df = TickAggregator.aggregate_ticks(tick_objects, interval)
            agg_duration = (datetime.now() - agg_start).total_seconds()

            # Save
            bars_dir = Path("tick_data/databento/bars")
            bars_dir.mkdir(parents=True, exist_ok=True)

            bars_file = bars_dir / f"MNQ_{interval}t_bars.csv"
            bars_df.to_csv(bars_file)

            interval_duration = (datetime.now() - interval_start).total_seconds()
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] Completed in {interval_duration:.1f}s")

            print(f"    [SAVED] {len(bars_df)} bars -> {bars_file}")
            print(f"    Sample: O={bars_df.iloc[0]['open']:.2f}, "
                  f"H={bars_df.iloc[0]['high']:.2f}, "
                  f"L={bars_df.iloc[0]['low']:.2f}, "
                  f"C={bars_df.iloc[0]['close']:.2f}, "
                  f"V={bars_df.iloc[0]['volume']}")

        # Calculate total time
        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds()
        total_minutes = total_duration / 60

        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"\n[{total_end.strftime('%H:%M:%S')}] Total time: {total_duration:.1f}s ({total_minutes:.1f} minutes)")
        print(f"\nData saved to: tick_data/databento/")
        print(f"  - Raw tick data: tick_data/databento/*.parquet")
        print(f"  - Aggregated bars: tick_data/databento/bars/*.csv")
        print(f"\nNext steps:")
        print(f"1. Run backtest: python quick_tick_backtest.py")
        print(f"2. Compare intervals: 250t vs 500t vs 1000t")
        print(f"3. Optimize strategy parameters for tick data")
        print("=" * 80)

    else:
        print("\n[FAILED] No data downloaded. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
