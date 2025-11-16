"""
Generate 100-tick bars from existing raw tick data.

This script loads the existing raw tick parquet file and aggregates it
to 100-tick bars, which is faster than re-downloading from Databento.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from src.tick_aggregator import TickAggregator, Tick


def load_latest_tick_data():
    """Load the most recent raw tick data parquet file."""
    tick_dir = Path("tick_data/databento")

    # Find all parquet files
    parquet_files = list(tick_dir.glob("MNQ_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError("No raw tick data files found. Run download_databento_ticks.py first.")

    # Sort by file modification time and get the latest
    latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)

    print(f"[*] Loading raw tick data from: {latest_file.name}")
    print(f"    File size: {latest_file.stat().st_size / 1024 / 1024:.2f} MB")

    df = pd.read_parquet(latest_file)

    print(f"[OK] Loaded {len(df):,} raw ticks")
    print(f"    Date range: {df.index[0]} to {df.index[-1]}")

    return df


def convert_to_tick_format(df: pd.DataFrame) -> list:
    """
    Convert Databento DataFrame to tick format for aggregation.

    Args:
        df: Databento DataFrame with tick data

    Returns:
        List of tick dictionaries
    """
    print(f"\n[*] Converting {len(df):,} ticks to aggregator format...")

    # Map sides using vectorized operation
    side_map = {'B': 'buy', 'A': 'sell', 'N': 'unknown'}
    df['side_mapped'] = df['side'].map(side_map).fillna('unknown')

    # Convert to list of dicts
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

    return ticks


def generate_100t_bars(ticks: list):
    """
    Generate 100-tick bars from raw ticks.

    Args:
        ticks: List of tick dictionaries

    Returns:
        DataFrame with 100-tick bars
    """
    print(f"\n[*] Generating 100-tick bars...")
    print(f"    Converting {len(ticks):,} ticks to Tick objects...")

    # Convert to Tick objects
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

    # Aggregate to 100-tick bars
    print(f"\n[*] Aggregating to 100-tick bars...")
    agg_start = datetime.now()

    bars_df = TickAggregator.aggregate_ticks(tick_objects, ticks_per_bar=100)

    agg_duration = (datetime.now() - agg_start).total_seconds()
    print(f"[OK] Aggregation completed in {agg_duration:.1f}s")

    return bars_df


def main():
    print("=" * 80)
    print("GENERATE 100-TICK BARS FROM EXISTING DATA")
    print("=" * 80)
    print("\nThis will create 100t bars from the existing raw tick data,")
    print("which is much faster than re-downloading from Databento.")
    print("=" * 80)

    start_time = datetime.now()

    # Load existing raw tick data
    df = load_latest_tick_data()

    # Convert to tick format
    ticks = convert_to_tick_format(df)

    # Generate 100t bars
    bars_df = generate_100t_bars(ticks)

    # Save to CSV
    bars_dir = Path("tick_data/databento/bars")
    bars_dir.mkdir(parents=True, exist_ok=True)

    bars_file = bars_dir / "MNQ_100t_bars.csv"
    bars_df.to_csv(bars_file)

    # Calculate total time
    total_duration = (datetime.now() - start_time).total_seconds()
    total_minutes = total_duration / 60

    print("\n" + "=" * 80)
    print("100T BARS GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n[OK] Created {len(bars_df):,} bars")
    print(f"[OK] Saved to: {bars_file}")
    print(f"[OK] File size: {bars_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n[PERFORMANCE]")
    print(f"  Total time: {total_duration:.1f}s ({total_minutes:.1f} minutes)")
    print(f"  Average: {len(bars_df) / total_duration:.0f} bars/second")

    print(f"\n[SAMPLE BAR]")
    print(f"  Open:   {bars_df.iloc[0]['open']:.2f}")
    print(f"  High:   {bars_df.iloc[0]['high']:.2f}")
    print(f"  Low:    {bars_df.iloc[0]['low']:.2f}")
    print(f"  Close:  {bars_df.iloc[0]['close']:.2f}")
    print(f"  Volume: {bars_df.iloc[0]['volume']}")
    print(f"  Delta:  {bars_df.iloc[0]['volume_delta']}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Run backtest with 100t bars")
    print("2. Compare 100t vs 250t vs 500t vs 1000t performance")
    print("3. Identify optimal tick interval for strategy")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
