"""
Create Minute Bars from Tick Data

Converts tick bars to standard minute timeframes:
- M1 (1 minute)
- M2 (2 minutes)
- M3 (3 minutes)
- M4 (4 minutes)
- M5 (5 minutes)
- M10 (10 minutes)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_minute_bars(tick_data_path, output_dir, timeframes=[1, 2, 3, 4, 5, 10]):
    """
    Create minute bars from tick bar data.

    Args:
        tick_data_path: Path to tick bar CSV file
        output_dir: Directory to save minute bar files
        timeframes: List of minute timeframes to create
    """
    print(f"[*] Loading tick data from {tick_data_path}...")
    df = pd.read_csv(tick_data_path, index_col=0, parse_dates=True)
    print(f"[OK] Loaded {len(df)} tick bars")
    print(f"    Date range: {df.index[0]} to {df.index[-1]}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for minutes in timeframes:
        print(f"\n[*] Creating {minutes}-minute bars...")

        # Resample to minute bars
        resampled = df.resample(f'{minutes}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'tick_count': 'sum',
            'buy_volume': 'sum',
            'sell_volume': 'sum',
            'volume_delta': 'sum',  # Sum of deltas
        })

        # Remove rows with NaN (incomplete bars)
        resampled = resampled.dropna()

        # Recalculate volume_delta from summed buy/sell volumes
        resampled['volume_delta'] = resampled['buy_volume'] - resampled['sell_volume']

        # Save to CSV
        output_path = output_dir / f"MNQ_M{minutes}_bars.csv"
        resampled.to_csv(output_path)

        print(f"[OK] Created {len(resampled)} M{minutes} bars")
        print(f"     Saved to: {output_path}")

    print(f"\n[COMPLETE] All minute bars created successfully!")


def main():
    print("=" * 80)
    print("CREATE MINUTE BARS FROM TICK DATA")
    print("=" * 80)
    print("\nConverting tick bars to minute timeframes...")
    print("Timeframes: M1, M2, M3, M4, M5, M10")
    print("=" * 80)

    # Use 250-tick bars (most granular available)
    tick_data_path = Path("tick_data/databento/bars/MNQ_250t_bars.csv")
    output_dir = Path("tick_data/databento/minute_bars")

    # Create minute bars
    create_minute_bars(
        tick_data_path=tick_data_path,
        output_dir=output_dir,
        timeframes=[1, 2, 3, 4, 5, 10]
    )

    print("\n" + "=" * 80)
    print("MINUTE BARS READY FOR BACKTESTING")
    print("=" * 80)


if __name__ == "__main__":
    main()
