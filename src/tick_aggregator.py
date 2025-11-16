"""
Tick Aggregator - Converts individual tick data into tick-based bars.

This module aggregates individual ticks into OHLCV bars based on tick count,
creating true tick charts (e.g., 100-tick, 250-tick, 500-tick, 1000-tick bars).
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Dict
import pandas as pd
from project_x_py import TradingSuite


@dataclass
class Tick:
    """Individual tick data."""
    instrument: str
    price: Decimal
    size: int
    timestamp: datetime
    side: str  # "buy", "sell", or "unknown"


@dataclass
class TickBar:
    """Aggregated tick bar (OHLCV)."""
    timestamp: datetime  # Timestamp of bar close (last tick)
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int  # Number of ticks in this bar
    buy_volume: int  # Volume from buy-side ticks
    sell_volume: int  # Volume from sell-side ticks


class TickAggregator:
    """
    Aggregates individual ticks into tick-based bars.

    Supports configurable tick intervals: 100t, 250t, 500t, 1000t, etc.
    """

    def __init__(self, ticks_per_bar: int = 500):
        """
        Initialize tick aggregator.

        Args:
            ticks_per_bar: Number of ticks to aggregate into one bar
        """
        self.ticks_per_bar = ticks_per_bar
        self.current_ticks: List[Tick] = []
        self.completed_bars: List[TickBar] = []

    def add_tick(self, tick: Tick) -> TickBar | None:
        """
        Add a tick to the aggregator.

        Returns:
            TickBar if a new bar was completed, None otherwise
        """
        self.current_ticks.append(tick)

        if len(self.current_ticks) >= self.ticks_per_bar:
            bar = self._create_bar(self.current_ticks)
            self.completed_bars.append(bar)
            self.current_ticks = []
            return bar

        return None

    def _create_bar(self, ticks: List[Tick]) -> TickBar:
        """
        Create a tick bar from a list of ticks.

        Args:
            ticks: List of ticks to aggregate

        Returns:
            TickBar with OHLCV data
        """
        if not ticks:
            raise ValueError("Cannot create bar from empty tick list")

        prices = [float(t.price) for t in ticks]

        # Calculate OHLC
        open_price = prices[0]
        high_price = max(prices)
        low_price = min(prices)
        close_price = prices[-1]

        # Calculate volume metrics
        total_volume = sum(t.size for t in ticks)
        buy_volume = sum(t.size for t in ticks if t.side == "buy")
        sell_volume = sum(t.size for t in ticks if t.side == "sell")

        # Use timestamp of last tick (bar close)
        timestamp = ticks[-1].timestamp

        return TickBar(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            tick_count=len(ticks),
            buy_volume=buy_volume,
            sell_volume=sell_volume
        )

    def flush(self) -> TickBar | None:
        """
        Flush remaining ticks into a partial bar.

        Returns:
            TickBar if there are remaining ticks, None otherwise
        """
        if self.current_ticks:
            bar = self._create_bar(self.current_ticks)
            self.completed_bars.append(bar)
            self.current_ticks = []
            return bar
        return None

    def get_bars_dataframe(self) -> pd.DataFrame:
        """
        Convert completed bars to pandas DataFrame.

        Returns:
            DataFrame with OHLCV data compatible with backtester
        """
        if not self.completed_bars:
            return pd.DataFrame()

        data = {
            'timestamp': [bar.timestamp for bar in self.completed_bars],
            'open': [bar.open for bar in self.completed_bars],
            'high': [bar.high for bar in self.completed_bars],
            'low': [bar.low for bar in self.completed_bars],
            'close': [bar.close for bar in self.completed_bars],
            'volume': [bar.volume for bar in self.completed_bars],
            'tick_count': [bar.tick_count for bar in self.completed_bars],
            'buy_volume': [bar.buy_volume for bar in self.completed_bars],
            'sell_volume': [bar.sell_volume for bar in self.completed_bars],
        }

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        # Calculate volume delta (buy - sell pressure)
        df['volume_delta'] = df['buy_volume'] - df['sell_volume']

        return df

    @staticmethod
    def aggregate_ticks(ticks: List[Tick], ticks_per_bar: int) -> pd.DataFrame:
        """
        Static method to aggregate a list of ticks into bars.

        Args:
            ticks: List of Tick objects
            ticks_per_bar: Number of ticks per bar

        Returns:
            DataFrame with OHLCV data
        """
        aggregator = TickAggregator(ticks_per_bar)

        for tick in ticks:
            aggregator.add_tick(tick)

        # Flush remaining ticks
        aggregator.flush()

        return aggregator.get_bars_dataframe()


class TickDataManager:
    """
    Manages tick data download and aggregation for backtesting.

    Uses TradingSuite API to fetch individual ticks and aggregates them
    into tick-based bars.
    """

    def __init__(self, symbols: List[str], ticks_per_bar: int = 500):
        """
        Initialize tick data manager.

        Args:
            symbols: List of instrument symbols
            ticks_per_bar: Number of ticks per bar
        """
        self.symbols = symbols
        self.ticks_per_bar = ticks_per_bar
        self.tick_data: Dict[str, pd.DataFrame] = {}

    async def download_tick_data(
        self,
        symbol: str,
        tick_count: int = 100000
    ) -> pd.DataFrame:
        """
        Download tick data and aggregate into bars.

        Args:
            symbol: Instrument symbol
            tick_count: Number of recent ticks to fetch

        Returns:
            DataFrame with aggregated tick bars
        """
        print(f"\n[*] Downloading {tick_count} ticks for {symbol}...")

        # Create TradingSuite and subscribe to tick data
        suite = await TradingSuite.create([symbol])

        try:
            instrument_data = suite[symbol].data

            # Subscribe to trades (ticks)
            await instrument_data.subscribe_to_trades()

            # Get recent ticks
            print(f"[*] Fetching recent ticks...")
            ticks = await instrument_data.get_recent_ticks(count=tick_count)

            print(f"[*] Retrieved {len(ticks)} ticks")

            if not ticks:
                print(f"[!] No tick data available for {symbol}")
                return pd.DataFrame()

            # Convert to Tick objects
            tick_objects = [
                Tick(
                    instrument=tick.instrument,
                    price=tick.price,
                    size=tick.size,
                    timestamp=tick.timestamp,
                    side=tick.side
                )
                for tick in ticks
            ]

            # Aggregate into bars
            print(f"[*] Aggregating {len(tick_objects)} ticks into {self.ticks_per_bar}-tick bars...")
            df = TickAggregator.aggregate_ticks(tick_objects, self.ticks_per_bar)

            print(f"[*] Created {len(df)} bars from {len(tick_objects)} ticks")
            print(f"[*] Time range: {df.index[0]} to {df.index[-1]}")

            # Cache the data
            self.tick_data[symbol] = df

            return df

        finally:
            await suite.disconnect()

    async def download_all_symbols(self, tick_count: int = 100000) -> Dict[str, pd.DataFrame]:
        """
        Download tick data for all configured symbols.

        Args:
            tick_count: Number of recent ticks to fetch per symbol

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        for symbol in self.symbols:
            await self.download_tick_data(symbol, tick_count)

        return self.tick_data

    def get_data(self, symbol: str) -> pd.DataFrame:
        """
        Get cached tick data for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            DataFrame with tick bars
        """
        return self.tick_data.get(symbol, pd.DataFrame())


def compare_tick_intervals(
    ticks: List[Tick],
    intervals: List[int] = [100, 250, 500, 1000]
) -> Dict[int, pd.DataFrame]:
    """
    Compare different tick aggregation intervals.

    Args:
        ticks: List of Tick objects
        intervals: List of tick intervals to test

    Returns:
        Dictionary mapping interval to resulting DataFrame
    """
    results = {}

    print("\n" + "=" * 80)
    print("Tick Interval Comparison")
    print("=" * 80)

    for interval in intervals:
        df = TickAggregator.aggregate_ticks(ticks, interval)
        results[interval] = df

        print(f"\n{interval}-tick bars:")
        print(f"  - Total bars: {len(df)}")
        print(f"  - Avg volume/bar: {df['volume'].mean():.0f}")
        print(f"  - Avg buy volume: {df['buy_volume'].mean():.0f}")
        print(f"  - Avg sell volume: {df['sell_volume'].mean():.0f}")

        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            print(f"  - Avg time/bar: {time_diffs.mean()}")
            print(f"  - Min time/bar: {time_diffs.min()}")
            print(f"  - Max time/bar: {time_diffs.max()}")

    print("\n" + "=" * 80)

    return results
