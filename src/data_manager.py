"""
Data Manager Module
Handles historical data retrieval, real-time data streaming, and indicator calculations
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from project_x_py import ProjectX, TradingSuite, EventType

from src.config import config


class DataManager:
    """
    Manages all data operations including:
    - Historical data download
    - Real-time data streaming
    - Technical indicator calculations
    - Data caching
    """

    def __init__(self, client: Optional[ProjectX] = None):
        """
        Initialize Data Manager

        Args:
            client: ProjectX client instance (optional, will create if not provided)
        """
        self.client = client
        self.suites: Dict[str, TradingSuite] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the data manager and authenticate"""
        if self._initialized:
            return

        # Create client if not provided
        if self.client is None:
            # Get the async context manager
            cm = ProjectX.from_env()
            # Enter the context manager
            self.client = await cm.__aenter__()
            # Store the context manager for cleanup
            self._cm = cm
            await self.client.authenticate()

        print(f"[OK] Data Manager initialized")
        self._initialized = True

    async def setup_instruments(self, symbols: List[str]):
        """
        Setup trading suites for multiple instruments

        Args:
            symbols: List of instrument symbols (e.g., ['MNQ', 'MGC', 'MYM'])
        """
        print(f"\n[*] Setting up instruments: {', '.join(symbols)}")

        for symbol in symbols:
            try:
                # Create TradingSuite for each instrument
                suite = await TradingSuite.create(
                    symbol,
                    timeframes=config.strategy.timeframes
                )

                self.suites[symbol] = suite
                print(f"  [OK] {symbol} suite created")

                # Setup event handlers for real-time data
                await suite.on(EventType.TICK, self._on_tick_callback(symbol))
                await suite.on(EventType.NEW_BAR, self._on_new_bar_callback(symbol))

            except Exception as e:
                print(f"  [ERROR] Error setting up {symbol}: {e}")

    def _on_tick_callback(self, symbol: str):
        """Create tick callback for a specific symbol"""
        async def callback(tick):
            self.current_prices[symbol] = tick.price
        return callback

    def _on_new_bar_callback(self, symbol: str):
        """Create new bar callback for a specific symbol"""
        async def callback(bar, timeframe):
            # Update historical data with new bar
            if symbol in self.historical_data:
                new_row = pd.DataFrame([{
                    'timestamp': bar.t,
                    'open': bar.o,
                    'high': bar.h,
                    'low': bar.l,
                    'close': bar.c,
                    'volume': bar.v
                }])
                self.historical_data[symbol] = pd.concat([
                    self.historical_data[symbol],
                    new_row
                ], ignore_index=True)

        return callback

    async def download_historical_data(
        self,
        symbol: str,
        days: int = 90,
        interval: int = 5
    ) -> pd.DataFrame:
        """
        Download historical data for an instrument

        Args:
            symbol: Instrument symbol (e.g., 'MNQ')
            days: Number of days to download
            interval: Bar interval in minutes

        Returns:
            DataFrame with OHLCV data
        """
        print(f"\n[*] Downloading {days} days of {interval}min data for {symbol}...")

        try:
            # Get historical bars using client (returns Polars DataFrame)
            polars_df = await self.client.get_bars(
                symbol=symbol,
                days=days,
                interval=interval
            )

            # Convert Polars DataFrame to Pandas DataFrame
            df = polars_df.to_pandas()

            # Make sure we have the right columns and timestamp index
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            # Store in cache
            self.historical_data[symbol] = df

            print(f"  [OK] Downloaded {len(df)} bars for {symbol}")
            print(f"  [*] Date range: {df.index[0]} to {df.index[-1]}")

            return df

        except Exception as e:
            print(f"  [ERROR] Error downloading data for {symbol}: {e}")
            raise

    async def download_all_historical_data(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for all configured instruments

        Args:
            days: Number of days to download

        Returns:
            Dictionary of DataFrames, keyed by symbol
        """
        # Parse interval from timeframe string (e.g., '5min' -> 5)
        interval_str = config.strategy.timeframe
        interval = int(''.join(filter(str.isdigit, interval_str)))

        for symbol in config.instruments.symbols:
            await self.download_historical_data(symbol, days, interval)

        return self.historical_data

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators on OHLCV data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # EMA calculations
        df['ema_fast'] = df['close'].ewm(span=config.strategy.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=config.strategy.ema_slow, adjust=False).mean()

        # RSI calculation
        df['rsi'] = self._calculate_rsi(df['close'], config.strategy.rsi_period)

        # ATR calculation
        df['atr'] = self._calculate_atr(df, config.strategy.atr_period)

        # Trading signals
        df['ema_bullish'] = df['ema_fast'] > df['ema_slow']
        df['ema_bearish'] = df['ema_fast'] < df['ema_slow']

        # Crossover detection
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range

        Args:
            df: DataFrame with OHLC data
            period: ATR period

        Returns:
            ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def get_latest_data(self, symbol: str, bars: int = 100) -> pd.DataFrame:
        """
        Get latest N bars with indicators

        Args:
            symbol: Instrument symbol
            bars: Number of bars to return

        Returns:
            DataFrame with latest data and indicators
        """
        if symbol not in self.historical_data:
            raise ValueError(f"No historical data for {symbol}. Run download_historical_data first.")

        df = self.historical_data[symbol].tail(bars).copy()
        df = self.calculate_indicators(df)

        return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for an instrument

        Args:
            symbol: Instrument symbol

        Returns:
            Current price or None if not available
        """
        return self.current_prices.get(symbol)

    async def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent completed bar

        Args:
            symbol: Instrument symbol

        Returns:
            Dictionary with OHLCV data
        """
        if symbol not in self.historical_data or len(self.historical_data[symbol]) == 0:
            return None

        latest = self.historical_data[symbol].iloc[-1]
        return {
            'timestamp': latest.name,
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'close': latest['close'],
            'volume': latest['volume']
        }

    def get_signal(self, symbol: str) -> Tuple[Optional[str], Dict]:
        """
        Get current trading signal for an instrument

        Args:
            symbol: Instrument symbol

        Returns:
            Tuple of (signal, context)
            signal: 'BUY', 'SELL', or None
            context: Dictionary with signal details
        """
        try:
            df = self.get_latest_data(symbol, bars=50)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            context = {
                'price': latest['close'],
                'ema_fast': latest['ema_fast'],
                'ema_slow': latest['ema_slow'],
                'rsi': latest['rsi'],
                'atr': latest['atr'],
                'timestamp': latest.name
            }

            # Check for valid signals
            signal = None

            # Bullish crossover + RSI not overbought
            if latest['ema_cross_up'] and latest['rsi'] < config.strategy.rsi_overbought:
                if latest['atr'] > config.strategy.min_atr_threshold:
                    signal = 'BUY'
                    context['reason'] = 'EMA bullish crossover + RSI confirmation'

            # Bearish crossover + RSI not oversold
            elif latest['ema_cross_down'] and latest['rsi'] > config.strategy.rsi_oversold:
                if latest['atr'] > config.strategy.min_atr_threshold:
                    signal = 'SELL'
                    context['reason'] = 'EMA bearish crossover + RSI confirmation'

            return signal, context

        except Exception as e:
            print(f"Error getting signal for {symbol}: {e}")
            return None, {}

    async def cleanup(self):
        """Cleanup resources"""
        print("\n[*] Cleaning up Data Manager...")

        # Disconnect all suites
        for symbol, suite in self.suites.items():
            try:
                await suite.disconnect()
                print(f"  [OK] Disconnected {symbol} suite")
            except Exception as e:
                print(f"  [WARN] Error disconnecting {symbol}: {e}")

        # Close client if we created it
        if hasattr(self, '_cm') and self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
                print(f"  [OK] Closed ProjectX client")
            except Exception as e:
                print(f"  [WARN] Error closing client: {e}")

        self._initialized = False


# Convenience function for testing
async def test_data_manager():
    """Test the data manager functionality"""
    print("=" * 60)
    print("TESTING DATA MANAGER")
    print("=" * 60)

    dm = DataManager()

    try:
        # Initialize
        await dm.initialize()

        # Download historical data for one symbol
        symbol = 'MNQ'
        df = await dm.download_historical_data(symbol, days=30, interval=5)

        print(f"\n📊 Data summary for {symbol}:")
        print(df.tail())

        # Calculate indicators
        print(f"\n📈 Calculating indicators...")
        df_with_indicators = dm.calculate_indicators(df)
        print(df_with_indicators[['close', 'ema_fast', 'ema_slow', 'rsi', 'atr']].tail())

        # Get signal
        print(f"\n🎯 Current signal:")
        signal, context = dm.get_signal(symbol)
        print(f"  Signal: {signal}")
        print(f"  Context: {context}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await dm.cleanup()


if __name__ == "__main__":
    asyncio.run(test_data_manager())
