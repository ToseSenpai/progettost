"""
Volumetric Trading Strategy
Uses volume-based indicators for high-accuracy trading on futures
Indicators: VWAP, Money Flow Index (MFI), Volume Delta, SuperTrend
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

from src.config import config
from src.strategy import TradeSignal  # For backtester compatibility


@dataclass
class VolumetricSignal:
    """Trading signal from volumetric strategy"""
    signal: str  # 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'HOLD'
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0  # 0-1 scale


class VolumetricStrategy:
    """
    Volumetric Trading Strategy

    Combines volume-based indicators for institutional-grade entries:
    - VWAP: Volume Weighted Average Price (trend + dynamic S/R)
    - MFI: Money Flow Index (RSI with volume)
    - Volume Delta: Buy/Sell pressure
    - SuperTrend: ATR-based trend filter
    """

    def __init__(self):
        """Initialize strategy with config parameters"""
        self.name = "Volumetric Strategy"

        # VWAP settings
        self.vwap_session = config.strategy.vwap_session if hasattr(config.strategy, 'vwap_session') else 'D'

        # MFI settings (RSI with volume)
        self.mfi_period = config.strategy.mfi_period if hasattr(config.strategy, 'mfi_period') else 14
        self.mfi_overbought = config.strategy.mfi_overbought if hasattr(config.strategy, 'mfi_overbought') else 80
        self.mfi_oversold = config.strategy.mfi_oversold if hasattr(config.strategy, 'mfi_oversold') else 20

        # SuperTrend settings
        self.st_period = config.strategy.st_period if hasattr(config.strategy, 'st_period') else 10
        self.st_multiplier = config.strategy.st_multiplier if hasattr(config.strategy, 'st_multiplier') else 3.0

        # Volume Delta settings
        self.vd_threshold = config.strategy.vd_threshold if hasattr(config.strategy, 'vd_threshold') else 0.3

        # Risk management
        self.stop_multiplier = config.strategy.atr_stop_multiplier
        self.target_multiplier = config.strategy.atr_target_multiplier
        self.atr_period = config.strategy.atr_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all volumetric indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # 1. VWAP (Volume Weighted Average Price)
        df['vwap'] = self._calculate_vwap(df)

        # 2. Money Flow Index (MFI) - RSI with volume
        df['mfi'] = self._calculate_mfi(df)

        # 3. Volume Delta (Buy vs Sell pressure)
        df['volume_delta'] = self._calculate_volume_delta(df)
        df['cumulative_delta'] = df['volume_delta'].cumsum()

        # 4. SuperTrend (ATR-based trend filter)
        df = self._calculate_supertrend(df)

        # 5. ATR for stops/targets
        df['atr'] = self._calculate_atr(df, self.atr_period)

        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price)

        VWAP = Sum(Price * Volume) / Sum(Volume)
        Resets each session (daily by default)
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # For intraday, reset VWAP at start of each day
        if self.vwap_session == 'D':
            # Get timestamp from either column or index
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp'])
                df_date = timestamps.dt.date
            else:
                # Index is already DatetimeIndex
                timestamps = df.index
                if isinstance(timestamps, pd.DatetimeIndex):
                    df_date = timestamps.date
                else:
                    df_date = pd.to_datetime(timestamps).dt.date
            vwap = []

            # Get unique dates (handle both Series and numpy array)
            if hasattr(df_date, 'unique'):
                unique_dates = df_date.unique()
            else:
                unique_dates = np.unique(df_date)

            for date in unique_dates:
                mask = df_date == date
                tp = typical_price[mask]
                vol = df.loc[mask, 'volume']

                # Cumulative VWAP for this session
                cum_vol = vol.cumsum()
                cum_tp_vol = (tp * vol).cumsum()
                session_vwap = cum_tp_vol / cum_vol

                vwap.extend(session_vwap.values)

            return pd.Series(vwap, index=df.index)
        else:
            # Simple rolling VWAP
            cum_vol = df['volume'].cumsum()
            cum_tp_vol = (typical_price * df['volume']).cumsum()
            return cum_tp_vol / cum_vol

    def _calculate_mfi(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Money Flow Index (MFI)

        MFI is like RSI but incorporates volume:
        1. Typical Price = (H + L + C) / 3
        2. Money Flow = Typical Price × Volume
        3. Positive/Negative Money Flow based on price direction
        4. Money Ratio = Positive MF / Negative MF
        5. MFI = 100 - (100 / (1 + Money Ratio))
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Identify up/down periods
        tp_change = typical_price.diff()

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        positive_flow[tp_change > 0] = money_flow[tp_change > 0]
        negative_flow[tp_change < 0] = money_flow[tp_change < 0]

        # Calculate MFI using rolling sums
        positive_mf = positive_flow.rolling(window=self.mfi_period).sum()
        negative_mf = negative_flow.rolling(window=self.mfi_period).sum()

        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    def _calculate_volume_delta(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Delta (Buy Volume - Sell Volume estimate)

        Since we don't have tick data, we estimate:
        - If close > open: bullish bar, assign more volume to buys
        - If close < open: bearish bar, assign more volume to sells
        - Weight by bar range position
        """
        volume_delta = pd.Series(0.0, index=df.index)

        for i in df.index:
            close = df.loc[i, 'close']
            open_price = df.loc[i, 'open']
            high = df.loc[i, 'high']
            low = df.loc[i, 'low']
            volume = df.loc[i, 'volume']

            if high == low:  # Avoid division by zero
                volume_delta[i] = 0
                continue

            # Calculate close position in range (0 to 1)
            close_position = (close - low) / (high - low)

            # Delta = volume weighted by close position
            # close_position > 0.5 = more buying
            # close_position < 0.5 = more selling
            delta = (2 * close_position - 1) * volume
            volume_delta[i] = delta

        return volume_delta

    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator

        SuperTrend = Moving support/resistance based on ATR
        - Uses ATR to set dynamic distance from price
        - Trend changes when price crosses bands
        """
        df = df.copy()

        # Calculate ATR
        atr = self._calculate_atr(df, self.st_period)

        # Calculate basic upper and lower bands
        hl_avg = (df['high'] + df['low']) / 2

        basic_ub = hl_avg + (self.st_multiplier * atr)
        basic_lb = hl_avg - (self.st_multiplier * atr)

        # Initialize - use NumPy arrays for speed
        basic_ub_arr = basic_ub.values
        basic_lb_arr = basic_lb.values
        close_arr = df['close'].values

        final_ub_arr = np.zeros(len(df))
        final_lb_arr = np.zeros(len(df))
        supertrend_arr = np.zeros(len(df))

        for i in range(len(df)):
            if i == 0:
                final_ub_arr[i] = basic_ub_arr[i]
                final_lb_arr[i] = basic_lb_arr[i]
            else:
                # Final Upper Band
                if basic_ub_arr[i] < final_ub_arr[i-1] or close_arr[i-1] > final_ub_arr[i-1]:
                    final_ub_arr[i] = basic_ub_arr[i]
                else:
                    final_ub_arr[i] = final_ub_arr[i-1]

                # Final Lower Band
                if basic_lb_arr[i] > final_lb_arr[i-1] or close_arr[i-1] < final_lb_arr[i-1]:
                    final_lb_arr[i] = basic_lb_arr[i]
                else:
                    final_lb_arr[i] = final_lb_arr[i-1]

            # SuperTrend
            if i == 0:
                supertrend_arr[i] = final_ub_arr[i]
            else:
                if supertrend_arr[i-1] == final_ub_arr[i-1]:
                    if close_arr[i] <= final_ub_arr[i]:
                        supertrend_arr[i] = final_ub_arr[i]
                    else:
                        supertrend_arr[i] = final_lb_arr[i]
                else:
                    if close_arr[i] >= final_lb_arr[i]:
                        supertrend_arr[i] = final_lb_arr[i]
                    else:
                        supertrend_arr[i] = final_ub_arr[i]

        # Convert back to Series
        supertrend = pd.Series(supertrend_arr, index=df.index)

        df['supertrend'] = supertrend
        df['st_direction'] = np.where(df['close'] > df['supertrend'], 1, -1)

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def generate_signal(self, df: pd.DataFrame, current_position: Optional[str] = None) -> VolumetricSignal:
        """
        Generate trading signal based on volumetric indicators

        Entry Logic (ALL conditions must be met):
        LONG:
        - Price ABOVE VWAP (institutional support)
        - MFI < 80 (not overbought)
        - Volume Delta positive and increasing
        - SuperTrend = LONG (close > supertrend)
        - Confluence score >= 0.7

        SHORT:
        - Price BELOW VWAP (institutional resistance)
        - MFI > 20 (not oversold)
        - Volume Delta negative and decreasing
        - SuperTrend = SHORT (close < supertrend)
        - Confluence score >= 0.7

        Args:
            df: DataFrame with calculated indicators
            current_position: Current position ('LONG', 'SHORT', or None)

        Returns:
            VolumetricSignal with trading decision
        """
        if len(df) < max(self.mfi_period, self.st_period, self.atr_period) + 1:
            return VolumetricSignal('HOLD', 0.0, reason="Insufficient data")

        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = latest['close']
        vwap = latest['vwap']
        mfi = latest['mfi']
        volume_delta = latest['volume_delta']
        cum_delta = latest['cumulative_delta']
        prev_cum_delta = prev['cumulative_delta']
        st_direction = latest['st_direction']
        supertrend = latest['supertrend']
        atr = latest['atr']

        # Check for NaN values
        if pd.isna([vwap, mfi, volume_delta, st_direction, atr]).any():
            return VolumetricSignal('HOLD', close, reason="Indicator calculation incomplete")

        # Calculate confluence score (0-1)
        confluence_long = 0.0
        confluence_short = 0.0
        reasons_long = []
        reasons_short = []

        # === LONG Conditions ===

        # 1. Price vs VWAP (25% weight)
        if close > vwap:
            confluence_long += 0.25
            reasons_long.append(f"Price above VWAP ({close:.2f} > {vwap:.2f})")

        # 2. MFI (25% weight)
        if mfi < self.mfi_overbought:
            confluence_long += 0.25 * (1 - mfi / 100)  # More weight when MFI is lower
            reasons_long.append(f"MFI not overbought ({mfi:.1f})")

        # 3. Volume Delta (25% weight)
        if volume_delta > 0 and cum_delta > prev_cum_delta:
            confluence_long += 0.25
            reasons_long.append(f"Positive volume delta ({volume_delta:.0f})")

        # 4. SuperTrend (25% weight)
        if st_direction > 0:
            confluence_long += 0.25
            reasons_long.append("SuperTrend LONG")

        # === SHORT Conditions ===

        # 1. Price vs VWAP (25% weight)
        if close < vwap:
            confluence_short += 0.25
            reasons_short.append(f"Price below VWAP ({close:.2f} < {vwap:.2f})")

        # 2. MFI (25% weight)
        if mfi > self.mfi_oversold:
            confluence_short += 0.25 * (mfi / 100)  # More weight when MFI is higher
            reasons_short.append(f"MFI not oversold ({mfi:.1f})")

        # 3. Volume Delta (25% weight)
        if volume_delta < 0 and cum_delta < prev_cum_delta:
            confluence_short += 0.25
            reasons_short.append(f"Negative volume delta ({volume_delta:.0f})")

        # 4. SuperTrend (25% weight)
        if st_direction < 0:
            confluence_short += 0.25
            reasons_short.append("SuperTrend SHORT")

        # === Exit Conditions ===

        if current_position == 'LONG':
            # Exit LONG if SuperTrend flips or price crosses below VWAP
            if st_direction < 0:
                return VolumetricSignal(
                    'EXIT_LONG',
                    close,
                    reason="SuperTrend flipped to SHORT",
                    confidence=1.0
                )
            if close < vwap and mfi > self.mfi_overbought:
                return VolumetricSignal(
                    'EXIT_LONG',
                    close,
                    reason="Price below VWAP + MFI overbought",
                    confidence=0.9
                )

        elif current_position == 'SHORT':
            # Exit SHORT if SuperTrend flips or price crosses above VWAP
            if st_direction > 0:
                return VolumetricSignal(
                    'EXIT_SHORT',
                    close,
                    reason="SuperTrend flipped to LONG",
                    confidence=1.0
                )
            if close > vwap and mfi < self.mfi_oversold:
                return VolumetricSignal(
                    'EXIT_SHORT',
                    close,
                    reason="Price above VWAP + MFI oversold",
                    confidence=0.9
                )

        # === Entry Decisions ===

        # Require minimum confluence of 0.75 (3 out of 4 conditions)
        MIN_CONFLUENCE = 0.75

        if confluence_long >= MIN_CONFLUENCE and current_position != 'LONG':
            stop_loss = close - (self.stop_multiplier * atr)
            take_profit = close + (self.target_multiplier * atr)

            return VolumetricSignal(
                'LONG',
                close,
                stop_loss,
                take_profit,
                reason=" | ".join(reasons_long),
                confidence=confluence_long
            )

        elif confluence_short >= MIN_CONFLUENCE and current_position != 'SHORT':
            stop_loss = close + (self.stop_multiplier * atr)
            take_profit = close - (self.target_multiplier * atr)

            return VolumetricSignal(
                'SHORT',
                close,
                stop_loss,
                take_profit,
                reason=" | ".join(reasons_short),
                confidence=confluence_short
            )

        # No signal
        return VolumetricSignal(
            'HOLD',
            close,
            reason=f"No confluence (LONG: {confluence_long:.2f}, SHORT: {confluence_short:.2f})",
            confidence=max(confluence_long, confluence_short)
        )

    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Analyze market data and generate trading signal
        Compatible interface with backtester

        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data

        Returns:
            TradeSignal if entry conditions met, None otherwise
        """
        # Calculate indicators
        df_with_indicators = self.calculate_indicators(df)

        # Get current position (if any)
        # Note: backtester handles position tracking separately
        vol_signal = self.generate_signal(df_with_indicators, current_position=None)

        # Convert VolumetricSignal to TradeSignal for backtester compatibility
        # Use the last timestamp from the dataframe (tz-aware)
        current_timestamp = df.index[-1]

        if vol_signal.signal == 'LONG':
            return TradeSignal(
                symbol=symbol,
                direction='LONG',
                entry_price=vol_signal.entry_price,
                stop_loss=vol_signal.stop_loss,
                take_profit=vol_signal.take_profit,
                size=1,  # Default size, backtester will override
                timestamp=current_timestamp,  # Use tz-aware timestamp from dataframe
                reason=vol_signal.reason,
                confidence=vol_signal.confidence,
                indicators={}  # Indicators dict, can be populated if needed
            )
        elif vol_signal.signal == 'SHORT':
            return TradeSignal(
                symbol=symbol,
                direction='SHORT',
                entry_price=vol_signal.entry_price,
                stop_loss=vol_signal.stop_loss,
                take_profit=vol_signal.take_profit,
                size=1,  # Default size, backtester will override
                timestamp=current_timestamp,  # Use tz-aware timestamp from dataframe
                reason=vol_signal.reason,
                confidence=vol_signal.confidence,
                indicators={}  # Indicators dict, can be populated if needed
            )
        else:
            return None

    def should_close_position(
        self,
        position_side: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        df: pd.DataFrame
    ) -> tuple[bool, str]:
        """
        Check if position should be closed
        Compatible interface with backtester

        Args:
            position_side: 'LONG' or 'SHORT'
            entry_price: Entry price of position
            current_price: Current market price
            stop_loss: Stop loss price
            take_profit: Take profit price
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (should_close: bool, reason: str)
        """
        # Calculate indicators
        df_with_indicators = self.calculate_indicators(df)

        # Check stop loss / take profit
        if position_side == 'LONG':
            if current_price <= stop_loss:
                return True, f"Stop loss hit (${current_price:.2f} <= ${stop_loss:.2f})"
            if current_price >= take_profit:
                return True, f"Take profit hit (${current_price:.2f} >= ${take_profit:.2f})"
        else:  # SHORT
            if current_price >= stop_loss:
                return True, f"Stop loss hit (${current_price:.2f} >= ${stop_loss:.2f})"
            if current_price <= take_profit:
                return True, f"Take profit hit (${current_price:.2f} <= ${take_profit:.2f})"

        # Check exit signals from volumetric indicators
        vol_signal = self.generate_signal(df_with_indicators, current_position=position_side)

        if vol_signal.signal == 'EXIT_LONG' and position_side == 'LONG':
            return True, vol_signal.reason
        elif vol_signal.signal == 'EXIT_SHORT' and position_side == 'SHORT':
            return True, vol_signal.reason

        return False, "Position still valid"

    def get_strategy_info(self) -> dict:
        """Get strategy configuration and info"""
        return {
            'name': self.name,
            'version': '1.0',
            'parameters': {
                'timeframe': config.strategy.timeframe,
                'vwap_session': self.vwap_session,
                'mfi_period': self.mfi_period,
                'mfi_overbought': self.mfi_overbought,
                'mfi_oversold': self.mfi_oversold,
                'st_period': self.st_period,
                'st_multiplier': self.st_multiplier,
                'vd_threshold': self.vd_threshold,
                'atr_stop_multiplier': self.stop_multiplier,
                'atr_target_multiplier': self.target_multiplier,
            }
        }

    def __str__(self):
        return f"{self.name} (VWAP + MFI{self.mfi_period} + VD + ST{self.st_period})"
