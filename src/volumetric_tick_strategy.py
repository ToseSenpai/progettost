"""
PURE Volumetric Strategy for Tick Data
Uses ONLY order flow indicators - NO price-based indicators!

PHILOSOPHY: Order flow tells us what's REALLY happening in the market.
Price is just the result. We trade based on institutional footprint.

VOLUMETRIC INDICATORS ONLY:
- VWAP - volume-weighted price (institutional benchmark)
- Volume Delta/Cumulative Delta - buy vs sell pressure
- Volume Imbalance - normalized footprint metric
- Buy/Sell Ratio - market aggression
- Cumulative Delta Slope - order flow trend (NOT price trend!)
- Stacked Imbalance - professional pattern
- ATR - only for stop/target sizing

NO PRICE-BASED INDICATORS:
- NO EMA, SMA, moving averages
- NO SuperTrend, RSI, MACD
- NO price divergence
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.volumetric_strategy import VolumetricStrategy, VolumetricSignal
from src.config import config


class VolumetricTickStrategy(VolumetricStrategy):
    """
    PURE Volumetric Strategy - Order Flow Only.

    Trades based on institutional footprint, not price action.
    """

    def __init__(self):
        """Initialize with volumetric-only parameters"""
        super().__init__()
        self.name = "Pure Volumetric Strategy (Order Flow Only)"

        # Volume thresholds (VERY strict to avoid overtrading)
        self.min_volume_imbalance = 0.25  # 25% imbalance required
        self.strong_imbalance = 0.40      # 40% = strong signal

        # Delta slope settings (order flow trend)
        self.delta_slope_period = 100  # Longer period for stable trend

        # Stacked imbalance settings - 3:1 ratio (from research!)
        self.min_stacked_bars = 5  # 5 consecutive bars
        self.stacked_imbalance_ratio = 3.0  # 3:1 buy/sell ratio (professional standard)

        # Buy/Sell ratio thresholds (configurable for optimization)
        self.buy_sell_long_threshold = 1.5  # Buyers dominant for LONG
        self.buy_sell_short_threshold = 0.6  # Sellers dominant for SHORT
        self.buy_sell_exit_long = 1.0  # Exit LONG when neutral
        self.buy_sell_exit_short = 1.0  # Exit SHORT when neutral

        # Volume ratio threshold (configurable for optimization)
        self.volume_ratio_threshold = 2.0  # High volume = 2x average

        # Confluence thresholds (configurable for optimization)
        self.min_confluence = 0.85  # High quality entry (allows more trades)
        self.exit_confluence_threshold = 0.70  # Conservative exits (prevents premature exits)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ONLY volumetric indicators.

        NO price-based indicators!
        ALL indicators derived from volume data.

        Args:
            df: DataFrame with OHLCV + tick volume data

        Returns:
            DataFrame with volumetric indicators only
        """
        df = df.copy()

        # Verify tick data
        has_tick_data = all(col in df.columns for col in ['buy_volume', 'sell_volume', 'volume_delta'])
        if not has_tick_data:
            raise ValueError("Tick data required! Missing buy_volume, sell_volume, or volume_delta")

        # RTH Filter (from research) - Regular Trading Hours only
        # 9:30-16:00 ET = 14:30-21:00 UTC (MNQ futures)
        # Eliminates overnight noise and improves signal quality
        df['hour_utc'] = df.index.hour
        df['is_rth'] = (df['hour_utc'] >= 14) & (df['hour_utc'] < 21)  # RTH in UTC

        # === VOLUMETRIC INDICATORS ===

        # 1. VWAP - Volume Weighted Average Price (volumetric!)
        df['vwap'] = self._calculate_vwap(df)

        # 1b. VWAP Bands (Standard Deviation) - from research!
        df['vwap_std'] = self._calculate_vwap_std(df)
        df['vwap_upper_1std'] = df['vwap'] + df['vwap_std']
        df['vwap_lower_1std'] = df['vwap'] - df['vwap_std']
        df['vwap_upper_2std'] = df['vwap'] + (2 * df['vwap_std'])
        df['vwap_lower_2std'] = df['vwap'] - (2 * df['vwap_std'])

        # 2. Volume Delta - PRIMARY order flow signal
        df['cumulative_delta'] = df['volume_delta'].cumsum()

        # 3. Buy/Sell Ratio - Market aggression
        df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1)

        # 4. Volume Imbalance (normalized) - Footprint metric
        df['volume_imbalance'] = df['volume_delta'] / (df['volume'] + 1)

        # 5. Cumulative Delta Slope - ORDER FLOW TREND (not price trend!)
        df['delta_slope'] = df['cumulative_delta'].diff(self.delta_slope_period)
        df['delta_trend'] = np.where(df['delta_slope'] > 0, 1, -1)

        # 6. Stacked Imbalance Detection (institutional pattern)
        df['stacked_imbalance'] = self._detect_stacked_imbalance(df)

        # 6b. CVD Divergence Detection (HIGH-PROBABILITY reversal signal) - from research!
        df['cvd_divergence'] = self._detect_cvd_divergence(df, lookback=20)

        # 7. Volume Profile (POC, VAH, VAL) - from research!
        # Point of Control, Value Area High/Low = key support/resistance
        profile = self._calculate_volume_profile(df, window=50)
        df['poc'] = profile['poc']
        df['vah'] = profile['vah']
        df['val'] = profile['val']

        # 8. Volume Profile metrics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1)  # Current vs average

        # 9. Buy/Sell Pressure Momentum
        df['buy_pressure_ma'] = df['buy_volume'].rolling(window=10).mean()
        df['sell_pressure_ma'] = df['sell_volume'].rolling(window=10).mean()
        df['pressure_momentum'] = (df['buy_pressure_ma'] - df['sell_pressure_ma']) / (df['volume_ma'] + 1)

        # 10. ATR - Risk management only (NOT for entry signals!)
        df['atr'] = self._calculate_atr(df, self.atr_period)

        return df

    def _calculate_vwap_std(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP Standard Deviation (from research).

        Used for VWAP bands (±1σ, ±2σ) to identify overbought/oversold conditions.
        Mean reversion from 2σ is a high-probability setup.
        """
        # Calculate squared price deviations from VWAP, weighted by volume
        price_dev_squared = ((df['close'] - df['vwap']) ** 2) * df['volume']
        cum_volume = df['volume'].cumsum()
        cum_price_dev_squared = price_dev_squared.cumsum()

        # Variance = cumulative (price_dev^2 * volume) / cumulative volume
        variance = cum_price_dev_squared / (cum_volume + 1)

        # Standard deviation = sqrt(variance)
        return np.sqrt(variance)

    def _calculate_volume_profile(self, df: pd.DataFrame, window: int = 50) -> dict:
        """
        Calculate Volume Profile: POC, VAH, VAL (from research).

        POC (Point of Control) = price level with most volume
        VAH/VAL (Value Area High/Low) = 70% volume range
        These become key support/resistance levels.

        Args:
            df: DataFrame with OHLC data
            window: Rolling window for volume profile calculation

        Returns:
            dict with 'poc', 'vah', 'val' as pd.Series
        """
        poc_list = []
        vah_list = []
        val_list = []

        for i in range(len(df)):
            if i < window - 1:
                # Not enough data yet
                poc_list.append(np.nan)
                vah_list.append(np.nan)
                val_list.append(np.nan)
                continue

            # Get window data
            window_df = df.iloc[i - window + 1:i + 1]

            # Create price levels (use close prices rounded to tick size)
            # For MNQ, tick size is 0.25
            tick_size = 0.25
            prices = np.round(window_df['close'] / tick_size) * tick_size
            volumes = window_df['volume']

            # Build volume profile (price -> total volume)
            from collections import defaultdict
            volume_by_price = defaultdict(float)

            for price, volume in zip(prices, volumes):
                volume_by_price[price] += volume

            if not volume_by_price:
                poc_list.append(np.nan)
                vah_list.append(np.nan)
                val_list.append(np.nan)
                continue

            # Find POC (price with most volume)
            poc = max(volume_by_price.items(), key=lambda x: x[1])[0]

            # Calculate Value Area (70% of total volume)
            total_volume = sum(volume_by_price.values())
            target_volume = total_volume * 0.70

            # Sort prices by volume (descending)
            sorted_prices = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)

            # Build value area starting from POC
            value_area_volume = 0
            value_area_prices = []

            for price, volume in sorted_prices:
                value_area_prices.append(price)
                value_area_volume += volume
                if value_area_volume >= target_volume:
                    break

            # VAH = highest price in value area, VAL = lowest price
            vah = max(value_area_prices) if value_area_prices else poc
            val = min(value_area_prices) if value_area_prices else poc

            poc_list.append(poc)
            vah_list.append(vah)
            val_list.append(val)

        return {
            'poc': pd.Series(poc_list, index=df.index),
            'vah': pd.Series(vah_list, index=df.index),
            'val': pd.Series(val_list, index=df.index)
        }

    def _detect_stacked_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect stacked imbalances (professional footprint pattern) - IMPROVED with 3:1 ratio.

        From research: "Un rapporto comune per definire uno squilibrio è 300% o 400% (3:1 o 4:1)"

        3+ consecutive bars with 3:1 buy/sell ratio = institutional activity.
        This is MORE STRICT than simple threshold - professional standard!
        """
        # Calculate buy/sell ratio per bar
        buy_sell_ratio = df['buy_volume'] / (df['sell_volume'] + 1)

        # Bullish stack: N:1 ratio (buy_volume > N * sell_volume)
        # e.g., ratio=3.0 means buyers are 3x more aggressive (3:1)
        bullish = (buy_sell_ratio > self.stacked_imbalance_ratio).astype(int)
        bullish_streak = bullish.rolling(window=self.min_stacked_bars).sum()

        # Bearish stack: 1:N ratio (sell_volume > N * buy_volume)
        # e.g., ratio=3.0 -> 1/3.0=0.33 means sellers are 3x more aggressive (1:3)
        bearish = (buy_sell_ratio < (1 / self.stacked_imbalance_ratio)).astype(int)
        bearish_streak = bearish.rolling(window=self.min_stacked_bars).sum()

        stacked = pd.Series(0, index=df.index)
        stacked[bullish_streak >= self.min_stacked_bars] = 1
        stacked[bearish_streak >= self.min_stacked_bars] = -1

        return stacked

    def _detect_cvd_divergence(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Detect CVD (Cumulative Volume Delta) Divergence - HIGH PROBABILITY reversal signal!

        From research: "Divergenze tra prezzo e CVD possono segnalare inversioni di tendenza"

        Bearish Divergence: Price makes higher high BUT CVD makes lower high
            -> Institutional selling despite price rise = REVERSAL DOWN

        Bullish Divergence: Price makes lower low BUT CVD makes higher low
            -> Institutional buying despite price fall = REVERSAL UP

        Args:
            df: DataFrame with 'close' and 'cumulative_delta'
            lookback: Period to look for divergences

        Returns:
            Series with: 1 = bullish divergence, -1 = bearish divergence, 0 = none
        """
        divergence = pd.Series(0, index=df.index)

        for i in range(lookback, len(df)):
            # Get recent window
            window_close = df['close'].iloc[i - lookback:i + 1]
            window_cvd = df['cumulative_delta'].iloc[i - lookback:i + 1]

            # Find highs and lows in the window
            price_high_idx = window_close.idxmax()
            price_low_idx = window_close.idxmin()
            cvd_high_idx = window_cvd.idxmax()
            cvd_low_idx = window_cvd.idxmin()

            current_idx = df.index[i]

            # BEARISH DIVERGENCE: Price higher high + CVD lower high
            # Price makes new high at current bar
            if price_high_idx == current_idx:
                # Find previous high in CVD
                prev_window_cvd = window_cvd.iloc[:-1]  # Exclude current bar
                if len(prev_window_cvd) > 0:
                    prev_cvd_high = prev_window_cvd.max()
                    current_cvd = window_cvd.iloc[-1]

                    # If current CVD is lower than previous CVD high = BEARISH DIVERGENCE
                    if current_cvd < prev_cvd_high:
                        divergence.iloc[i] = -1  # Bearish divergence

            # BULLISH DIVERGENCE: Price lower low + CVD higher low
            # Price makes new low at current bar
            if price_low_idx == current_idx:
                # Find previous low in CVD
                prev_window_cvd = window_cvd.iloc[:-1]  # Exclude current bar
                if len(prev_window_cvd) > 0:
                    prev_cvd_low = prev_window_cvd.min()
                    current_cvd = window_cvd.iloc[-1]

                    # If current CVD is higher than previous CVD low = BULLISH DIVERGENCE
                    if current_cvd > prev_cvd_low:
                        divergence.iloc[i] = 1  # Bullish divergence

        return divergence

    def generate_signal(self, df: pd.DataFrame, current_position: Optional[str] = None) -> VolumetricSignal:
        """
        Generate signals using PURE order flow logic.

        LONG Entry (ALL required):
        - Price > VWAP (institutional support)
        - Volume Imbalance > 20% (strong buying)
        - Buy/Sell Ratio > 1.3 (buyer aggression)
        - Delta Trend = bullish (order flow up)
        - Confluence >= 0.80 (high quality only)

        SHORT Entry (ALL required):
        - Price < VWAP (institutional resistance)
        - Volume Imbalance < -20% (strong selling)
        - Buy/Sell Ratio < 0.7 (seller aggression)
        - Delta Trend = bearish (order flow down)
        - Confluence >= 0.80 (high quality only)

        Args:
            df: DataFrame with calculated indicators
            current_position: Current position

        Returns:
            VolumetricSignal
        """
        if len(df) < max(self.delta_slope_period, self.atr_period) + 10:
            return VolumetricSignal('HOLD', 0.0, reason="Insufficient data")

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # RTH Filter (from research) - only trade during Regular Trading Hours
        if not latest['is_rth']:
            return VolumetricSignal('HOLD', 0.0, reason="Outside RTH (9:30-16:00 ET)")

        # TOXIC HOURS FILTER (from stop loss analysis) - avoid hours with excessive SL rates
        # Hour 14 UTC (8:00 AM ET) = Market Open chop (-$3,256 loss)
        # Hour 15 UTC (9:00 AM ET) = High SL rate 65.5%, marginal profit +$1,198
        # Hour 16 UTC (10:00 AM ET) = WORST hour: 70% SL rate, -$1,434 loss (LOSING!)
        # Hour 18 UTC (12:00 PM ET) = Pre-lunch chop (-$652 loss)
        # Hour 19 UTC (13:00 PM ET) = Pre-close volatility (-$2,649 loss)
        current_hour = latest.name.hour
        if current_hour in [14, 15, 16, 18, 19]:
            return VolumetricSignal('HOLD', 0.0, reason=f"Toxic hour {current_hour}:00 UTC (filtered)")

        close = latest['close']
        vwap = latest['vwap']
        volume_imbalance = latest['volume_imbalance']
        buy_sell_ratio = latest['buy_sell_ratio']
        delta_trend = latest['delta_trend']
        delta_slope = latest['delta_slope']
        stacked_imbalance = latest['stacked_imbalance']
        volume_ratio = latest['volume_ratio']
        pressure_momentum = latest['pressure_momentum']
        atr = latest['atr']

        # Check for NaN
        if pd.isna([vwap, volume_imbalance, buy_sell_ratio, delta_trend, atr]).any():
            return VolumetricSignal('HOLD', close, reason="Indicator incomplete")

        # === EXIT LOGIC - CONFLUENCE BASED (anti-premature exit!) ===
        if current_position == 'LONG':
            # Count bearish signals (need MULTIPLE to exit)
            exit_confluence = 0
            exit_reasons = []

            # 1. Delta trend reversed (30% weight)
            if delta_trend == -1:
                exit_confluence += 0.30
                exit_reasons.append("Delta trend bearish")

            # 2. Strong selling pressure (40% weight)
            if volume_imbalance < -self.strong_imbalance:
                exit_confluence += 0.40
                exit_reasons.append(f"Strong selling ({volume_imbalance:.1%})")

            # 3. Sellers aggressive (30% weight)
            if buy_sell_ratio < self.buy_sell_exit_long:
                exit_confluence += 0.30
                exit_reasons.append(f"Sellers control ({buy_sell_ratio:.2f})")

            # REQUIRE exit_confluence_threshold to exit (prevents premature exits!)
            if exit_confluence >= self.exit_confluence_threshold:
                reason = " + ".join(exit_reasons)
                return VolumetricSignal('EXIT_LONG', close, reason=reason)

        elif current_position == 'SHORT':
            # Count bullish signals (need MULTIPLE to exit)
            exit_confluence = 0
            exit_reasons = []

            # 1. Delta trend reversed (30% weight)
            if delta_trend == 1:
                exit_confluence += 0.30
                exit_reasons.append("Delta trend bullish")

            # 2. Strong buying pressure (40% weight)
            if volume_imbalance > self.strong_imbalance:
                exit_confluence += 0.40
                exit_reasons.append(f"Strong buying ({volume_imbalance:.1%})")

            # 3. Buyers aggressive (30% weight)
            if buy_sell_ratio > self.buy_sell_exit_short:
                exit_confluence += 0.30
                exit_reasons.append(f"Buyers control ({buy_sell_ratio:.2f})")

            # REQUIRE exit_confluence_threshold to exit (prevents premature exits!)
            if exit_confluence >= self.exit_confluence_threshold:
                reason = " + ".join(exit_reasons)
                return VolumetricSignal('EXIT_SHORT', close, reason=reason)

        # === ENTRY LOGIC ===
        if current_position is None or current_position == 'HOLD':

            confluence_long = 0.0
            confluence_short = 0.0
            reasons_long = []
            reasons_short = []

            # === LONG CONDITIONS (Pure Order Flow) === IMPROVED with Research!

            # Get new indicators from research
            poc = latest['poc']
            vah = latest['vah']
            val = latest['val']
            cvd_div = latest['cvd_divergence']
            vwap_upper_2std = latest['vwap_upper_2std']
            vwap_lower_2std = latest['vwap_lower_2std']

            # 1. VWAP (15% weight) - Institutional level
            if close > vwap:
                confluence_long += 0.15
                reasons_long.append(f"Price above VWAP")

            # 2. Volume Imbalance (30% weight - REDUCED from 35%) - PRIMARY signal
            if volume_imbalance > self.min_volume_imbalance:
                weight = 0.30 * min(volume_imbalance / self.strong_imbalance, 1.0)
                confluence_long += weight
                reasons_long.append(f"Buy pressure ({volume_imbalance:.1%})")

            # 3. Buy/Sell Ratio (20% weight - REDUCED from 25%) - Aggression
            if buy_sell_ratio > self.buy_sell_long_threshold:
                weight = 0.20 * min((buy_sell_ratio - 1) / 0.5, 1.0)
                confluence_long += weight
                reasons_long.append(f"Buyers aggressive ({buy_sell_ratio:.2f})")

            # 4. Volume Profile (15% weight) - NEW! Institutional levels from research
            if not pd.isna(poc):
                # Price above POC (bullish positioning)
                if close > poc:
                    confluence_long += 0.08
                    reasons_long.append("Above POC")
                # Near VAL support (bounce opportunity)
                if not pd.isna(val) and abs(close - val) < (0.3 * atr):
                    confluence_long += 0.07
                    reasons_long.append("Near VAL support")

            # 5. CVD Divergence (15% weight) - NEW! High-probability reversal from research
            if cvd_div == 1:  # Bullish divergence
                confluence_long += 0.15
                reasons_long.append("CVD BULLISH DIVERGENCE!")

            # 6. Delta Trend (10% weight - REDUCED from 15%, now WEIGHT not FILTER!)
            if delta_trend == 1:
                confluence_long += 0.10
                reasons_long.append("Delta trend aligned")
            elif delta_trend == -1:
                confluence_long -= 0.05  # Penalty but not blocking!
                # No reason added for penalty

            # 7. Pressure Momentum (10% weight) - Sustained buying
            if pressure_momentum > 0:
                weight = 0.10 * min(abs(pressure_momentum) / 0.1, 1.0)
                confluence_long += weight
                reasons_long.append("Buy pressure sustained")

            # BONUS: Stacked Imbalance (10% extra - REDUCED from 15%) - Institutional signal
            if stacked_imbalance == 1:
                confluence_long += 0.10
                reasons_long.append("STACKED IMBALANCE (institutional)")

            # BONUS: VWAP Mean Reversion (10% extra) - NEW! From research
            if not pd.isna(vwap_lower_2std) and close < vwap_lower_2std:
                # Oversold + buy pressure starting = mean reversion LONG
                if volume_imbalance > self.min_volume_imbalance:
                    confluence_long += 0.10
                    reasons_long.append("Mean reversion from -2σ VWAP!")

            # BONUS: High Volume (5% extra) - Confirmation
            if volume_ratio > self.volume_ratio_threshold:
                confluence_long += 0.05
                reasons_long.append("High volume confirmation")

            # === SHORT CONDITIONS (Pure Order Flow) === IMPROVED with Research!

            # Get new indicators from research (same as LONG)
            poc = latest['poc']
            vah = latest['vah']
            val = latest['val']
            cvd_div = latest['cvd_divergence']
            vwap_upper_2std = latest['vwap_upper_2std']
            vwap_lower_2std = latest['vwap_lower_2std']

            # 1. VWAP (15% weight) - Institutional level
            if close < vwap:
                confluence_short += 0.15
                reasons_short.append(f"Price below VWAP")

            # 2. Volume Imbalance (30% weight - REDUCED from 35%) - PRIMARY signal
            if volume_imbalance < -self.min_volume_imbalance:
                weight = 0.30 * min(abs(volume_imbalance) / self.strong_imbalance, 1.0)
                confluence_short += weight
                reasons_short.append(f"Sell pressure ({volume_imbalance:.1%})")

            # 3. Buy/Sell Ratio (20% weight - REDUCED from 25%) - Aggression
            if buy_sell_ratio < self.buy_sell_short_threshold:
                weight = 0.20 * min((1 - buy_sell_ratio) / 0.5, 1.0)
                confluence_short += weight
                reasons_short.append(f"Sellers aggressive ({buy_sell_ratio:.2f})")

            # 4. Volume Profile (15% weight) - NEW! Institutional levels from research
            if not pd.isna(poc):
                # Price below POC (bearish positioning)
                if close < poc:
                    confluence_short += 0.08
                    reasons_short.append("Below POC")
                # Near VAH resistance (rejection opportunity)
                if not pd.isna(vah) and abs(close - vah) < (0.3 * atr):
                    confluence_short += 0.07
                    reasons_short.append("Near VAH resistance")

            # 5. CVD Divergence (15% weight) - NEW! High-probability reversal from research
            if cvd_div == -1:  # Bearish divergence
                confluence_short += 0.15
                reasons_short.append("CVD BEARISH DIVERGENCE!")

            # 6. Delta Trend (10% weight - REDUCED from 15%, now WEIGHT not FILTER!)
            if delta_trend == -1:
                confluence_short += 0.10
                reasons_short.append("Delta trend aligned")
            elif delta_trend == 1:
                confluence_short -= 0.05  # Penalty but not blocking!
                # No reason added for penalty

            # 7. Pressure Momentum (10% weight) - Sustained selling
            if pressure_momentum < 0:
                weight = 0.10 * min(abs(pressure_momentum) / 0.1, 1.0)
                confluence_short += weight
                reasons_short.append("Sell pressure sustained")

            # BONUS: Stacked Imbalance (10% extra - REDUCED from 15%) - Institutional signal
            if stacked_imbalance == -1:
                confluence_short += 0.10
                reasons_short.append("STACKED IMBALANCE (institutional)")

            # BONUS: VWAP Mean Reversion (10% extra) - NEW! From research
            if not pd.isna(vwap_upper_2std) and close > vwap_upper_2std:
                # Overbought + sell pressure starting = mean reversion SHORT
                if volume_imbalance < -self.min_volume_imbalance:
                    confluence_short += 0.10
                    reasons_short.append("Mean reversion from +2σ VWAP!")

            # BONUS: High Volume (5% extra) - Confirmation
            if volume_ratio > self.volume_ratio_threshold:
                confluence_short += 0.05
                reasons_short.append("High volume confirmation")

            # === DECISION ===

            # Pure Confluence-Based Entry (Delta Trend is now weighted signal, NOT hard filter!)
            if confluence_long >= self.min_confluence and confluence_long > confluence_short:
                # LONG ENTRY (based on confluence score only)
                stop_loss = close - (self.stop_multiplier * atr)
                take_profit = close + (self.target_multiplier * atr)

                return VolumetricSignal(
                    'LONG', close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=" | ".join(reasons_long),
                    confidence=confluence_long
                )

            elif confluence_short >= self.min_confluence and confluence_short > confluence_long:
                # SHORT ENTRY (based on confluence score only)
                stop_loss = close + (self.stop_multiplier * atr)
                take_profit = close - (self.target_multiplier * atr)

                return VolumetricSignal(
                    'SHORT', close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=" | ".join(reasons_short),
                    confidence=confluence_short
                )

        # Default: HOLD
        return VolumetricSignal('HOLD', close, reason="No order flow confluence")

    def get_strategy_info(self) -> dict:
        """Return strategy configuration"""
        return {
            'name': self.name,
            'type': 'pure_volumetric',
            'philosophy': 'Order flow only - NO price indicators',
            'indicators': 'VWAP, Volume Delta, Buy/Sell Ratio, Delta Slope, Stacked Imbalance',
            'removed': 'MFI, SuperTrend, EMA, all price-based indicators',
            'vwap_session': self.vwap_session,
            'delta_slope_period': self.delta_slope_period,
            'min_volume_imbalance': f'{self.min_volume_imbalance:.0%}',
            'strong_imbalance': f'{self.strong_imbalance:.0%}',
            'min_stacked_bars': self.min_stacked_bars,
            'stacked_imbalance_ratio': f'{self.stacked_imbalance_ratio:.1f}:1',
            'min_confluence': f'{self.min_confluence:.0%}',
            'stop_multiplier': self.stop_multiplier,
            'target_multiplier': self.target_multiplier,
        }
