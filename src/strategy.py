"""
Trading Strategy Module
Implements the trend-following strategy with EMA crossover, RSI, and ATR
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

from src.config import config


@dataclass
class TradeSignal:
    """Represents a trading signal"""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    size: int
    timestamp: datetime
    reason: str
    confidence: float  # 0.0 to 1.0
    indicators: Dict

    @property
    def risk_amount(self) -> float:
        """Calculate risk amount in price points"""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_amount(self) -> float:
        """Calculate reward amount in price points"""
        return abs(self.take_profit - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio"""
        if self.risk_amount == 0:
            return 0
        return self.reward_amount / self.risk_amount


class TrendFollowingStrategy:
    """
    Trend Following Strategy using EMA Crossover with RSI and ATR

    Entry Rules:
    - LONG: Fast EMA crosses above Slow EMA + RSI < 70
    - SHORT: Fast EMA crosses below Slow EMA + RSI > 30

    Exit Rules:
    - Stop Loss: Entry ± (ATR × stop_multiplier)
    - Take Profit: Entry ± (ATR × target_multiplier)

    Filters:
    - Minimum ATR threshold (avoid low volatility periods)
    - Trading hours restriction
    """

    def __init__(self):
        """Initialize the strategy"""
        self.name = "Trend Following EMA-RSI-ATR"
        self.version = "1.0"

    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Analyze market data and generate trading signal

        Args:
            symbol: Instrument symbol
            df: DataFrame with OHLCV data and indicators

        Returns:
            TradeSignal if valid signal found, None otherwise
        """
        if len(df) < config.strategy.ema_slow + 10:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Extract indicator values
        price = latest['close']
        ema_fast = latest['ema_fast']
        ema_slow = latest['ema_slow']
        rsi = latest['rsi']
        atr = latest['atr']

        # Check for valid ATR (minimum volatility filter)
        if atr < config.strategy.min_atr_threshold:
            return None

        # Check trading hours
        if not self._is_trading_hours(latest.name):
            return None

        # Detect crossovers
        bullish_cross = latest['ema_cross_up']
        bearish_cross = latest['ema_cross_down']

        signal = None

        # LONG SIGNAL
        if bullish_cross and rsi < config.strategy.rsi_overbought:
            stop_loss = price - (atr * config.strategy.atr_stop_multiplier)
            take_profit = price + (atr * config.strategy.atr_target_multiplier)

            # Calculate confidence based on indicator alignment
            confidence = self._calculate_confidence(
                ema_trend='bullish',
                rsi=rsi,
                atr=atr,
                df=df
            )

            signal = TradeSignal(
                symbol=symbol,
                direction='BUY',
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=1,  # Will be adjusted by risk manager
                timestamp=latest.name if isinstance(latest.name, datetime) else datetime.now(),
                reason='Bullish EMA crossover with RSI confirmation',
                confidence=confidence,
                indicators={
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'rsi': rsi,
                    'atr': atr
                }
            )

        # SHORT SIGNAL
        elif bearish_cross and rsi > config.strategy.rsi_oversold:
            stop_loss = price + (atr * config.strategy.atr_stop_multiplier)
            take_profit = price - (atr * config.strategy.atr_target_multiplier)

            # Calculate confidence
            confidence = self._calculate_confidence(
                ema_trend='bearish',
                rsi=rsi,
                atr=atr,
                df=df
            )

            signal = TradeSignal(
                symbol=symbol,
                direction='SELL',
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                size=1,
                timestamp=latest.name if isinstance(latest.name, datetime) else datetime.now(),
                reason='Bearish EMA crossover with RSI confirmation',
                confidence=confidence,
                indicators={
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'rsi': rsi,
                    'atr': atr
                }
            )

        # Validate signal meets minimum risk/reward ratio
        if signal and signal.risk_reward_ratio < config.risk.min_risk_reward_ratio:
            return None

        return signal

    def _calculate_confidence(
        self,
        ema_trend: str,
        rsi: float,
        atr: float,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate confidence score for the signal (0.0 to 1.0)

        Factors:
        - RSI distance from neutral (50)
        - ATR relative to recent ATR
        - Trend strength (EMA separation)
        """
        confidence = 0.5  # Base confidence

        # RSI factor
        if ema_trend == 'bullish':
            # Prefer RSI between 40-60 for longs
            if 40 <= rsi <= 60:
                confidence += 0.2
            elif rsi < 40:
                confidence += 0.1
        else:  # bearish
            # Prefer RSI between 40-60 for shorts
            if 40 <= rsi <= 60:
                confidence += 0.2
            elif rsi > 60:
                confidence += 0.1

        # ATR factor (higher volatility = higher confidence in trend)
        avg_atr = df['atr'].tail(20).mean()
        if atr > avg_atr:
            confidence += 0.15
        elif atr > avg_atr * 0.8:
            confidence += 0.05

        # EMA separation (stronger trend = higher confidence)
        latest = df.iloc[-1]
        ema_diff_pct = abs(latest['ema_fast'] - latest['ema_slow']) / latest['close'] * 100

        if ema_diff_pct > 0.5:
            confidence += 0.15
        elif ema_diff_pct > 0.3:
            confidence += 0.1

        return min(confidence, 1.0)

    @staticmethod
    def _is_trading_hours(timestamp: datetime) -> bool:
        """
        Check if current time is within trading hours

        Args:
            timestamp: Current timestamp

        Returns:
            True if within trading hours
        """
        hour = timestamp.hour

        # Check if within configured trading hours
        if config.risk.trading_start_hour <= hour < config.risk.trading_end_hour:
            return True

        return False

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
        Check if position should be closed based on exit rules

        Args:
            position_side: 'LONG' or 'SHORT'
            entry_price: Entry price
            current_price: Current price
            stop_loss: Stop loss price
            take_profit: Take profit price
            df: Current market data

        Returns:
            Tuple of (should_close, reason)
        """
        latest = df.iloc[-1]

        # Check stop loss
        if position_side == 'LONG':
            if current_price <= stop_loss:
                return True, "Stop loss hit"
            if current_price >= take_profit:
                return True, "Take profit hit"

            # Exit if bearish crossover occurs
            if latest['ema_cross_down']:
                return True, "Bearish crossover - exit long"

        else:  # SHORT
            if current_price >= stop_loss:
                return True, "Stop loss hit"
            if current_price <= take_profit:
                return True, "Take profit hit"

            # Exit if bullish crossover occurs
            if latest['ema_cross_up']:
                return True, "Bullish crossover - exit short"

        # Check if outside trading hours
        if not self._is_trading_hours(latest.name if isinstance(latest.name, datetime) else datetime.now()):
            return True, "Outside trading hours"

        return False, ""

    def get_strategy_info(self) -> Dict:
        """Get strategy configuration and info"""
        return {
            'name': self.name,
            'version': self.version,
            'parameters': {
                'timeframe': config.strategy.timeframe,
                'ema_fast': config.strategy.ema_fast,
                'ema_slow': config.strategy.ema_slow,
                'rsi_period': config.strategy.rsi_period,
                'rsi_overbought': config.strategy.rsi_overbought,
                'rsi_oversold': config.strategy.rsi_oversold,
                'atr_period': config.strategy.atr_period,
                'atr_stop_multiplier': config.strategy.atr_stop_multiplier,
                'atr_target_multiplier': config.strategy.atr_target_multiplier,
                'min_atr_threshold': config.strategy.min_atr_threshold
            },
            'risk_management': {
                'min_risk_reward': config.risk.min_risk_reward_ratio,
                'max_position_size': config.risk.max_position_size,
                'max_daily_loss': config.risk.max_daily_loss
            }
        }


def test_strategy():
    """Test the strategy with sample data"""
    print("=" * 60)
    print("TESTING TRADING STRATEGY")
    print("=" * 60)

    strategy = TrendFollowingStrategy()

    # Print strategy info
    info = strategy.get_strategy_info()
    print(f"\nStrategy: {info['name']} v{info['version']}")
    print("\nParameters:")
    for key, value in info['parameters'].items():
        print(f"  {key}: {value}")

    print("\nStrategy ready for backtesting and live trading!")


if __name__ == "__main__":
    test_strategy()
