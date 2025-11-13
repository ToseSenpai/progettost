"""
Configuration module for ProjectX Trading Bot
Manages all settings for strategy, risk management, and API connection
"""

import os
from dataclasses import dataclass
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API Configuration"""
    username: str
    api_key: str
    trading_mode: str  # 'demo' or 'live'
    environment: str  # 'demo' or 'topstepx'

    @classmethod
    def from_env(cls):
        """Create APIConfig from environment variables"""
        return cls(
            username=os.getenv('PROJECT_X_USERNAME', ''),
            api_key=os.getenv('PROJECT_X_API_KEY', ''),
            trading_mode=os.getenv('TRADING_MODE', 'demo'),
            environment=os.getenv('PROJECT_X_ENVIRONMENT', 'demo')
        )

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.username or not self.api_key:
            raise ValueError("PROJECT_X_USERNAME and PROJECT_X_API_KEY must be set in .env file")

        if self.trading_mode not in ['demo', 'live']:
            raise ValueError("TRADING_MODE must be 'demo' or 'live'")

        if self.environment not in ['demo', 'topstepx']:
            raise ValueError("PROJECT_X_ENVIRONMENT must be 'demo' or 'topstepx'")

        return True


@dataclass
class InstrumentConfig:
    """Configuration for trading instruments"""
    # Instruments to trade
    symbols: List[str] = None

    # Contract IDs (will be fetched dynamically)
    contract_ids: Dict[str, str] = None

    def __post_init__(self):
        if self.symbols is None:
            # Best symbol with drawdown < 3%:
            # MET: +9.59% return, -1.03% drawdown, 75% win rate, 6.79 profit factor
            self.symbols = ['MET']
        if self.contract_ids is None:
            self.contract_ids = {}


@dataclass
class StrategyConfig:
    """Trading Strategy Configuration"""

    # Timeframe
    timeframe: str = '5min'  # Optimal timeframe for MET
    timeframes: List[str] = None  # Multiple timeframes for analysis

    # EMA Settings (OPTIMIZED FOR MET on 5min - 300 trials)
    ema_fast: int = 12
    ema_slow: int = 92

    # RSI Settings (OPTIMIZED FOR MET - 300 trials)
    rsi_period: int = 12
    rsi_overbought: float = 76.5529
    rsi_oversold: float = 32.8699

    # ATR Settings (OPTIMIZED FOR MET - for dynamic stops - 300 trials)
    atr_period: int = 13
    atr_stop_multiplier: float = 1.6518  # Stop loss = 1.65x ATR
    atr_target_multiplier: float = 4.4606  # Take profit = 4.46x ATR (2.70:1 R/R)

    # Minimum volatility filter (OPTIMIZED FOR MET - in ATR units - 300 trials)
    min_atr_threshold: float = 0.6334  # Minimum ATR to take trades (high filter)

    # === VOLUMETRIC STRATEGY PARAMETERS ===
    # Strategy selector
    strategy_type: str = 'volumetric'  # 'classic' (EMA/RSI/ATR) or 'volumetric' (VWAP/MFI/VD/ST)

    # VWAP Settings
    vwap_session: str = 'D'  # 'D' for daily reset, 'W' for weekly

    # Money Flow Index (MFI) Settings - RSI with Volume
    mfi_period: int = 14
    mfi_overbought: float = 80
    mfi_oversold: float = 20

    # SuperTrend Settings
    st_period: int = 10  # ATR period for SuperTrend
    st_multiplier: float = 3.0  # ATR multiplier for bands

    # Volume Delta Settings
    vd_threshold: float = 0.3  # Threshold for significant volume imbalance

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['1min', '5min', '15min']


@dataclass
class RiskManagementConfig:
    """Risk Management Configuration"""

    # Position Sizing
    max_position_size: int = 1  # Max contracts per instrument
    max_total_positions: int = 3  # Max total positions (1 per instrument)

    # Loss Limits
    max_daily_loss: float = 300.0  # Maximum daily loss in dollars
    max_trade_loss: float = 100.0  # Maximum loss per trade
    max_drawdown_pct: float = 3.0  # Maximum drawdown percentage (3%)

    # Risk/Reward
    min_risk_reward_ratio: float = 2.0  # Minimum 1:2 risk/reward

    # Time-based controls
    trading_start_hour: int = 9  # Start trading at 9 AM ET (14:00 UTC)
    trading_end_hour: int = 15  # Stop trading at 3 PM ET (20:00 UTC)
    intraday_only: bool = True  # Close all positions before market close
    close_positions_hour: int = 15  # Close all positions at 3 PM ET (1 hour before close)

    # Account protection
    emergency_close_loss: float = 400.0  # Emergency close all if daily loss exceeds this
    max_consecutive_losses: int = 3  # Stop trading after X consecutive losses

    # Position management
    use_trailing_stop: bool = False  # Enable trailing stops
    trailing_stop_activation: float = 1.5  # Activate trailing after 1.5x ATR profit
    trailing_stop_distance: float = 1.0  # Trail by 1x ATR


@dataclass
class BacktestConfig:
    """Backtesting Configuration"""

    # Historical data period
    lookback_days: int = 90  # 3 months of data

    # Starting capital
    initial_capital: float = 10000.0

    # Commission and slippage
    commission_per_contract: float = 2.50  # Round trip commission
    slippage_ticks: int = 1  # Assume 1 tick slippage

    # Output settings
    save_results: bool = True
    generate_charts: bool = True
    export_trades: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and Logging Configuration"""

    # Logging
    log_level: str = 'INFO'  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_to_console: bool = True

    # Database
    db_path: str = 'data/trading_bot.db'

    # Notifications (future feature)
    enable_notifications: bool = False
    notification_type: str = 'console'  # console, email, telegram


class Config:
    """Main Configuration Class"""

    def __init__(self):
        self.api = APIConfig.from_env()
        self.instruments = InstrumentConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskManagementConfig()
        self.backtest = BacktestConfig()
        self.monitoring = MonitoringConfig()

    def validate(self):
        """Validate all configurations"""
        self.api.validate()

        # Additional validation
        if self.risk.max_total_positions > len(self.instruments.symbols):
            print(f"Warning: max_total_positions ({self.risk.max_total_positions}) "
                  f"is greater than number of instruments ({len(self.instruments.symbols)})")

        if self.risk.max_daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")

        if self.strategy.ema_fast >= self.strategy.ema_slow:
            raise ValueError("ema_fast must be less than ema_slow")

        return True

    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("PROJECTX TRADING BOT CONFIGURATION")
        print("=" * 60)
        print(f"\nAPI Configuration:")
        print(f"  Username: {self.api.username}")
        print(f"  Trading Mode: {self.api.trading_mode.upper()}")
        print(f"  Environment: {self.api.environment.upper()}")

        print(f"\nInstruments:")
        print(f"  Symbols: {', '.join(self.instruments.symbols)}")

        print(f"\nStrategy:")
        print(f"  Timeframe: {self.strategy.timeframe}")
        print(f"  EMA: {self.strategy.ema_fast}/{self.strategy.ema_slow}")
        print(f"  RSI Period: {self.strategy.rsi_period}")
        print(f"  ATR Period: {self.strategy.atr_period}")
        print(f"  Stop Loss: {self.strategy.atr_stop_multiplier}x ATR")
        print(f"  Take Profit: {self.strategy.atr_target_multiplier}x ATR")

        print(f"\nRisk Management:")
        print(f"  Max Daily Loss: ${self.risk.max_daily_loss}")
        print(f"  Max Position Size: {self.risk.max_position_size} contract(s)")
        print(f"  Max Total Positions: {self.risk.max_total_positions}")
        print(f"  Trading Hours: {self.risk.trading_start_hour}:00 - {self.risk.trading_end_hour}:00 ET")

        print(f"\nBacktest Settings:")
        print(f"  Lookback Days: {self.backtest.lookback_days}")
        print(f"  Initial Capital: ${self.backtest.initial_capital:,.2f}")
        print(f"  Commission: ${self.backtest.commission_per_contract} per contract")

        print("=" * 60)


# Create global config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    try:
        config.validate()
        config.print_config()
        print("\n✅ Configuration validated successfully!")
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
