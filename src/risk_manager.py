"""
Risk Management Module
Handles all risk-related decisions including position sizing, loss limits, and emergency controls
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, date
from collections import defaultdict

from src.config import config
from src.strategy import TradeSignal


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    size: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float, tick_value: float = 5.0):
        """
        Update unrealized P&L

        Args:
            current_price: Current market price
            tick_value: Dollar value per point (default $5 for micro contracts)
        """
        if self.direction == 'LONG':
            pnl = (current_price - self.entry_price) * self.size * tick_value
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.size * tick_value

        self.unrealized_pnl = pnl


@dataclass
class Trade:
    """Represents a closed trade"""
    symbol: str
    direction: str
    size: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: str
    commission: float = 0.0

    @property
    def net_pnl(self) -> float:
        """Net P&L after commission"""
        return self.pnl - self.commission

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.net_pnl > 0


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: date
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0

    def add_trade(self, trade: Trade):
        """Add a trade to daily stats"""
        self.trades.append(trade)
        self.total_pnl += trade.net_pnl

        if trade.is_winner:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1


class RiskManager:
    """
    Risk Manager for the trading system

    Responsibilities:
    - Position sizing
    - Daily loss limits
    - Maximum position limits
    - Emergency stop mechanisms
    - Trade approval/rejection
    - Performance tracking
    """

    def __init__(self):
        """Initialize the Risk Manager"""
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.daily_stats: Dict[date, DailyStats] = defaultdict(lambda: DailyStats(date=date.today()))
        self.total_pnl: float = 0.0
        self.trading_enabled: bool = True
        self.emergency_stop_triggered: bool = False

        # Tick values for different instruments
        self.tick_values = {
            'MNQ': 2.0,    # Micro Nasdaq: $2 per point
            'MES': 5.0,    # Micro S&P: $5 per point
            'MYM': 0.5,    # Micro Dow: $0.50 per point
            'MGC': 1.0,    # Micro Gold: $1 per point
            'MBT': 10.0,   # Micro Bitcoin: $10 per point
            'MET': 10.0,   # Micro Ether: $10 per point
            'MCL': 10.0,   # Micro Crude Oil: $10 per point
            'MHG': 12.5,   # Micro Copper: $12.50 per point
        }

    def can_open_position(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Check if a new position can be opened

        Args:
            signal: Trading signal

        Returns:
            Tuple of (can_open, reason)
        """
        # Check if trading is enabled
        if not self.trading_enabled:
            return False, "Trading is disabled"

        # Check emergency stop
        if self.emergency_stop_triggered:
            return False, "Emergency stop triggered"

        # Check if already have position in this instrument
        if signal.symbol in self.positions:
            return False, f"Already have open position in {signal.symbol}"

        # Check maximum total positions
        if len(self.positions) >= config.risk.max_total_positions:
            return False, f"Maximum total positions ({config.risk.max_total_positions}) reached"

        # Check daily loss limit
        today = date.today()
        daily_pnl = self.daily_stats[today].total_pnl

        if daily_pnl <= -config.risk.max_daily_loss:
            self.trading_enabled = False
            return False, f"Daily loss limit reached: ${daily_pnl:.2f}"

        # Check consecutive losses
        consecutive_losses = self.daily_stats[today].consecutive_losses
        if consecutive_losses >= config.risk.max_consecutive_losses:
            self.trading_enabled = False
            return False, f"Maximum consecutive losses ({consecutive_losses}) reached"

        # Check emergency loss limit
        if daily_pnl <= -config.risk.emergency_close_loss:
            self.emergency_stop_triggered = True
            return False, f"EMERGENCY: Daily loss exceeds ${config.risk.emergency_close_loss}"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: TradeSignal,
        account_balance: float
    ) -> int:
        """
        Calculate appropriate position size based on risk parameters

        Args:
            signal: Trading signal
            account_balance: Current account balance

        Returns:
            Position size in contracts
        """
        # For now, use fixed position size from config
        # In future, can implement Kelly Criterion or fixed fractional sizing
        size = config.risk.max_position_size

        # Check if risk per trade exceeds maximum
        tick_value = self.tick_values.get(signal.symbol, 2.0)
        risk_per_contract = signal.risk_amount * tick_value

        if risk_per_contract > config.risk.max_trade_loss:
            # Reduce size to meet max trade loss
            max_size = int(config.risk.max_trade_loss / risk_per_contract)
            size = max(1, min(size, max_size))

        return size

    def open_position(self, signal: TradeSignal) -> Position:
        """
        Open a new position

        Args:
            signal: Trading signal

        Returns:
            Position object
        """
        position = Position(
            symbol=signal.symbol,
            direction='LONG' if signal.direction == 'BUY' else 'SHORT',
            size=signal.size,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=signal.timestamp
        )

        self.positions[signal.symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str,
        exit_time: Optional[datetime] = None
    ) -> Optional[Trade]:
        """
        Close an existing position

        Args:
            symbol: Instrument symbol
            exit_price: Exit price
            exit_reason: Reason for exit
            exit_time: Exit timestamp (default: now)

        Returns:
            Trade object or None if position doesn't exist
        """
        if symbol not in self.positions:
            return None

        position = self.positions.pop(symbol)

        if exit_time is None:
            exit_time = datetime.now()

        # Calculate P&L
        tick_value = self.tick_values.get(symbol, 2.0)

        if position.direction == 'LONG':
            pnl = (exit_price - position.entry_price) * position.size * tick_value
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.size * tick_value

        # Commission (round trip)
        commission = config.backtest.commission_per_contract * position.size

        trade = Trade(
            symbol=symbol,
            direction=position.direction,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            exit_reason=exit_reason,
            commission=commission
        )

        # Update statistics
        self.closed_trades.append(trade)
        self.total_pnl += trade.net_pnl

        today = exit_time.date()
        self.daily_stats[today].add_trade(trade)

        return trade

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update all open positions with current prices

        Args:
            current_prices: Dictionary of current prices by symbol
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                tick_value = self.tick_values.get(symbol, 2.0)
                position.update_pnl(current_prices[symbol], tick_value)

    def get_open_positions(self) -> List[Position]:
        """Get list of all open positions"""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)

    def close_all_positions(self, current_prices: Dict[str, float], reason: str = "Emergency close"):
        """
        Close all open positions

        Args:
            current_prices: Dictionary of current prices
            reason: Reason for closing all positions
        """
        symbols = list(self.positions.keys())

        for symbol in symbols:
            if symbol in current_prices:
                self.close_position(
                    symbol=symbol,
                    exit_price=current_prices[symbol],
                    exit_reason=reason
                )

    def get_daily_pnl(self, target_date: Optional[date] = None) -> float:
        """
        Get P&L for a specific day

        Args:
            target_date: Target date (default: today)

        Returns:
            Daily P&L
        """
        if target_date is None:
            target_date = date.today()

        return self.daily_stats[target_date].total_pnl

    def get_statistics(self) -> Dict:
        """Get comprehensive trading statistics"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

        winning_trades = [t for t in self.closed_trades if t.is_winner]
        losing_trades = [t for t in self.closed_trades if not t.is_winner]

        total_wins = sum(t.net_pnl for t in winning_trades)
        total_losses = abs(sum(t.net_pnl for t in losing_trades))

        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) * 100,
            'total_pnl': self.total_pnl,
            'avg_win': total_wins / len(winning_trades) if winning_trades else 0,
            'avg_loss': total_losses / len(losing_trades) if losing_trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'largest_win': max([t.net_pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.net_pnl for t in losing_trades]) if losing_trades else 0,
            'open_positions': len(self.positions)
        }

    def reset_daily_limits(self):
        """Reset daily trading limits (call at start of new trading day)"""
        self.trading_enabled = True
        # Note: Don't reset emergency_stop_triggered - requires manual intervention

    def enable_emergency_mode(self):
        """Enable emergency mode - stops all trading"""
        self.emergency_stop_triggered = True
        self.trading_enabled = False

    def should_close_for_drawdown(self, current_equity: float, peak_equity: float) -> tuple[bool, str]:
        """
        Check if positions should be closed due to drawdown limit

        Args:
            current_equity: Current account equity
            peak_equity: Peak equity (highest value reached)

        Returns:
            Tuple of (should_close, reason)
        """
        if peak_equity <= 0:
            return False, "No peak equity set"

        # Calculate current drawdown percentage
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100

        if drawdown_pct >= config.risk.max_drawdown_pct:
            return True, f"Drawdown limit reached: {drawdown_pct:.2f}% >= {config.risk.max_drawdown_pct}%"

        return False, "Drawdown within limits"

    def should_close_for_eod(self, current_time: datetime) -> tuple[bool, str]:
        """
        Check if positions should be closed for end-of-day (intraday only mode)

        Args:
            current_time: Current timestamp

        Returns:
            Tuple of (should_close, reason)
        """
        if not config.risk.intraday_only:
            return False, "Intraday mode not enabled"

        # Get current hour in ET (UTC-5 or UTC-4 depending on DST)
        # For simplicity, we check the hour component
        current_hour = current_time.hour

        # Close all positions at the close_positions_hour
        if current_hour >= config.risk.close_positions_hour:
            return True, f"End of day: closing positions at {config.risk.close_positions_hour}:00"

        return False, "Not yet end of day"

    def disable_emergency_mode(self):
        """Disable emergency mode - allows trading to resume"""
        self.emergency_stop_triggered = False
        self.trading_enabled = True

    def print_status(self):
        """Print current risk manager status"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("RISK MANAGER STATUS")
        print("=" * 60)

        print(f"\nTrading Status:")
        print(f"  Enabled: {self.trading_enabled}")
        print(f"  Emergency Stop: {self.emergency_stop_triggered}")

        print(f"\nPositions:")
        print(f"  Open Positions: {len(self.positions)}/{config.risk.max_total_positions}")

        for symbol, pos in self.positions.items():
            print(f"    {symbol}: {pos.direction} {pos.size} @ ${pos.entry_price:.2f} | P&L: ${pos.unrealized_pnl:.2f}")

        print(f"\nDaily Stats ({date.today()}):")
        today = date.today()
        daily = self.daily_stats[today]
        print(f"  Trades: {len(daily.trades)}")
        print(f"  P&L: ${daily.total_pnl:.2f}")
        print(f"  Consecutive Losses: {daily.consecutive_losses}")
        print(f"  Remaining Loss Capacity: ${config.risk.max_daily_loss + daily.total_pnl:.2f}")

        print(f"\nOverall Statistics:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")

        print("=" * 60)


def test_risk_manager():
    """Test the risk manager"""
    print("=" * 60)
    print("TESTING RISK MANAGER")
    print("=" * 60)

    from src.strategy import TradeSignal

    rm = RiskManager()

    # Create test signal
    signal = TradeSignal(
        symbol='MNQ',
        direction='BUY',
        entry_price=19500.0,
        stop_loss=19480.0,
        take_profit=19540.0,
        size=1,
        timestamp=datetime.now(),
        reason='Test signal',
        confidence=0.8,
        indicators={}
    )

    # Test position approval
    can_open, reason = rm.can_open_position(signal)
    print(f"\nCan open position: {can_open}")
    print(f"Reason: {reason}")

    if can_open:
        # Calculate position size
        size = rm.calculate_position_size(signal, account_balance=10000)
        print(f"Position size: {size} contract(s)")

        signal.size = size

        # Open position
        position = rm.open_position(signal)
        print(f"\n✅ Position opened: {position}")

    # Print status
    rm.print_status()


if __name__ == "__main__":
    test_risk_manager()
