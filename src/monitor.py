"""
Monitoring and Logging Module
Handles logging, event tracking, and performance monitoring
"""

import logging
import asyncio
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json

from src.config import config


class TradingMonitor:
    """
    Monitor and logging system for the trading bot

    Features:
    - File and console logging
    - Database logging for trades and events
    - Performance tracking
    - Error logging
    """

    def __init__(self):
        """Initialize the monitor"""
        self.logger = None
        self.db_path = config.monitoring.db_path
        self.db = None

    async def initialize(self):
        """Initialize logging and database"""
        # Setup file logging
        self._setup_logging()

        # Setup database
        await self._setup_database()

        self.logger.info("=" * 60)
        self.logger.info("Trading Monitor Initialized")
        self.logger.info("=" * 60)

    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger('TradingBot')
        self.logger.setLevel(getattr(logging, config.monitoring.log_level))

        # Remove existing handlers
        self.logger.handlers.clear()

        # File handler
        if config.monitoring.log_to_file:
            log_file = log_dir / f'trading_{datetime.now().strftime("%Y%m%d")}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Console handler
        if config.monitoring.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    async def _setup_database(self):
        """Setup SQLite database for logging"""
        # Create data directory if it doesn't exist
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(exist_ok=True)

        # Connect to database
        self.db = await aiosqlite.connect(self.db_path)

        # Create tables
        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                size INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                pnl REAL NOT NULL,
                commission REAL NOT NULL,
                net_pnl REAL NOT NULL,
                exit_reason TEXT NOT NULL
            )
        ''')

        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL
            )
        ''')

        await self.db.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                total_pnl REAL NOT NULL,
                largest_win REAL NOT NULL,
                largest_loss REAL NOT NULL
            )
        ''')

        await self.db.commit()

        self.logger.info(f"Database initialized: {self.db_path}")

    async def log_trade(self, trade):
        """
        Log a completed trade to database

        Args:
            trade: Trade object from RiskManager
        """
        try:
            await self.db.execute('''
                INSERT INTO trades (
                    timestamp, symbol, direction, size,
                    entry_price, exit_price, entry_time, exit_time,
                    pnl, commission, net_pnl, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                trade.symbol,
                trade.direction,
                trade.size,
                trade.entry_price,
                trade.exit_price,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.pnl,
                trade.commission,
                trade.net_pnl,
                trade.exit_reason
            ))

            await self.db.commit()

            self.logger.info(f"Trade logged: {trade.symbol} {trade.direction} | P&L: ${trade.net_pnl:.2f}")

        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")

    async def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event to database

        Args:
            event_type: Type of event (e.g., 'signal_detected', 'position_opened')
            data: Event data as dictionary
        """
        try:
            await self.db.execute('''
                INSERT INTO events (timestamp, event_type, data)
                VALUES (?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                event_type,
                json.dumps(data, default=str)
            ))

            await self.db.commit()

            self.logger.debug(f"Event logged: {event_type}")

        except Exception as e:
            self.logger.error(f"Error logging event: {e}")

    async def update_daily_stats(self, date: str, stats: Dict[str, Any]):
        """
        Update daily statistics

        Args:
            date: Date string (YYYY-MM-DD)
            stats: Dictionary with daily statistics
        """
        try:
            await self.db.execute('''
                INSERT OR REPLACE INTO daily_stats (
                    date, total_trades, winning_trades, losing_trades,
                    total_pnl, largest_win, largest_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                date,
                stats['total_trades'],
                stats['winning_trades'],
                stats['losing_trades'],
                stats['total_pnl'],
                stats['largest_win'],
                stats['largest_loss']
            ))

            await self.db.commit()

        except Exception as e:
            self.logger.error(f"Error updating daily stats: {e}")

    async def get_trades(self, limit: int = 100) -> List[Dict]:
        """
        Get recent trades from database

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        try:
            cursor = await self.db.execute('''
                SELECT * FROM trades
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            rows = await cursor.fetchall()

            trades = []
            for row in rows:
                trades.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'symbol': row[2],
                    'direction': row[3],
                    'size': row[4],
                    'entry_price': row[5],
                    'exit_price': row[6],
                    'entry_time': row[7],
                    'exit_time': row[8],
                    'pnl': row[9],
                    'commission': row[10],
                    'net_pnl': row[11],
                    'exit_reason': row[12]
                })

            return trades

        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return []

    async def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """
        Get daily statistics

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily stats dictionaries
        """
        try:
            cursor = await self.db.execute('''
                SELECT * FROM daily_stats
                ORDER BY date DESC
                LIMIT ?
            ''', (days,))

            rows = await cursor.fetchall()

            stats = []
            for row in rows:
                stats.append({
                    'date': row[1],
                    'total_trades': row[2],
                    'winning_trades': row[3],
                    'losing_trades': row[4],
                    'total_pnl': row[5],
                    'largest_win': row[6],
                    'largest_loss': row[7]
                })

            return stats

        except Exception as e:
            self.logger.error(f"Error getting daily stats: {e}")
            return []

    def log_info(self, message: str):
        """Log info message"""
        if self.logger:
            self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        if self.logger:
            self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message"""
        if self.logger:
            self.logger.error(message)

    def log_debug(self, message: str):
        """Log debug message"""
        if self.logger:
            self.logger.debug(message)

    async def cleanup(self):
        """Cleanup resources"""
        if self.db:
            await self.db.close()
            self.logger.info("Database closed")


async def test_monitor():
    """Test the monitor"""
    print("=" * 60)
    print("TESTING MONITOR")
    print("=" * 60)

    monitor = TradingMonitor()
    await monitor.initialize()

    # Test logging
    monitor.log_info("This is an info message")
    monitor.log_warning("This is a warning message")
    monitor.log_debug("This is a debug message")

    # Test event logging
    await monitor.log_event('test_event', {'data': 'test'})

    # Test getting trades
    trades = await monitor.get_trades(10)
    print(f"\nRecent trades: {len(trades)}")

    # Cleanup
    await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(test_monitor())
