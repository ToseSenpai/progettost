"""
Trading Engine Module
Handles live trading operations with ProjectX API
"""

import asyncio
from datetime import datetime, time
from typing import Dict, List, Optional
from project_x_py import TradingSuite, EventType

from src.config import config
from src.data_manager import DataManager
from src.strategy import TrendFollowingStrategy, TradeSignal
from src.risk_manager import RiskManager
from src.monitor import TradingMonitor


class TradingEngine:
    """
    Main trading engine for live trading

    Responsibilities:
    - Connect to ProjectX API
    - Monitor market data in real-time
    - Execute strategy signals
    - Manage open positions
    - Handle order execution
    - Risk management integration
    """

    def __init__(self):
        """Initialize the trading engine"""
        self.data_manager = DataManager()
        self.strategy = TrendFollowingStrategy()
        self.risk_manager = RiskManager()
        self.monitor = TradingMonitor()

        self.suites: Dict[str, TradingSuite] = {}
        self.is_running = False
        self.last_check_time = {}

    async def initialize(self):
        """Initialize all components"""
        print("=" * 60)
        print("INITIALIZING TRADING ENGINE")
        print("=" * 60)

        # Validate configuration
        config.validate()
        config.print_config()

        # Initialize data manager
        print("\n📊 Initializing data manager...")
        await self.data_manager.initialize()

        # Setup instruments
        await self.data_manager.setup_instruments(config.instruments.symbols)

        # Download initial historical data
        print("\n📥 Downloading initial historical data...")
        await self.data_manager.download_all_historical_data(days=30)  # Need data for indicators

        # Store suites reference
        self.suites = self.data_manager.suites

        # Setup event handlers
        for symbol in config.instruments.symbols:
            suite = self.suites[symbol]

            # Register event handlers
            await suite.on(EventType.NEW_BAR, self._on_new_bar_handler(symbol))
            await suite.on(EventType.ORDER_FILLED, self._on_order_filled_handler(symbol))
            await suite.on(EventType.TICK, self._on_tick_handler(symbol))

        # Initialize monitor
        await self.monitor.initialize()

        print("\n✅ Trading engine initialized successfully!")

    def _on_new_bar_handler(self, symbol: str):
        """Create new bar event handler for a symbol"""
        async def handler(bar, timeframe):
            # Only process on primary timeframe
            if timeframe == config.strategy.timeframe:
                await self._process_bar(symbol, bar)

        return handler

    def _on_order_filled_handler(self, symbol: str):
        """Create order filled event handler for a symbol"""
        async def handler(order):
            print(f"\n✅ Order filled for {symbol}:")
            print(f"   Order ID: {order.orderId if hasattr(order, 'orderId') else 'N/A'}")
            print(f"   Status: Filled")

            # Log to monitor
            await self.monitor.log_event('order_filled', {
                'symbol': symbol,
                'order': str(order)
            })

        return handler

    def _on_tick_handler(self, symbol: str):
        """Create tick event handler for a symbol"""
        async def handler(tick):
            # Update current prices
            self.data_manager.current_prices[symbol] = tick.price

            # Check existing positions for stops/targets
            if symbol in self.risk_manager.positions:
                await self._check_position_exit(symbol, tick.price)

        return handler

    async def _process_bar(self, symbol: str, bar):
        """
        Process new bar and check for trading signals

        Args:
            symbol: Instrument symbol
            bar: New bar data
        """
        # Avoid processing too frequently
        current_time = datetime.now()
        if symbol in self.last_check_time:
            time_diff = (current_time - self.last_check_time[symbol]).total_seconds()
            if time_diff < 30:  # Wait at least 30 seconds between checks
                return

        self.last_check_time[symbol] = current_time

        # Get latest data with indicators
        df = self.data_manager.get_latest_data(symbol, bars=100)

        # Check if we have an open position
        if symbol in self.risk_manager.positions:
            position = self.risk_manager.positions[symbol]
            current_price = df.iloc[-1]['close']

            # Check if should close position
            should_close, reason = self.strategy.should_close_position(
                position_side=position.direction,
                entry_price=position.entry_price,
                current_price=current_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                df=df
            )

            if should_close:
                await self._close_position(symbol, reason)

        else:
            # Look for new signal
            signal = self.strategy.analyze(symbol, df)

            if signal:
                await self._process_signal(signal)

    async def _process_signal(self, signal: TradeSignal):
        """
        Process a trading signal

        Args:
            signal: Trading signal from strategy
        """
        print(f"\n🎯 Signal detected for {signal.symbol}:")
        print(f"   Direction: {signal.direction}")
        print(f"   Entry: ${signal.entry_price:.2f}")
        print(f"   Stop: ${signal.stop_loss:.2f}")
        print(f"   Target: ${signal.take_profit:.2f}")
        print(f"   R/R: 1:{signal.risk_reward_ratio:.2f}")
        print(f"   Confidence: {signal.confidence:.1%}")
        print(f"   Reason: {signal.reason}")

        # Check risk management approval
        can_open, reason = self.risk_manager.can_open_position(signal)

        if not can_open:
            print(f"   ❌ Position rejected: {reason}")
            await self.monitor.log_event('signal_rejected', {
                'symbol': signal.symbol,
                'reason': reason,
                'signal': str(signal)
            })
            return

        # Calculate position size
        account_balance = await self._get_account_balance()
        signal.size = self.risk_manager.calculate_position_size(signal, account_balance)

        print(f"   ✅ Opening position: {signal.size} contract(s)")

        # Execute order
        success = await self._execute_entry_order(signal)

        if success:
            # Open position in risk manager
            self.risk_manager.open_position(signal)

            await self.monitor.log_event('position_opened', {
                'symbol': signal.symbol,
                'signal': str(signal)
            })

            print(f"   ✅ Position opened successfully!")
        else:
            print(f"   ❌ Failed to open position")

    async def _execute_entry_order(self, signal: TradeSignal) -> bool:
        """
        Execute entry order with bracket (stop loss + take profit)

        Args:
            signal: Trading signal

        Returns:
            True if successful
        """
        try:
            suite = self.suites[signal.symbol]

            # Place bracket order (entry + stop loss + take profit)
            order = await suite.orders.place_bracket_order(
                contract_id=suite.instrument_id,
                side=0 if signal.direction == 'BUY' else 1,
                size=signal.size,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            if hasattr(order, 'success') and order.success:
                return True
            else:
                error_msg = order.errorMessage if hasattr(order, 'errorMessage') else 'Unknown error'
                print(f"   ❌ Order failed: {error_msg}")
                return False

        except Exception as e:
            print(f"   ❌ Error executing order: {e}")
            return False

    async def _close_position(self, symbol: str, reason: str):
        """
        Close an open position

        Args:
            symbol: Instrument symbol
            reason: Reason for closing
        """
        if symbol not in self.risk_manager.positions:
            return

        position = self.risk_manager.positions[symbol]

        print(f"\n📤 Closing position for {symbol}")
        print(f"   Reason: {reason}")
        print(f"   Position: {position.direction} {position.size} @ ${position.entry_price:.2f}")

        try:
            suite = self.suites[symbol]

            # Close position via API
            await suite.positions.close_position(symbol)

            # Get current price for P&L calculation
            current_price = self.data_manager.current_prices.get(symbol, position.entry_price)

            # Close in risk manager
            trade = self.risk_manager.close_position(
                symbol=symbol,
                exit_price=current_price,
                exit_reason=reason
            )

            if trade:
                print(f"   ✅ Position closed")
                print(f"   Exit: ${trade.exit_price:.2f}")
                print(f"   P&L: ${trade.net_pnl:.2f}")

                await self.monitor.log_event('position_closed', {
                    'symbol': symbol,
                    'trade': str(trade)
                })

                # Update statistics
                self.risk_manager.print_status()

        except Exception as e:
            print(f"   ❌ Error closing position: {e}")

    async def _check_position_exit(self, symbol: str, current_price: float):
        """
        Check if position should be exited based on current price

        Args:
            symbol: Instrument symbol
            current_price: Current market price
        """
        if symbol not in self.risk_manager.positions:
            return

        position = self.risk_manager.positions[symbol]

        # Check stop loss
        if position.direction == 'LONG':
            if current_price <= position.stop_loss:
                await self._close_position(symbol, "Stop loss hit")
            elif current_price >= position.take_profit:
                await self._close_position(symbol, "Take profit hit")

        else:  # SHORT
            if current_price >= position.stop_loss:
                await self._close_position(symbol, "Stop loss hit")
            elif current_price <= position.take_profit:
                await self._close_position(symbol, "Take profit hit")

    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            # Get account info from first suite (they all use same account)
            suite = list(self.suites.values())[0]
            account_info = await suite.client.account_info

            if hasattr(account_info, 'balance'):
                return float(account_info.balance)
            else:
                return config.backtest.initial_capital

        except Exception as e:
            print(f"   ⚠️  Error getting account balance: {e}")
            return config.backtest.initial_capital

    async def run(self):
        """
        Run the trading engine

        Main trading loop that monitors market and executes strategy
        """
        print("\n" + "=" * 60)
        print("🚀 STARTING LIVE TRADING")
        print("=" * 60)

        self.is_running = True

        try:
            print("\n📊 Monitoring market data...")
            print("   Press Ctrl+C to stop\n")

            # Main trading loop
            while self.is_running:
                # Check if within trading hours
                now = datetime.now().time()
                start_time = time(config.risk.trading_start_hour, 0)
                end_time = time(config.risk.trading_end_hour, 0)

                if start_time <= now < end_time:
                    if not self.risk_manager.trading_enabled:
                        # Reset at start of new trading day
                        self.risk_manager.reset_daily_limits()

                    # Print periodic status
                    await self._print_status()

                else:
                    if len(self.risk_manager.positions) > 0:
                        print("\n⏰ Outside trading hours - closing all positions")
                        current_prices = self.data_manager.current_prices
                        self.risk_manager.close_all_positions(current_prices, "Outside trading hours")

                # Sleep before next iteration
                await asyncio.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n\n⏹️  Received stop signal")
            await self.stop()

        except Exception as e:
            print(f"\n❌ Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
            await self.stop()

    async def _print_status(self):
        """Print periodic status update"""
        # Print every 5 minutes
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = datetime.now()

        time_diff = (datetime.now() - self._last_status_time).total_seconds()

        if time_diff >= 300:  # 5 minutes
            print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")

            # Print current prices
            for symbol, price in self.data_manager.current_prices.items():
                print(f"   {symbol}: ${price:.2f}")

            # Print positions
            if self.risk_manager.positions:
                print(f"\n   Open Positions: {len(self.risk_manager.positions)}")
                for symbol, pos in self.risk_manager.positions.items():
                    print(f"      {symbol}: {pos.direction} {pos.size} @ ${pos.entry_price:.2f} | P&L: ${pos.unrealized_pnl:.2f}")
            else:
                print(f"   No open positions")

            # Print daily P&L
            daily_pnl = self.risk_manager.get_daily_pnl()
            print(f"   Daily P&L: ${daily_pnl:.2f}")

            self._last_status_time = datetime.now()

    async def stop(self):
        """Stop the trading engine gracefully"""
        print("\n🛑 Stopping trading engine...")

        self.is_running = False

        # Close all open positions
        if self.risk_manager.positions:
            print("   Closing all open positions...")
            current_prices = self.data_manager.current_prices
            self.risk_manager.close_all_positions(current_prices, "System shutdown")

        # Print final statistics
        self.risk_manager.print_status()

        # Cleanup
        await self.data_manager.cleanup()
        await self.monitor.cleanup()

        print("\n✅ Trading engine stopped")

    async def emergency_stop(self):
        """Emergency stop - immediately close all positions"""
        print("\n🚨 EMERGENCY STOP TRIGGERED!")

        self.risk_manager.enable_emergency_mode()

        # Close all positions immediately
        if self.risk_manager.positions:
            current_prices = self.data_manager.current_prices
            self.risk_manager.close_all_positions(current_prices, "EMERGENCY STOP")

        await self.stop()


async def run_trading_engine():
    """Main function to run the trading engine"""
    engine = TradingEngine()

    try:
        # Initialize
        await engine.initialize()

        # Run trading
        await engine.run()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure cleanup
        if engine.is_running:
            await engine.stop()


if __name__ == "__main__":
    asyncio.run(run_trading_engine())
