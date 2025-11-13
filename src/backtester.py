"""
Backtesting Engine
Simulates trading strategy on historical data and generates performance reports
"""

import asyncio
from datetime import datetime, date
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import config
from src.data_manager import DataManager
from src.strategy import TrendFollowingStrategy, TradeSignal
from src.volumetric_strategy import VolumetricStrategy, VolumetricSignal
from src.risk_manager import RiskManager, Trade


class BacktestResult:
    """Container for backtest results"""

    def __init__(
        self,
        trades: List[Trade],
        equity_curve: pd.DataFrame,
        statistics: Dict,
        parameters: Dict
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.statistics = statistics
        self.parameters = parameters


class Backtester:
    """
    Backtesting engine for trading strategies

    Simulates trading on historical data with:
    - Realistic order execution
    - Commission and slippage
    - Risk management rules
    - Performance metrics
    """

    def __init__(self):
        """Initialize the backtester"""
        self.data_manager = None

        # Select strategy based on config
        strategy_type = getattr(config.strategy, 'strategy_type', 'classic')
        if strategy_type == 'volumetric':
            self.strategy = VolumetricStrategy()
        else:
            self.strategy = TrendFollowingStrategy()

        self.risk_manager = RiskManager()
        self.results = None

    async def run_backtest(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            symbols: List of symbols to backtest (default: from config)
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            BacktestResult object
        """
        if symbols is None:
            symbols = config.instruments.symbols

        print("=" * 60)
        print("BACKTESTING ENGINE")
        print("=" * 60)
        print(f"\nSymbols: {', '.join(symbols)}")
        print(f"Lookback: {config.backtest.lookback_days} days")
        print(f"Initial Capital: ${config.backtest.initial_capital:,.2f}")
        print(f"Strategy: {self.strategy.name}")

        # Initialize data manager
        print("\n[*] Initializing data...")
        self.data_manager = DataManager()
        await self.data_manager.initialize()

        # Download historical data for all symbols
        await self.data_manager.download_all_historical_data(days=config.backtest.lookback_days)

        # Run simulation
        print("\n[*] Running simulation...")
        equity_curve = await self._simulate(symbols, start_date, end_date)

        # Calculate statistics
        print("\n[*] Calculating statistics...")
        statistics = self._calculate_statistics(equity_curve)

        # Create result object
        self.results = BacktestResult(
            trades=self.risk_manager.closed_trades,
            equity_curve=equity_curve,
            statistics=statistics,
            parameters=self.strategy.get_strategy_info()
        )

        # Print summary
        self._print_summary()

        # Cleanup
        await self.data_manager.cleanup()

        return self.results

    async def _simulate(
        self,
        symbols: List[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Run the trading simulation

        Args:
            symbols: List of symbols to trade
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with equity curve
        """
        # Initialize tracking variables
        equity = config.backtest.initial_capital
        peak_equity = equity  # Track peak equity for drawdown calculation
        equity_history = []

        # Get all historical data
        all_data = {}
        for symbol in symbols:
            df = self.data_manager.historical_data[symbol].copy()
            df = self.data_manager.calculate_indicators(df)
            all_data[symbol] = df

        # Get common date range
        min_len = min(len(df) for df in all_data.values())

        # Simulate bar by bar
        for i in range(config.strategy.ema_slow + 10, min_len):
            current_time = None
            current_prices = {}

            # Process each symbol
            for symbol in symbols:
                df = all_data[symbol]
                current_bar = df.iloc[i]
                current_time = current_bar.name if isinstance(current_bar.name, datetime) else datetime.now()
                current_price = current_bar['close']
                current_prices[symbol] = current_price

                # Update existing positions
                if symbol in self.risk_manager.positions:
                    position = self.risk_manager.positions[symbol]

                    # Check if should close position
                    should_close, reason = self.strategy.should_close_position(
                        position_side=position.direction,
                        entry_price=position.entry_price,
                        current_price=current_price,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        df=df.iloc[:i+1]
                    )

                    if should_close:
                        # Apply slippage
                        exit_price = self._apply_slippage(
                            current_price,
                            position.direction,
                            'exit'
                        )

                        trade = self.risk_manager.close_position(
                            symbol=symbol,
                            exit_price=exit_price,
                            exit_reason=reason,
                            exit_time=current_time
                        )

                        if trade:
                            equity += trade.net_pnl

                # Check for new signals (only if no position)
                elif symbol not in self.risk_manager.positions:
                    # Get signal from strategy
                    signal = self.strategy.analyze(symbol, df.iloc[:i+1])

                    if signal:
                        # Check if can open position
                        can_open, reason = self.risk_manager.can_open_position(signal)

                        if can_open:
                            # Calculate position size
                            signal.size = self.risk_manager.calculate_position_size(signal, equity)

                            # Apply slippage to entry
                            entry_price = self._apply_slippage(
                                signal.entry_price,
                                signal.direction,
                                'entry'
                            )

                            # Update signal with slippage
                            signal.entry_price = entry_price

                            # Open position
                            self.risk_manager.open_position(signal)

            # Update positions with current prices
            self.risk_manager.update_positions(current_prices)

            # Calculate current equity (capital + unrealized P&L)
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.risk_manager.positions.values())
            current_equity = equity + unrealized_pnl

            # Update peak equity
            if current_equity > peak_equity:
                peak_equity = current_equity

            # Check for drawdown limit
            should_close_dd, reason_dd = self.risk_manager.should_close_for_drawdown(current_equity, peak_equity)

            # Check for end-of-day closure
            should_close_eod, reason_eod = self.risk_manager.should_close_for_eod(current_time)

            # Close all positions if drawdown or EOD trigger
            if should_close_dd or should_close_eod:
                reason = reason_dd if should_close_dd else reason_eod
                # Close all open positions
                positions_to_close = list(self.risk_manager.positions.keys())
                for symbol in positions_to_close:
                    position = self.risk_manager.positions[symbol]
                    current_price = current_prices.get(symbol, position.entry_price)

                    # Apply slippage
                    exit_price = self._apply_slippage(
                        current_price,
                        position.direction,
                        'exit'
                    )

                    trade = self.risk_manager.close_position(
                        symbol=symbol,
                        exit_price=exit_price,
                        exit_reason=reason,
                        exit_time=current_time
                    )

                    if trade:
                        equity += trade.net_pnl

            # Record equity
            equity_history.append({
                'timestamp': current_time,
                'equity': current_equity,
                'cash': equity,
                'unrealized_pnl': unrealized_pnl,
                'open_positions': len(self.risk_manager.positions)
            })

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_history)
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1

        return equity_df

    def _apply_slippage(self, price: float, direction: str, action: str) -> float:
        """
        Apply slippage to price

        Args:
            price: Original price
            direction: Trade direction ('BUY' or 'SELL')
            action: 'entry' or 'exit'

        Returns:
            Price with slippage applied
        """
        slippage_ticks = config.backtest.slippage_ticks
        tick_size = 0.25  # Standard tick size for micro contracts

        if (direction == 'BUY' and action == 'entry') or (direction == 'SELL' and action == 'exit'):
            # Pay the ask - slippage hurts
            return price + (slippage_ticks * tick_size)
        else:
            # Sell at bid - slippage hurts
            return price - (slippage_ticks * tick_size)

    def _calculate_statistics(self, equity_curve: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive backtest statistics

        Args:
            equity_curve: DataFrame with equity history

        Returns:
            Dictionary of statistics
        """
        trades = self.risk_manager.closed_trades

        if not trades:
            return {'error': 'No trades executed'}

        # Basic trade statistics
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]

        total_wins = sum(t.net_pnl for t in winning_trades)
        total_losses = abs(sum(t.net_pnl for t in losing_trades))

        # Equity curve statistics
        final_equity = equity_curve['equity'].iloc[-1]
        initial_equity = config.backtest.initial_capital
        total_return = (final_equity - initial_equity) / initial_equity * 100

        # Calculate drawdown
        equity_curve['cummax'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['cummax']) / equity_curve['cummax'] * 100
        max_drawdown = equity_curve['drawdown'].min()

        # Calculate Sharpe Ratio (annualized)
        returns = equity_curve['returns'].dropna()
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate win/loss streaks
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for trade in trades:
            if trade.is_winner:
                if streak >= 0:
                    streak += 1
                else:
                    streak = 1
                max_win_streak = max(max_win_streak, streak)
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    streak = -1
                max_loss_streak = max(max_loss_streak, abs(streak))

        # Duration statistics
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        avg_duration = np.mean(durations)

        statistics = {
            # Overall Performance
            'initial_capital': initial_equity,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_pnl': final_equity - initial_equity,

            # Trade Statistics
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': len(winning_trades) / len(trades) * 100 if trades else 0,

            # P&L Statistics
            'gross_profit': total_wins,
            'gross_loss': total_losses,
            'net_profit': total_wins - total_losses,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),

            # Trade Averages
            'avg_win': total_wins / len(winning_trades) if winning_trades else 0,
            'avg_loss': total_losses / len(losing_trades) if losing_trades else 0,
            'avg_trade': sum(t.net_pnl for t in trades) / len(trades),
            'avg_duration_hours': avg_duration,

            # Extremes
            'largest_win': max([t.net_pnl for t in winning_trades]) if winning_trades else 0,
            'largest_loss': min([t.net_pnl for t in losing_trades]) if losing_trades else 0,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,

            # Risk Metrics
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': (len(winning_trades) / len(trades) * (total_wins / len(winning_trades)) -
                          len(losing_trades) / len(trades) * (total_losses / len(losing_trades))) if trades else 0,

            # Costs
            'total_commission': sum(t.commission for t in trades),
        }

        return statistics

    def _print_summary(self):
        """Print backtest summary"""
        if not self.results:
            return

        stats = self.results.statistics

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)

        print(f"\n[*] Performance:")
        print(f"  Initial Capital:     ${stats['initial_capital']:>12,.2f}")
        print(f"  Final Equity:        ${stats['final_equity']:>12,.2f}")
        print(f"  Total Return:        {stats['total_return_pct']:>12.2f}%")
        print(f"  Net Profit:          ${stats['net_profit']:>12,.2f}")

        print(f"\n[*] Trade Statistics:")
        print(f"  Total Trades:        {stats['total_trades']:>12}")
        print(f"  Winning Trades:      {stats['winning_trades']:>12}")
        print(f"  Losing Trades:       {stats['losing_trades']:>12}")
        print(f"  Win Rate:            {stats['win_rate_pct']:>12.2f}%")

        print(f"\n[*] P&L Analysis:")
        print(f"  Gross Profit:        ${stats['gross_profit']:>12,.2f}")
        print(f"  Gross Loss:          ${stats['gross_loss']:>12,.2f}")
        print(f"  Profit Factor:       {stats['profit_factor']:>12.2f}")
        print(f"  Average Win:         ${stats['avg_win']:>12,.2f}")
        print(f"  Average Loss:        ${stats['avg_loss']:>12,.2f}")
        print(f"  Average Trade:       ${stats['avg_trade']:>12,.2f}")

        print(f"\n[!] Risk Metrics:")
        print(f"  Max Drawdown:        {stats['max_drawdown_pct']:>12.2f}%")
        print(f"  Sharpe Ratio:        {stats['sharpe_ratio']:>12.2f}")
        print(f"  Expectancy:          ${stats['expectancy']:>12,.2f}")

        print(f"\n[*] Streaks:")
        print(f"  Max Win Streak:      {stats['max_win_streak']:>12}")
        print(f"  Max Loss Streak:     {stats['max_loss_streak']:>12}")

        print(f"\n[*] Costs:")
        print(f"  Total Commission:    ${stats['total_commission']:>12,.2f}")

        print("\n" + "=" * 60)

    def generate_report(self, output_dir: str = 'backtest_results'):
        """
        Generate comprehensive backtest report with charts

        Args:
            output_dir: Directory to save report files
        """
        if not self.results:
            print("[ERROR] No backtest results available. Run backtest first.")
            return

        print(f"\n[*] Generating backtest report...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save trades to CSV
        if self.results.trades:
            trades_df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'direction': t.direction,
                    'size': t.size,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'pnl': t.pnl,
                    'commission': t.commission,
                    'net_pnl': t.net_pnl,
                    'exit_reason': t.exit_reason
                }
                for t in self.results.trades
            ])
            trades_file = output_path / f'trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"  [OK] Trades saved to {trades_file}")

        # 2. Save equity curve
        equity_file = output_path / f'equity_curve_{timestamp}.csv'
        self.results.equity_curve.to_csv(equity_file)
        print(f"  [OK] Equity curve saved to {equity_file}")

        # 3. Generate charts
        if config.backtest.generate_charts:
            self._generate_charts(output_path, timestamp)

        print(f"\n[SUCCESS] Report generated in {output_dir}/")

    def _generate_charts(self, output_path: Path, timestamp: str):
        """
        Generate performance charts

        Args:
            output_path: Output directory
            timestamp: Timestamp for filenames
        """
        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (15, 10)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Backtest Results - {self.strategy.name}', fontsize=16, fontweight='bold')

        # 1. Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(self.results.equity_curve['timestamp'], self.results.equity_curve['equity'], linewidth=2)
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=config.backtest.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.legend()

        # 2. Drawdown
        ax2 = axes[0, 1]
        ax2.fill_between(
            self.results.equity_curve['timestamp'],
            self.results.equity_curve['drawdown'],
            0,
            alpha=0.3,
            color='red'
        )
        ax2.plot(self.results.equity_curve['timestamp'], self.results.equity_curve['drawdown'], color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Trade P&L Distribution
        ax3 = axes[1, 0]
        pnls = [t.net_pnl for t in self.results.trades]
        ax3.hist(pnls, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative P&L
        ax4 = axes[1, 1]
        cumulative_pnl = np.cumsum([t.net_pnl for t in self.results.trades])
        ax4.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        ax4.set_title('Cumulative P&L by Trade')
        ax4.set_xlabel('Trade Number')
        ax4.set_ylabel('Cumulative P&L ($)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # 5. Win/Loss by Symbol
        ax5 = axes[2, 0]
        symbol_pnl = {}
        for trade in self.results.trades:
            if trade.symbol not in symbol_pnl:
                symbol_pnl[trade.symbol] = 0
            symbol_pnl[trade.symbol] += trade.net_pnl

        symbols = list(symbol_pnl.keys())
        pnls = list(symbol_pnl.values())
        colors = ['green' if p > 0 else 'red' for p in pnls]

        ax5.bar(symbols, pnls, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_title('P&L by Symbol')
        ax5.set_xlabel('Symbol')
        ax5.set_ylabel('Net P&L ($)')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 6. Monthly Returns
        ax6 = axes[2, 1]
        equity_df = self.results.equity_curve.copy()
        equity_df['month'] = pd.to_datetime(equity_df['timestamp']).dt.to_period('M')
        monthly_returns = equity_df.groupby('month')['returns'].sum() * 100

        colors = ['green' if r > 0 else 'red' for r in monthly_returns.values]
        ax6.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_title('Monthly Returns')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Return (%)')
        ax6.set_xticks(range(len(monthly_returns)))
        ax6.set_xticklabels([str(m) for m in monthly_returns.index], rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Adjust layout and save
        plt.tight_layout()
        chart_file = output_path / f'backtest_charts_{timestamp}.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"  [OK] Charts saved to {chart_file}")
        plt.close()


if __name__ == "__main__":
    # This will be called from run_backtest.py
    pass
