"""
Symbol Comparison Script
Runs backtests on each symbol individually to identify the most profitable instruments
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtester import Backtester
from src.config import config
import copy


class SymbolComparator:
    """Compare performance across multiple trading symbols"""

    def __init__(self):
        self.results = []
        self.original_symbols = config.instruments.symbols.copy()

    async def test_symbol(self, symbol):
        """
        Run backtest for a single symbol

        Args:
            symbol: Symbol to test

        Returns:
            Dictionary with symbol and performance metrics
        """
        print(f"\n[*] Testing {symbol}...")

        # Temporarily set config to test only this symbol
        config.instruments.symbols = [symbol]

        try:
            # Run backtest
            backtester = Backtester()
            results = await backtester.run_backtest()

            # Extract key metrics
            stats = results.statistics

            return {
                'symbol': symbol,
                'total_return_pct': stats['total_return_pct'],
                'sharpe_ratio': stats['sharpe_ratio'],
                'profit_factor': stats['profit_factor'],
                'win_rate_pct': stats['win_rate_pct'],
                'total_trades': stats['total_trades'],
                'winning_trades': stats['winning_trades'],
                'losing_trades': stats['losing_trades'],
                'avg_win': stats['avg_win'],
                'avg_loss': stats['avg_loss'],
                'max_drawdown_pct': stats['max_drawdown_pct'],
                'expectancy': stats['expectancy'],
                'total_pnl': stats['total_pnl'],
                'final_equity': stats['final_equity'],
                'initial_capital': stats['initial_capital']
            }

        except Exception as e:
            print(f"    >> Error testing {symbol}: {e}")
            return {
                'symbol': symbol,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0,
                'win_rate_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown_pct': 0,
                'expectancy': 0,
                'total_pnl': 0,
                'final_equity': config.backtest.initial_capital,
                'initial_capital': config.backtest.initial_capital,
                'error': str(e)
            }

        finally:
            # Restore original symbols
            config.instruments.symbols = self.original_symbols.copy()

    async def compare_all(self):
        """Run comparison on all symbols"""
        print("\n" + "=" * 60)
        print("     SYMBOL PROFITABILITY COMPARISON")
        print("=" * 60)
        print(f"\nTesting {len(self.original_symbols)} symbols...")
        print(f"Symbols: {', '.join(self.original_symbols)}")
        print(f"Lookback: {config.backtest.lookback_days} days")
        print(f"Initial Capital: ${config.backtest.initial_capital:,.2f}")
        print("\n" + "=" * 60)

        # Test each symbol
        for symbol in self.original_symbols:
            result = await self.test_symbol(symbol)
            self.results.append(result)

        # Generate report
        self.print_results()
        self.save_results()

        return self.results

    def print_results(self):
        """Print comparison results"""
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)

        # Convert to DataFrame for easy sorting
        df = pd.DataFrame(self.results)

        # Sort by total return
        df_sorted = df.sort_values('total_return_pct', ascending=False)

        print("\n[*] Ranking by Total Return:")
        print("\n" + "-" * 80)
        print(f"{'Rank':<6} {'Symbol':<8} {'Return %':<12} {'Sharpe':<10} {'P.Factor':<10} {'Win %':<10} {'Trades':<8}")
        print("-" * 80)

        for i, row in enumerate(df_sorted.itertuples(), 1):
            print(f"{i:<6} {row.symbol:<8} {row.total_return_pct:>10.2f}%  "
                  f"{row.sharpe_ratio:>8.2f}  {row.profit_factor:>8.2f}  "
                  f"{row.win_rate_pct:>8.2f}%  {row.total_trades:>6}")

        print("-" * 80)

        # Top 3 performers
        print("\n[*] TOP 3 PERFORMERS:")
        for i, row in enumerate(df_sorted.head(3).itertuples(), 1):
            print(f"\n  #{i} - {row.symbol}")
            print(f"      Total Return: {row.total_return_pct:.2f}%")
            print(f"      Sharpe Ratio: {row.sharpe_ratio:.2f}")
            print(f"      Profit Factor: {row.profit_factor:.2f}")
            print(f"      Win Rate: {row.win_rate_pct:.2f}%")
            print(f"      Total Trades: {row.total_trades}")
            print(f"      Expectancy: ${row.expectancy:.2f}")
            print(f"      Max Drawdown: {row.max_drawdown_pct:.2f}%")

        # Bottom performers
        print("\n[*] WORST PERFORMERS:")
        for i, row in enumerate(df_sorted.tail(3).itertuples(), 1):
            print(f"\n  {row.symbol}: {row.total_return_pct:.2f}%")
            if 'error' in row._fields and hasattr(row, 'error'):
                print(f"      Error: {row.error}")

        # Additional rankings
        print("\n" + "=" * 60)
        print("ALTERNATIVE RANKINGS")
        print("=" * 60)

        # By Sharpe Ratio
        print("\n[*] Best Risk-Adjusted Returns (Sharpe Ratio):")
        df_sharpe = df.sort_values('sharpe_ratio', ascending=False).head(5)
        for i, row in enumerate(df_sharpe.itertuples(), 1):
            print(f"  {i}. {row.symbol}: {row.sharpe_ratio:.2f}")

        # By Profit Factor
        print("\n[*] Best Profit Factor:")
        df_pf = df.sort_values('profit_factor', ascending=False).head(5)
        for i, row in enumerate(df_pf.itertuples(), 1):
            print(f"  {i}. {row.symbol}: {row.profit_factor:.2f}")

        # By Win Rate
        print("\n[*] Highest Win Rate:")
        df_wr = df.sort_values('win_rate_pct', ascending=False).head(5)
        for i, row in enumerate(df_wr.itertuples(), 1):
            print(f"  {i}. {row.symbol}: {row.win_rate_pct:.2f}%")

        # Composite Score
        print("\n[*] Composite Score (Balanced):")
        df['composite_score'] = (
            df['total_return_pct'] * 0.3 +
            df['sharpe_ratio'] * 10 * 0.3 +
            (df['profit_factor'] - 1) * 10 * 0.2 +
            df['win_rate_pct'] * 0.2
        )
        df_composite = df.sort_values('composite_score', ascending=False).head(5)
        for i, row in enumerate(df_composite.itertuples(), 1):
            print(f"  {i}. {row.symbol}: {row.composite_score:.2f}")

        print("\n" + "=" * 60)

    def save_results(self):
        """Save comparison results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create comparison results directory
        results_dir = Path('comparison_results')
        results_dir.mkdir(exist_ok=True)

        # Save detailed CSV
        csv_file = results_dir / f'symbol_comparison_{timestamp}.csv'
        df = pd.DataFrame(self.results)
        df = df.sort_values('total_return_pct', ascending=False)
        df.to_csv(csv_file, index=False)
        print(f"\n[OK] Detailed results saved to {csv_file}")

        # Save summary report
        report_file = results_dir / f'comparison_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SYMBOL PROFITABILITY COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Lookback: {config.backtest.lookback_days} days\n")
            f.write(f"Initial Capital: ${config.backtest.initial_capital:,.2f}\n\n")

            f.write("RANKING BY TOTAL RETURN\n")
            f.write("-" * 60 + "\n")

            for i, row in enumerate(df.itertuples(), 1):
                f.write(f"\n#{i} - {row.symbol}\n")
                f.write(f"  Total Return: {row.total_return_pct:.2f}%\n")
                f.write(f"  Sharpe Ratio: {row.sharpe_ratio:.2f}\n")
                f.write(f"  Profit Factor: {row.profit_factor:.2f}\n")
                f.write(f"  Win Rate: {row.win_rate_pct:.2f}%\n")
                f.write(f"  Total Trades: {row.total_trades}\n")
                f.write(f"  Expectancy: ${row.expectancy:.2f}\n")
                f.write(f"  Max Drawdown: {row.max_drawdown_pct:.2f}%\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("=" * 60 + "\n\n")

            # Get top 3
            top_3 = df.head(3)
            f.write("Focus on these top 3 most profitable symbols:\n\n")
            for i, row in enumerate(top_3.itertuples(), 1):
                f.write(f"  {i}. {row.symbol} ({row.total_return_pct:.2f}% return)\n")

        print(f"[OK] Summary report saved to {report_file}")


async def main():
    """Main comparison function"""
    print("\n")
    print("=" * 60)
    print("     PROJECTX TRADING BOT - SYMBOL COMPARISON")
    print("=" * 60)
    print()

    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        print("\nPlease check your .env file and config.py")
        return

    # Create comparator
    comparator = SymbolComparator()

    try:
        # Run comparison
        results = await comparator.compare_all()

        print("\n" + "=" * 60)
        print("[SUCCESS] Symbol comparison completed!")
        print("=" * 60)

        # Show recommendation
        df = pd.DataFrame(results)
        df_sorted = df.sort_values('total_return_pct', ascending=False)
        top_symbols = df_sorted.head(3)['symbol'].tolist()

        print(f"\n[*] RECOMMENDATION:")
        print(f"\n    Focus trading on these symbols:")
        for i, symbol in enumerate(top_symbols, 1):
            return_pct = df_sorted[df_sorted['symbol'] == symbol]['total_return_pct'].values[0]
            print(f"    {i}. {symbol} ({return_pct:.2f}% return)")

        print(f"\n[*] To update config, edit src/config.py and set:")
        print(f"    symbols = {top_symbols}")

    except KeyboardInterrupt:
        print("\n\n[STOP] Comparison interrupted by user")

    except Exception as e:
        print(f"\n[ERROR] Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("\n[ERROR] .env file not found!")
        print("\nPlease create a .env file with your ProjectX credentials")
        sys.exit(1)

    # Run comparison
    asyncio.run(main())
