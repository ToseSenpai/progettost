"""
Trade Analysis Visualization Module

Generates comprehensive visualizations for trading strategy analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


class TradeAnalysisPlots:
    """Generate visualizations for trade analysis."""

    def __init__(self, trades_df: pd.DataFrame, equity_curve: pd.Series):
        """
        Initialize plot generator.

        Args:
            trades_df: DataFrame with trade data
            equity_curve: Series of equity over time
        """
        self.trades_df = trades_df
        self.equity_curve = equity_curve

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve with drawdown overlay."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Equity curve
        ax1.plot(self.equity_curve.index, self.equity_curve.values, linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.equity_curve.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
        ax1.fill_between(self.equity_curve.index, self.equity_curve.iloc[0], self.equity_curve.values,
                         where=self.equity_curve.values >= self.equity_curve.iloc[0],
                         alpha=0.3, color='green', label='Profit')
        ax1.fill_between(self.equity_curve.index, self.equity_curve.iloc[0], self.equity_curve.values,
                         where=self.equity_curve.values < self.equity_curve.iloc[0],
                         alpha=0.3, color='red', label='Loss')
        ax1.set_ylabel('Capital ($)')
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        running_peak = pd.Series(self.equity_curve).expanding().max()
        drawdown = (self.equity_curve - running_peak) / running_peak * 100

        ax2.fill_between(drawdown.index, 0, drawdown.values, alpha=0.5, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1.5)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Trade Number')
        ax2.set_title('Drawdown from Peak', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Equity curve: {save_path}")

        plt.close()

    def plot_pnl_distribution(self, save_path: Optional[str] = None):
        """Plot P&L distribution histogram."""
        fig, ax = plt.subplots(figsize=(12, 7))

        # Separate wins and losses
        wins = self.trades_df[self.trades_df['pnl'] > 0]['pnl']
        losses = self.trades_df[self.trades_df['pnl'] <= 0]['pnl']

        # Plot histograms
        ax.hist(wins, bins=50, alpha=0.6, color='green', label=f'Wins (n={len(wins)})', edgecolor='black')
        ax.hist(losses, bins=50, alpha=0.6, color='red', label=f'Losses (n={len(losses)})', edgecolor='black')

        # Add mean lines
        if len(wins) > 0:
            ax.axvline(wins.mean(), color='darkgreen', linestyle='--', linewidth=2,
                      label=f'Avg Win: ${wins.mean():.2f}')
        if len(losses) > 0:
            ax.axvline(losses.mean(), color='darkred', linestyle='--', linewidth=2,
                      label=f'Avg Loss: ${losses.mean():.2f}')

        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] P&L distribution: {save_path}")

        plt.close()

    def plot_exit_reasons(self, save_path: Optional[str] = None):
        """Plot exit reason breakdown."""
        exit_counts = self.trades_df['exit_reason'].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pie chart
        colors = plt.cm.Set3(range(len(exit_counts)))
        ax1.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Exit Reason Distribution', fontsize=14, fontweight='bold')

        # Bar chart with win rates
        exit_stats = []
        for reason in exit_counts.index:
            trades = self.trades_df[self.trades_df['exit_reason'] == reason]
            win_rate = (trades['pnl'] > 0).sum() / len(trades) * 100
            exit_stats.append({
                'reason': reason,
                'count': len(trades),
                'win_rate': win_rate
            })

        exit_df = pd.DataFrame(exit_stats).sort_values('count', ascending=False)

        bars = ax2.bar(range(len(exit_df)), exit_df['win_rate'], color=colors)
        ax2.set_xticks(range(len(exit_df)))
        ax2.set_xticklabels(exit_df['reason'], rotation=45, ha='right')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate by Exit Reason', fontsize=14, fontweight='bold')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Break-even')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, exit_df['count'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'n={count}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Exit reasons: {save_path}")

        plt.close()

    def plot_hourly_performance(self, save_path: Optional[str] = None):
        """Plot performance by hour of day."""
        # Calculate stats by hour
        if 'entry_hour' not in self.trades_df.columns:
            self.trades_df['entry_hour'] = self.trades_df['entry_time'].dt.hour

        hourly_stats = self.trades_df.groupby('entry_hour').agg({
            'pnl': ['sum', 'mean', 'count'],
        }).reset_index()
        hourly_stats.columns = ['hour', 'total_pnl', 'avg_pnl', 'count']

        # Calculate win rate by hour
        win_rates = []
        for hour in hourly_stats['hour']:
            trades = self.trades_df[self.trades_df['entry_hour'] == hour]
            wr = (trades['pnl'] > 0).sum() / len(trades) * 100 if len(trades) > 0 else 0
            win_rates.append(wr)
        hourly_stats['win_rate'] = win_rates

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Total P&L by hour
        colors = ['green' if x > 0 else 'red' for x in hourly_stats['total_pnl']]
        ax1.bar(hourly_stats['hour'], hourly_stats['total_pnl'], color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_ylabel('Total P&L ($)')
        ax1.set_title('Total P&L by Hour of Day (UTC)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Win rate by hour
        ax2.plot(hourly_stats['hour'], hourly_stats['win_rate'], marker='o', linewidth=2, color='#2E86AB')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Break-even')
        ax2.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='40% Target')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate by Hour', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Trade count by hour
        ax3.bar(hourly_stats['hour'], hourly_stats['count'], color='steelblue', alpha=0.7)
        ax3.set_xlabel('Hour of Day (UTC)')
        ax3.set_ylabel('Number of Trades')
        ax3.set_title('Trade Frequency by Hour', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Hourly performance: {save_path}")

        plt.close()

    def plot_trade_duration(self, save_path: Optional[str] = None):
        """Plot trade duration analysis."""
        if 'duration_minutes' not in self.trades_df.columns:
            self.trades_df['duration_minutes'] = (
                (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / 60
            )

        winners = self.trades_df[self.trades_df['pnl'] > 0]['duration_minutes']
        losers = self.trades_df[self.trades_df['pnl'] <= 0]['duration_minutes']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram comparison
        bins = np.linspace(0, max(self.trades_df['duration_minutes'].max(), 100), 40)
        ax1.hist(winners, bins=bins, alpha=0.6, color='green', label=f'Winners (avg: {winners.mean():.1f} min)')
        ax1.hist(losers, bins=bins, alpha=0.6, color='red', label=f'Losers (avg: {losers.mean():.1f} min)')
        ax1.set_xlabel('Duration (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Box plot comparison
        data_to_plot = [winners.dropna(), losers.dropna()]
        bp = ax2.boxplot(data_to_plot, labels=['Winners', 'Losers'], patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax2.set_ylabel('Duration (minutes)')
        ax2.set_title('Duration Comparison: Winners vs Losers', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Trade duration: {save_path}")

        plt.close()

    def plot_rolling_win_rate(self, window: int = 50, save_path: Optional[str] = None):
        """Plot rolling win rate over time."""
        # Calculate rolling win rate
        is_win = (self.trades_df['pnl'] > 0).astype(int)
        rolling_wr = is_win.rolling(window=window, min_periods=1).mean() * 100

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(range(len(rolling_wr)), rolling_wr, linewidth=2, color='#2E86AB', label=f'{window}-trade Rolling WR')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Break-even')
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='40% Target')

        overall_wr = (self.trades_df['pnl'] > 0).sum() / len(self.trades_df) * 100
        ax.axhline(y=overall_wr, color='red', linestyle='-', alpha=0.7,
                  label=f'Overall WR: {overall_wr:.1f}%')

        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title(f'{window}-Trade Rolling Win Rate', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Rolling win rate: {save_path}")

        plt.close()

    def plot_cumulative_pnl(self, save_path: Optional[str] = None):
        """Plot cumulative P&L over time."""
        cumulative_pnl = self.trades_df['pnl'].cumsum()

        fig, ax = plt.subplots(figsize=(14, 7))

        ax.plot(range(len(cumulative_pnl)), cumulative_pnl.values, linewidth=2, color='#2E86AB')
        ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                        where=cumulative_pnl.values >= 0, alpha=0.3, color='green')
        ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values,
                        where=cumulative_pnl.values < 0, alpha=0.3, color='red')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.set_title('Cumulative P&L Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SAVED] Cumulative P&L: {save_path}")

        plt.close()

    def create_all_plots(self, output_dir: str = "reports"):
        """Generate all plots and save to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n[*] Generating visualizations...")

        self.plot_equity_curve(str(output_path / "equity_curve.png"))
        self.plot_pnl_distribution(str(output_path / "pnl_distribution.png"))
        self.plot_exit_reasons(str(output_path / "exit_reasons.png"))
        self.plot_hourly_performance(str(output_path / "hourly_performance.png"))
        self.plot_trade_duration(str(output_path / "trade_duration.png"))
        self.plot_rolling_win_rate(50, str(output_path / "rolling_win_rate.png"))
        self.plot_cumulative_pnl(str(output_path / "cumulative_pnl.png"))

        print(f"\n[OK] All visualizations saved to {output_path}/")
