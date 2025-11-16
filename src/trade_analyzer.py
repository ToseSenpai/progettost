"""
Trade Analyzer Module

Provides comprehensive analysis of trading strategy performance.
Analyzes trades to identify strengths, weaknesses, and opportunities for improvement.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class TradeAnalyzer:
    """Analyzes trading performance from backtest results."""

    def __init__(self, trades_df: pd.DataFrame, equity_curve: pd.Series, price_data: pd.DataFrame = None):
        """
        Initialize TradeAnalyzer.

        Args:
            trades_df: DataFrame with columns: entry_time, exit_time, side, entry_price,
                      exit_price, pnl, exit_reason, confidence, etc.
            equity_curve: Series of capital over time
            price_data: Optional DataFrame with price/indicator data for context
        """
        self.trades_df = trades_df.copy()
        self.equity_curve = equity_curve
        self.price_data = price_data

        # Calculate derived metrics
        self._calculate_derived_metrics()

    def _calculate_derived_metrics(self):
        """Calculate additional metrics for each trade."""
        # Trade duration in minutes
        self.trades_df['duration_minutes'] = (
            (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / 60
        )

        # Extract hour and day of week
        self.trades_df['entry_hour'] = self.trades_df['entry_time'].dt.hour
        self.trades_df['entry_day'] = self.trades_df['entry_time'].dt.day_name()
        self.trades_df['entry_dayofweek'] = self.trades_df['entry_time'].dt.dayofweek

        # Win/loss classification
        self.trades_df['is_win'] = self.trades_df['pnl'] > 0
        self.trades_df['is_loss'] = self.trades_df['pnl'] < 0

        # Cumulative P&L
        self.trades_df['cumulative_pnl'] = self.trades_df['pnl'].cumsum()

    def get_summary_stats(self) -> Dict:
        """Get overall performance summary statistics."""
        total_trades = len(self.trades_df)
        winning_trades = self.trades_df[self.trades_df['is_win']]
        losing_trades = self.trades_df[self.trades_df['is_loss']]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': self.trades_df['pnl'].sum(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            'largest_win': self.trades_df['pnl'].max(),
            'largest_loss': self.trades_df['pnl'].min(),
        }

    def analyze_exit_reasons(self) -> pd.DataFrame:
        """Analyze performance by exit reason."""
        exit_stats = []

        for reason in self.trades_df['exit_reason'].unique():
            trades = self.trades_df[self.trades_df['exit_reason'] == reason]

            win_rate = (trades['is_win'].sum() / len(trades) * 100) if len(trades) > 0 else 0
            avg_pnl = trades['pnl'].mean()
            total_pnl = trades['pnl'].sum()

            exit_stats.append({
                'exit_reason': reason,
                'count': len(trades),
                'percentage': len(trades) / len(self.trades_df) * 100,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
            })

        return pd.DataFrame(exit_stats).sort_values('count', ascending=False)

    def analyze_pnl_distribution(self) -> Dict:
        """Analyze P&L distribution characteristics."""
        winning_trades = self.trades_df[self.trades_df['is_win']]
        losing_trades = self.trades_df[self.trades_df['is_loss']]

        return {
            'win_pnl_mean': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'win_pnl_median': winning_trades['pnl'].median() if len(winning_trades) > 0 else 0,
            'win_pnl_std': winning_trades['pnl'].std() if len(winning_trades) > 0 else 0,
            'loss_pnl_mean': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'loss_pnl_median': losing_trades['pnl'].median() if len(losing_trades) > 0 else 0,
            'loss_pnl_std': losing_trades['pnl'].std() if len(losing_trades) > 0 else 0,
            'pnl_skewness': self.trades_df['pnl'].skew(),
            'pnl_kurtosis': self.trades_df['pnl'].kurtosis(),
        }

    def analyze_by_hour(self) -> pd.DataFrame:
        """Analyze performance by hour of day."""
        hourly_stats = []

        for hour in sorted(self.trades_df['entry_hour'].unique()):
            trades = self.trades_df[self.trades_df['entry_hour'] == hour]

            win_rate = (trades['is_win'].sum() / len(trades) * 100) if len(trades) > 0 else 0
            avg_pnl = trades['pnl'].mean()
            total_pnl = trades['pnl'].sum()

            hourly_stats.append({
                'hour': hour,
                'count': len(trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
            })

        return pd.DataFrame(hourly_stats)

    def analyze_by_day(self) -> pd.DataFrame:
        """Analyze performance by day of week."""
        day_stats = []

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for day_idx, day_name in enumerate(day_names):
            trades = self.trades_df[self.trades_df['entry_dayofweek'] == day_idx]

            if len(trades) == 0:
                continue

            win_rate = (trades['is_win'].sum() / len(trades) * 100)
            avg_pnl = trades['pnl'].mean()
            total_pnl = trades['pnl'].sum()

            day_stats.append({
                'day': day_name,
                'count': len(trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
            })

        return pd.DataFrame(day_stats)

    def analyze_trade_duration(self) -> Dict:
        """Analyze trade duration patterns."""
        winning_trades = self.trades_df[self.trades_df['is_win']]
        losing_trades = self.trades_df[self.trades_df['is_loss']]

        return {
            'avg_duration_minutes': self.trades_df['duration_minutes'].mean(),
            'median_duration_minutes': self.trades_df['duration_minutes'].median(),
            'avg_winner_duration': winning_trades['duration_minutes'].mean() if len(winning_trades) > 0 else 0,
            'avg_loser_duration': losing_trades['duration_minutes'].mean() if len(losing_trades) > 0 else 0,
            'min_duration': self.trades_df['duration_minutes'].min(),
            'max_duration': self.trades_df['duration_minutes'].max(),
        }

    def analyze_streaks(self) -> Dict:
        """Analyze winning and losing streaks."""
        # Calculate streaks
        streaks = []
        current_streak = 0
        current_type = None

        for is_win in self.trades_df['is_win']:
            if is_win:
                if current_type == 'win':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append({'type': current_type, 'length': current_streak})
                    current_streak = 1
                    current_type = 'win'
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append({'type': current_type, 'length': current_streak})
                    current_streak = 1
                    current_type = 'loss'

        # Add final streak
        if current_streak > 0:
            streaks.append({'type': current_type, 'length': current_streak})

        streaks_df = pd.DataFrame(streaks)

        win_streaks = streaks_df[streaks_df['type'] == 'win']['length'] if len(streaks_df) > 0 else pd.Series([0])
        loss_streaks = streaks_df[streaks_df['type'] == 'loss']['length'] if len(streaks_df) > 0 else pd.Series([0])

        return {
            'longest_win_streak': win_streaks.max(),
            'longest_loss_streak': loss_streaks.max(),
            'avg_win_streak': win_streaks.mean(),
            'avg_loss_streak': loss_streaks.mean(),
        }

    def analyze_drawdowns(self) -> pd.DataFrame:
        """Analyze drawdown periods."""
        # Calculate running peak
        running_peak = self.equity_curve.expanding().max()

        # Calculate drawdown from peak
        drawdown = (self.equity_curve - running_peak) / running_peak * 100

        # Find drawdown periods > 10%
        in_drawdown = drawdown < -10

        # Identify drawdown periods
        drawdown_periods = []
        start_idx = None

        for idx, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = idx
            elif not is_dd and start_idx is not None:
                # Drawdown period ended
                dd_values = drawdown.iloc[start_idx:idx]
                drawdown_periods.append({
                    'start_index': start_idx,
                    'end_index': idx - 1,
                    'duration_bars': idx - start_idx,
                    'max_drawdown': dd_values.min(),
                })
                start_idx = None

        # If still in drawdown at end
        if start_idx is not None:
            dd_values = drawdown.iloc[start_idx:]
            drawdown_periods.append({
                'start_index': start_idx,
                'end_index': len(drawdown) - 1,
                'duration_bars': len(drawdown) - start_idx,
                'max_drawdown': dd_values.min(),
            })

        return pd.DataFrame(drawdown_periods)

    def get_best_trades(self, n: int = 20) -> pd.DataFrame:
        """Get top N best trades by P&L."""
        return self.trades_df.nlargest(n, 'pnl')

    def get_worst_trades(self, n: int = 20) -> pd.DataFrame:
        """Get top N worst trades by P&L."""
        return self.trades_df.nsmallest(n, 'pnl')

    def analyze_confidence_vs_performance(self) -> pd.DataFrame:
        """Analyze if confidence score correlates with performance."""
        if 'confidence' not in self.trades_df.columns:
            return pd.DataFrame()

        # Bin confidence into quartiles
        self.trades_df['confidence_bin'] = pd.qcut(
            self.trades_df['confidence'],
            q=4,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
            duplicates='drop'
        )

        confidence_stats = []

        for bin_label in self.trades_df['confidence_bin'].unique():
            trades = self.trades_df[self.trades_df['confidence_bin'] == bin_label]

            win_rate = (trades['is_win'].sum() / len(trades) * 100) if len(trades) > 0 else 0
            avg_pnl = trades['pnl'].mean()

            confidence_stats.append({
                'confidence_bin': bin_label,
                'count': len(trades),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': trades['pnl'].sum(),
            })

        return pd.DataFrame(confidence_stats)

    def generate_insights(self) -> Dict:
        """Generate actionable insights from analysis."""
        summary = self.get_summary_stats()
        exit_analysis = self.analyze_exit_reasons()
        hourly_analysis = self.analyze_by_hour()
        duration_analysis = self.analyze_trade_duration()

        insights = {
            'summary': summary,
            'key_findings': [],
            'recommendations': [],
        }

        # Analyze exit patterns
        if len(exit_analysis) > 0:
            stop_loss_pct = exit_analysis[exit_analysis['exit_reason'] == 'STOP_LOSS']['percentage'].values
            if len(stop_loss_pct) > 0 and stop_loss_pct[0] > 60:
                insights['key_findings'].append(
                    f"CRITICAL: {stop_loss_pct[0]:.1f}% of trades hit stop loss (too high)"
                )
                insights['recommendations'].append(
                    "Consider tightening entry criteria (raise min_confluence) or widening stops"
                )

        # Analyze win rate
        if summary['win_rate'] < 38:
            insights['key_findings'].append(
                f"Win rate ({summary['win_rate']:.1f}%) below minimum viable (38%) for 2:1 R/R"
            )
            insights['recommendations'].append(
                "Either improve win rate to 40%+ OR increase R/R target to 3:1"
            )

        # Analyze overtrading
        if summary['total_trades'] > 1000:
            insights['key_findings'].append(
                f"Overtrading detected: {summary['total_trades']} trades (likely too many)"
            )
            insights['recommendations'].append(
                "Increase min_confluence threshold to 0.85-0.90 for higher quality trades"
            )

        # Analyze worst hours
        if len(hourly_analysis) > 0:
            worst_hours = hourly_analysis.nsmallest(3, 'total_pnl')
            for _, row in worst_hours.iterrows():
                if row['total_pnl'] < -500:  # Significant loss
                    insights['key_findings'].append(
                        f"Hour {row['hour']}:00 UTC is toxic (${row['total_pnl']:.0f} loss, {row['win_rate']:.1f}% WR)"
                    )
                    insights['recommendations'].append(
                        f"Consider filtering out trades during hour {row['hour']}:00"
                    )

        # Analyze expectancy
        if summary['expectancy'] < 0:
            insights['key_findings'].append(
                f"Negative expectancy (${summary['expectancy']:.2f} per trade) - strategy is losing"
            )
            insights['recommendations'].append(
                "Strategy requires fundamental changes - consider revising confluence logic"
            )

        return insights
