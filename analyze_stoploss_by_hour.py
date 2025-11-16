"""
Analisi Stop Loss per Ora

Identifica se ci sono ore specifiche del giorno con eccesso di stop loss.
Aiuta a capire se ci sono periodi particolarmente volatili o problematici.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def analyze_stoploss_by_hour(trades_df):
    """Analyze stop loss patterns by hour of day."""

    # Convert times to datetime
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    # Extract hour
    trades_df['entry_hour'] = trades_df['entry_time'].dt.hour
    trades_df['entry_dayofweek'] = trades_df['entry_time'].dt.dayofweek

    # Identify stop loss trades
    trades_df['is_stop_loss'] = trades_df['exit_reason'] == 'STOP_LOSS'

    # Overall statistics
    total_trades = len(trades_df)
    total_sl = trades_df['is_stop_loss'].sum()
    sl_percentage = (total_sl / total_trades * 100)

    print("=" * 80)
    print("ANALISI STOP LOSS COMPLESSIVA")
    print("=" * 80)
    print(f"\nTotal Trades:        {total_trades}")
    print(f"Stop Loss Trades:    {total_sl}")
    print(f"Stop Loss %:         {sl_percentage:.2f}%")
    print(f"Avg SL Loss:         ${trades_df[trades_df['is_stop_loss']]['pnl'].mean():.2f}")
    print(f"Total SL Loss:       ${trades_df[trades_df['is_stop_loss']]['pnl'].sum():.2f}")

    # Analyze by hour
    hourly_stats = []

    for hour in range(24):
        hour_trades = trades_df[trades_df['entry_hour'] == hour]

        if len(hour_trades) == 0:
            continue

        hour_sl = hour_trades['is_stop_loss'].sum()
        hour_sl_pct = (hour_sl / len(hour_trades) * 100) if len(hour_trades) > 0 else 0
        hour_sl_total_loss = hour_trades[hour_trades['is_stop_loss']]['pnl'].sum()
        hour_sl_avg_loss = hour_trades[hour_trades['is_stop_loss']]['pnl'].mean() if hour_sl > 0 else 0
        hour_win_rate = (hour_trades['is_win'].sum() / len(hour_trades) * 100) if len(hour_trades) > 0 else 0
        hour_total_pnl = hour_trades['pnl'].sum()

        hourly_stats.append({
            'hour': hour,
            'total_trades': len(hour_trades),
            'stop_loss_count': hour_sl,
            'stop_loss_pct': hour_sl_pct,
            'stop_loss_total_loss': hour_sl_total_loss,
            'stop_loss_avg_loss': hour_sl_avg_loss,
            'win_rate': hour_win_rate,
            'total_pnl': hour_total_pnl
        })

    hourly_df = pd.DataFrame(hourly_stats).sort_values('hour')

    return hourly_df, trades_df


def identify_problematic_hours(hourly_df, threshold_pct=50):
    """Identify hours with excessive stop losses."""

    print("\n" + "=" * 80)
    print("ORE PROBLEMATICHE (ECCESSO STOP LOSS)")
    print("=" * 80)

    # Find hours with high SL percentage
    problematic = hourly_df[hourly_df['stop_loss_pct'] > threshold_pct].sort_values('stop_loss_pct', ascending=False)

    if len(problematic) > 0:
        print(f"\nOre con > {threshold_pct}% stop loss:\n")
        for _, row in problematic.iterrows():
            print(f"  Ora {int(row['hour']):02d}:00 UTC:")
            print(f"    - Total trades:     {int(row['total_trades'])}")
            print(f"    - Stop loss:        {int(row['stop_loss_count'])} ({row['stop_loss_pct']:.1f}%)")
            print(f"    - Total loss SL:    ${row['stop_loss_total_loss']:.2f}")
            print(f"    - Win rate:         {row['win_rate']:.1f}%")
            print(f"    - Total P&L:        ${row['total_pnl']:.2f}")
            print()
    else:
        print(f"\nNessuna ora con > {threshold_pct}% stop loss.")

    # Find hours with highest total loss from SL
    print("\n" + "=" * 80)
    print("TOP 5 ORE CON MAGGIOR PERDITA DA STOP LOSS")
    print("=" * 80)

    worst_hours = hourly_df.nsmallest(5, 'stop_loss_total_loss')

    print()
    for i, (_, row) in enumerate(worst_hours.iterrows(), 1):
        print(f"{i}. Ora {int(row['hour']):02d}:00 UTC:")
        print(f"   - Stop loss:        {int(row['stop_loss_count'])} ({row['stop_loss_pct']:.1f}%)")
        print(f"   - Total loss SL:    ${row['stop_loss_total_loss']:.2f}")
        print(f"   - Win rate:         {row['win_rate']:.1f}%")
        print()

    return problematic


def plot_stoploss_analysis(hourly_df, trades_df, save_path):
    """Generate comprehensive stop loss visualization."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Stop Loss Percentage by Hour
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['red' if x > 50 else 'orange' if x > 40 else 'yellow' if x > 30 else 'green'
              for x in hourly_df['stop_loss_pct']]
    bars1 = ax1.bar(hourly_df['hour'], hourly_df['stop_loss_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(50, color='red', linestyle='--', linewidth=2, label='50% (Critico)', alpha=0.7)
    ax1.axhline(40, color='orange', linestyle='--', linewidth=1.5, label='40% (Problematico)', alpha=0.5)

    # Add count labels on bars
    for bar, count in zip(bars1, hourly_df['stop_loss_count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'n={int(count)}', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('Ora del Giorno (UTC)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('% di Stop Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Percentuale Stop Loss per Ora del Giorno', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(range(24))

    # 2. Total Loss from SL by Hour
    ax2 = fig.add_subplot(gs[1, 0])
    colors2 = ['darkred' if x < -500 else 'red' if x < -300 else 'orange'
               for x in hourly_df['stop_loss_total_loss']]
    ax2.bar(hourly_df['hour'], hourly_df['stop_loss_total_loss'], color=colors2, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Ora del Giorno (UTC)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perdita Totale da SL ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Perdita Totale da Stop Loss per Ora', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(range(0, 24, 2))

    # 3. Win Rate by Hour
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(hourly_df['hour'], hourly_df['win_rate'], marker='o', linewidth=2, color='#2E86AB', markersize=6)
    ax3.axhline(30, color='gray', linestyle='--', alpha=0.5, label='30% (Min acceptable)')
    ax3.fill_between(hourly_df['hour'], 0, hourly_df['win_rate'],
                     where=hourly_df['win_rate'] < 30, alpha=0.3, color='red', label='< 30% (Problematico)')
    ax3.set_xlabel('Ora del Giorno (UTC)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Win Rate per Ora del Giorno', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 24, 2))

    # 4. Heatmap: Hour vs Day of Week for SL frequency
    ax4 = fig.add_subplot(gs[2, 0])

    # Create pivot table for heatmap
    trades_df['hour'] = trades_df['entry_hour']
    heatmap_data = trades_df[trades_df['is_stop_loss']].groupby(['entry_dayofweek', 'hour']).size().unstack(fill_value=0)

    # Ensure all hours are present
    for hour in range(24):
        if hour not in heatmap_data.columns:
            heatmap_data[hour] = 0
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

    sns.heatmap(heatmap_data, cmap='Reds', annot=True, fmt='d', cbar_kws={'label': 'N. Stop Loss'},
                ax=ax4, linewidths=0.5)
    ax4.set_xlabel('Ora (UTC)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Giorno Settimana', fontsize=12, fontweight='bold')

    # Set day labels dynamically based on actual data
    day_labels = ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom']
    actual_labels = [day_labels[i] for i in heatmap_data.index]
    ax4.set_yticklabels(actual_labels, rotation=0)

    ax4.set_title('Heatmap Stop Loss: Giorno x Ora', fontsize=14, fontweight='bold')

    # 5. Trade Count vs SL Count by Hour
    ax5 = fig.add_subplot(gs[2, 1])
    x = hourly_df['hour']
    width = 0.35

    bars1 = ax5.bar(x - width/2, hourly_df['total_trades'], width, label='Total Trades', alpha=0.6, color='blue')
    bars2 = ax5.bar(x + width/2, hourly_df['stop_loss_count'], width, label='Stop Loss', alpha=0.6, color='red')

    ax5.set_xlabel('Ora del Giorno (UTC)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax5.set_title('Confronto: Total Trades vs Stop Loss per Ora', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xticks(range(0, 24, 2))

    plt.suptitle('Analisi Completa Stop Loss per Ora', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[SAVED] {save_path}")
    plt.close()


def main():
    print("=" * 80)
    print("ANALISI STOP LOSS PER ORA")
    print("=" * 80)

    # Load trade log
    trade_log_path = Path("reports/trade_log.csv")

    if not trade_log_path.exists():
        print(f"\n[ERROR] File not found: {trade_log_path}")
        print("Run 'python visualize_baseline_strategy.py' first to generate trade log.")
        return

    print(f"\n[*] Loading trade log: {trade_log_path}")
    trades_df = pd.read_csv(trade_log_path)

    # Analyze stop losses by hour
    print("[*] Analyzing stop loss patterns by hour...")
    hourly_df, trades_df = analyze_stoploss_by_hour(trades_df)

    # Identify problematic hours
    print("[*] Identifying problematic hours...")
    problematic_hours = identify_problematic_hours(hourly_df, threshold_pct=45)

    # Generate visualization
    print("\n[*] Generating visualizations...")
    save_path = Path("reports/stoploss_analysis.png")
    plot_stoploss_analysis(hourly_df, trades_df, save_path)

    # Save hourly statistics to CSV
    hourly_stats_path = Path("reports/stoploss_hourly_stats.csv")
    hourly_df.to_csv(hourly_stats_path, index=False)
    print(f"[SAVED] {hourly_stats_path}")

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("RACCOMANDAZIONI")
    print("=" * 80)

    avg_sl_pct = hourly_df['stop_loss_pct'].mean()
    print(f"\nMedia stop loss %: {avg_sl_pct:.1f}%")

    if len(problematic_hours) > 0:
        print("\nOre con eccesso di stop loss identificate!")
        print("Considera di:")
        print("  1. Escludere queste ore dal trading (time filter)")
        print("  2. Usare stop loss più ampi in queste ore")
        print("  3. Ridurre la size in queste ore")
        print("\nOre consigliate da escludere:")
        for _, row in problematic_hours.iterrows():
            print(f"  - Ora {int(row['hour']):02d}:00 UTC ({row['stop_loss_pct']:.1f}% SL)")
    else:
        print("\nNessuna ora con eccesso critico di stop loss.")
        print("La distribuzione degli SL sembra uniforme.")

    print("\n" + "=" * 80)
    print("NOTA: Ore 14, 18, 19 UTC sono già ESCLUSE dal time filter!")
    print("=" * 80)


if __name__ == "__main__":
    main()
