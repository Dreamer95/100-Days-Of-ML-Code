import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates

def plot_weekly_tpm_analysis():
    """
    Vẽ biểu đồ area chart của TPM và line chart của play_counted 
    cho tuần gần nhất (thứ 5 và thứ 6)
    """
    
    # Load và preprocess data
    print("Loading data...")
    df = pd.read_csv('../../datasets/gamification_traffic_data.csv')
    
    # Chuyển đổi timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_name'] = df['timestamp'].dt.strftime('%A')
    
    print("Available dates in dataset:")
    unique_dates = df['date'].unique()
    for date in sorted(unique_dates):
        day_name = pd.to_datetime(date).strftime('%A')
        print(f"  {date} ({day_name})")
    
    # Lấy 2 ngày gần nhất (thứ 5 và thứ 6 gần nhất)
    # Từ dữ liệu: 2025-05-03 (Saturday) và 2025-05-09 (Friday)
    latest_dates = sorted(unique_dates)[-2:]  # 2 ngày gần nhất
    
    print(f"\nAnalyzing latest week data:")
    for date in latest_dates:
        day_name = pd.to_datetime(date).strftime('%A')
        print(f"  {date} ({day_name})")
    
    # Filter data cho 2 ngày này
    week_data = df[df['date'].isin(latest_dates)].copy()
    week_data = week_data.sort_values('timestamp')
    
    # Tạo figure và subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('TPM và Play Count Analysis - Tuần Gần Nhất', fontsize=16, fontweight='bold')
    
    # Subplot 1: TPM Area Chart với Play Count Line
    ax1_twin = ax1.twinx()  # Tạo trục y thứ 2 cho play_counted
    
    # Vẽ area chart cho TPM
    ax1.fill_between(week_data['timestamp'], week_data['tpm'], 
                     alpha=0.7, color='lightblue', label='TPM')
    ax1.plot(week_data['timestamp'], week_data['tpm'], 
             color='blue', linewidth=2, label='TPM Line')
    
    # Vẽ line chart cho play_counted trên trục y thứ 2
    ax1_twin.plot(week_data['timestamp'], week_data['play_counted'], 
                  color='red', linewidth=2, marker='o', markersize=3,
                  label='Play Counted', alpha=0.8)
    
    # Highlight push notifications
    push_notifications = week_data[week_data['is_push_notification'] == 1]
    if not push_notifications.empty:
        ax1.scatter(push_notifications['timestamp'], push_notifications['tpm'],
                   color='orange', s=100, marker='^', zorder=5,
                   label='Push Notifications')
    
    # Formatting cho subplot 1
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('TPM (Transactions Per Minute)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Play Counted (Cumulative)', fontsize=12, color='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Format x-axis để hiển thị ngày và giờ
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_title('TPM Area Chart với Play Count Trend', fontsize=14)
    
    # Subplot 2: Detailed view by day
    colors = ['lightcoral', 'lightgreen']
    day_names = []
    
    for i, date in enumerate(latest_dates):
        day_data = week_data[week_data['date'] == date]
        day_name = pd.to_datetime(date).strftime('%A, %m/%d')
        day_names.append(day_name)
        
        # Tạo time trong ngày cho x-axis
        day_data_copy = day_data.copy()
        day_data_copy['hour_minute'] = day_data_copy['timestamp'].dt.strftime('%H:%M')
        
        # Vẽ area chart cho từng ngày
        ax2.fill_between(range(len(day_data)), day_data['tpm'], 
                        alpha=0.6, color=colors[i], 
                        label=f'TPM - {day_name}')
        
        # Vẽ line cho play_counted (normalized để hiển thị cùng scale)
        play_normalized = (day_data['play_counted'] - day_data['play_counted'].min()) / \
                         (day_data['play_counted'].max() - day_data['play_counted'].min()) * \
                         day_data['tpm'].max()
        
        if i == 0:
            ax2.plot(range(len(day_data)), play_normalized,
                    color='darkred', linewidth=2, linestyle='--',
                    label='Play Count (Normalized)')
    
    # Formatting cho subplot 2
    ax2.set_xlabel('Time Points (5-minute intervals)', fontsize=12)
    ax2.set_ylabel('TPM', fontsize=12)
    ax2.set_title('TPM Comparison by Day', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Thêm thông tin thống kê
    stats_text = f"""
    Statistics for Latest Week:
    Total Days Analyzed: {len(latest_dates)}
    Date Range: {min(latest_dates)} to {max(latest_dates)}
    Max TPM: {week_data['tpm'].max():,.0f}
    Min TPM: {week_data['tpm'].min():,.0f}
    Avg TPM: {week_data['tpm'].mean():,.0f}
    Total Play Count: {week_data['play_counted'].max():,.0f}
    Push Notifications: {len(push_notifications)}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    # In thông tin chi tiết
    print(f"\n=== WEEKLY ANALYSIS SUMMARY ===")
    print(f"Analysis Period: {min(latest_dates)} to {max(latest_dates)}")
    print(f"Total Records: {len(week_data)}")
    
    for date in latest_dates:
        day_data = week_data[week_data['date'] == date]
        day_name = pd.to_datetime(date).strftime('%A')
        print(f"\n{day_name} ({date}):")
        print(f"  Records: {len(day_data)}")
        print(f"  TPM - Max: {day_data['tpm'].max():,}, Min: {day_data['tpm'].min():,}, Avg: {day_data['tpm'].mean():.0f}")
        print(f"  Play Count Range: {day_data['play_counted'].min():,} - {day_data['play_counted'].max():,}")
        
        # Tìm peak hours
        peak_hours = day_data.nlargest(5, 'tpm')[['timestamp', 'tpm']]
        print(f"  Top 5 Peak Hours:")
        for _, row in peak_hours.iterrows():
            print(f"    {row['timestamp'].strftime('%H:%M')}: {row['tpm']:,} TPM")


def plot_hourly_pattern_comparison():
    """
    Vẽ so sánh pattern theo giờ của 2 ngày
    """
    # Load data
    df = pd.read_csv('../../datasets/gamification_traffic_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    
    # Lấy 2 ngày gần nhất
    latest_dates = sorted(df['date'].unique())[-2:]
    week_data = df[df['date'].isin(latest_dates)]
    
    # Tạo hourly summary
    hourly_summary = week_data.groupby(['date', 'hour']).agg({
        'tpm': ['mean', 'max', 'min'],
        'play_counted': 'max'
    }).round(0)
    
    # Flatten column names
    hourly_summary.columns = ['tpm_mean', 'tpm_max', 'tpm_min', 'play_counted_max']
    hourly_summary = hourly_summary.reset_index()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = ['blue', 'green']
    for i, date in enumerate(latest_dates):
        day_data = hourly_summary[hourly_summary['date'] == date]
        day_name = pd.to_datetime(date).strftime('%A')
        
        # TPM pattern
        ax1.plot(day_data['hour'], day_data['tpm_mean'], 
                marker='o', linewidth=2, color=colors[i],
                label=f'{day_name} - Avg TPM')
        ax1.fill_between(day_data['hour'], day_data['tpm_min'], day_data['tpm_max'],
                        alpha=0.3, color=colors[i])
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('TPM')
    ax1.set_title('Hourly TPM Pattern Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))
    
    # Play count growth pattern
    for i, date in enumerate(latest_dates):
        day_data = hourly_summary[hourly_summary['date'] == date]
        day_name = pd.to_datetime(date).strftime('%A')
        
        ax2.plot(day_data['hour'], day_data['play_counted_max'],
                marker='s', linewidth=2, color=colors[i],
                label=f'{day_name} - Play Count')
    
    ax2.set_xlabel('Hour of Day') 
    ax2.set_ylabel('Cumulative Play Count')
    ax2.set_title('Hourly Play Count Growth Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.show()


# Sử dụng functions
if __name__ == "__main__":
    print("=== TPM & Play Count Analysis ===")
    plot_weekly_tpm_analysis()
    
    print("\n" + "="*50)
    print("=== Hourly Pattern Comparison ===")
    plot_hourly_pattern_comparison()