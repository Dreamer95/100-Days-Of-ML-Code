"""
Day6 Summary: Data Visualizer
Hàm visualizer để vẽ data lên chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def plot_tpm_area_chart(csv_file_path=None, data=None, figsize=(15, 8), save_plot=True, show_plot=True, output_dir="charts"):
    """
    Vẽ area chart cho TPM data theo thời gian
    
    Parameters:
    -----------
    csv_file_path : str
        Đường dẫn đến file CSV (nếu data=None)
    data : pandas.DataFrame
        DataFrame chứa data (nếu csv_file_path=None)
    figsize : tuple
        Kích thước figure
    save_plot : bool
        Có lưu plot không
    show_plot : bool
        Có hiển thị plot không
    output_dir : str
        Thư mục lưu charts
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    
    # Load data
    if data is None:
        if csv_file_path is None:
            raise ValueError("Phải cung cấp csv_file_path hoặc data")
        data = pd.read_csv(csv_file_path)
    
    # Prepare data
    df = data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot area chart
    ax.fill_between(df['timestamp'], df['tpm'], alpha=0.7, color='skyblue', label='TPM')
    ax.plot(df['timestamp'], df['tpm'], color='darkblue', linewidth=1, alpha=0.8)
    
    # Customize plot
    ax.set_title('TPM (Transactions Per Minute) Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('TPM', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    # Add statistics text
    stats_text = f"""Statistics:
    Max TPM: {df['tpm'].max():.1f}
    Min TPM: {df['tpm'].min():.1f}
    Mean TPM: {df['tpm'].mean():.1f}
    Std TPM: {df['tpm'].std():.1f}
    Data Points: {len(df):,}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tpm_area_chart_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Area chart saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Error saving chart: {e}")
    
    if show_plot:
        plt.show()
    
    return fig


def plot_tpm_heatmap_hourly(csv_file_path=None, data=None, start_date=None, end_date=None,
                            figsize=(16, 10), save_plot=True, show_plot=True, output_dir="day6_charts",
                            aggregation_method='mean', color_scheme='RdYlBu_r'):
    """
    Vẽ heatmap TPM theo giờ trong ngày và ngày trong tuần với enhanced features
    Style giống như Src/Day5/visualizer.py với improvements

    Parameters:
    -----------
    csv_file_path : str
        Đường dẫn đến file CSV (nếu data=None)
    data : pandas.DataFrame
        DataFrame chứa data (nếu csv_file_path=None)
    start_date : str, optional
        Ngày bắt đầu để filter dữ liệu (format: 'YYYY-MM-DD')
    end_date : str, optional
        Ngày kết thúc để filter dữ liệu (format: 'YYYY-MM-DD')
    figsize : tuple
        Kích thước figure
    save_plot : bool
        Có lưu plot không
    show_plot : bool
        Có hiển thị plot không
    output_dir : str
        Thư mục lưu charts
    aggregation_method : str
        Phương thức tính toán ('mean', 'max', 'median', 'sum')
    color_scheme : str
        Color scheme cho heatmap ('RdYlBu_r', 'YlOrRd', 'viridis', 'plasma')

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    import glob

    # Setup enhanced style
    plt.style.use('default')
    if color_scheme == 'RdYlBu_r':
        sns.set_palette("RdYlBu_r")
    else:
        sns.set_palette("husl")

    # Load data with auto-detection
    if data is None:
        if csv_file_path is None:
            # Auto-detect latest CSV file in datasets
            dataset_files = glob.glob(os.path.join('datasets', '*.csv'))
            if not dataset_files:
                raise ValueError("❌ Không tìm thấy file dữ liệu trong thư mục datasets/")
            csv_file_path = max(dataset_files, key=os.path.getmtime)
            print(f"📊 Auto-detected file: {os.path.basename(csv_file_path)}")

        try:
            data = pd.read_csv(csv_file_path)
            print(f"✅ Đọc thành công {len(data)} records từ {os.path.basename(csv_file_path)}")
        except Exception as e:
            raise ValueError(f"❌ Lỗi đọc file: {e}")

    # Prepare data
    df = data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Handle timezone - keep GMT+7
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        print("🔧 Removed timezone info, keeping original GMT+7 time")
    else:
        print("📅 Using original timestamp (GMT+7)")

    # Filter by date range if provided
    original_count = len(df)
    if start_date or end_date:
        if start_date:
            start_datetime = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_datetime]
            print(f"🔍 Filtered từ ngày bắt đầu: {start_date}")
        if end_date:
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            df = df[df['timestamp'] <= end_datetime]
            print(f"🔍 Filtered đến ngày kết thúc: {end_date}")
        print(f"📈 Filtered từ {original_count} xuống {len(df)} records")
    else:
        print(f"📊 Sử dụng toàn bộ data: {len(df)} records")

    if df.empty:
        raise ValueError("❌ Không có dữ liệu trong khoảng thời gian đã chọn")

    # Extract time features (GMT+7)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date

    # Create pivot table with aggregation method
    aggregation_config = {
        'mean': ('mean', 'Average TPM', 'Average TPM'),
        'max': ('max', 'Maximum TPM', 'Maximum TPM'),
        'median': ('median', 'Median TPM', 'Median TPM'),
        'sum': ('sum', 'Total TPM', 'Total TPM')
    }

    agg_func, title_suffix, cbar_label = aggregation_config.get(aggregation_method, aggregation_config['mean'])

    # Group and aggregate data
    heatmap_data = df.groupby(['day_of_week', 'hour'])['tpm'].agg(agg_func).unstack(fill_value=0)

    # Reorder days of week properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    # Create enhanced figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create beautiful heatmap with enhanced styling
    heatmap = sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=color_scheme,
        annot=True,
        fmt='.0f',
        cbar_kws={
            'label': cbar_label,
            'shrink': 0.8,
            'pad': 0.02
        },
        linewidths=0.8,
        linecolor='white',
        square=False,
        annot_kws={
            'size': 9,
            'weight': 'bold',
            'color': 'white'
        },
        robust=True  # Better color scaling
    )

    # Enhanced title with date range info
    date_range_text = ""
    if start_date or end_date:
        if start_date and end_date:
            date_range_text = f" ({start_date} to {end_date})"
        elif start_date:
            date_range_text = f" (from {start_date})"
        elif end_date:
            date_range_text = f" (to {end_date})"

    ax.set_title(
        f'🔥 TPM Heatmap - {title_suffix}{date_range_text} (GMT+7)',
        fontsize=18,
        fontweight='bold',
        pad=25,
        color='darkblue'
    )

    # Enhanced axis labels
    ax.set_xlabel('Hour of Day (GMT+7)', fontsize=14, fontweight='bold', color='darkgreen')
    ax.set_ylabel('Day of Week', fontsize=14, fontweight='bold', color='darkgreen')

    # Beautiful tick labels
    hour_labels = [f'{i:02d}:00' for i in range(24)]
    ax.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_yticklabels(day_order, rotation=0, fontsize=11, fontweight='bold')

    # Add statistics text box
    total_records = len(df)
    date_range_str = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    peak_hour = heatmap_data.max(axis=0).idxmax()
    peak_day = heatmap_data.max(axis=1).idxmax()
    peak_value = heatmap_data.max().max()

    stats_text = f"""📊 Data Statistics:
    Records: {total_records:,}
    Date Range: {date_range_str}
    Peak: {peak_day} at {peak_hour:02d}:00
    Peak TPM: {peak_value:.0f}
    Method: {aggregation_method.title()}"""

    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='lightblue',
            alpha=0.8,
            edgecolor='navy',
            linewidth=1
        ),
        fontweight='bold'
    )

    # Add colorbar enhancements
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, colors='darkblue')
    cbar.ax.yaxis.label.set_color('darkblue')
    cbar.ax.yaxis.label.set_fontweight('bold')

    # Enhanced grid and styling
    ax.grid(False)  # Remove default grid

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('darkblue')

    plt.tight_layout()

    # Save plot with enhanced metadata
    if save_plot:
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Enhanced filename with metadata
            date_suffix = ""
            if start_date and end_date:
                date_suffix = f"_{start_date}_{end_date}"
            elif start_date:
                date_suffix = f"_from_{start_date}"
            elif end_date:
                date_suffix = f"_to_{end_date}"

            filename = f"tpm_heatmap_{aggregation_method}_{color_scheme}{date_suffix}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)

            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.2
            )
            print(f"💾 Enhanced heatmap saved to: {filepath}")

        except Exception as e:
            print(f"⚠️ Error saving chart: {e}")

    if show_plot:
        plt.show()

    # Print summary insights
    print(f"\n🎯 Heatmap Insights:")
    print(f"   📈 Peak Day: {peak_day}")
    print(f"   ⏰ Peak Hour: {peak_hour:02d}:00")
    print(f"   🔥 Peak TPM: {peak_value:.0f}")
    print(f"   📊 Color Scheme: {color_scheme}")
    print(f"   🧮 Aggregation: {aggregation_method.title()}")

    return fig

def plot_tpm_comparison_charts(csv_file_path=None, data=None, comparison_type='day_of_week', figsize=(16, 10), save_plot=True, show_plot=True, output_dir="charts"):
    """
    Vẽ comparison charts cho TPM theo các tiêu chí khác nhau
    
    Parameters:
    -----------
    csv_file_path : str
        Đường dẫn đến file CSV (nếu data=None)
    data : pandas.DataFrame
        DataFrame chứa data (nếu csv_file_path=None)
    comparison_type : str
        Loại comparison ('day_of_week', 'hour_of_day')
    figsize : tuple
        Kích thước figure
    save_plot : bool
        Có lưu plot không
    show_plot : bool
        Có hiển thị plot không
    output_dir : str
        Thư mục lưu charts
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    
    # Load data
    if data is None:
        if csv_file_path is None:
            raise ValueError("Phải cung cấp csv_file_path hoặc data")
        data = pd.read_csv(csv_file_path)
    
    # Prepare data
    df = data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'TPM Analysis by {comparison_type.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    
    if comparison_type == 'day_of_week':
        # Extract day of week
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['day_num'] = df['timestamp'].dt.dayofweek
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Plot 1: Box plot
        ax1 = axes[0, 0]
        df_ordered = df.set_index('day_of_week').reindex(day_order).reset_index()
        sns.boxplot(data=df_ordered, x='day_of_week', y='tpm', ax=ax1)
        ax1.set_title('TPM Distribution by Day of Week')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # Plot 2: Mean TPM by day
        ax2 = axes[0, 1]
        daily_mean = df.groupby('day_of_week')['tpm'].mean().reindex(day_order)
        daily_mean.plot(kind='bar', ax=ax2, color='lightcoral')
        ax2.set_title('Average TPM by Day of Week')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: TPM over time colored by day
        ax3 = axes[1, 0]
        for day in day_order:
            day_data = df[df['day_of_week'] == day]
            ax3.scatter(day_data['timestamp'], day_data['tpm'], label=day, alpha=0.6, s=1)
        ax3.set_title('TPM Over Time by Day of Week')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 4: Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics table
        stats_data = []
        for day in day_order:
            day_data = df[df['day_of_week'] == day]['tpm']
            stats_data.append([
                day,
                f"{day_data.mean():.1f}",
                f"{day_data.std():.1f}",
                f"{day_data.min():.1f}",
                f"{day_data.max():.1f}",
                f"{len(day_data)}"
            ])
        
        table = ax4.table(cellText=stats_data,
                         colLabels=['Day', 'Mean', 'Std', 'Min', 'Max', 'Count'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Statistics by Day of Week')
        
    elif comparison_type == 'hour_of_day':
        # Extract hour of day
        df['hour'] = df['timestamp'].dt.hour
        
        # Plot 1: Box plot
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='hour', y='tpm', ax=ax1)
        ax1.set_title('TPM Distribution by Hour of Day')
        
        # Plot 2: Mean TPM by hour
        ax2 = axes[0, 1]
        hourly_mean = df.groupby('hour')['tpm'].mean()
        hourly_mean.plot(kind='line', ax=ax2, marker='o', color='green')
        ax2.set_title('Average TPM by Hour of Day')
        ax2.set_xlabel('Hour')
        
        # Plot 3: Heatmap by hour and day of week
        ax3 = axes[1, 0]
        df['day_of_week'] = df['timestamp'].dt.day_name()
        pivot_data = df.groupby(['day_of_week', 'hour'])['tpm'].mean().unstack()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_data = pivot_data.reindex(day_order)
        sns.heatmap(pivot_data, cmap='viridis', ax=ax3, cbar_kws={'label': 'Avg TPM'})
        ax3.set_title('TPM Heatmap: Hour vs Day of Week')
        
        # Plot 4: Peak hours analysis
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Find peak hours
        hourly_stats = df.groupby('hour')['tpm'].agg(['mean', 'std', 'count']).round(1)
        top_hours = hourly_stats.nlargest(5, 'mean')
        
        stats_text = "Top 5 Peak Hours:\n\n"
        for hour, row in top_hours.iterrows():
            stats_text += f"Hour {hour:02d}: {row['mean']:.1f} TPM\n"
        
        stats_text += f"\nOverall Statistics:\n"
        stats_text += f"Peak Hour: {hourly_stats['mean'].idxmax():02d}:00\n"
        stats_text += f"Lowest Hour: {hourly_stats['mean'].idxmin():02d}:00\n"
        stats_text += f"Peak TPM: {hourly_stats['mean'].max():.1f}\n"
        stats_text += f"Lowest TPM: {hourly_stats['mean'].min():.1f}"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('Peak Hours Analysis')
    
    plt.tight_layout()
    
    # Save plot
    if save_plot:
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tpm_comparison_{comparison_type}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Comparison chart saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Error saving chart: {e}")
    
    if show_plot:
        plt.show()
    
    return fig

def create_comprehensive_visualization_suite(csv_file_path=None, data=None, output_dir="charts"):
    """
    Tạo bộ visualization hoàn chỉnh cho TPM data
    
    Parameters:
    -----------
    csv_file_path : str
        Đường dẫn đến file CSV (nếu data=None)
    data : pandas.DataFrame
        DataFrame chứa data (nếu csv_file_path=None)
    output_dir : str
        Thư mục lưu charts
        
    Returns:
    --------
    dict
        Dictionary chứa các figure objects
    """
    
    print("🚀 Tạo bộ Comprehensive Visualization Suite...")
    
    figures = {}
    
    # 1. TPM Area Chart
    print("📈 1. Creating TPM Area Chart...")
    try:
        fig_area = plot_tpm_area_chart(csv_file_path, data, output_dir=output_dir)
        if fig_area:
            figures['area_chart'] = fig_area
            print("✅ TPM Area Chart created successfully")
    except Exception as e:
        print(f"❌ Error creating TPM Area Chart: {e}")
    
    # 2. TPM Heatmap Hourly
    print("🔥 2. Creating TPM Hourly Heatmap...")
    try:
        fig_heatmap = plot_tpm_heatmap_hourly(csv_file_path, data, output_dir=output_dir)
        if fig_heatmap:
            figures['hourly_heatmap'] = fig_heatmap
            print("✅ TPM Hourly Heatmap created successfully")
    except Exception as e:
        print(f"❌ Error creating TPM Hourly Heatmap: {e}")
    
    # 4. Hour of Day Comparison
    print("🕐 4. Creating Hour of Day Comparison...")
    try:
        fig_hod = plot_tpm_comparison_charts(csv_file_path, data, 'hour_of_day', output_dir=output_dir)
        if fig_hod:
            figures['hour_of_day_comparison'] = fig_hod
            print("✅ Hour of Day Comparison created successfully")
    except Exception as e:
        print(f"❌ Error creating Hour of Day Comparison: {e}")
    
    print(f"🎯 Comprehensive Visualization Suite completed! Created {len(figures)} visualizations.")
    return figures

def main():
    """
    Demo function để test visualization
    """
    print("🚀 Day6 - Data Visualizer Demo")
    print("=" * 50)
    
    # Tạo sample data để demo
    print("📊 Creating sample data for demo...")
    
    # Generate sample TPM data
    from datetime import datetime, timedelta
    import random
    
    # Create 7 days of hourly data
    start_date = datetime.now() - timedelta(days=7)
    timestamps = []
    tpm_values = []
    
    for i in range(7 * 24):  # 7 days * 24 hours
        timestamp = start_date + timedelta(hours=i)
        
        # Simulate realistic TPM pattern
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base TPM with daily and weekly patterns
        base_tpm = 100
        
        # Daily pattern (higher during business hours)
        if 8 <= hour <= 18:
            base_tpm += 200
        elif 19 <= hour <= 22:
            base_tpm += 100
        
        # Weekly pattern (lower on weekends)
        if day_of_week >= 5:  # Weekend
            base_tpm *= 0.6
        
        # Add some randomness
        tpm = base_tpm + random.gauss(0, 50)
        tpm = max(10, tpm)  # Minimum TPM
        
        timestamps.append(timestamp)
        tpm_values.append(tpm)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'tpm': tpm_values
    })
    
    print(f"✅ Created sample data: {len(sample_data)} records")
    print(f"📅 Time range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
    
    try:
        # Create comprehensive visualization suite
        figures = create_comprehensive_visualization_suite(
            data=sample_data,
            output_dir="day6_charts"
        )
        
        print(f"\n✅ Demo thành công!")
        print(f"📊 Created {len(figures)} visualizations")
        print(f"💾 Charts saved to: day6_charts/")
        
    except Exception as e:
        print(f"❌ Lỗi trong demo: {e}")

if __name__ == "__main__":
    main()