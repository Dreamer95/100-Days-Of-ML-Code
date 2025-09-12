"""
Day6: Advanced New Relic Data Collector with Maximum Granularity
Thu thập data với độ chi tiết cao nhất theo quy tắc New Relic API
Cải tiến từ Day5 với strategy tối ưu cho từng khoảng thời gian và tuổi data
"""

import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta, timezone
import pytz
import numpy as np

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def calculate_optimal_granularity_strategy(time_range_hours, data_age_days):
    """
    Calculate optimal granularity based on exact New Relic API rules for maximum precision

    New Relic Rules:
    Data age ≤ 8 days:
    - ≤3 hours: 1 minute
    - 3-6 hours: 2 minutes
    - 6-14 hours: 5 minutes
    - 14-24 hours: 10 minutes
    - 1-4 days: 30 minutes
    - 4-7 days: 1 hour

    Data age > 8 days:
    - ≤24 hours: 10 evenly spaced data points
    - 1-4 days: 10 evenly spaced data points
    - 4-7 days: 1 hour
    - 7 days-3 weeks: 3 hours
    - 3-6 weeks: 6 hours
    - 6-9 weeks: 12 hours
    - 63+ days: 3 days
    """

    if data_age_days <= 8:
        # Recent data - high precision available
        if time_range_hours <= 3:
            return 60, int(time_range_hours * 60), 5, "ultra_high_1min"  # 1 minute
        elif time_range_hours <= 6:
            return 120, int(time_range_hours * 30), 4, "very_high_2min"  # 2 minutes
        elif time_range_hours <= 14:
            return 300, int(time_range_hours * 12), 4, "high_5min"  # 5 minutes
        elif time_range_hours <= 24:
            return 600, int(time_range_hours * 6), 3, "medium_high_10min"  # 10 minutes
        elif data_age_days <= 4:
            return 1800, int(time_range_hours * 2), 3, "medium_30min"  # 30 minutes
        else:  # 4-7 days
            return 3600, int(time_range_hours), 2, "medium_low_1hour"  # 1 hour
    else:
        # Historical data - limited precision
        if data_age_days <= 63:
            if time_range_hours <= 24:
                # 10 evenly spaced points
                period = int((time_range_hours * 3600) / 10)
                return period, 10, 2, f"limited_10points_{time_range_hours:.0f}h"
            elif data_age_days <= 7:
                return 3600, int(time_range_hours), 2, "historical_1hour"  # 1 hour
            elif data_age_days <= 21:  # 3 weeks
                return 10800, int(time_range_hours / 3), 1, "historical_3hour"  # 3 hours
            elif data_age_days <= 42:  # 6 weeks
                return 21600, int(time_range_hours / 6), 1, "historical_6hour"  # 6 hours
            elif data_age_days <= 63:  # 9 weeks
                return 43200, int(time_range_hours / 12), 1, "historical_12hour"  # 12 hours
        else:
            # Very old data
            return 259200, int(time_range_hours / 72), 1, "very_historical_3day"  # 3 days

    # Fallback
    return 3600, int(time_range_hours), 1, "fallback_1hour"


def collect_newrelic_data_with_optimal_granularity(api_key, app_id, metrics, start_time, end_time,
                                                   week_priority=1, data_age_days=0):
    """
    Collect metrics data from New Relic API with optimal granularity strategy for maximum precision

    Parameters:
    -----------
    api_key : str
        New Relic API key for authentication
    app_id : str
        Application ID in New Relic
    metrics : list
        List of metrics to collect
    start_time : datetime
        Start time for data collection (UTC)
    end_time : datetime
        End time for data collection (UTC)
    week_priority : int
        Priority weight for this week's data
    data_age_days : int
        Age of the data in days

    Returns:
    --------
    dict
        JSON response from New Relic API with enhanced metadata
    """
    # Convert datetime objects to New Relic API format
    from_time = start_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    to_time = end_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # Prepare the API endpoint
    url = f"https://api.newrelic.com/v2/applications/{app_id}/metrics/data.json"

    # Prepare headers with API key
    headers = {
        "X-Api-Key": api_key,
        "Content-Type": "application/json"
    }

    # Calculate time range in hours
    time_range_hours = (end_time - start_time).total_seconds() / 3600

    # Get optimal granularity strategy for maximum precision
    optimal_period, expected_points, ml_weight, strategy = calculate_optimal_granularity_strategy(
        time_range_hours, data_age_days
    )

    print(f"🕐 Collecting data: {from_time} to {to_time}")
    print(f"📏 Time range: {time_range_hours:.1f} hours, Data age: {data_age_days} days")
    print(f"🎯 Strategy: {strategy}")
    print(f"⚙️  Period: {optimal_period}s, Expected points: {expected_points}")
    print(f"🏆 Week priority: {week_priority}, ML weight: {ml_weight}")

    # Prepare parameters
    params = {
        "names[]": metrics,
        "from": from_time,
        "to": to_time,
        "summarize": "false"
    }

    # Add period if specified (None means New Relic auto-selects)
    if optimal_period:
        params["period"] = optimal_period

    print(f"📡 API Parameters: {params}")

    # Make the API request
    response = requests.get(url, headers=headers, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        # Add enhanced metadata for processing
        result['_metadata'] = {
            'week_priority': week_priority,
            'data_age_days': data_age_days,
            'time_range_hours': time_range_hours,
            'requested_period': optimal_period,
            'expected_data_points': expected_points,
            'ml_weight': max(week_priority, ml_weight),  # Use higher weight
            'collection_strategy': strategy,
            'is_recent_data': data_age_days <= 8,
            'query_start_time': from_time,
            'query_end_time': to_time
        }
        return result
    else:
        print(f"❌ API Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def process_newrelic_data_with_enhanced_metadata(data):
    """
    Process New Relic API response and create enhanced DataFrame with maximum features
    """
    if not data or 'metric_data' not in data:
        print("❌ No valid data to process")
        return None

    metadata = data.get('_metadata', {})
    processed_rows = []

    gmt7_tz = pytz.timezone('Asia/Ho_Chi_Minh')

    try:
        for metric in data['metric_data']['metrics']:
            metric_name = metric['name']
            timeslices = metric.get('timeslices', [])

            for timeslice in timeslices:
                from_time = datetime.fromisoformat(timeslice['from'].replace('Z', '+00:00'))
                to_time = datetime.fromisoformat(timeslice['to'].replace('Z', '+00:00'))

                # Convert to GMT+7
                from_gmt7 = from_time.astimezone(gmt7_tz)
                to_gmt7 = to_time.astimezone(gmt7_tz)

                values = timeslice.get('values', {})

                # Calculate interval precision
                actual_interval_seconds = (to_time - from_time).total_seconds()
                interval_minutes = actual_interval_seconds / 60

                # Enhanced data quality categorization
                is_minute_level = interval_minutes <= 1
                is_sub_5min = interval_minutes <= 5
                is_high_precision = metadata.get('data_age_days', 999) <= 8 and interval_minutes <= 10

                # Precision category
                if metadata.get('data_age_days', 999) <= 8:
                    if interval_minutes <= 1:
                        precision_category = "ultra_precise"
                    elif interval_minutes <= 5:
                        precision_category = "high_precise"
                    elif interval_minutes <= 30:
                        precision_category = "medium_precise"
                    else:
                        precision_category = "low_precise"
                else:
                    precision_category = "historical"

                # Calculate performance scores
                tpm = values.get('requests_per_minute', 0)
                response_time = values.get('average_response_time', 0)
                call_count = values.get('call_count', 0)
                std_dev = values.get('standard_deviation', 0)

                # Throughput score
                throughput_score = 0
                if tpm > 0 and call_count > 0:
                    throughput_score = min(tpm / 100.0, 1.0) * min(call_count / 60.0, 1.0)

                # Efficiency score (lower response time = higher efficiency)
                efficiency_score = 0.5  # neutral default
                if response_time > 0:
                    efficiency_score = max(0, (2000 - response_time) / 1900)
                    efficiency_score = min(efficiency_score, 1.0)

                # Stability score (lower variance = higher stability)
                stability_score = 0.5  # neutral default
                if response_time > 0:
                    cv = std_dev / response_time if response_time > 0 else 0
                    stability_score = max(0, 1 - cv)
                    stability_score = min(stability_score, 1.0)

                processed_row = {
                    # Time information
                    'timestamp': from_gmt7,
                    'timestamp_to': to_gmt7,
                    'timestamp_utc': from_time,
                    'timestamp_to_utc': to_time,
                    'interval_seconds': actual_interval_seconds,
                    'interval_minutes': interval_minutes,

                    # Metric information
                    'metric_name': metric_name,
                    'tpm': tpm,
                    'response_time': response_time,
                    'call_count': call_count,
                    'min_response_time': values.get('min_response_time', 0),
                    'max_response_time': values.get('max_response_time', 0),
                    'total_call_time': values.get('total_call_time', 0),
                    'total_call_time_per_minute': values.get('total_call_time_per_minute', 0),
                    'standard_deviation': std_dev,
                    'score': values.get('score', 0),

                    # Time features
                    'hour': from_gmt7.hour,
                    'minute': from_gmt7.minute,
                    'day_of_week': from_gmt7.weekday(),
                    'day_name': from_gmt7.strftime('%A'),
                    'is_weekend': int(from_gmt7.weekday() >= 5),
                    'is_business_hours': int(9 <= from_gmt7.hour <= 17),

                    # Data quality and collection metadata
                    'collection_strategy': metadata.get('collection_strategy', 'unknown'),
                    'data_quality': precision_category,
                    'ml_weight': metadata.get('ml_weight', 1),
                    'week_priority': metadata.get('week_priority', 1),
                    'data_age_days': metadata.get('data_age_days', 0),
                    'optimal_period_seconds': metadata.get('requested_period', actual_interval_seconds),

                    # Precision indicators
                    'is_minute_level': int(is_minute_level),
                    'is_sub_5min': int(is_sub_5min),
                    'is_high_precision': int(is_high_precision),
                    'is_recent_data': int(metadata.get('data_age_days', 999) <= 8),
                    'precision_category': precision_category,

                    # Performance metrics
                    'throughput_score': throughput_score,
                    'efficiency_score': efficiency_score,
                    'stability_score': stability_score,
                }

                processed_rows.append(processed_row)

    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

    if not processed_rows:
        return None

    # Create DataFrame
    df = pd.DataFrame(processed_rows)

    # Sort by timestamp for proper time series
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Add enhanced features
    df = enhance_dataframe_with_newrelic_optimized_features(df)

    return df


def enhance_dataframe_with_newrelic_optimized_features(df):
    """
    Enhance DataFrame with advanced features optimized for New Relic data patterns
    """
    if df.empty:
        return df

    df = df.copy()

    # Add cyclical time encoding for better time series modeling
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Add lag features for time series with smart handling for variable intervals
    for lag in [1, 2, 3, 5, 10]:
        df[f'tpm_lag_{lag}'] = df['tpm'].shift(lag)
        df[f'response_time_lag_{lag}'] = df['response_time'].shift(lag)
        df[f'throughput_score_lag_{lag}'] = df['throughput_score'].shift(lag)

    # Add rolling statistics with adaptive windows
    for window in [3, 5, 10, 15]:
        df[f'tpm_ma_{window}'] = df['tpm'].rolling(window=window, min_periods=1).mean()
        df[f'response_time_ma_{window}'] = df['response_time'].rolling(window=window, min_periods=1).mean()
        df[f'tpm_std_{window}'] = df['tpm'].rolling(window=window, min_periods=2).std()
        df[f'tpm_min_{window}'] = df['tpm'].rolling(window=window, min_periods=1).min()
        df[f'tpm_max_{window}'] = df['tpm'].rolling(window=window, min_periods=1).max()

    # Add change and trend features
    df['tpm_change'] = df['tpm'].diff()
    df['tpm_change_rate'] = df['tpm'].pct_change()
    df['response_time_change'] = df['response_time'].diff()

    # Add advanced push notification simulation
    df = simulate_realistic_push_notification_campaigns(df)

    # Add anomaly detection features
    df = add_anomaly_detection_features(df)

    # Fill NaN values intelligently
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    return df


def simulate_realistic_push_notification_campaigns(df):
    """
    Simulate realistic push notification campaigns with multiple strategies and realistic patterns
    """
    df = df.copy()
    df['push_notification_active'] = 0
    df['minutes_since_push'] = 9999
    df['push_campaign_type'] = 'none'
    df['push_intensity'] = 0.0

    # Define realistic campaign patterns
    campaigns = [
        {'hour': 8, 'minute': 0, 'intensity': 0.8, 'type': 'morning_commute', 'duration': 15},
        {'hour': 12, 'minute': 0, 'intensity': 1.2, 'type': 'lunch_peak', 'duration': 20},
        {'hour': 15, 'minute': 30, 'intensity': 0.9, 'type': 'afternoon_break', 'duration': 12},
        {'hour': 18, 'minute': 0, 'intensity': 1.0, 'type': 'evening_commute', 'duration': 18},
        {'hour': 20, 'minute': 0, 'intensity': 0.7, 'type': 'evening_leisure', 'duration': 10}
    ]

    for idx, row in df.iterrows():
        current_time = row['timestamp']

        # Check for campaign activation
        for campaign in campaigns:
            if (current_time.hour == campaign['hour'] and
                    current_time.minute == campaign['minute']):

                df.loc[idx, 'push_notification_active'] = 1
                df.loc[idx, 'push_campaign_type'] = campaign['type']
                df.loc[idx, 'push_intensity'] = campaign['intensity']
                df.loc[idx, 'minutes_since_push'] = 0

                # Apply campaign effects for duration
                duration = campaign['duration']
                for future_idx in range(idx + 1, min(idx + duration + 1, len(df))):
                    if future_idx < len(df):
                        minutes_elapsed = future_idx - idx
                        df.loc[future_idx, 'minutes_since_push'] = minutes_elapsed
                        df.loc[future_idx, 'push_campaign_type'] = campaign['type']

                        # Exponential decay effect
                        decay_factor = np.exp(-minutes_elapsed / (duration / 2))
                        current_intensity = campaign['intensity'] * decay_factor
                        df.loc[future_idx, 'push_intensity'] = current_intensity

                        # Apply traffic boost
                        boost = current_intensity * 0.25  # Up to 25% boost
                        df.loc[future_idx, 'tpm'] *= (1 + boost)
                break

    return df


def add_anomaly_detection_features(df):
    """
    Add anomaly detection features to identify unusual patterns
    """
    df = df.copy()

    # Statistical anomalies for TPM
    tpm_mean = df['tpm'].mean()
    tpm_std = df['tpm'].std()

    df['tpm_zscore'] = (df['tpm'] - tpm_mean) / tpm_std if tpm_std > 0 else 0
    df['is_tpm_anomaly'] = (np.abs(df['tpm_zscore']) > 2).astype(int)

    # Response time anomalies
    rt_mean = df['response_time'].mean()
    rt_std = df['response_time'].std()

    df['response_time_zscore'] = (df['response_time'] - rt_mean) / rt_std if rt_std > 0 else 0
    df['is_rt_anomaly'] = (np.abs(df['response_time_zscore']) > 2).astype(int)

    # Combined anomaly score
    df['anomaly_score'] = (np.abs(df['tpm_zscore']) + np.abs(df['response_time_zscore'])) / 2
    df['is_anomaly'] = ((df['is_tpm_anomaly'] == 1) | (df['is_rt_anomaly'] == 1)).astype(int)

    return df


def get_intelligent_time_periods(weeks_back=6, timezone_str='Asia/Ho_Chi_Minh'):
    """
    Generate intelligent time periods with maximum granularity based on New Relic API rules
    Split periods into optimal chunks to get the highest possible data resolution
    """
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    now_utc = datetime.now(timezone.utc)

    periods = []

    print(f"🎯 Generating periods with maximum granularity strategy...")
    print(f"📅 Target weeks back: {weeks_back}")

    for week in range(weeks_back):
        week_start = now - timedelta(weeks=week)

        # Get Monday to Sunday for full week coverage
        days_since_monday = week_start.weekday()
        monday = week_start - timedelta(days=days_since_monday)
        monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        # Sunday end of week
        sunday = monday + timedelta(days=6)
        sunday = sunday.replace(hour=23, minute=59, second=59, microsecond=0)

        # Convert to UTC
        monday_utc = monday.astimezone(timezone.utc)
        sunday_utc = sunday.astimezone(timezone.utc)

        # Skip future dates
        if monday_utc > now_utc:
            continue

        # Adjust end time if in future
        if sunday_utc > now_utc:
            sunday_utc = now_utc

        # Calculate data age at start of week
        data_age_days = (now - monday).days
        week_priority = weeks_back - week

        print(f"\n📅 Week {week + 1}: {monday.strftime('%Y-%m-%d')} to {sunday.strftime('%Y-%m-%d')} GMT+7")
        print(f"   Age: {data_age_days} days, Priority: {week_priority}")

        # Split week into optimal periods based on data age and New Relic rules
        week_periods = split_week_into_optimal_periods(
            monday_utc, sunday_utc, data_age_days, week_priority
        )

        periods.extend(week_periods)
        print(f"   📊 Generated {len(week_periods)} periods for optimal granularity")

    print(f"\n🎯 Total periods generated: {len(periods)}")
    return periods


def split_week_into_optimal_periods(start_utc, end_utc, data_age_days, week_priority):
    """
    Split a week into optimal periods based on New Relic granularity rules
    - data_age <= 8: Use 3-hour chunks for 1-minute granularity
    - data_age > 8: Use 7-day chunks for 1-hour granularity
    """
    periods = []
    current_start = start_utc

    while current_start < end_utc:
        if data_age_days <= 8:
            # Recent data - use 3-hour chunks to get 1-minute granularity
            chunk_hours = 3
        else:
            # Historical data - use 7-day chunks to get 1-hour granularity
            chunk_hours = 7 * 24  # 7 days = 168 hours

        # Calculate chunk end time
        chunk_end = min(current_start + timedelta(hours=chunk_hours), end_utc)

        # Calculate time range for this chunk
        time_range_hours = (chunk_end - current_start).total_seconds() / 3600

        # Get strategy info for logging
        _, _, _, strategy = calculate_optimal_granularity_strategy(time_range_hours, data_age_days)

        periods.append((current_start, chunk_end, week_priority, data_age_days))

        print(f"     📏 {current_start.strftime('%m-%d %H:%M')} → {chunk_end.strftime('%m-%d %H:%M')} "
              f"({time_range_hours:.1f}h) → {strategy}")

        current_start = chunk_end

    return periods


def determine_recent_data_chunk_size(data_age_days):
    """
    Determine optimal chunk size for recent data (≤8 days) to maximize granularity
    """
    if data_age_days <= 4:
        # Very recent data - use small chunks to get high granularity
        if data_age_days <= 1:
            return 3  # 3-hour chunks → 1-2 minute granularity
        else:
            return 6  # 6-hour chunks → 2-5 minute granularity
    else:
        # 4-8 days old - use medium chunks
        return 12  # 12-hour chunks → 5-10 minute granularity


def determine_historical_data_chunk_size(data_age_days):
    """
    Determine optimal chunk size for historical data (>8 days)
    """
    if data_age_days <= 63:
        if data_age_days <= 21:  # ≤3 weeks
            return 6  # 6-hour chunks → 10 points or 1-3 hour granularity
        elif data_age_days <= 42:  # ≤6 weeks
            return 24  # 24-hour chunks → 3-6 hour granularity
        else:  # ≤9 weeks
            return 48  # 48-hour chunks → 6-12 hour granularity
    else:
        # Very old data (>63 days)
        return 168  # 1-week chunks → 3-day granularity


def get_ultra_high_precision_periods(hours_back=24, timezone_str='Asia/Ho_Chi_Minh'):
    """
    Generate ultra-high precision periods for very recent data (last 24 hours)
    Optimized for 1-minute granularity collection
    """
    tz = pytz.timezone(timezone_str)
    now = datetime.now(tz)
    now_utc = datetime.now(timezone.utc)

    periods = []

    # Start from hours_back ago
    start_time = now - timedelta(hours=hours_back)
    start_utc = start_time.astimezone(timezone.utc)

    print(f"🔥 Ultra-High Precision Mode: Last {hours_back} hours")
    print(f"📅 {start_time.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')} GMT+7")

    current_start = start_utc
    period_count = 0

    while current_start < now_utc:
        # Use 3-hour chunks for 1-minute granularity (≤3 hours = 1 minute interval)
        chunk_end = min(current_start + timedelta(hours=3), now_utc)

        data_age_days = (now_utc - current_start).days
        time_range_hours = (chunk_end - current_start).total_seconds() / 3600
        week_priority = 10  # Highest priority for recent data

        # Get expected granularity
        period_seconds, _, _, strategy = calculate_optimal_granularity_strategy(time_range_hours, data_age_days)
        expected_interval_min = period_seconds / 60

        periods.append((current_start, chunk_end, week_priority, data_age_days))
        period_count += 1

        print(
            f"   📏 Period {period_count}: {current_start.strftime('%m-%d %H:%M')} → {chunk_end.strftime('%m-%d %H:%M')} "
            f"({time_range_hours:.1f}h) → {expected_interval_min:.1f}min intervals")

        current_start = chunk_end

    print(f"🎯 Generated {len(periods)} ultra-high precision periods")
    return periods


def get_mixed_granularity_periods(weeks_back=4, include_recent_hours=24, timezone_str='Asia/Ho_Chi_Minh'):
    """
    Generate mixed granularity periods combining ultra-high precision recent data
    with optimized historical periods
    """
    print("🚀 Mixed Granularity Strategy:")
    print("=" * 50)

    all_periods = []

    # 1. Ultra-high precision for last 24 hours (1-minute intervals)
    if include_recent_hours > 0:
        recent_periods = get_ultra_high_precision_periods(include_recent_hours, timezone_str)
        all_periods.extend(recent_periods)
        print(f"✅ Recent periods: {len(recent_periods)} (1-5 min intervals)")

    # 2. Optimized periods for historical weeks
    historical_periods = get_intelligent_time_periods(weeks_back, timezone_str)

    # Filter out overlapping periods
    tz = pytz.timezone(timezone_str)
    cutoff_time = datetime.now(tz) - timedelta(hours=include_recent_hours)
    cutoff_utc = cutoff_time.astimezone(timezone.utc)

    filtered_historical = [
        period for period in historical_periods
        if period[1] <= cutoff_utc  # end_time <= cutoff
    ]

    all_periods.extend(filtered_historical)
    print(f"✅ Historical periods: {len(filtered_historical)} (5min-3day intervals)")

    # Sort by start time
    all_periods.sort(key=lambda x: x[0])

    print(f"\n🎯 Total mixed granularity periods: {len(all_periods)}")
    print(f"📊 Strategy: Ultra-precision recent + Optimized historical")

    return all_periods


def collect_comprehensive_newrelic_data_optimized(api_key=None, app_id=None, weeks_back=4,
                                                  filename="newrelic_maximum_precision_data.csv"):
    """
    Main function to collect comprehensive New Relic data with maximum precision strategy
    Similar structure to Day5 but with enhanced precision focus
    """

    print("🚀 Day6: Advanced New Relic Data Collection with Maximum Precision")
    print("=" * 70)

    # Get credentials
    if not api_key:
        api_key = os.getenv("NEWRELIC_API_KEY") or os.getenv("NEW_RELIC_API_KEY")
    if not app_id:
        app_id = os.getenv("NEWRELIC_APP_ID") or os.getenv("NEW_RELIC_APP_ID")

    if not api_key or not app_id:
        print("❌ Missing New Relic credentials")
        return None

    print(f"📊 Target weeks: {weeks_back}")
    print(f"🎯 Strategy: Maximum granularity per New Relic API rules")
    print(f"💾 Output: datasets/{filename}")

    # Define priority metrics (similar to Day5)
    priority_metrics = [
        "HttpDispatcher",  # Main web transaction metric
    ]

    # Get optimized periods
    periods = get_intelligent_time_periods(weeks_back)

    if not periods:
        print("❌ No valid periods found")
        return None

    print(f"📊 Found {len(periods)} periods to collect")

    # Show collection strategy summary
    recent_periods = sum(1 for _, _, _, age in periods if age <= 8)
    historical_periods = len(periods) - recent_periods
    print(f"\n📋 Collection Strategy Summary:")
    print(f"   Recent periods (≤8 days): {recent_periods} → Ultra-high granularity (1-30 min intervals)")
    print(
        f"   Historical periods (>8 days): {historical_periods} → Limited granularity (10 points or 1+ hour intervals)")

    all_dataframes = []
    successful_collections = 0
    total_data_points = 0

    # Collect data for each period
    for i, (start_time, end_time, week_priority, data_age_days) in enumerate(periods):
        print(f"\n📈 Processing period {i + 1}/{len(periods)}")
        print(f"   📅 {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   🏆 Priority: {week_priority}, Age: {data_age_days} days")

        # Collect data with maximum precision
        data = collect_newrelic_data_with_optimal_granularity(
            api_key, app_id, priority_metrics, start_time, end_time,
            week_priority=week_priority, data_age_days=data_age_days
        )

        if data:
            # Process data with enhanced metadata
            df = process_newrelic_data_with_enhanced_metadata(data)
            if df is not None and not df.empty:
                all_dataframes.append(df)
                data_points = len(df)
                total_data_points += data_points
                successful_collections += 1

                # Quality metrics
                high_precision_pct = (df['is_high_precision'].sum() / data_points) * 100
                minute_level_pct = (df['is_minute_level'].sum() / data_points) * 100
                avg_interval = df['interval_minutes'].mean()

                print(f"   ✅ Collected {data_points} points (avg interval: {avg_interval:.2f}min)")
                print(f"   🎯 High precision: {high_precision_pct:.1f}%, Minute-level: {minute_level_pct:.1f}%")

                # Show strategy distribution
                strategy_counts = df['collection_strategy'].value_counts()
                print(f"   📊 Strategies: {dict(strategy_counts.head(3))}")
            else:
                print(f"   ⚠️  No processed data for period {i + 1}")
        else:
            print(f"   ❌ Failed to collect data for period {i + 1}")

        # Rate limiting
        time.sleep(1.5)

    if not all_dataframes:
        print("❌ No data collected from any period")
        return None

    # Combine all data
    print(f"\n🔄 Combining data from {successful_collections} successful periods...")
    final_df = pd.concat(all_dataframes, ignore_index=True)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)

    # Add sequence number for time series
    final_df['sequence_id'] = range(len(final_df))

    # Create datasets directory
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    # Save to datasets folder
    output_path = os.path.join(datasets_dir, filename)
    final_df.to_csv(output_path, index=False)

    # Enhanced summary (similar to Day5 style)
    print(f"\n📊 Enhanced Dataset Summary:")
    print(f"   📈 Total records: {len(final_df)}")
    print(f"   📅 Time range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    print(f"   🚀 TPM range: {final_df['tpm'].min():.0f} - {final_df['tpm'].max():.0f}")
    print(
        f"   ⏱️  Response time range: {final_df['response_time'].min():.1f} - {final_df['response_time'].max():.1f}ms")
    print(
        f"   📏 Interval range: {final_df['interval_minutes'].min():.1f} - {final_df['interval_minutes'].max():.1f} minutes")

    # Quality analysis
    print(f"\n📈 Data Quality Analysis:")
    print(
        f"   Minute-level data: {final_df['is_minute_level'].sum()} points ({final_df['is_minute_level'].mean() * 100:.1f}%)")
    print(f"   Sub-5min data: {final_df['is_sub_5min'].sum()} points ({final_df['is_sub_5min'].mean() * 100:.1f}%)")
    print(
        f"   High precision data: {final_df['is_high_precision'].sum()} points ({final_df['is_high_precision'].mean() * 100:.1f}%)")
    print(
        f"   Recent data (≤8 days): {final_df['is_recent_data'].sum()} points ({final_df['is_recent_data'].mean() * 100:.1f}%)")

    # Strategy distribution
    print(f"\n🎯 Collection Strategy Distribution:")
    strategy_dist = final_df['collection_strategy'].value_counts()
    for strategy, count in strategy_dist.items():
        print(f"   {strategy}: {count} records ({count / len(final_df) * 100:.1f}%)")

    # Precision category distribution
    print(f"\n📏 Precision Category Distribution:")
    precision_dist = final_df['precision_category'].value_counts()
    for category, count in precision_dist.items():
        avg_interval = final_df[final_df['precision_category'] == category]['interval_minutes'].mean()
        print(f"   {category}: {count} records (avg: {avg_interval:.1f} min intervals)")

    # High priority data summary
    high_priority_data = final_df[final_df['ml_weight'] >= 4]
    print(
        f"\n⭐ High priority data (weight ≥ 4): {len(high_priority_data)} records ({len(high_priority_data) / len(final_df) * 100:.1f}%)")

    print(f"\n✅ Data collection completed!")
    print(f"💾 Saved to: {output_path}")
    print(f"📋 Features: {len(final_df.columns)} columns")

    return final_df


def main():
    """
    Main demo function - similar structure to Day5
    """
    print("🚀 Day6 - Advanced New Relic Data Collector Demo")
    print("=" * 60)

    try:
        # Collect comprehensive data với maximum precision
        df = collect_comprehensive_newrelic_data_optimized(
            weeks_back=3,  # 3 weeks for demo
            filename="day6_maximum_precision_data.csv"
        )

        if df is not None:
            print(f"\n🎉 Demo completed successfully!")
            print(f"📊 Dataset shape: {df.shape}")

            # Show key statistics
            print(f"\n📋 Key Statistics:")
            print(f"   TPM range: {df['tpm'].min():.1f} - {df['tpm'].max():.1f}")
            print(f"   Response time range: {df['response_time'].min():.1f} - {df['response_time'].max():.1f}ms")
            print(f"   Average interval: {df['interval_minutes'].mean():.2f} minutes")
            print(f"   Push notifications: {df['push_notification_active'].sum()} events")
            print(f"   Anomalies detected: {df['is_anomaly'].sum()} points")

            # Sample data
            key_columns = ['timestamp', 'tpm', 'response_time', 'collection_strategy',
                           'precision_category', 'push_notification_active']
            print(f"\n📋 Sample Data:")
            print(df[key_columns].head(10))

        else:
            print("❌ Demo failed - no data collected")

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()