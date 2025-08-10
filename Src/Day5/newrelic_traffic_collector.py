import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta, timezone, time as dtime
import pytz
# Add numpy import for circular encoding
import numpy as np
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def calculate_optimal_granularity_strategy(time_range_hours, data_age_days):
    """
    Calculate optimal data collection strategy based on New Relic API rules
    
    Parameters:
    -----------
    time_range_hours : float
        Duration of the query in hours (48+ hours for Thu-Sun)
    data_age_days : int
        How many days old the data is
    
    Returns:
    --------
    tuple: (period_seconds, expected_data_points, ml_weight, collection_strategy)
    """
    
    if data_age_days <= 8:  # Recent data - can get fine granularity
        if time_range_hours <= 3:
            return 60, int(time_range_hours * 60), 10, "1min_granularity"
        elif time_range_hours <= 6:
            return 120, int(time_range_hours * 30), 9, "2min_granularity"
        elif time_range_hours <= 14:
            return 300, int(time_range_hours * 12), 8, "5min_granularity"
        elif time_range_hours <= 24:
            return 600, int(time_range_hours * 6), 7, "10min_granularity"
        elif time_range_hours <= 96:  # 4 days - our weekend is ~48 hours
            return 1800, int(time_range_hours * 2), 6, "30min_granularity"
        else:
            return 3600, int(time_range_hours), 5, "1hour_granularity"
    
    else:  # Historical data (>8 days) - limited granularity
        if time_range_hours <= 96:  # ≤ 4 days (our weekend periods)
            return None, 10, 4, "10_evenly_spaced_points"  # New Relic auto-selects 10 points
        elif time_range_hours <= 168:  # 7 days
            return 3600, int(time_range_hours), 3, "1hour_intervals"
        elif time_range_hours <= 504:  # 3 weeks
            return 10800, int(time_range_hours / 3), 2, "3hour_intervals"
        elif time_range_hours <= 1008:  # 6 weeks
            return 21600, int(time_range_hours / 6), 2, "6hour_intervals"
        elif time_range_hours <= 1512:  # 9 weeks
            return 43200, int(time_range_hours / 12), 1, "12hour_intervals"
        else:  # > 63 days
            return 259200, int(time_range_hours / 72), 1, "3day_intervals"

def collect_newrelic_data_with_optimal_granularity(api_key, app_id, metrics, start_time, end_time, 
                                                  week_priority=1, data_age_days=0):
    """
    Collect metrics data from New Relic API with optimal granularity strategy
    
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
    
    # Get optimal granularity strategy
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
    Process the data received from New Relic API with enhanced metadata and timestamp_to
    
    Parameters:
    -----------
    data : dict
        JSON response from New Relic API with metadata
    
    Returns:
    --------
    pandas.DataFrame
        Processed data with enhanced features including timestamp_to
    """
    if not data or 'metric_data' not in data:
        print("No metric_data found in response")
        return None
    
    # Extract metadata
    metadata = data.get('_metadata', {})
    ml_weight = metadata.get('ml_weight', 1)
    week_priority = metadata.get('week_priority', 1)
    data_age_days = metadata.get('data_age_days', 0)
    collection_strategy = metadata.get('collection_strategy', 'unknown')
    
    # Extract the metrics data
    metrics_data = data['metric_data']
    
    if 'metrics' not in metrics_data or not metrics_data['metrics']:
        print("No metrics found in metric_data")
        return None
    
    # Initialize lists to store the processed data
    all_data = []
    
    # Setup GMT+7 timezone for conversion
    gmt7 = pytz.timezone('Asia/Bangkok')
    
    # Process each metric
    for metric in metrics_data['metrics']:
        metric_name = metric['name']
        
        if 'timeslices' in metric:
            for timeslice in metric['timeslices']:
                # Parse FROM timestamp
                timestamp_from_str = timeslice['from']
                timestamp_from_utc = datetime.fromisoformat(timestamp_from_str.replace('+00:00', '+00:00'))
                
                if timestamp_from_utc.tzinfo is None:
                    timestamp_from_utc = timestamp_from_utc.replace(tzinfo=timezone.utc)
                
                # Parse TO timestamp - ADDED
                timestamp_to_str = timeslice['to']
                timestamp_to_utc = datetime.fromisoformat(timestamp_to_str.replace('+00:00', '+00:00'))
                
                if timestamp_to_utc.tzinfo is None:
                    timestamp_to_utc = timestamp_to_utc.replace(tzinfo=timezone.utc)
                
                # Convert to GMT+7
                timestamp_from_gmt7 = timestamp_from_utc.astimezone(gmt7)
                timestamp_to_gmt7 = timestamp_to_utc.astimezone(gmt7)
                
                # Calculate actual interval duration
                interval_minutes = (timestamp_to_utc - timestamp_from_utc).total_seconds() / 60
                
                # Extract values
                values = timeslice['values']

                # Safe extraction of requested_period with None handling
                requested_period = metadata.get('requested_period')
                if requested_period is None:
                    # For historical data with auto-selected points, calculate from actual interval
                    requested_period = interval_minutes * 60  # Convert to seconds

                # Create enhanced data entry
                data_entry = {
                    # Timestamps - ENHANCED
                    'timestamp': timestamp_from_gmt7,  # Start time in GMT+7
                    'timestamp_to': timestamp_to_gmt7,  # End time in GMT+7 - NEW
                    'timestamp_utc': timestamp_from_utc,  # Start time in UTC
                    'timestamp_to_utc': timestamp_to_utc,  # End time in UTC - NEW
                    'interval_minutes': interval_minutes,  # Actual interval duration - NEW
                    'metric_name': metric_name,
                    
                    # Traffic metrics - Enhanced extraction
                    'tpm': values.get('requests_per_minute', values.get('calls_per_minute', 0)),
                    'response_time': values.get('average_response_time', values.get('average_call_time', 0)),
                    'call_count': values.get('call_count', 0),
                    'min_response_time': values.get('min_response_time', 0),
                    'max_response_time': values.get('max_response_time', 0),
                    'standard_deviation': values.get('standard_deviation', 0),
                    'total_call_time_per_minute': values.get('total_call_time_per_minute', 0),
                    
                    # Time features
                    'hour': timestamp_from_gmt7.hour,
                    'minute': timestamp_from_gmt7.minute,
                    'day_of_week': timestamp_from_gmt7.weekday(),
                    'is_weekend': 1 if timestamp_from_gmt7.weekday() >= 5 else 0,
                    'day_name': timestamp_from_gmt7.strftime('%A'),  # NEW
                    
                    # ML Training weights and metadata - ENHANCED
                    'ml_weight': ml_weight,
                    'week_priority': week_priority,
                    'data_age_days': data_age_days,
                    'data_quality': 'high' if ml_weight >= 8 else 'medium' if ml_weight >= 5 else 'low',
                    'collection_strategy': collection_strategy,  # NEW
                    'is_recent_data': 1 if data_age_days <= 8 else 0,
                    
                    # Granularity indicators - ENHANCED
                    'granularity_minutes': requested_period / 60,
                    'is_high_granularity': 1 if requested_period <= 300 else 0,  # ≤5 min
                    'is_fine_granularity': 1 if requested_period <= 120 else 0,  # ≤2 min - NEW
                    'granularity_category': (  # NEW
                        'minute' if requested_period <= 60 else
                        'multi_minute' if requested_period <= 600 else
                        'hourly' if requested_period <= 3600 else
                        'multi_hour'
                    )
                }
                
                all_data.append(data_entry)
    
    if not all_data:
        print("No data extracted from metrics")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    def compute_push_features(ts_gmt7):
        # ts_gmt7: pandas.Timestamp (tz-aware, GMT+7)
        hour = ts_gmt7.hour
        minute = ts_gmt7.minute

        # Xác định lần push gần nhất
        if hour > 12 or (hour == 12 and minute >= 0):
            # Sau hoặc đúng 12:00 hôm nay -> lần push gần nhất là 12:00 hôm nay
            last_push_date = ts_gmt7.date()
            last_push_time = dtime(12, 0)
        elif hour > 3 or (hour == 3 and minute >= 0):
            # Từ 03:00 đến trước 12:00 -> lần push gần nhất là 03:00 hôm nay
            last_push_date = ts_gmt7.date()
            last_push_time = dtime(3, 0)
        else:
            # Trước 03:00 -> lần push gần nhất là 12:00 ngày hôm qua
            last_push_date = (ts_gmt7 - timedelta(days=1)).date()
            last_push_time = dtime(12, 0)

        # Tạo datetime tz-aware tại GMT+7 cho last push
        last_push_dt = df.loc[0, 'timestamp'].tz  # lấy tzinfo GMT+7 sẵn có
        last_push_dt = datetime.combine(last_push_date, last_push_time).replace(tzinfo=last_push_dt)

        minutes_since = int((ts_gmt7 - last_push_dt).total_seconds() // 60)
        push_active = 1 if minutes_since == 0 else 0
        return pd.Series({'push_notification_active': push_active, 'minutes_since_push': minutes_since})

    push_df = df['timestamp'].apply(compute_push_features)
    df['push_notification_active'] = push_df['push_notification_active'].astype(int)
    df['minutes_since_push'] = push_df['minutes_since_push'].astype(int)

    # Add time-based features for ML
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)  # Circular encoding
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    # Add lag features (important for prediction)
    for i in [1, 2, 3, 5]:  # Adjusted for different granularities
        if len(df) > i:
            df[f'tpm_lag_{i}'] = df['tpm'].shift(i)
            df[f'response_time_lag_{i}'] = df['response_time'].shift(i)
    
    # Add rolling statistics
    for window in [3, 5]:  # Smaller windows for variable granularity
        if len(df) >= window:
            df[f'tpm_ma_{window}'] = df['tpm'].rolling(window=window, min_periods=1).mean()
            df[f'response_time_ma_{window}'] = df['response_time'].rolling(window=window, min_periods=1).mean()
            df[f'tpm_std_{window}'] = df['tpm'].rolling(window=window, min_periods=1).std()
    
    # Add change features
    df['tpm_change'] = df['tpm'].diff()
    df['tpm_change_rate'] = df['tpm'].pct_change()
    df['response_time_change'] = df['response_time'].diff()
    
    # Fill missing values
    df = df.fillna(0)
    
    print(f"✅ Processed {len(df)} data points with enhanced metadata")
    print(f"🎯 Collection strategy: {collection_strategy}")
    print(f"📊 Interval range: {df['interval_minutes'].min():.1f} - {df['interval_minutes'].max():.1f} minutes")
    print(f"🏆 Weight distribution: {df['ml_weight'].value_counts().sort_index().to_dict()}")
    print(f"🎨 Data quality: {df['data_quality'].value_counts().to_dict()}")
    
    return df


def get_weekend_traffic_periods(weeks_back=12, focus_recent_weeks=True):
    """
    Get weekend traffic periods: Thursday 23:50 to Sunday 00:00 (GMT+7)
    Total duration: 48 hours 10 minutes per week

    Parameters:
    -----------
    weeks_back : int
        Total weeks to go back for data collection
    focus_recent_weeks : bool
        If True, prioritize recent weeks with more granular time periods

    Returns:
    --------
    list
        List of (start_time_utc, end_time_utc, week_priority, data_age_days) tuples
    """
    from datetime import datetime, timedelta, timezone
    import pytz

    # GMT+7 timezone
    gmt7 = pytz.timezone('Asia/Ho_Chi_Minh')

    # Current time in GMT+7
    now_gmt7 = datetime.now(gmt7)

    periods = []

    print(f"🕐 Weekend Period Definition: Thursday 23:50 to Sunday 00:00 (GMT+7)")
    print(f"⏱️  Total duration per week: 48 hours 10 minutes")
    print(f"🔄 Converting all times to UTC for New Relic API")

    for week in range(weeks_back):
        # Find the most recent Thursday 23:50 GMT+7
        days_since_thursday = (now_gmt7.weekday() - 3) % 7
        current_thursday = now_gmt7 - timedelta(days=days_since_thursday, weeks=week)

        # Set exact weekend start: Thursday 23:50 GMT+7
        weekend_start_gmt7 = current_thursday.replace(
            hour=23, minute=50, second=0, microsecond=0
        )

        # Set exact weekend end: Sunday 00:00 GMT+7 (next week)
        weekend_end_gmt7 = weekend_start_gmt7 + timedelta(days=2, hours=0, minutes=10)
        weekend_end_gmt7 = weekend_end_gmt7.replace(hour=23, minute=59, second=59, microsecond=999_999)

        # Convert to UTC for New Relic API
        weekend_start_utc = weekend_start_gmt7.astimezone(timezone.utc)
        weekend_end_utc = weekend_end_gmt7.astimezone(timezone.utc)

        # Calculate data age and priority
        data_age_days = (now_gmt7 - weekend_start_gmt7).days
        week_priority = max(1, 13 - week)

        print(f"\n📅 Week {week + 1}:")
        print(
            f"   GMT+7: {weekend_start_gmt7.strftime('%Y-%m-%d %H:%M')} to {weekend_end_gmt7.strftime('%Y-%m-%d %H:%M')}")
        print(
            f"   UTC:   {weekend_start_utc.strftime('%Y-%m-%d %H:%M')} to {weekend_end_utc.strftime('%Y-%m-%d %H:%M')}")
        print(f"   📊 Age: {data_age_days} days, Priority: {week_priority}")

        # Calculate total weekend duration
        total_hours = (weekend_end_utc - weekend_start_utc).total_seconds() / 3600
        print(f"   ⏱️  Duration: {total_hours:.2f} hours")

        # Determine collection strategy based on data age and New Relic granularity table
        if focus_recent_weeks and data_age_days <= 8:
            # RECENT DATA (≤8 days): High-resolution strategy
            print(f"   🔥 RECENT DATA: Using high-resolution collection")

            # Split into optimal periods based on New Relic granularity table
            # ≤3 hours = 1-minute granularity, 3-6 hours = 2-minute granularity
            current_time = weekend_start_utc
            period_count = 0

            while current_time < weekend_end_utc:
                # Use 3-hour chunks for 1-minute granularity (180 points)
                period_duration = min(3.0, (weekend_end_utc - current_time).total_seconds() / 3600)
                period_end = current_time + timedelta(hours=period_duration)
                period_end = min(period_end, weekend_end_utc)

                periods.append((
                    current_time,
                    period_end,
                    week_priority,
                    data_age_days
                ))

                period_duration_actual = (period_end - current_time).total_seconds() / 3600
                print(f"     ⏰ Period {period_count + 1}: {period_duration_actual:.1f}h → 1-min granularity")

                current_time = period_end
                period_count += 1

                if period_count >= 20:  # Safety limit
                    break

        elif data_age_days <= 14:
            # MEDIUM-TERM DATA (8-14 days): Moderate resolution
            print(f"   📊 MEDIUM-TERM DATA: Using moderate-resolution collection")

            # Use 6-hour chunks for 5-minute granularity (72 points)
            current_time = weekend_start_utc
            period_count = 0

            while current_time < weekend_end_utc:
                period_duration = min(6.0, (weekend_end_utc - current_time).total_seconds() / 3600)
                period_end = current_time + timedelta(hours=period_duration)
                period_end = min(period_end, weekend_end_utc)

                periods.append((
                    current_time,
                    period_end,
                    week_priority,
                    data_age_days
                ))

                period_duration_actual = (period_end - current_time).total_seconds() / 3600
                print(f"     ⏰ Period {period_count + 1}: {period_duration_actual:.1f}h → 5-min granularity")

                current_time = period_end
                period_count += 1

                if period_count >= 10:
                    break

        else:
            # HISTORICAL DATA (>14 days): Low resolution
            print(f"   📈 HISTORICAL DATA: Using low-resolution collection")

            # Use larger chunks for 30-minute or 1-hour granularity
            if total_hours <= 24:
                # Single period with 30-minute granularity (48 points)
                periods.append((
                    weekend_start_utc,
                    weekend_end_utc,
                    week_priority,
                    data_age_days
                ))
                print(f"     ⏰ Single period: {total_hours:.1f}h → 30-min granularity")
            else:
                # Split into 2 periods of ~24 hours each with 1-hour granularity
                mid_point = weekend_start_utc + timedelta(hours=24)

                periods.extend([
                    (weekend_start_utc, mid_point, week_priority, data_age_days),
                    (mid_point, weekend_end_utc, week_priority, data_age_days)
                ])
                print(f"     ⏰ Period 1: 24.0h → 1-hour granularity")
                print(
                    f"     ⏰ Period 2: {(weekend_end_utc - mid_point).total_seconds() / 3600:.1f}h → 1-hour granularity")

    print(f"\n📊 Collection Summary:")
    print(f"   Total periods: {len(periods)}")

    # Analyze by data age
    recent_periods = [p for p in periods if p[3] <= 8]
    medium_periods = [p for p in periods if 8 < p[3] <= 14]
    historical_periods = [p for p in periods if p[3] > 14]

    print(f"   🔥 Recent (≤8 days): {len(recent_periods)} periods → 1-2min intervals")
    print(f"   📊 Medium (8-14 days): {len(medium_periods)} periods → 5-15min intervals")
    print(f"   📈 Historical (>14 days): {len(historical_periods)} periods → 30-60min intervals")

    # Sort by priority (higher priority first) then by recency (newer first)
    periods.sort(key=lambda x: (-x[2], x[3]))

    return periods


def collect_and_save_newrelic_data(api_key, app_id="1080863725", weeks_back=8, filename="newrelic_weekend_traffic.csv"):
    """
    Collect weekend traffic data with focus on recent high-resolution data

    Parameters:
    -----------
    api_key : str
        New Relic API key
    app_id : str
        Application ID
    weeks_back : int
        Number of weeks to collect (default 8 to focus on recent data)
    filename : str
        Output CSV filename

    Returns:
    --------
    bool
        Success status
    """
    print(f"🚀 Collecting RECENT-FOCUSED weekend traffic data for ML training...")
    print(f"📅 Collecting {weeks_back} weeks with PRIORITY on last 8 days (data_age <= 8)")
    print(f"🎯 Using NEW RELIC granularity table for optimal collection strategy")

    # Get weekend periods with focus on recent data
    periods = get_weekend_traffic_periods(weeks_back, focus_recent_weeks=True)

    if not periods:
        print("❌ No valid periods found")
        return False

    print(f"\n📊 Found {len(periods)} periods to collect")

    # Analyze collection strategy by data age
    recent_periods = [p for p in periods if p[3] <= 8]
    medium_periods = [p for p in periods if 8 < p[3] <= 14]
    historical_periods = [p for p in periods if p[3] > 14]

    print(f"\n📋 Enhanced Collection Strategy:")
    print(f"   🔥 RECENT periods (≤8 days): {len(recent_periods)} → 1-2min granularity")
    print(f"   📊 MEDIUM periods (8-14 days): {len(medium_periods)} → 5-15min granularity")
    print(f"   📈 HISTORICAL periods (>14 days): {len(historical_periods)} → 30-60min granularity")

    # Define metrics for comprehensive traffic analysis
    metrics = [
        "HttpDispatcher",  # Main web transaction metric
    ]

    all_dataframes = []
    successful_collections = 0
    total_data_points = 0

    # Process periods with priority order (recent first)
    for i, (start_time, end_time, week_priority, data_age_days) in enumerate(periods):
        print(f"\n📈 Collecting period {i + 1}/{len(periods)}")
        print(f"   📅 {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   🏆 Week priority: {week_priority}, Data age: {data_age_days} days")

        # Show expected granularity based on data age
        if data_age_days <= 8:
            expected_granularity = "1-2 minutes"
            priority_level = "🔥 HIGH"
        elif data_age_days <= 14:
            expected_granularity = "5-15 minutes"
            priority_level = "📊 MEDIUM"
        else:
            expected_granularity = "30-60 minutes"
            priority_level = "📈 LOW"

        print(f"   {priority_level} priority → Expected: {expected_granularity}")

        # Collect data for this period
        data = collect_newrelic_data_with_optimal_granularity(
            api_key, app_id, metrics, start_time, end_time,
            week_priority=week_priority, data_age_days=data_age_days
        )

        if data:
            # Process data with enhanced metadata
            df = process_newrelic_data_with_enhanced_metadata(data)
            if df is not None and not df.empty:
                all_dataframes.append(df)
                avg_interval = df['interval_minutes'].mean()
                data_points = len(df)
                total_data_points += data_points
                successful_collections += 1

                print(f"   ✅ Collected {data_points} points (avg interval: {avg_interval:.1f} min)")

                # Show quality metrics for recent data
                if data_age_days <= 8:
                    high_res_points = len(df[df['interval_minutes'] <= 5])
                    print(
                        f"   🎯 High-resolution points (≤5min): {high_res_points} ({high_res_points / data_points * 100:.1f}%)")
            else:
                print(f"   ⚠️  No data processed for this period")
        else:
            print(f"   ❌ Failed to collect data for this period")

        # Enhanced rate limiting with priority consideration
        if data_age_days <= 8:
            time.sleep(0.3)  # Faster for recent data
        else:
            time.sleep(0.5)  # Standard rate limiting

    if not all_dataframes:
        print("❌ No data collected from any period")
        return False

    # Combine all data with priority weighting
    print(f"\n🔄 Combining data from {successful_collections} successful periods...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Sort by timestamp for time series analysis
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

    # Add sequence number for time series
    combined_df['sequence_id'] = range(len(combined_df))

    # Enhanced summary with focus on data quality distribution
    print(f"\n📊 RECENT-FOCUSED Dataset Summary:")
    print(f"   📈 Total records: {len(combined_df):,}")
    print(f"   📅 Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"   🚀 TPM range: {combined_df['tpm'].min():.0f} - {combined_df['tpm'].max():.0f}")
    print(
        f"   ⏱️  Response time range: {combined_df['response_time'].min():.1f} - {combined_df['response_time'].max():.1f}ms")
    print(
        f"   📏 Interval range: {combined_df['interval_minutes'].min():.1f} - {combined_df['interval_minutes'].max():.1f} minutes")

    # Data age distribution analysis
    print(f"\n🎯 Data Age Distribution (ML Training Priority):")
    age_groups = [
        ("Recent (≤8 days)", combined_df['data_age_days'] <= 8),
        ("Medium (8-14 days)", (combined_df['data_age_days'] > 8) & (combined_df['data_age_days'] <= 14)),
        ("Historical (>14 days)", combined_df['data_age_days'] > 14)
    ]

    for group_name, mask in age_groups:
        group_data = combined_df[mask]
        if len(group_data) > 0:
            avg_interval = group_data['interval_minutes'].mean()
            avg_weight = group_data['ml_weight'].mean()
            print(f"   {group_name}: {len(group_data):,} records ({len(group_data) / len(combined_df) * 100:.1f}%)")
            print(f"      → Avg interval: {avg_interval:.1f}min, Avg ML weight: {avg_weight:.1f}")

    # High-resolution data analysis
    high_res_data = combined_df[combined_df['interval_minutes'] <= 5]
    minute_data = combined_df[combined_df['interval_minutes'] <= 2]

    print(f"\n⭐ High-Resolution Data Analysis:")
    print(
        f"   Fine granularity (≤5min): {len(high_res_data):,} records ({len(high_res_data) / len(combined_df) * 100:.1f}%)")
    print(f"   Ultra-fine (≤2min): {len(minute_data):,} records ({len(minute_data) / len(combined_df) * 100:.1f}%)")

    # Strategy effectiveness summary
    print(f"\n🎯 Collection Strategy Distribution:")
    strategy_dist = combined_df['collection_strategy'].value_counts()
    for strategy, count in strategy_dist.items():
        print(f"   {strategy}: {count:,} records ({count / len(combined_df) * 100:.1f}%)")

    # High priority data summary (for ML training)
    high_priority_data = combined_df[combined_df['ml_weight'] >= 8]
    print(
        f"\n⭐ Premium ML Training Data (weight ≥ 8): {len(high_priority_data):,} records ({len(high_priority_data) / len(combined_df) * 100:.1f}%)")

    # Save enhanced dataset
    try:
        os.makedirs('datasets', exist_ok=True)
        filepath = os.path.join('datasets', filename)
        combined_df.to_csv(filepath, index=False)
        print(f"\n✅ Recent-focused dataset saved to {filepath}")

        # Enhanced summary report with focus metrics
        summary_filename = filename.replace('.csv', '_recent_focused_summary.txt')
        summary_filepath = os.path.join('datasets', summary_filename)
        with open(summary_filepath, 'w') as f:
            f.write(f"Recent-Focused Weekend Traffic Data Collection Summary\n")
            f.write(f"====================================================\n\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Collection Strategy: RECENT-FOCUSED with New Relic granularity optimization\n")
            f.write(f"Focus Period: Last 8 days (data_age <= 8) with 1-2 minute intervals\n")
            f.write(f"Total Records: {len(combined_df):,}\n")
            f.write(f"Time Range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}\n")
            f.write(f"Weeks Collected: {weeks_back}\n")
            f.write(f"Total Periods: {len(periods)}\n")
            f.write(f"Successful Collections: {successful_collections}\n\n")

            f.write("Data Age Distribution:\n")
            for group_name, mask in age_groups:
                group_data = combined_df[mask]
                if len(group_data) > 0:
                    f.write(
                        f"  {group_name}: {len(group_data):,} records ({len(group_data) / len(combined_df) * 100:.1f}%)\n")

            f.write(f"\nHigh-Resolution Data:\n")
            f.write(
                f"  Fine granularity (≤5min): {len(high_res_data):,} records ({len(high_res_data) / len(combined_df) * 100:.1f}%)\n")
            f.write(
                f"  Ultra-fine (≤2min): {len(minute_data):,} records ({len(minute_data) / len(combined_df) * 100:.1f}%)\n")

            f.write(f"\nCollection Strategy Distribution:\n")
            for strategy, count in strategy_dist.items():
                f.write(f"  {strategy}: {count:,} records ({count / len(combined_df) * 100:.1f}%)\n")

            f.write(f"\nML Training Quality Metrics:\n")
            f.write(f"  Premium data (weight ≥ 8): {len(high_priority_data):,} records\n")
            f.write(f"  Average ML weight: {combined_df['ml_weight'].mean():.2f}\n")
            f.write(
                f"  Recent data percentage: {len(combined_df[combined_df['data_age_days'] <= 8]) / len(combined_df) * 100:.1f}%\n")

        print(f"✅ Enhanced summary saved to {summary_filepath}")

        print(f"\n🎉 RECENT-FOCUSED collection completed successfully!")
        print(f"📊 Collected {len(combined_df):,} total records")
        print(f"🔥 {len(combined_df[combined_df['data_age_days'] <= 8]):,} recent high-quality records for ML training")

        return True

    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

def test_weekend_periods():
    """Enhanced test function to verify weekend period calculations and strategies"""
    print("=== Testing Enhanced Weekend Period Calculation ===")
    
    periods = get_weekend_traffic_periods(4)  # Test with 4 weeks
    
    gmt7 = pytz.timezone('Asia/Bangkok')
    
    for i, (start_utc, end_utc, priority, age) in enumerate(periods):
        start_gmt7 = start_utc.astimezone(gmt7)
        end_gmt7 = end_utc.astimezone(gmt7)
        
        duration_hours = (end_utc - start_utc).total_seconds() / 3600
        
        # Get strategy for this period
        period_seconds, expected_points, ml_weight, strategy = calculate_optimal_granularity_strategy(
            duration_hours, age
        )
        
        print(f"\nPeriod {i+1}:")
        print(f"  📅 GMT+7: {start_gmt7.strftime('%Y-%m-%d %a %H:%M')} to {end_gmt7.strftime('%Y-%m-%d %a %H:%M')}")
        print(f"  🌍 UTC: {start_utc.strftime('%Y-%m-%d %H:%M')} to {end_utc.strftime('%Y-%m-%d %H:%M')}")
        print(f"  ⏱️  Duration: {duration_hours:.1f} hours ({duration_hours/24:.1f} days)")
        print(f"  🏆 Priority: {priority}, Age: {age} days")
        print(f"  🎯 Strategy: {strategy}")
        print(f"  📏 Period: {period_seconds}s, Expected points: {expected_points}, ML weight: {ml_weight}")

def plot_tpm_area_chart_fixed(csv_file_path=None, start_date=None, end_date=None, figsize=(15, 8),
                              save_plot=True, show_plot=True, output_dir="charts"):
    """
    Vẽ area chart của dữ liệu TPM theo thời gian từ file CSV đã collect - VERSION ĐÃ SỬA TIMEZONE BUG

    Parameters:
    -----------
    csv_file_path : str, optional
        Đường dẫn đến file CSV. Nếu None, sẽ tìm file mới nhất trong thư mục datasets
    start_date : str, optional
        Ngày bắt đầu để filter dữ liệu (format: 'YYYY-MM-DD')
    end_date : str, optional
        Ngày kết thúc để filter dữ liệu (format: 'YYYY-MM-DD')
    figsize : tuple
        Kích thước figure (width, height)
    save_plot : bool
        Có lưu chart thành file không
    show_plot : bool
        Có hiển thị chart không
    output_dir : str
        Thư mục để lưu chart

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object của chart đã tạo
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    import seaborn as sns
    import glob
    import os

    # Setup style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Tìm file CSV nếu không được cung cấp
    if csv_file_path is None:
        dataset_files = glob.glob(os.path.join('datasets', 'newrelic_weekend_traffic*.csv'))
        if not dataset_files:
            print("❌ Không tìm thấy file dữ liệu trong thư mục datasets/")
            return None
        csv_file_path = max(dataset_files, key=os.path.getmtime)  # File mới nhất
        print(f"📊 Sử dụng file: {os.path.basename(csv_file_path)}")

    # Đọc dữ liệu
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Đọc thành công {len(df)} records từ {os.path.basename(csv_file_path)}")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None

    # Chuyển đổi timestamp và chuẩn hóa timezone
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Giữ nguyên timezone GMT+7, chỉ remove timezone info nếu có
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        print("🔧 Removed timezone info, keeping original GMT+7 time")
    else:
        print("📅 Using original timestamp (GMT+7)")

    # Filter theo khoảng thời gian nếu được cung cấp
    if start_date or end_date:
        original_count = len(df)
        if start_date:
            # Tạo timezone-naive datetime để so sánh
            start_datetime = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(
                start_date).tz is None else pd.to_datetime(start_date).tz_localize(None)
            df = df[df['timestamp'] >= start_datetime]
        if end_date:
            # Tạo timezone-naive datetime để so sánh
            end_datetime = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(
                end_date).tz is None else pd.to_datetime(end_date).tz_localize(None)
            end_datetime = end_datetime + pd.Timedelta(days=1)  # Include whole end day
            df = df[df['timestamp'] <= end_datetime]
        print(f"🔍 Filtered từ {original_count} xuống {len(df)} records")

        if df.empty:
            print("❌ Không có dữ liệu trong khoảng thời gian đã chọn")
            return None

    # Sắp xếp theo thời gian
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Tạo figure và subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    fig.suptitle('🚀 New Relic Weekend Traffic Analysis - TPM Area Chart (Fixed)',
                 fontsize=16, fontweight='bold', y=0.98)

    # Màu sắc cho các granularity khác nhau
    granularity_colors = {
        'minute': '#2E86AB',  # Blue
        'multi_minute': '#A23B72',  # Purple
        'hourly': '#F18F01',  # Orange
        'multi_hour': '#C73E1D'  # Red
    }

    # Main area chart - TPM over time
    ax1 = axes[0]

    # Tạo area chart cho từng granularity category
    if 'granularity_category' in df.columns:
        categories = df['granularity_category'].unique()

        for i, category in enumerate(sorted(categories)):
            category_data = df[df['granularity_category'] == category].copy()
            if not category_data.empty:
                color = granularity_colors.get(category, f'C{i}')
                ax1.fill_between(category_data['timestamp'],
                                 category_data['tpm'],
                                 alpha=0.7,
                                 color=color,
                                 label=f'{category.replace("_", " ").title()} Granularity')
    else:
        # Fallback: single area chart
        ax1.fill_between(df['timestamp'], df['tpm'],
                         alpha=0.7, color='#2E86AB', label='TPM')

    # Thêm line cho moving average
    if len(df) > 10:
        df['tpm_ma_10'] = df['tpm'].rolling(window=10, min_periods=1).mean()
        ax1.plot(df['timestamp'], df['tpm_ma_10'],
                 color='red', linewidth=2, alpha=0.8,
                 label='10-period Moving Average')

    # Customize main chart
    ax1.set_ylabel('TPM (Transactions Per Minute)', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'📈 TPM Area Chart - {df["timestamp"].min().strftime("%Y-%m-%d")} to {df["timestamp"].max().strftime("%Y-%m-%d")}',
        fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    # Thêm thông tin về peak values
    max_tpm = df['tpm'].max()
    max_tpm_time = df.loc[df['tpm'].idxmax(), 'timestamp']
    ax1.annotate(f'Peak: {max_tpm:.0f} TPM',
                 xy=(max_tpm_time, max_tpm),
                 xytext=(10, 20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Response Time subplot
    ax2 = axes[1]
    if 'response_time' in df.columns:
        ax2.fill_between(df['timestamp'], df['response_time'],
                         alpha=0.6, color='orange', label='Response Time')
        ax2.set_ylabel('Response Time (ms)', fontsize=10)
        ax2.set_xlabel('Time (GMT+7)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))

    # Rotate x-axis labels for better readability
    for ax in axes:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Add statistics text box
    stats_text = f"""📊 Dataset Statistics:
• Total Records: {len(df):,}
• Time Range: {(df['timestamp'].max() - df['timestamp'].min()).days} days  
• TPM Range: {df['tpm'].min():.0f} - {df['tpm'].max():.0f}
• Avg TPM: {df['tpm'].mean():.1f}"""

    if 'response_time' in df.columns:
        stats_text += f"\n• Avg Response Time: {df['response_time'].mean():.1f}ms"

    # Add granularity info
    if 'granularity_category' in df.columns:
        granularity_dist = df['granularity_category'].value_counts()
        stats_text += f"\n• Data Quality: {len(granularity_dist)} granularity levels"

    # Add statistics box to the plot
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # Save plot nếu được yêu cầu
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tpm_area_chart_fixed_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Chart saved: {filepath}")

    # Show plot nếu được yêu cầu
    if show_plot:
        plt.show()

    return fig


def plot_tpm_heatmap_hourly(csv_file_path=None, start_date=None, end_date=None,
                            figsize=(14, 8), save_plot=True, show_plot=True,
                            output_dir="charts", aggregation_method='mean'):
    """
    Vẽ heatmap của TPM theo khung thời gian mỗi giờ từ training data đã thu thập
    Chỉ vẽ 1 heatmap với trục ngang là giờ trong ngày (00h-24h) và trục dọc là TPM
    Giữ nguyên thời gian GMT+7, không convert về UTC

    Parameters:
    -----------
    csv_file_path : str, optional
        Đường dẫn đến file CSV. Nếu None, sẽ tìm file mới nhất trong thư mục datasets
    start_date : str, optional
        Ngày bắt đầu để filter dữ liệu (format: 'YYYY-MM-DD')
    end_date : str, optional
        Ngày kết thúc để filter dữ liệu (format: 'YYYY-MM-DD')
    figsize : tuple
        Kích thước figure (width, height)
    save_plot : bool
        Có lưu chart thành file không
    show_plot : bool
        Có hiển thị chart không
    output_dir : str
        Thư mục để lưu chart
    aggregation_method : str
        Phương thức tính toán ('mean', 'max', 'median', 'sum')

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object của heatmap đã tạo
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import glob
    import os
    from datetime import datetime

    # Setup style
    plt.style.use('default')
    sns.set_palette("RdYlBu_r")

    # Tìm file CSV nếu không được cung cấp
    if csv_file_path is None:
        dataset_files = glob.glob(os.path.join('datasets', 'newrelic_weekend_traffic*.csv'))
        if not dataset_files:
            print("❌ Không tìm thấy file dữ liệu trong thư mục datasets/")
            return None
        csv_file_path = max(dataset_files, key=os.path.getmtime)  # File mới nhất
        print(f"📊 Sử dụng file: {os.path.basename(csv_file_path)}")

    # Đọc dữ liệu
    try:
        df = pd.read_csv(csv_file_path)
        print(f"✅ Đọc thành công {len(df)} records từ {os.path.basename(csv_file_path)}")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None

    # Chuyển đổi timestamp và GIỮ NGUYÊN timezone GMT+7
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Chỉ remove timezone info nếu có, KHÔNG convert về UTC
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        print("🔧 Removed timezone info, keeping original GMT+7 time")
    else:
        print("📅 Using original timestamp (GMT+7)")

    # Filter theo khoảng thời gian nếu được cung cấp
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
        print(f"📊 Sử dụng toàn bộ training data: {len(df)} records")

    if df.empty:
        print("❌ Không có dữ liệu trong khoảng thời gian đã chọn")
        return None

    # Tạo các cột thời gian để group (GMT+7)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date

    # Tạo pivot table cho heatmap
    if aggregation_method == 'mean':
        heatmap_data = df.groupby(['day_of_week', 'hour'])['tpm'].mean().unstack(fill_value=0)
        title_suffix = "Average TPM"
        cbar_label = "Average TPM"
    elif aggregation_method == 'max':
        heatmap_data = df.groupby(['day_of_week', 'hour'])['tpm'].max().unstack(fill_value=0)
        title_suffix = "Maximum TPM"
        cbar_label = "Maximum TPM"
    elif aggregation_method == 'median':
        heatmap_data = df.groupby(['day_of_week', 'hour'])['tpm'].median().unstack(fill_value=0)
        title_suffix = "Median TPM"
        cbar_label = "Median TPM"
    else:  # sum
        heatmap_data = df.groupby(['day_of_week', 'hour'])['tpm'].sum().unstack(fill_value=0)
        title_suffix = "Total TPM"
        cbar_label = "Total TPM"

    # Sắp xếp lại thứ tự ngày trong tuần
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)

    # Tạo figure và axes
    fig, ax = plt.subplots(figsize=figsize)

    # Vẽ heatmap
    sns.heatmap(heatmap_data,
                ax=ax,
                cmap='RdYlBu_r',
                annot=True,
                fmt='.0f',
                cbar_kws={'label': cbar_label},
                linewidths=0.8,
                square=False,
                annot_kws={'size': 8})

    # Tùy chỉnh tiêu đề và labels
    date_range_text = ""
    if start_date or end_date:
        if start_date and end_date:
            date_range_text = f" ({start_date} to {end_date})"
        elif start_date:
            date_range_text = f" (from {start_date})"
        elif end_date:
            date_range_text = f" (to {end_date})"

    ax.set_title(f'🔥 TPM Heatmap - {title_suffix}{date_range_text} (GMT+7)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day (GMT+7)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Day of Week', fontsize=14, fontweight='bold')

    # Tùy chỉnh tick labels
    ax.set_xticklabels([f'{i:02d}:00' for i in range(24)], rotation=45)
    ax.set_yticklabels(day_order, rotation=0)

    # Thêm thống kê tổng quan
    stats_text = f"""📊 Statistics (GMT+7):
Records: {len(df):,}
Date Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}
TPM Range: {df['tpm'].min():.0f} - {df['tpm'].max():.0f}
Avg TPM: {df['tpm'].mean():.1f}"""

    # Đặt text box ở góc dưới trái
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')

    # Điều chỉnh layout
    plt.tight_layout()

    # Save plot nếu được yêu cầu
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)

        # Tạo filename với timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        date_suffix = ""
        if start_date or end_date:
            if start_date and end_date:
                date_suffix = f"_{start_date}_to_{end_date}"
            elif start_date:
                date_suffix = f"_from_{start_date}"
            elif end_date:
                date_suffix = f"_to_{end_date}"

        filename = f"tpm_heatmap_hourly_{aggregation_method}{date_suffix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"💾 Heatmap saved: {filepath}")

    # Show plot nếu được yêu cầu
    if show_plot:
        plt.show()

    return fig


def plot_tpm_comparison_charts(csv_file_path=None, comparison_type='day_of_week',
                               figsize=(16, 10), save_plot=True, show_plot=True,
                               output_dir="charts"):
    """
    Vẽ comparison charts để so sánh TPM theo different dimensions

    Parameters:
    -----------
    csv_file_path : str, optional
        Đường dẫn đến file CSV
    comparison_type : str
        Loại comparison: 'day_of_week', 'hour_of_day', 'granularity', 'weekend_vs_weekday'
    figsize : tuple
        Kích thước figure
    save_plot : bool
        Có lưu chart không
    show_plot : bool
        Có hiển thị chart không
    output_dir : str
        Thư mục lưu chart

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import glob

    # Tìm file CSV nếu không được cung cấp
    if csv_file_path is None:
        dataset_files = glob.glob(os.path.join('datasets', 'newrelic_weekend_traffic*.csv'))
        if not dataset_files:
            print("❌ Không tìm thấy file dữ liệu")
            return None
        csv_file_path = max(dataset_files, key=os.path.getmtime)

    # Đọc dữ liệu
    try:
        df = pd.read_csv(csv_file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ Đọc {len(df)} records cho comparison analysis")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None

    plt.style.use('seaborn-v0_8')

    if comparison_type == 'day_of_week':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('📅 TPM Analysis by Day of Week', fontsize=16, fontweight='bold')

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        df['day_name'] = df['day_of_week'].map(dict(enumerate(day_names)))

        # Box plot by day
        sns.boxplot(data=df, x='day_name', y='tpm', ax=axes[0,0])
        axes[0,0].set_title('TPM Distribution by Day')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Average TPM by day
        daily_avg = df.groupby('day_name')['tpm'].mean().reindex(day_names)
        axes[0,1].bar(daily_avg.index, daily_avg.values, color='skyblue')
        axes[0,1].set_title('Average TPM by Day')
        axes[0,1].tick_params(axis='x', rotation=45)

        # TPM heatmap by day and hour
        pivot_data = df.pivot_table(values='tpm', index='hour', columns='day_name', aggfunc='mean')
        pivot_data = pivot_data.reindex(columns=day_names)
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1,0])
        axes[1,0].set_title('TPM Heatmap (Hour vs Day)')

        # Response time by day if available
        if 'response_time' in df.columns:
            daily_response = df.groupby('day_name')['response_time'].mean().reindex(day_names)
            axes[1,1].bar(daily_response.index, daily_response.values, color='orange')
            axes[1,1].set_title('Avg Response Time by Day')
            axes[1,1].tick_params(axis='x', rotation=45)

    elif comparison_type == 'hour_of_day':
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🕐 TPM Analysis by Hour of Day', fontsize=16, fontweight='bold')

        # TPM by hour - line chart
        hourly_avg = df.groupby('hour')['tpm'].agg(['mean', 'std']).reset_index()
        axes[0,0].plot(hourly_avg['hour'], hourly_avg['mean'], marker='o', linewidth=2)
        axes[0,0].fill_between(hourly_avg['hour'],
                               hourly_avg['mean'] - hourly_avg['std'],
                               hourly_avg['mean'] + hourly_avg['std'],
                               alpha=0.3)
        axes[0,0].set_title('TPM by Hour (with std deviation)')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].grid(True, alpha=0.3)

        # Box plot by hour
        sns.boxplot(data=df, x='hour', y='tpm', ax=axes[0,1])
        axes[0,1].set_title('TPM Distribution by Hour')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Violin plot for detailed distribution
        sns.violinplot(data=df, x='hour', y='tpm', ax=axes[1,0])
        axes[1,0].set_title('TPM Density by Hour')
        axes[1,0].tick_params(axis='x', rotation=90)

        # Peak vs off-peak comparison
        peak_hours = df['hour'].between(10, 16)
        df['period'] = peak_hours.map({True: 'Peak (10-16h)', False: 'Off-Peak'})
        period_stats = df.groupby('period')['tpm'].agg(['mean', 'std', 'count'])

        x_pos = range(len(period_stats))
        axes[1,1].bar(x_pos, period_stats['mean'],
                      yerr=period_stats['std'], capsize=5, color=['red', 'blue'])
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(period_stats.index)
        axes[1,1].set_title('Peak vs Off-Peak TPM')

    plt.tight_layout()

    # Save if requested
    if save_plot:
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tpm_comparison_{comparison_type}_{timestamp_suffix}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"💾 Comparison chart saved to: {filepath}")
        except Exception as e:
            print(f"⚠️ Error saving chart: {e}")

    if show_plot:
        plt.show()

    return fig


def create_traffic_visualization_suite_enhanced(csv_file_path=None, start_date=None, end_date=None):
    """
    Tạo bộ visualization hoàn chỉnh bao gồm cả heatmap analysis
    """
    print("🚀 Tạo bộ Traffic Visualization Suite Enhanced...")

    figures = {}

    # 1. TPM Area Chart
    print("📈 1. Creating TPM Area Chart...")
    try:
        fig_area = plot_tpm_area_chart_fixed(csv_file_path, start_date, end_date)
        if fig_area:
            figures['area_chart'] = fig_area
            print("✅ TPM Area Chart created successfully")
    except Exception as e:
        print(f"❌ Error creating TPM Area Chart: {e}")

    # 3. TPM Heatmap Hourly
    print("🔥 3. Creating TPM Hourly Heatmap...")
    try:
        fig_heatmap = plot_tpm_heatmap_hourly(csv_file_path)
        if fig_heatmap:
            figures['hourly_heatmap'] = fig_heatmap
            print("✅ TPM Hourly Heatmap created successfully")
    except Exception as e:
        print(f"❌ Error creating TPM Hourly Heatmap: {e}")


    # 4. Advanced Multi-dimensional Heatmap
    print("🔥 4. Creating Advanced Multi-dimensional Heatmap...")
    try:
        fig_advanced = plot_tpm_comparison_charts(csv_file_path, comparison_type='day_of_week')
        if fig_advanced:
            figures['advanced_heatmap'] = fig_advanced
            print("✅ Advanced Heatmap created successfully")
    except Exception as e:
        print(f"❌ Error creating Advanced Heatmap: {e}")

    print(f"🎯 Enhanced Traffic Visualization Suite completed! Created {len(figures)} visualizations.")
    return figures

# ===== Helper for live inference: fetch last 3 hours at 1-minute granularity =====

def get_recent_3h_1min_dataframe(api_key=None, app_id=None, metric_name="HttpDispatcher"):
    """
    Fetch last 3 hours of New Relic metric data at ~1-minute granularity and return
    a minimal DataFrame suitable for model feature creation.

    Parameters:
    - api_key: New Relic API Key (falls back to env NEWRELIC_API_KEY/NEW_RELIC_API_KEY)
    - app_id: New Relic Application ID (falls back to env NEWRELIC_APP_ID/NEW_RELIC_APP_ID)
    - metric_name: Metric to query (default 'HttpDispatcher')

    Returns:
    - pandas.DataFrame with at least ['timestamp', 'tpm'] columns, sorted by timestamp.
    """
    # Lazy imports and env fallbacks
    if api_key is None:
        api_key = os.getenv("NEWRELIC_API_KEY") or os.getenv("NEW_RELIC_API_KEY")
    if app_id is None:
        app_id = os.getenv("NEWRELIC_APP_ID") or os.getenv("NEW_RELIC_APP_ID")

    if not api_key or not app_id:
        print("❌ Missing New Relic credentials. Set NEWRELIC_API_KEY and NEWRELIC_APP_ID or pass via arguments.")
        return None

    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=3)

        data = collect_newrelic_data_with_optimal_granularity(
            api_key=api_key,
            app_id=app_id,
            metrics=[metric_name],
            start_time=start_time,
            end_time=end_time,
            week_priority=10,
            data_age_days=0,
        )

        if not data:
            print("❌ No data returned from New Relic API for the recent 3h window.")
            return None

        df = process_newrelic_data_with_enhanced_metadata(data)
        if df is None or df.empty:
            print("⚠️ No timeslices processed for the recent 3h window.")
            return None

        # Ensure required columns exist
        if 'timestamp' not in df.columns or 'tpm' not in df.columns:
            print("❌ Processed data missing required columns 'timestamp' and 'tpm'.")
            return None

        # Keep only necessary columns for the predictor's feature creation
        df_min = df[['timestamp', 'tpm']].copy()
        df_min = df_min.sort_values('timestamp').reset_index(drop=True)

        # Clean/clip TPM values and forward/back fill if there are rare gaps
        df_min['tpm'] = df_min['tpm'].ffill().bfill().clip(lower=0)

        return df_min
    except Exception as e:
        print(f"❌ Error fetching recent 3h data: {e}")
        return None

def main():
    print("\n" + "=" * 60)

    # Load API key
    API_KEY = os.getenv("NEWRELIC_API_KEY")

    if not API_KEY:
        print("❌ NEWRELIC_API_KEY not found in environment variables")
        exit(1)

    print(f"✅ API Key loaded: {API_KEY[:10]}...{API_KEY[-4:] if len(API_KEY) > 14 else ''}")

    # Collect weekend traffic data with optimal strategy
    success = collect_and_save_newrelic_data(
        API_KEY,
        weeks_back=12,  # 3 months
        filename="newrelic_weekend_traffic_enhanced.csv"
    )

if __name__ == "__main__":
    main()