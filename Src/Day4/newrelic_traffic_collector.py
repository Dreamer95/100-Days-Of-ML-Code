import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta, timezone
import pytz
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

def get_weekend_traffic_periods(weeks_back=12):
    """
    Generate optimized time periods for weekend traffic collection (Thu 23:50 - Sun 00:00 GMT+7)
    
    Parameters:
    -----------
    weeks_back : int
        Number of weeks to go back (default: 12 weeks ≈ 3 months)
    
    Returns:
    --------
    list: List of tuples (start_time_utc, end_time_utc, week_priority, data_age_days)
    """
    gmt7 = pytz.timezone('Asia/Bangkok')
    utc = pytz.timezone('UTC')
    
    periods = []
    now_gmt7 = datetime.now(gmt7)

    for week in range(weeks_back):
        # Calculate the target Thursday for this week
        days_back = week * 7
        target_date = now_gmt7 - timedelta(days=days_back)

        # Find the Thursday of this week
        days_since_thursday = (target_date.weekday() - 3) % 7
        thursday = target_date - timedelta(days=days_since_thursday)

        # Set to 23:50 Thursday GMT+7
        start_time_gmt7 = thursday.replace(hour=23, minute=50, second=0, microsecond=0)

        # End time: Sunday 00:00 GMT+7 (48 hours 10 minutes later)
        sunday = thursday + timedelta(days=3)  # Thursday + 3 days = Sunday
        end_time_gmt7 = sunday.replace(hour=0, minute=0, second=0, microsecond=0)

        # Convert to UTC
        start_time_utc = start_time_gmt7.astimezone(utc).replace(tzinfo=timezone.utc)
        end_time_utc = end_time_gmt7.astimezone(utc).replace(tzinfo=timezone.utc)

        # Skip future dates
        if start_time_utc > datetime.now(timezone.utc):
            continue

        # Calculate priority (most recent week = highest priority)
        week_priority = weeks_back - week

        # Calculate data age
        data_age_days = (datetime.now(timezone.utc) - start_time_utc).days

        periods.append((start_time_utc, end_time_utc, week_priority, data_age_days))

    return periods

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

def collect_and_save_newrelic_data(api_key, app_id="1080863725", weeks_back=12, filename="newrelic_weekend_traffic.csv"):
    """
    Collect weekend traffic data from New Relic with optimal granularity strategy
    """
    print(f"🚀 Collecting optimized weekend traffic data for ML training...")
    print(f"📅 Collecting {weeks_back} weeks of weekend periods (Thu 23:50 - Sun 00:00 GMT+7)")
    print(f"🎯 Using optimal granularity strategy based on New Relic API rules")
    
    # Get weekend periods
    periods = get_weekend_traffic_periods(weeks_back)
    
    if not periods:
        print("❌ No valid periods found")
        return False
    
    print(f"📊 Found {len(periods)} weekend periods to collect")
    
    # Show collection strategy summary
    print(f"\n📋 Collection Strategy Summary:")
    recent_periods = sum(1 for _, _, _, age in periods if age <= 8)
    historical_periods = len(periods) - recent_periods
    print(f"   Recent periods (≤8 days): {recent_periods} → High granularity (5-30 min intervals)")
    print(f"   Historical periods (>8 days): {historical_periods} → Limited granularity (10 points or 1+ hour intervals)")
    
    # Define metrics for traffic prediction
    metrics = [
        "HttpDispatcher",  # Main web transaction metric
    ]
    
    all_dataframes = []
    
    for i, (start_time, end_time, week_priority, data_age_days) in enumerate(periods):
        print(f"\n📈 Collecting period {i+1}/{len(periods)}")
        print(f"   📅 {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   🏆 Week priority: {week_priority}, Data age: {data_age_days} days")
        
        # Collect data for this period with optimal strategy
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
                print(f"   ✅ Collected {len(df)} data points (avg interval: {avg_interval:.1f} min)")
            else:
                print(f"   ⚠️  No data processed for this period")
        else:
            print(f"   ❌ Failed to collect data for this period")
        
        # Rate limiting
        time.sleep(0.5)
    
    if not all_dataframes:
        print("❌ No data collected from any period")
        return False
    
    # Combine all data
    print(f"\n🔄 Combining data from {len(all_dataframes)} periods...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by timestamp for time series analysis
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Add sequence number for time series
    combined_df['sequence_id'] = range(len(combined_df))
    
    # Enhanced summary
    print(f"\n📊 Enhanced Dataset Summary:")
    print(f"   📈 Total records: {len(combined_df)}")
    print(f"   📅 Time range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"   🚀 TPM range: {combined_df['tpm'].min():.0f} - {combined_df['tpm'].max():.0f}")
    print(f"   ⏱️  Response time range: {combined_df['response_time'].min():.1f} - {combined_df['response_time'].max():.1f}ms")
    print(f"   📏 Interval range: {combined_df['interval_minutes'].min():.1f} - {combined_df['interval_minutes'].max():.1f} minutes")
    
    # Strategy distribution
    print(f"\n🎯 Collection Strategy Distribution:")
    strategy_dist = combined_df['collection_strategy'].value_counts()
    for strategy, count in strategy_dist.items():
        print(f"   {strategy}: {count} records ({count/len(combined_df)*100:.1f}%)")
    
    # Granularity analysis
    print(f"\n📏 Granularity Analysis:")
    granularity_dist = combined_df['granularity_category'].value_counts()
    for category, count in granularity_dist.items():
        avg_interval = combined_df[combined_df['granularity_category'] == category]['interval_minutes'].mean()
        print(f"   {category}: {count} records (avg: {avg_interval:.1f} min intervals)")
    
    # High priority data summary
    high_priority_data = combined_df[combined_df['ml_weight'] >= 7]
    print(f"\n⭐ High priority data (weight ≥ 7): {len(high_priority_data)} records ({len(high_priority_data)/len(combined_df)*100:.1f}%)")
    
    # Save to CSV with enhanced columns
    try:
        os.makedirs('datasets', exist_ok=True)
        filepath = os.path.join('datasets', filename)
        combined_df.to_csv(filepath, index=False)
        print(f"\n✅ Enhanced data saved to {filepath}")
        
        # Enhanced summary report
        summary_filename = filename.replace('.csv', '_enhanced_summary.txt')
        summary_filepath = os.path.join('datasets', summary_filename)
        with open(summary_filepath, 'w') as f:
            f.write(f"Enhanced Weekend Traffic Data Collection Summary\n")
            f.write(f"===============================================\n\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Collection Strategy: Optimal granularity based on New Relic API rules\n")
            f.write(f"Total Records: {len(combined_df)}\n")
            f.write(f"Time Range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}\n")
            f.write(f"Weeks Collected: {weeks_back}\n")
            f.write(f"Weekend Periods: {len(periods)}\n\n")
            
            f.write("Collection Strategy Distribution:\n")
            for strategy, count in strategy_dist.items():
                f.write(f"  {strategy}: {count} records ({count/len(combined_df)*100:.1f}%)\n")
            
            f.write(f"\nGranularity Distribution:\n")
            for category, count in granularity_dist.items():
                avg_interval = combined_df[combined_df['granularity_category'] == category]['interval_minutes'].mean()
                f.write(f"  {category}: {count} records (avg: {avg_interval:.1f} min intervals)\n")
            
            f.write(f"\nData Quality Distribution:\n")
            for quality, count in combined_df['data_quality'].value_counts().items():
                f.write(f"  {quality}: {count} records ({count/len(combined_df)*100:.1f}%)\n")
            
            f.write(f"\nML Weight Distribution:\n")
            for weight, count in combined_df['ml_weight'].value_counts().sort_index().items():
                f.write(f"  Weight {weight}: {count} records\n")
            
            f.write(f"\nEnhanced Features Available:\n")
            timestamp_cols = [col for col in combined_df.columns if 'timestamp' in col.lower()]
            for col in timestamp_cols:
                f.write(f"  - {col}\n")
            
            feature_cols = [col for col in combined_df.columns if any(x in col for x in ['tpm', 'response_time', 'lag', 'ma', 'change', 'sin', 'cos'])]
            f.write(f"\nML Features Available ({len(feature_cols)} total):\n")
            for col in sorted(feature_cols):
                f.write(f"  - {col}\n")
        
        print(f"📋 Enhanced summary report saved to {summary_filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
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

# Add numpy import for circular encoding
import numpy as np

if __name__ == "__main__":
    # Test period calculation
    test_weekend_periods()
    
    print("\n" + "="*60)
    
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
    
    if success:
        print("\n🎉 Enhanced weekend traffic data collection completed!")
        print("🔮 Ready for advanced ML model training with optimal granularity!")
    else:
        print("\n❌ Data collection failed")