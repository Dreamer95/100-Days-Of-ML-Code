#!/usr/bin/env python3
"""
Example script demonstrating how to use the New Relic Traffic Collector
"""

import os
import argparse
from datetime import datetime
import pytz
from Src.Day4.newrelic_traffic_collector import NewRelicTrafficCollector

def main():
    """
    Main function to demonstrate the usage of the New Relic Traffic Collector
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect traffic data from New Relic for prediction model')
    parser.add_argument('--api-key', required=True, help='New Relic API key')
    parser.add_argument('--app-id', default='1080863725', help='New Relic application ID (default: 1080863725)')
    parser.add_argument('--start-date', default='2025-05-01', help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-07-31', help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--output-file', default='traffic_prediction_data.csv', help='Output CSV filename')

    args = parser.parse_args()

    # Parse dates
    gmt7 = pytz.timezone('Asia/Bangkok')
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').replace(tzinfo=gmt7)
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=gmt7)

    print(f"New Relic Traffic Collector Example")
    print(f"===================================")
    print(f"Collecting data for application {args.app_id}")
    print(f"Time range: {start_date} to {end_date} (GMT+7)")
    print(f"Output file: {args.output_file}")
    print()

    # Create collector
    collector = NewRelicTrafficCollector(args.api_key, args.app_id)

    # Collect and process data
    print("Starting data collection and processing...")
    df = collector.collect_and_process_data(start_date, end_date, args.output_file)

    if df.empty:
        print("No data collected. Check error messages above.")
        return

    # Display summary statistics
    print("\nData Collection Summary:")
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Average requests per minute: {df['requests_per_minute'].mean():.2f}")
    
    # Count records by granularity
    granularity_counts = df['granularity_minutes'].value_counts()
    print("\nRecords by granularity:")
    for granularity, count in granularity_counts.items():
        print(f"  {granularity} minutes: {count} records")
    
    # Count records by day of week
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().sort_index()
    print("\nRecords by day of week:")
    for day_num, count in day_counts.items():
        print(f"  {day_names[day_num]}: {count} records")
    
    # Count push notification events
    push_count = df[df['push_notification_active'] == 1]['timestamp'].dt.date.nunique()
    print(f"\nPush notification events: {push_count}")
    
    # Display preview of the data
    print("\nPreview of collected data:")
    preview_columns = ['timestamp', 'requests_per_minute', 'hour', 'day_of_week', 
                      'granularity_minutes', 'push_notification_active']
    print(df[preview_columns].head())
    
    # Get the full path to the saved file
    filepath = os.path.join(os.getcwd(), 'datasets', args.output_file)
    print(f"\nData saved to: {filepath}")
    print(f"Data collection and processing completed successfully!")

if __name__ == "__main__":
    main()