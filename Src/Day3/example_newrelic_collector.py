#!/usr/bin/env python3
"""
Example script demonstrating how to use the New Relic Data Collector
"""

from Src.Day3.newrelic_data_collector import collect_and_save_newrelic_data
import os
import argparse

def main():
    """
    Main function to demonstrate the usage of the New Relic Data Collector
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect data from New Relic and save to CSV')
    parser.add_argument('--api-key', required=True, help='New Relic API key')
    parser.add_argument('--app-id', default='1080863725', help='New Relic application ID (default: 1080863725)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to collect (default: 7)')
    parser.add_argument('--filename', default='newrelic_metrics.csv', help='Output CSV filename (default: newrelic_metrics.csv)')

    args = parser.parse_args()

    # Collect and save data
    print(f"Collecting data for application {args.app_id} for the last {args.days} days...")
    success = collect_and_save_newrelic_data(
        api_key=args.api_key,
        app_id=args.app_id,
        days=args.days,
        filename=args.filename
    )

    if success:
        # Get the full path to the saved file
        filepath = os.path.join(os.getcwd(), 'datasets', args.filename)
        print(f"Data collection completed successfully!")
        print(f"Data saved to: {filepath}")

        # Display the first few rows of the CSV file
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            print("\nPreview of the collected data:")
            print(df.head())

            print(f"\nTotal records: {len(df)}")
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Average TPM: {df['tpm'].mean():.2f}")

            if 'error_rate' in df.columns:
                print(f"Average error rate: {df['error_rate'].mean():.4f}")

            if 'response_time' in df.columns:
                print(f"Average response time: {df['response_time'].mean():.4f} seconds")
        except Exception as e:
            print(f"Error displaying data preview: {e}")
    else:
        print("Data collection failed. Check the error messages above for details.")

if __name__ == "__main__":
    main()
