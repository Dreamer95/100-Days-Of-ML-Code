# New Relic Data Collector

This script collects metrics data from New Relic, processes it, and saves it to a CSV file for model training.

## Features

- Collects TPM (Transactions Per Minute) and other metrics from New Relic
- Processes the data into a format suitable for model training
- Saves the data to a CSV file in the datasets directory

## Requirements

- Python 3.6+
- Required packages: requests, pandas

## Installation

```bash
pip install requests pandas
```

## Usage

1. Replace the placeholder API key in the script with your actual New Relic API key:

```python
API_KEY = "YOUR_NEW_RELIC_API_KEY"
```

2. Run the script:

```bash
python Src/Day3/newrelic_data_collector.py
```

3. Alternatively, you can import and use the functions in your own code:

```python
from Src.Day3.newrelic_data_collector import collect_and_save_newrelic_data

# Replace with your actual New Relic API key
API_KEY = "YOUR_NEW_RELIC_API_KEY"

# Collect data for the last 7 days and save to newrelic_metrics.csv
collect_and_save_newrelic_data(API_KEY)

# Or customize the parameters
collect_and_save_newrelic_data(
    api_key=API_KEY,
    app_id="1080863725",  # Your New Relic application ID
    days=30,  # Collect data for the last 30 days
    filename="custom_metrics.csv"  # Custom filename
)
```

## Function Reference

### collect_and_save_newrelic_data

```python
collect_and_save_newrelic_data(api_key, app_id="1080863725", days=7, filename="newrelic_metrics.csv")
```

Main function to collect data from New Relic, process it, and save to CSV.

Parameters:
- `api_key` (str): New Relic API key for authentication
- `app_id` (str, optional): Application ID in New Relic. Default is "1080863725".
- `days` (int, optional): Number of days of data to collect. Default is 7.
- `filename` (str, optional): Name of the CSV file to save. Default is "newrelic_metrics.csv".

Returns:
- `bool`: True if successful, False otherwise

## Output Format

The script generates a CSV file with the following columns:

- `timestamp`: Date and time of the data point
- `tpm`: Transactions Per Minute
- `error_rate`: Error rate (if available)
- `response_time`: Response time in seconds (if available)
- `is_push_notification`: Binary flag (0 or 1) indicating if a push notification was sent
- `play_counted`: Cumulative count of plays/transactions

## New Relic API Documentation

For more information about the New Relic API and time ranges, see:
- [New Relic API Documentation](https://docs.newrelic.com/docs/apis/rest-api-v2/basic-functions/extract-metric-timeslice-data/)