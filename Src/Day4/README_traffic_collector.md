# New Relic Traffic Collector for Prediction Model

This module collects traffic data from New Relic API for training a traffic prediction model. The data is optimized for predicting website traffic trends in 5, 10, 20, and 30-minute intervals.

## Features

- Collects data from New Relic API for the specified time range
- Filters data for specific hours (11:00 PM Wednesday to 1:00 AM Sunday each week)
- Applies granularity rules (5 minutes for recent week, 1 hour for older data)
- Extracts requests_per_minute from the API response
- Handles timezone conversion from UTC to GMT+7
- Implements rate limiting to avoid API throttling
- Creates a CSV with features optimized for time series prediction
- Simulates push notification events and their effects on traffic

## Requirements

- Python 3.6+
- Required packages: requests, pandas, numpy, pytz

## Installation

```bash
pip install requests pandas numpy pytz
```

## Usage

### Basic Usage

```python
from datetime import datetime
import pytz
from Src.Day4.newrelic_traffic_collector import NewRelicTrafficCollector

# Replace with your actual New Relic API key
API_KEY = "YOUR_NEW_RELIC_API_KEY"

# Create collector
collector = NewRelicTrafficCollector(API_KEY)

# Collect and process data for May-July 2025
start_date = datetime(2025, 5, 1, tzinfo=pytz.timezone('Asia/Bangkok'))
end_date = datetime(2025, 7, 31, 23, 59, 59, tzinfo=pytz.timezone('Asia/Bangkok'))

df = collector.collect_and_process_data(start_date, end_date, "traffic_prediction_data.csv")

print(f"Data collection and processing completed with {len(df)} records")
```

### Advanced Usage

You can also use the individual methods for more control:

```python
from datetime import datetime
import pytz
from Src.Day4.newrelic_traffic_collector import NewRelicTrafficCollector

# Create collector
collector = NewRelicTrafficCollector("YOUR_API_KEY")

# Define date range
start_date = datetime(2025, 5, 1, tzinfo=pytz.timezone('Asia/Bangkok'))
end_date = datetime(2025, 7, 31, tzinfo=pytz.timezone('Asia/Bangkok'))

# Collect raw data
data = collector.collect_data(start_date, end_date)

# Process data for prediction
df = collector.process_data_for_prediction(data)

# Simulate push notifications
df = collector.simulate_push_notifications(df)

# Save to CSV
import os
os.makedirs('datasets', exist_ok=True)
df.to_csv('datasets/custom_traffic_data.csv', index=False)
```

## Output Format

The script generates a CSV file with the following columns:

- `timestamp`: Date and time of the data point (in GMT+7 timezone)
- `requests_per_minute`: Number of requests per minute
- `granularity_minutes`: Granularity of the data point (5 or 60 minutes)
- `hour`: Hour of the day (0-23)
- `day_of_week`: Day of the week (0=Monday, 6=Sunday)
- `is_weekend`: Binary flag (0 or 1) indicating if the day is a weekend
- `week_of_year`: Week number of the year
- `data_age_days`: Age of the data in days from the most recent timestamp
- `requests_lag_1`, `requests_lag_2`, `requests_lag_3`: Previous values of requests_per_minute
- `requests_rolling_mean_5`, `requests_rolling_mean_10`: Rolling means of requests_per_minute
- `push_notification_active`: Binary flag (0 or 1) indicating if a push notification is active
- `minutes_since_push`: Minutes elapsed since the last push notification
- `target_5min`, `target_10min`, `target_20min`, `target_30min`: Future values of requests_per_minute

## Time Range Details

- **Period**: 3 months (May 2025 to July 2025)
- **Specific Hours**: Only collects data from 11:00 PM Wednesday to 1:00 AM Sunday each week (GMT+7 timezone)
- **Granularity Rules**:
  - Most recent week: 5 minutes per record
  - All other weeks: 1 hour per record

## Push Notification Feature

The script simulates push notification events at 11:00 AM each day and their effects on traffic:

- `push_notification_active`: Set to 1 at 11:00 AM and for the next 10 minutes
- `minutes_since_push`: Minutes elapsed since the last push notification
- When a push notification occurs, requests_per_minute typically increases steadily for the next 10 minutes

## API Details

- **Endpoint**: https://api.newrelic.com/v2/applications/1080863725/metrics/data.json
- **Method**: GET
- **Authentication**: API-Key header required
- **Metric**: HttpDispatcher

## Error Handling

The script includes error handling for API requests and data processing. If an error occurs, it will be printed to the console.

## Rate Limiting

The script implements rate limiting to avoid API throttling. By default, it waits 1 second between API calls, but this can be adjusted by modifying the `rate_limit_delay` attribute.