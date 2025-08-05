# Traffic Prediction Model

This module provides a machine learning model for predicting website traffic in 5, 10, 20, and 30-minute intervals based on historical data collected from New Relic.

## Features

- Predicts website traffic for 5, 10, 20, and 30 minutes into the future
- Takes into account time-based patterns (hour of day, day of week, weekend)
- Considers recent traffic trends through lag features and moving averages
- Handles push notification effects on traffic
- Uses ml_weight to give more importance to recent, high-quality data
- Supports multiple machine learning algorithms (Random Forest, Gradient Boosting, Linear Regression)
- Provides visualization of prediction results

## Requirements

- Python 3.6+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

### Basic Usage

```python
from Src.Day4.traffic_prediction_model import TrafficPredictionModel

# Create and train a new model
model = TrafficPredictionModel()
model.train_and_save()

# Make predictions
prediction = model.predict(
    current_tpm=500,
    hour=14,
    day_of_week=4,  # Friday
    previous_tpm_values=[480, 450, 420, 400, 380],
    push_notification_active=0
)

# Print predictions
for target, value in prediction.items():
    print(f"{target}: {value:.2f} TPM")
```

### Loading Pre-trained Models

```python
from Src.Day4.traffic_prediction_model import TrafficPredictionModel

# Load pre-trained models
model = TrafficPredictionModel()
model.load_models()

# Make predictions
prediction = model.predict(
    current_tpm=500,
    hour=14,
    day_of_week=4,
    previous_tpm_values=[480, 450, 420, 400, 380]
)
```

### Predicting with Push Notifications

```python
# Predict traffic with active push notification
prediction = model.predict(
    current_tpm=500,
    hour=14,
    day_of_week=4,
    previous_tpm_values=[480, 450, 420, 400, 380],
    push_notification_active=1,
    minutes_since_push=0
)

# Predict traffic 5 minutes after push notification
prediction = model.predict(
    current_tpm=550,
    hour=14,
    day_of_week=4,
    previous_tpm_values=[500, 480, 450, 420, 400],
    push_notification_active=0,
    minutes_since_push=5
)
```

## Model Details

### Input Features

The model uses the following features for prediction:

- **Time features**: hour, day_of_week, is_weekend, hour_sin, hour_cos
- **Current traffic**: tpm (transactions per minute), response_time
- **Historical traffic**: tpm_lag_1, tpm_lag_2, tpm_lag_3, tpm_lag_5
- **Statistical features**: tpm_ma_3, tpm_ma_5 (moving averages), tpm_std_3, tpm_std_5 (standard deviations)
- **Change metrics**: tpm_change, tpm_change_rate
- **Push notification**: push_notification_active, minutes_since_push

### Training Process

The model training process includes:

1. Loading and preprocessing the data
2. Feature selection and engineering
3. Splitting data into training and testing sets
4. Training multiple models (Random Forest, Gradient Boosting, Linear Regression)
5. Evaluating models using MSE, MAE, and R² metrics
6. Selecting the best performing model for each prediction target
7. Saving the trained models for future use

### Push Notification Effects

The model handles push notification effects as follows:

- When a push notification is active (just sent), traffic is expected to increase over the next 5-15 minutes
- The boost factor varies by prediction time:
  - 5-minute prediction: 20% increase
  - 10-minute prediction: 40% increase
  - 15-minute prediction: 50% increase
  - 20-30 minute prediction: 30% increase
- For predictions after a push notification, the effect diminishes over time

## File Structure

- `traffic_prediction_model.py`: Main implementation of the traffic prediction model
- `models/`: Directory containing saved models
  - `target_5min_model.pkl`: Model for 5-minute predictions
  - `target_10min_model.pkl`: Model for 10-minute predictions
  - `target_20min_model.pkl`: Model for 20-minute predictions
  - `target_30min_model.pkl`: Model for 30-minute predictions
  - `scaler.pkl`: Feature scaler
  - `feature_columns.pkl`: List of feature columns

## Example Output

```
=== Training new models ===
Loading data from Src/Day4/datasets/newrelic_weekend_traffic_enhanced.csv...
Loaded 576 records.
Preprocessing data...
Selected 22 features and 4 targets.
Training models...

Training model for target_5min...
  Training RandomForest...
    MSE: 1234.56, MAE: 23.45, R²: 0.92
  Training GradientBoosting...
    MSE: 1345.67, MAE: 25.67, R²: 0.91
  Training LinearRegression...
    MSE: 2345.67, MAE: 35.67, R²: 0.85
Best model for target_5min: MSE=1234.56, R²=0.92

[... similar output for other targets ...]

=== Making predictions ===

Prediction for current traffic (no push notification):
  target_5min: 510.25 TPM
  target_10min: 520.50 TPM
  target_20min: 515.75 TPM
  target_30min: 505.25 TPM

Prediction for current traffic (with active push notification):
  target_5min: 612.30 TPM
  target_10min: 728.70 TPM
  target_20min: 670.48 TPM
  target_30min: 656.83 TPM
```

## Visualization

The model includes a visualization function that creates scatter plots of actual vs. predicted values for each target. The visualization is saved to `Src/Day4/prediction_visualization.png`.