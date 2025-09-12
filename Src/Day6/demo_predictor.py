"""
Day6 Summary: Enhanced Demo Predictor
Hàm demo từ model vừa training với chức năng:
- Load model mới nhất từ time_series_trainer.py
- Lấy data 3h gần nhất từ newrelic_data_collector.py
- Dự đoán TPM cho 5, 10, 15 phút tiếp theo
- So sánh scenarios có/không có push notification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import sys
import glob

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Import our Day6 modules
try:
    from time_series_trainer import TimeSeriesTrainer
    from newrelic_data_collector import collect_newrelic_data_with_optimal_granularity, process_newrelic_data_with_enhanced_metadata
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure time_series_trainer.py and newrelic_data_collector.py are in the same directory.")


class EnhancedDemoPredictor:
    """
    Enhanced Demo Predictor cho TPM prediction với real New Relic data
    """

    def __init__(self):
        self.trainer = None
        self.is_loaded = False
        self.data_collector_available = True

    def load_latest_trained_model(self, model_dir="day6_models"):
        """
        Load model mới nhất đã được train từ time_series_trainer.py

        Parameters:
        -----------
        model_dir : str
            Thư mục chứa models

        Returns:
        --------
        bool
            True nếu load thành công
        """

        print("🔍 Loading latest trained model from time_series_trainer...")

        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return False

        # Find latest model files
        try:
            model_files = glob.glob(os.path.join(model_dir, "*time_series_model*.joblib"))
            metadata_files = glob.glob(os.path.join(model_dir, "*metadata*.joblib"))

            if not model_files or not metadata_files:
                print(f"❌ No time_series_trainer model files found in {model_dir}")
                print(f"   Model files: {len(model_files)}, Metadata files: {len(metadata_files)}")
                return False

            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            metadata_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            latest_model = model_files[0]
            latest_metadata = metadata_files[0]

            print(f"✅ Found latest model files:")
            print(f"   Model: {os.path.basename(latest_model)}")
            print(f"   Metadata: {os.path.basename(latest_metadata)}")

            # Initialize trainer and load
            self.trainer = TimeSeriesTrainer()
            success = self.trainer.load_model(latest_model, latest_metadata)

            if success:
                self.is_loaded = True
                print("🚀 Model loaded successfully!")
                print(f"   Features: {len(self.trainer.feature_cols)}")
                print(f"   Models: {list(self.trainer.models.keys())}")
                return True
            else:
                print("❌ Failed to load model")
                return False

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def get_recent_3h_data(self, api_key=None, app_id=None):
        """
        Lấy data 3 giờ gần nhất từ newrelic_data_collector.py làm input

        Parameters:
        -----------
        api_key : str
            New Relic API key (nếu None sẽ lấy từ environment variables)
        app_id : str
            New Relic App ID (nếu None sẽ lấy từ environment variables)

        Returns:
        --------
        pd.DataFrame
            DataFrame chứa data 3 giờ gần nhất hoặc None nếu không có data
        """

        print("📊 Getting recent 3-hour data...")

        # Get credentials from environment variables if not provided
        if not api_key:
            api_key = os.getenv("NEWRELIC_API_KEY") or os.getenv("NEW_RELIC_API_KEY")
        if not app_id:
            app_id = os.getenv("NEWRELIC_APP_ID") or os.getenv("NEW_RELIC_APP_ID")

        if not api_key or not app_id:
            print("❌ Missing New Relic credentials. Set NEWRELIC_API_KEY and NEWRELIC_APP_ID environment variables.")
            return None

        try:
            # Get current time
            current_time = datetime.now()
            start_time = current_time - timedelta(hours=3)

            print(f"📅 Collecting data from {start_time.strftime('%H:%M')} to {current_time.strftime('%H:%M')}")

            # Use the collector from newrelic_data_collector
            # Calculate data age for optimal granularity
            data_age_days = 0  # Recent data

            raw_data = collect_newrelic_data_with_optimal_granularity(
                api_key=api_key,
                app_id=app_id,
                metrics=["HttpDispatcher"],
                start_time=start_time,
                end_time=current_time,
                week_priority=10,  # High priority for recent data
                data_age_days=data_age_days
            )

            if raw_data:
                # Process the raw data into DataFrame
                data = process_newrelic_data_with_enhanced_metadata(raw_data)
                if data is not None and not data.empty:
                    print(f"✅ Collected {len(data)} data points from New Relic")
                    return data
                else:
                    print("❌ No processed data from New Relic")
                    return None
            else:
                print("❌ No raw data from New Relic")
                return None

        except Exception as e:
            print(f"❌ Error collecting from New Relic: {e}")
            return None


    def predict_next_tpm_from_3h_data(self, historical_data, minutes_ahead=(5, 10, 15)):
        """
        Dự đoán TPM cho 5, 10, 15 phút tiếp theo từ data 3h

        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame chứa 3h data với columns ['timestamp', 'tpm', 'response_time', ...]
        minutes_ahead : tuple
            Tuple thời gian dự đoán (5, 10, 15)

        Returns:
        --------
        dict
            Kết quả dự đoán
        """

        if not self.is_loaded:
            raise ValueError("❌ Model chưa được load. Hãy gọi load_latest_trained_model() trước.")

        if historical_data.empty:
            raise ValueError("❌ Historical data không thể rỗng")

        # Sort by timestamp
        df = historical_data.sort_values('timestamp').reset_index(drop=True)

        # Get current values (latest data point)
        current_row = df.iloc[-1]
        current_time = pd.to_datetime(current_row['timestamp'])
        current_tpm = current_row['tpm']
        current_response_time = current_row.get('response_time', 100.0)

        # Get push info
        push_active = current_row.get('push_notification_active', 0)
        minutes_since_push = current_row.get('minutes_since_push', 999)

        print(f"🔮 Predicting TPM from 3h historical data")
        print(f"   Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Current TPM: {current_tpm:.1f}")
        print(f"   Push active: {push_active}, Minutes since: {minutes_since_push}")
        print(f"   Historical data points: {len(df)}")

        # Create features using time_series_trainer's method
        try:
            # Add required columns if missing
            for col in ['hour', 'minute', 'day_of_week', 'is_weekend']:
                if col not in df.columns:
                    if col == 'hour':
                        df[col] = pd.to_datetime(df['timestamp']).dt.hour
                    elif col == 'minute':
                        df[col] = pd.to_datetime(df['timestamp']).dt.minute
                    elif col == 'day_of_week':
                        df[col] = pd.to_datetime(df['timestamp']).dt.dayofweek
                    elif col == 'is_weekend':
                        df[col] = (pd.to_datetime(df['timestamp']).dt.dayofweek >= 5).astype(int)

            # Create time series features
            features_df = self.trainer.create_time_series_features(df)

            if features_df.empty:
                raise ValueError("No features could be created from historical data")

            # Get latest feature vector
            latest_features = features_df.iloc[-1:][self.trainer.feature_cols]

            # Make predictions
            predictions = {}
            model_name = self.trainer.best_models['tpm']
            model = self.trainer.models['tpm'][model_name]

            # Base prediction
            base_pred = model.predict(latest_features)[0]
            base_pred = max(0.0, float(base_pred))

            # Create horizon-specific predictions
            for minutes in minutes_ahead:
                # Time decay factor
                time_decay = 1.0 - (minutes - 5) * 0.015
                time_decay = max(0.85, time_decay)

                # Push effect calculation
                future_minutes_since_push = minutes_since_push + minutes
                push_effect = 0.0

                hour = current_time.hour
                is_daytime = 7 <= hour <= 21
                is_nighttime = 1 <= hour <= 6

                if push_active or future_minutes_since_push <= 15:
                    if is_daytime:
                        if future_minutes_since_push <= 15:
                            decay = np.exp(-future_minutes_since_push / 5.0)
                            push_effect = decay * 0.6  # 60% boost max
                    elif not is_nighttime:
                        if future_minutes_since_push <= 15:
                            decay = np.exp(-future_minutes_since_push / 10.0)
                            push_effect = decay * 0.2  # 20% boost max

                # Apply effects
                horizon_pred = base_pred * time_decay * (1 + push_effect)
                predictions[f'tpm_{minutes}min'] = max(0.0, horizon_pred)

            # Classify predictions
            labels = {}
            for pred_key, pred_value in predictions.items():
                labels[pred_key] = self.trainer.classify_tpm_value(pred_value, self.trainer.tpm_thresholds)

            results = {
                'predictions': predictions,
                'labels': labels,
                'current_values': {
                    'timestamp': current_time,
                    'tpm': current_tpm,
                    'response_time': current_response_time,
                    'push_active': push_active,
                    'minutes_since_push': minutes_since_push
                },
                'model_info': {
                    'model_used': model_name,
                    'features_count': len(self.trainer.feature_cols),
                    'data_points_used': len(df)
                }
            }

            print(f"   ✅ Predictions generated using {model_name}")
            for pred_key, pred_value in predictions.items():
                minutes = pred_key.split('_')[1]
                print(f"   📈 {minutes}: {pred_value:.1f} ({labels[pred_key]})")

            return results

        except Exception as e:
            print(f"❌ Error creating predictions: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_push_scenarios(self, historical_data, minutes_ahead=(5, 10, 15)):
        """
        So sánh predictions khi có và không có push notification

        Parameters:
        -----------
        historical_data : pd.DataFrame
            DataFrame chứa 3h data
        minutes_ahead : tuple
            Tuple thời gian dự đoán

        Returns:
        --------
        dict
            So sánh kết quả có/không có push
        """

        print("\n🔄 Comparing push notification scenarios...")

        # Scenario 1: No Push (set push inactive)
        df_no_push = historical_data.copy()
        df_no_push['push_notification_active'] = 0
        df_no_push['minutes_since_push'] = 999  # Long time since push

        # Scenario 2: With Push (set push active just now)
        df_with_push = historical_data.copy()
        df_with_push.iloc[-1, df_with_push.columns.get_loc('push_notification_active')] = 1
        df_with_push.iloc[-1, df_with_push.columns.get_loc('minutes_since_push')] = 0

        print("📊 Scenario 1: No Push Notification")
        results_no_push = self.predict_next_tpm_from_3h_data(df_no_push, minutes_ahead)

        print("\n📊 Scenario 2: With Push Notification (just sent)")
        results_with_push = self.predict_next_tpm_from_3h_data(df_with_push, minutes_ahead)

        if not results_no_push or not results_with_push:
            print("❌ Failed to generate comparison results")
            return None

        # Calculate differences
        comparison = {
            'no_push': results_no_push,
            'with_push': results_with_push,
            'differences': {},
            'percentage_changes': {},
            'label_changes': {}
        }

        print(f"\n🎯 COMPARISON RESULTS:")
        print(f"{'Horizon':<8} {'No Push':<8} {'With Push':<9} {'Diff':<6} {'Change':<8} {'Labels'}")
        print("-" * 55)

        for minutes in minutes_ahead:
            key = f'tpm_{minutes}min'
            no_push_val = results_no_push['predictions'][key]
            with_push_val = results_with_push['predictions'][key]

            diff = with_push_val - no_push_val
            pct_change = (diff / no_push_val * 100) if no_push_val > 0 else 0

            no_push_label = results_no_push['labels'][key]
            with_push_label = results_with_push['labels'][key]

            comparison['differences'][key] = diff
            comparison['percentage_changes'][key] = pct_change
            comparison['label_changes'][key] = {
                'no_push': no_push_label,
                'with_push': with_push_label,
                'changed': no_push_label != with_push_label
            }

            print(
                f"{minutes:>2}min    {no_push_val:>6.1f}   {with_push_val:>7.1f}   {diff:>+5.1f}   {pct_change:>+5.1f}%   {no_push_label} → {with_push_label}")

        # Summary insights
        avg_boost = np.mean(list(comparison['percentage_changes'].values()))
        max_boost = max(comparison['percentage_changes'].values())

        print(f"\n📈 PUSH NOTIFICATION IMPACT:")
        print(f"   Average boost: {avg_boost:+.1f}%")
        print(f"   Maximum boost: {max_boost:+.1f}%")
        print(
            f"   Label changes: {sum(1 for lc in comparison['label_changes'].values() if lc['changed'])}/{len(minutes_ahead)}")

        return comparison

    def demo_real_time_prediction(self, api_key=None, app_id="1080863725"):
        """
        Demo prediction với real-time data
        """

        if not self.is_loaded:
            print("❌ Model chưa được load. Hãy gọi load_latest_trained_model() trước.")
            return

        print("\n🚀 DEMO: Real-time TPM Prediction")
        print("=" * 60)

        # Get 3-hour data
        print("📊 Step 1: Getting 3-hour historical data...")
        historical_data = self.get_recent_3h_data(api_key, app_id)

        if historical_data is None or historical_data.empty:
            print("❌ No historical data available")
            return

        # Basic prediction
        print("\n📈 Step 2: Basic TPM Prediction")
        results = self.predict_next_tpm_from_3h_data(historical_data)

        if results:
            current = results['current_values']
            predictions = results['predictions']
            labels = results['labels']

            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Current Time: {current['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Current TPM: {current['tpm']:.1f}")
            print(f"   Current Response Time: {current['response_time']:.1f}ms")
            push_status = 'Active' if current['push_active'] else f'{current["minutes_since_push"]} min ago'
            print(f"   Push Status: {push_status}")
            print(f"\n📈 Predictions:")

            for horizon in [5, 10, 15]:
                key = f'tpm_{horizon}min'
                pred_val = predictions[key]
                label = labels[key]
                change_pct = ((pred_val - current['tpm']) / current['tpm'] * 100)
                print(f"   {horizon:>2} min: {pred_val:>6.1f} ({label}) [{change_pct:>+5.1f}%]")

            # Comparison scenarios
            print("\n🔄 Step 3: Push Notification Impact Analysis")
            comparison = self.compare_push_scenarios(historical_data)

            if comparison:
                print("\n✅ Impact analysis completed!")

            print(f"\n🎉 Real-time prediction demo completed!")

    def demo_multiple_scenarios(self):
        """
        Demo multiple prediction scenarios
        """

        if not self.is_loaded:
            print("❌ Model chưa được load. Hãy gọi load_latest_trained_model() trước.")
            return

        print("\n🎯 DEMO: Multiple Time Scenarios")
        print("=" * 60)

        scenarios = [
            {
                'name': 'Morning Peak (9 AM)',
                'hour': 9,
                'base_tpm': 800,
                'trend': 'increasing'
            },
            {
                'name': 'Lunch Break (12 PM)',
                'hour': 12,
                'base_tpm': 400,
                'trend': 'stable'
            },
            {
                'name': 'Afternoon Rush (3 PM)',
                'hour': 15,
                'base_tpm': 900,
                'trend': 'peak'
            },
            {
                'name': 'Evening Decline (8 PM)',
                'hour': 20,
                'base_tpm': 300,
                'trend': 'decreasing'
            },
            {
                'name': 'Night Time (2 AM)',
                'hour': 2,
                'base_tpm': 80,
                'trend': 'low'
            }
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")

            # Create scenario data
            current_time = datetime.now().replace(
                hour=scenario['hour'],
                minute=30,
                second=0,
                microsecond=0
            )

            # Generate 3h data for this scenario
            scenario_data = self._create_scenario_data(
                current_time,
                scenario['base_tpm'],
                scenario['trend']
            )

            # Run predictions
            try:
                results = self.predict_next_tpm_from_3h_data(scenario_data)

                if results:
                    current = results['current_values']
                    predictions = results['predictions']
                    labels = results['labels']

                    print(f"   Current TPM: {current['tpm']:.1f}")
                    print(f"   Predictions: ", end="")
                    for horizon in [5, 10, 15]:
                        key = f'tpm_{horizon}min'
                        print(f"{horizon}min={predictions[key]:.0f}({labels[key]}) ", end="")
                    print()

                # Quick push comparison for peak hours
                if scenario['hour'] in [9, 15]:
                    comp = self.compare_push_scenarios(scenario_data)
                    if comp:
                        avg_boost = np.mean(list(comp['percentage_changes'].values()))
                        print(f"   Push impact: {avg_boost:+.1f}% average boost")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        print(f"\n✅ Multiple scenarios demo completed!")

    def _create_scenario_data(self, current_time, base_tpm, trend_type):
        """
        Tạo scenario data cho demo
        """

        start_time = current_time - timedelta(hours=3)

        timestamps = []
        tpm_values = []
        response_times = []
        push_active_values = []
        minutes_since_push_values = []

        for i in range(180):  # 3 hours = 180 minutes
            timestamp = start_time + timedelta(minutes=i)

            # Apply trend
            if trend_type == 'increasing':
                tpm = base_tpm + i * 2 + np.random.normal(0, 20)
            elif trend_type == 'decreasing':
                tpm = base_tpm - i * 1 + np.random.normal(0, 15)
            elif trend_type == 'peak':
                peak_factor = np.sin(i / 60 * np.pi)  # Peak in middle of 3h
                tpm = base_tpm + peak_factor * 200 + np.random.normal(0, 25)
            elif trend_type == 'stable':
                tpm = base_tpm + np.sin(i / 30) * 50 + np.random.normal(0, 20)
            else:  # low
                tpm = base_tpm + np.random.normal(0, 10)

            tpm = max(20, tpm)

            # Response time inversely correlated
            response_time = 250 - (tpm / 10) + np.random.normal(0, 15)
            response_time = max(50, response_time)

            # Push notifications every 90 minutes
            if i % 90 == 45:  # At 45min and 135min
                push_active = 1
                minutes_since = 0
            else:
                push_active = 0
                minutes_since = (i % 90) - 45 if i % 90 >= 45 else (90 - (45 - (i % 90)))
                minutes_since = max(0, minutes_since)

            timestamps.append(timestamp)
            tpm_values.append(tpm)
            response_times.append(response_time)
            push_active_values.append(push_active)
            minutes_since_push_values.append(minutes_since)

        return pd.DataFrame({
            'timestamp': timestamps,
            'tpm': tpm_values,
            'response_time': response_times,
            'push_notification_active': push_active_values,
            'minutes_since_push': minutes_since_push_values,
            'hour': [ts.hour for ts in timestamps],
            'minute': [ts.minute for ts in timestamps],
            'day_of_week': [ts.weekday() for ts in timestamps],
            'is_weekend': [(ts.weekday() >= 5) for ts in timestamps]
        })


def main():
    """
    Main demo function với full workflow
    """
    print("🚀 Day6 - Enhanced Demo Predictor")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = EnhancedDemoPredictor()

        # Step 1: Load trained model
        print("📂 Step 1: Loading latest trained model...")
        success = predictor.load_latest_trained_model()

        if not success:
            print("❌ No trained model found. Please train a model first using time_series_trainer.py")
            return

        if predictor.is_loaded:
            print("\n🎯 Running enhanced prediction demos...")

            # Demo 1: Real-time prediction
            predictor.demo_real_time_prediction()

            # Demo 2: Multiple scenarios
            predictor.demo_multiple_scenarios()

            print(f"\n🎉 All enhanced demos completed successfully!")
            print(f"🔧 Key Features Demonstrated:")
            print(f"   ✅ Load latest trained model from time_series_trainer")
            print(f"   ✅ Get 3-hour recent data (sample/real)")
            print(f"   ✅ Predict TPM for 5, 10, 15 minutes ahead")
            print(f"   ✅ Compare scenarios with/without push notifications")
            print(f"   ✅ Multiple time-of-day scenarios")

        else:
            print("❌ Could not load or train model for demo")

    except Exception as e:
        print(f"❌ Error in enhanced demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
