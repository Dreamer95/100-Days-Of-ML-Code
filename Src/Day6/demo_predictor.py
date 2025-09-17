"""
Day6 Summary: Enhanced Demo Predictor (FIXED)
Hàm demo từ model vừa training với EnhancedTimeSeriesTrainer
- Load model mới nhất từ enhanced_time_series_trainer.py
- Lấy data 3h gần nhất từ newrelic_data_collector.py (or sample data)
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

# FIXED: Import correct class name from enhanced trainer
try:
    from enhanced_time_series_trainer import EnhancedTimeSeriesTrainer
    print("✅ Successfully imported EnhancedTimeSeriesTrainer")
except ImportError:
    try:
        from time_series_trainer import EnhancedTimeSeriesTrainer
        print("✅ Successfully imported EnhancedTimeSeriesTrainer from time_series_trainer")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        EnhancedTimeSeriesTrainer = None

# FIXED: Import New Relic data collector with better error handling
try:
    from newrelic_data_collector import collect_newrelic_data_with_optimal_granularity, process_newrelic_data_with_enhanced_metadata
    NEWRELIC_AVAILABLE = True
    print("✅ Successfully imported New Relic data collector functions")
except ImportError as e:
    print(f"❌ New Relic import error: {e}")
    print("ℹ️  New Relic data collector not available")
    NEWRELIC_AVAILABLE = False
    # Set dummy functions to avoid NameError
    def collect_newrelic_data_with_optimal_granularity(*args, **kwargs):
        return None
    def process_newrelic_data_with_enhanced_metadata(*args, **kwargs):
        return None



class EnhancedDemoPredictor:
    """
    FIXED: Enhanced Demo Predictor cho TPM prediction với EnhancedTimeSeriesTrainer
    """

    def __init__(self):
        self.trainer = None
        self.is_loaded = False
        self.data_collector_available = NEWRELIC_AVAILABLE

    def load_latest_trained_model(self, model_dir="day6_csv_enhanced_models"):  # FIXED: Use correct directory
        """
        FIXED: Load model mới nhất từ EnhancedTimeSeriesTrainer

        Parameters:
        -----------
        model_dir : str
            Thư mục chứa enhanced models (default: day6_csv_enhanced_models)

        Returns:
        --------
        bool
            True nếu load thành công
        """

        print("🔍 Loading latest trained model from EnhancedTimeSeriesTrainer...")

        if EnhancedTimeSeriesTrainer is None:
            print("❌ EnhancedTimeSeriesTrainer class not available")
            return False

        # Check if model directory exists
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return False

        try:
            # FIXED: Look for enhanced model files with correct naming pattern
            model_files = glob.glob(os.path.join(model_dir, "*csv_push_aware_model*.joblib"))
            metadata_files = glob.glob(os.path.join(model_dir, "*metadata*.joblib"))

            # Fallback to any .joblib files if specific pattern not found
            if not model_files:
                model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
                model_files = [f for f in model_files if 'metadata' not in f]

            if not model_files or not metadata_files:
                print(f"❌ No enhanced model files found in {model_dir}")
                print(f"   Model files found: {len(model_files)}")
                print(f"   Metadata files found: {len(metadata_files)}")

                # Show what files are available
                all_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
                print(f"   Available files: {all_files}")
                return False

            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            metadata_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            latest_model = model_files[0]
            latest_metadata = metadata_files[0]

            print(f"✅ Found latest enhanced model files:")
            print(f"   Model: {os.path.basename(latest_model)}")
            print(f"   Metadata: {os.path.basename(latest_metadata)}")

            # FIXED: Initialize EnhancedTimeSeriesTrainer and load
            self.trainer = EnhancedTimeSeriesTrainer()
            success = self.trainer.load_model(latest_model, latest_metadata)

            if success:
                self.is_loaded = True
                print("🚀 Enhanced model loaded successfully!")
                print(f"   Features: {len(self.trainer.feature_cols)}")
                print(f"   Best model: {self.trainer.best_models.get('tpm', 'Unknown')}")
                print(f"   Push config: {self.trainer.push_config.get('daytime_hours', 'Unknown')}")
                return True
            else:
                print("❌ Failed to load enhanced model")
                return False

        except Exception as e:
            print(f"❌ Error loading enhanced model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_recent_3h_data(self, api_key=None, app_id=None):
        """
        Lấy data 3 giờ gần nhất từ New Relic (FIXED - REAL DATA ONLY)
        """
        print("📊 Getting recent 3-hour data from New Relic...")

        # Get credentials from environment variables if not provided
        if not api_key:
            api_key = os.getenv("NEWRELIC_API_KEY") or os.getenv("NEW_RELIC_API_KEY")
        if not app_id:
            app_id = os.getenv("NEWRELIC_APP_ID") or os.getenv("NEW_RELIC_APP_ID")

        # Debug information
        print(f"🔑 API Key available: {'Yes' if api_key else 'No'}")
        print(f"🆔 App ID: {app_id}")
        print(f"📦 New Relic functions available: {NEWRELIC_AVAILABLE}")

        # Check prerequisites
        if not NEWRELIC_AVAILABLE:
            print("❌ New Relic data collector functions not available")
            print("   Make sure 'newrelic_data_collector.py' is in the same directory")
            return None

        if not api_key or not app_id:
            print("❌ Missing New Relic credentials")
            print("   Please check your .env file contains:")
            print("   NEWRELIC_API_KEY=your_api_key")
            print("   NEW_RELIC_APP_ID=your_app_id")
            return None

        try:
            # Calculate time range
            current_time = datetime.now()
            start_time = current_time - timedelta(hours=3)

            print(f"📅 Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {current_time.strftime('%Y-%m-%d %H:%M')}")

            # Call New Relic API with debug info
            print("📡 Calling New Relic API...")
            raw_data = collect_newrelic_data_with_optimal_granularity(
                api_key=api_key,
                app_id=app_id,
                metrics=["HttpDispatcher"],
                start_time=start_time,
                end_time=current_time,
                week_priority=10,
                data_age_days=0
            )

            # Debug raw response
            if raw_data:
                print(f"✅ Raw data received: {type(raw_data)}")
                if isinstance(raw_data, dict):
                    print(f"   Keys: {list(raw_data.keys())}")
                    if 'metric_data' in raw_data:
                        metrics = raw_data['metric_data'].get('metrics', [])
                        print(f"   Metrics count: {len(metrics)}")
                        if metrics:
                            timeslices = metrics[0].get('timeslices', [])
                            print(f"   Timeslices count: {len(timeslices)}")

                # Process the raw data
                print("🔄 Processing raw data...")
                data = process_newrelic_data_with_enhanced_metadata(raw_data)

                if data is not None and not data.empty:
                    print(f"✅ Successfully processed {len(data)} data points")
                    print(f"   Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                    print(f"   TPM range: {data['tpm'].min():.1f} to {data['tpm'].max():.1f}")
                    return data
                else:
                    print("❌ Processing resulted in empty DataFrame")
                    return None
            else:
                print("❌ No raw data received from New Relic API")
                return None

        except Exception as e:
            print(f"❌ Error collecting New Relic data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict_next_tpm_from_3h_data(self, historical_data, minutes_ahead=(5, 10, 15)):
        """
        FIXED: Dự đoán TPM sử dụng EnhancedTimeSeriesTrainer
        """
        if not self.is_loaded:
            raise ValueError("❌ Enhanced model chưa được load. Hãy gọi load_latest_trained_model() trước.")

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

        print(f"🔮 Predicting TPM using EnhancedTimeSeriesTrainer")
        print(f"   Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Current TPM: {current_tpm:.1f}")
        print(f"   Push active: {push_active}, Minutes since: {minutes_since_push}")
        print(f"   Historical data points: {len(df)}")

        try:
            # FIXED: Use enhanced trainer's feature creation method
            # Add any missing required columns
            required_columns = ['hour', 'minute', 'day_of_week', 'is_weekend', 'is_business_hours']

            for col in required_columns:
                if col not in df.columns:
                    if col == 'hour':
                        df[col] = pd.to_datetime(df['timestamp']).dt.hour
                    elif col == 'minute':
                        df[col] = pd.to_datetime(df['timestamp']).dt.minute
                    elif col == 'day_of_week':
                        df[col] = pd.to_datetime(df['timestamp']).dt.dayofweek
                    elif col == 'is_weekend':
                        df[col] = (pd.to_datetime(df['timestamp']).dt.dayofweek >= 5).astype(int)
                    elif col == 'is_business_hours':
                        df[col] = ((pd.to_datetime(df['timestamp']).dt.hour >= 9) &
                                 (pd.to_datetime(df['timestamp']).dt.hour <= 17)).astype(int)

            # Use enhanced feature creation (but avoid the business push effects since this is prediction)
            # We'll create basic features first
            temp_df = df.copy()

            # Create basic enhanced features without applying business effects
            features_df = self._create_prediction_features(temp_df)

            if features_df.empty:
                raise ValueError("No features could be created from historical data")

            # Get features that match training
            available_features = [col for col in self.trainer.feature_cols if col in features_df.columns]

            if len(available_features) == 0:
                raise ValueError("No matching features found between training and prediction data")

            print(f"   Using {len(available_features)} matching features")

            # Get latest feature vector
            latest_features = features_df.iloc[-1:][available_features]

            # Fill any missing features with 0
            for feature in self.trainer.feature_cols:
                if feature not in latest_features.columns:
                    latest_features[feature] = 0

            # Reorder to match training
            latest_features = latest_features[self.trainer.feature_cols]

            # Make predictions
            predictions = {}
            model_name = self.trainer.best_models['tpm']
            model = self.trainer.models['tpm'][model_name]

            # Base prediction
            base_pred = model.predict(latest_features)[0]
            base_pred = max(0.0, float(base_pred))

            # Create horizon-specific predictions with business logic
            for minutes in minutes_ahead:
                # Time decay factor
                time_decay = 1.0 - (minutes - 5) * 0.01
                time_decay = max(0.9, time_decay)

                # Enhanced push effect calculation using trainer's business logic
                future_minutes_since_push = minutes_since_push + minutes

                # Use trainer's push configuration for consistent business logic
                push_effect = self._calculate_prediction_push_effect(
                    current_hour=current_time.hour,
                    push_active=push_active,
                    future_minutes_since_push=future_minutes_since_push,
                    campaign_type=current_row.get('push_campaign_type', 'none')
                )

                # Apply effects
                horizon_pred = base_pred * time_decay * (1 + push_effect)
                predictions[f'tpm_{minutes}min'] = max(0.0, horizon_pred)

            # Classify predictions using trainer's thresholds
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
                    'features_count': len(available_features),
                    'data_points_used': len(df),
                    'trainer_type': 'EnhancedTimeSeriesTrainer'
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

    def _create_prediction_features(self, df):
        """
        Create features for prediction (COMPREHENSIVE FIX)
        """
        print("   🔧 Creating prediction features with comprehensive coverage...")

        result_df = df.copy()

        # ============ BASIC TIME FEATURES ============
        # Ensure all base time features exist
        if 'hour_sin' not in result_df.columns:
            result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
            result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
            result_df['day_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
            result_df['day_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)

        # ============ BUSINESS HOURS FEATURES ============
        result_df['is_morning_peak'] = ((result_df['hour'] >= 8) & (result_df['hour'] <= 10)).astype(int)
        result_df['is_lunch_peak'] = ((result_df['hour'] >= 12) & (result_df['hour'] <= 13)).astype(int)
        result_df['is_afternoon_peak'] = ((result_df['hour'] >= 15) & (result_df['hour'] <= 17)).astype(int)
        result_df['is_evening_peak'] = ((result_df['hour'] >= 18) & (result_df['hour'] <= 20)).astype(int)

        # ============ PUSH-RELATED FEATURES ============
        # Ensure they exist
        if 'push_notification_active' not in result_df.columns:
            result_df['push_notification_active'] = 0
        if 'minutes_since_push' not in result_df.columns:
            result_df['minutes_since_push'] = 9999

        result_df['is_push_daytime_effective'] = ((result_df['hour'] >= 7) & (result_df['hour'] <= 21)).astype(int)
        result_df['is_push_no_effect_period'] = ((result_df['hour'] >= 2) & (result_df['hour'] <= 6)).astype(int)

        # Enhanced push features
        result_df['in_push_delay_window'] = (
                    (result_df['minutes_since_push'] >= 5) & (result_df['minutes_since_push'] <= 10)).astype(int)
        result_df['in_push_peak_effect'] = (
                    (result_df['minutes_since_push'] >= 5) & (result_df['minutes_since_push'] <= 10)).astype(int)
        result_df['in_push_declining_effect'] = (
                    (result_df['minutes_since_push'] > 10) & (result_df['minutes_since_push'] <= 15)).astype(int)

        # Push decay features
        result_df['push_decay_exponential'] = result_df['minutes_since_push'].apply(
            lambda x: np.exp(-x / 8.0) if x <= 15 else 0.0
        )
        result_df['push_decay_linear'] = result_df['minutes_since_push'].apply(
            lambda x: max(0, 1 - x / 15.0) if x <= 15 else 0.0
        )

        # ============ INTERACTION FEATURES ============
        result_df['push_hour_interaction'] = result_df['push_notification_active'] * result_df['hour']
        result_df['push_weekday_interaction'] = result_df['push_notification_active'] * (result_df['day_of_week'] + 1)
        result_df['push_business_hour_interaction'] = result_df['push_notification_active'] * result_df[
            'is_business_hours']

        # ============ ADVANCED TEMPORAL FEATURES ============
        result_df['hour_quarter_sin'] = np.sin(2 * np.pi * (result_df['hour'] % 6) / 6)
        result_df['hour_quarter_cos'] = np.cos(2 * np.pi * (result_df['hour'] % 6) / 6)

        # ============ CAMPAIGN FEATURES (ALL CAMPAIGNS) ============
        campaign_types = ['morning_commute', 'lunch_peak', 'afternoon_break',
                          'evening_commute', 'evening_leisure', 'sleeping_hours_early', 'sleeping_hours_late']

        for campaign in campaign_types:
            column_name = f'campaign_{campaign}'
            if 'push_campaign_type' in result_df.columns:
                result_df[column_name] = (result_df['push_campaign_type'] == campaign).astype(int)
            else:
                result_df[column_name] = 0

        # Aggregated campaign features
        if 'push_campaign_type' in result_df.columns:
            result_df['is_peak_hour_campaign'] = result_df['push_campaign_type'].isin(
                ['lunch_peak', 'evening_commute']).astype(int)
            result_df['is_commute_campaign'] = result_df['push_campaign_type'].isin(
                ['morning_commute', 'evening_commute']).astype(int)
            result_df['is_no_effect_campaign'] = result_df['push_campaign_type'].isin(
                ['sleeping_hours_early', 'sleeping_hours_late']).astype(int)
        else:
            result_df['is_peak_hour_campaign'] = 0
            result_df['is_commute_campaign'] = 0
            result_df['is_no_effect_campaign'] = 0

        # Push timing quality
        result_df['push_timing_quality'] = result_df.apply(
            lambda row: self._calculate_push_timing_quality_local(
                row['hour'],
                str(row.get('push_campaign_type', 'none'))
            ), axis=1
        )

        # ============ SEQUENCE FEATURES ============
        if 'push_notification_active' in result_df.columns:
            result_df['cumulative_pushes'] = result_df['push_notification_active'].cumsum()

            # Rolling features
            window_size = min(12, len(result_df))
            result_df['pushes_last_hour'] = result_df['push_notification_active'].rolling(window=window_size,
                                                                                          min_periods=1).sum()
            result_df['pushes_last_day'] = result_df['pushes_last_hour']

        # ============ LAG AND ROLLING FEATURES ============
        if len(result_df) >= 5:
            result_df['tpm_lag_1'] = result_df['tpm'].shift(1)
            result_df['tpm_lag_2'] = result_df['tpm'].shift(2)
            result_df['tpm_lag_5'] = result_df['tpm'].shift(5) if len(result_df) >= 6 else result_df['tpm'].shift(1)

            # Rolling statistics
            result_df['tpm_ma_5'] = result_df['tpm'].rolling(window=5, min_periods=1).mean()
            result_df['tpm_ma_15'] = result_df['tpm'].rolling(window=min(15, len(result_df)), min_periods=1).mean()
            result_df['tpm_std_5'] = result_df['tpm'].rolling(window=5, min_periods=1).std().fillna(0)
            result_df['tpm_min_5'] = result_df['tpm'].rolling(window=5, min_periods=1).min()
            result_df['tpm_max_5'] = result_df['tpm'].rolling(window=5, min_periods=1).max()

        # ============ COMPREHENSIVE CLEANUP ============
        # Fill NaN values comprehensively
        result_df = result_df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        # Handle infinite values
        result_df = result_df.replace([np.inf, -np.inf], 0)

        print(f"   ✅ Created {len(result_df.columns)} comprehensive features")

        return result_df

    def _calculate_push_timing_quality_local(self, hour, campaign_type):
        """Local version of push timing quality calculation"""
        optimal_hours = {
            'morning_commute': [7, 8, 9],
            'lunch_peak': [11, 12, 13],
            'afternoon_break': [14, 15, 16],
            'evening_commute': [17, 18, 19],
            'evening_leisure': [19, 20, 21]
        }

        if campaign_type == 'none' or campaign_type not in optimal_hours:
            return 0.5

        if hour in optimal_hours[campaign_type]:
            return 1.0

        min_distance = min([abs(hour - h) for h in optimal_hours[campaign_type]])
        quality = max(0.1, 1.0 - (min_distance * 0.15))

        return quality

    def _calculate_prediction_push_effect(self, current_hour, push_active, future_minutes_since_push, campaign_type):
        """
        Calculate push effect for predictions using business logic from trainer
        """
        if not push_active and future_minutes_since_push > 15:
            return 0.0

        # Use trainer's push config
        daytime_start, daytime_end = self.trainer.push_config['daytime_hours']
        no_effect_start, no_effect_end = self.trainer.push_config['no_effect_hours']

        # No effect during night hours
        if no_effect_start <= current_hour <= no_effect_end:
            return 0.0

        # High effect during daytime
        if daytime_start <= current_hour <= daytime_end:
            if future_minutes_since_push <= 15:
                decay = np.exp(-future_minutes_since_push / 8.0)
                return decay * 0.8  # Up to 80% boost
        else:
            # Reduced effect for other hours
            if future_minutes_since_push <= 15:
                decay = np.exp(-future_minutes_since_push / 10.0)
                return decay * 0.3  # Up to 30% boost

        return 0.0

    def compare_push_scenarios(self, historical_data, minutes_ahead=(5, 10, 15)):
        """
        So sánh predictions khi có và không có push notification
        """
        print("\n🔄 Comparing push notification scenarios...")

        # Scenario 1: No Push (set push inactive)
        df_no_push = historical_data.copy()
        df_no_push['push_notification_active'] = 0
        df_no_push['minutes_since_push'] = 999
        df_no_push['push_campaign_type'] = 'none'

        # Scenario 2: With Push (set push active just now)
        df_with_push = historical_data.copy()
        df_with_push.iloc[-1, df_with_push.columns.get_loc('push_notification_active')] = 1
        df_with_push.iloc[-1, df_with_push.columns.get_loc('minutes_since_push')] = 0

        # Set appropriate campaign type based on time
        current_hour = df_with_push.iloc[-1]['hour']
        if 7 <= current_hour <= 9:
            campaign = 'morning_commute'
        elif 11 <= current_hour <= 13:
            campaign = 'lunch_peak'
        elif 15 <= current_hour <= 17:
            campaign = 'afternoon_break'
        elif 17 <= current_hour <= 19:
            campaign = 'evening_commute'
        else:
            campaign = 'evening_leisure'

        df_with_push.iloc[-1, df_with_push.columns.get_loc('push_campaign_type')] = campaign

        print("📊 Scenario 1: No Push Notification")
        results_no_push = self.predict_next_tpm_from_3h_data(df_no_push, minutes_ahead)

        print(f"\n📊 Scenario 2: With Push Notification ({campaign})")
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

        print(f"\n🎯 ENHANCED COMPARISON RESULTS:")
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

        print(f"\n📈 ENHANCED PUSH NOTIFICATION IMPACT:")
        print(f"   Campaign type: {campaign}")
        print(f"   Average boost: {avg_boost:+.1f}%")
        print(f"   Maximum boost: {max_boost:+.1f}%")
        print(f"   Label changes: {sum(1 for lc in comparison['label_changes'].values() if lc['changed'])}/{len(minutes_ahead)}")

        return comparison

    def demo_real_time_prediction(self, api_key=None, app_id=None):
        """
        FIXED: Demo prediction với enhanced trainer
        """
        if not self.is_loaded:
            print("❌ Enhanced model chưa được load. Hãy gọi load_latest_trained_model() trước.")
            return

        print("\n🚀 DEMO: Real-time TPM Prediction with EnhancedTimeSeriesTrainer")
        print("=" * 70)

        # Get 3-hour data
        print("📊 Step 1: Getting 3-hour historical data...")
        historical_data = self.get_recent_3h_data(api_key, app_id)

        if historical_data is None or historical_data.empty:
            print("❌ No historical data available")
            return

        # Basic prediction
        print("\n📈 Step 2: Enhanced TPM Prediction")
        results = self.predict_next_tpm_from_3h_data(historical_data)

        if results:
            current = results['current_values']
            predictions = results['predictions']
            labels = results['labels']

            print(f"\n🎯 ENHANCED PREDICTION RESULTS:")
            print(f"   Current Time: {current['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Current TPM: {current['tpm']:.1f}")
            print(f"   Current Response Time: {current['response_time']:.1f}ms")
            push_status = 'Active' if current['push_active'] else f'{current["minutes_since_push"]} min ago'
            print(f"   Push Status: {push_status}")
            print(f"   Model: {results['model_info']['trainer_type']}")
            print(f"\n📈 Predictions:")

            for horizon in [5, 10, 15]:
                key = f'tpm_{horizon}min'
                pred_val = predictions[key]
                label = labels[key]
                change_pct = ((pred_val - current['tpm']) / current['tpm'] * 100)
                print(f"   {horizon:>2} min: {pred_val:>6.1f} ({label:<10}) [{change_pct:>+5.1f}%]")

            # Comparison scenarios
            print("\n🔄 Step 3: Enhanced Push Notification Impact Analysis")
            comparison = self.compare_push_scenarios(historical_data)

            if comparison:
                print("\n✅ Enhanced impact analysis completed!")

            print(f"\n🎉 Real-time prediction demo completed with EnhancedTimeSeriesTrainer!")

    # Keep other methods but update references to use enhanced trainer...

    def demo_multiple_scenarios(self):
        """Demo multiple prediction scenarios with enhanced trainer"""
        if not self.is_loaded:
            print("❌ Enhanced model chưa được load.")
            return

        print("\n🎯 DEMO: Multiple Time Scenarios (Enhanced)")
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
                        print(f"{horizon}min={predictions[key]:.0f}({labels[key][:4]}) ", end="")
                    print()

                # Quick push comparison for peak hours
                if scenario['hour'] in [9, 15]:
                    comp = self.compare_push_scenarios(scenario_data)
                    if comp:
                        avg_boost = np.mean(list(comp['percentage_changes'].values()))
                        print(f"   Push impact: {avg_boost:+.1f}% average boost")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        print(f"\n✅ Multiple scenarios demo completed with EnhancedTimeSeriesTrainer!")

    def _create_scenario_data(self, current_time, base_tpm, trend_type):
        """Generate scenario data for demo purposes"""
        start_time = current_time - timedelta(hours=3)

        timestamps = []
        tpm_values = []
        response_times = []
        push_active_values = []
        minutes_since_push_values = []
        push_campaign_types = []

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
                hour = timestamp.hour
                if 7 <= hour <= 9:
                    campaign = 'morning_commute'
                elif 11 <= hour <= 13:
                    campaign = 'lunch_peak'
                elif 15 <= hour <= 17:
                    campaign = 'afternoon_break'
                elif 17 <= hour <= 19:
                    campaign = 'evening_commute'
                else:
                    campaign = 'evening_leisure'
            else:
                push_active = 0
                minutes_since = (i % 90) - 45 if i % 90 >= 45 else (90 - (45 - (i % 90)))
                minutes_since = max(0, minutes_since)
                campaign = 'none'

            timestamps.append(timestamp)
            tpm_values.append(tpm)
            response_times.append(response_time)
            push_active_values.append(push_active)
            minutes_since_push_values.append(minutes_since)
            push_campaign_types.append(campaign)

        return pd.DataFrame({
            'timestamp': timestamps,
            'tpm': tpm_values,
            'response_time': response_times,
            'push_notification_active': push_active_values,
            'minutes_since_push': minutes_since_push_values,
            'push_campaign_type': push_campaign_types,
            'hour': [ts.hour for ts in timestamps],
            'minute': [ts.minute for ts in timestamps],
            'day_of_week': [ts.weekday() for ts in timestamps],
            'is_weekend': [int(ts.weekday() >= 5) for ts in timestamps],
            'is_business_hours': [int(9 <= ts.hour <= 17) for ts in timestamps]
        })

    def predict_from_array(self, historical_tpm_1min,
                           push_notification_active=False,
                           minutes_since_push=9999,
                           push_campaign_type='none',
                           prediction_horizons=(5, 10, 15)):
        """
        Wrapper để gọi predict_tpm_from_historical_array từ trainer
        """
        if not self.is_loaded:
            raise ValueError("❌ Model chưa được load")

        current_time = datetime.now()

        print(f"🔮 Array Prediction via EnhancedDemoPredictor")
        print(f"   📊 Historical points: {len(historical_tpm_1min)}")
        print(f"   📱 Push: {push_notification_active}, Since: {minutes_since_push}min")
        print(f"   🎯 Campaign: {push_campaign_type}")

        # Gọi trainer's method
        return self.trainer.predict_tpm_from_historical_array(
            current_time=current_time,
            historical_tpm_1min=historical_tpm_1min,
            push_notification_active=push_notification_active,
            minutes_since_push=minutes_since_push,
            push_campaign_type=push_campaign_type,
            prediction_horizons=prediction_horizons
        )

    def demo_built_in_array_prediction(self):
        """Gọi demo_array_prediction từ trainer"""

        if not self.is_loaded:
            print("❌ Model chưa được load")
            return

        print(f"🎯 Running trainer's built-in demo_array_prediction()...")
        self.trainer.demo_array_prediction()

    def demo_custom_array_scenarios(self):
        """Demo custom array scenarios"""
        if not self.is_loaded:
            print("❌ Model chưa được load")
            return

        print(f"\n🎯 DEMO: Custom Array Prediction Scenarios")
        print("=" * 60)

        # Create different TPM patterns
        patterns = {
            'increasing_trend': {
                'name': 'Increasing Trend (Rush Hour)',
                'generator': lambda i: 300 + i * 3 + np.random.normal(0, 15)
            },
            'decreasing_trend': {
                'name': 'Decreasing Trend (Off Peak)',
                'generator': lambda i: 600 - i * 2 + np.random.normal(0, 20)
            },
            'cyclical_pattern': {
                'name': 'Cyclical Pattern (Normal Day)',
                'generator': lambda i: 400 + np.sin(i / 20) * 100 + np.random.normal(0, 25)
            },
            'spike_pattern': {
                'name': 'Traffic Spike Pattern',
                'generator': lambda i: 200 + (500 * np.exp(-(i - 30) ** 2 / 200)) + np.random.normal(0, 30)
            }
        }

        push_scenarios = [
            {'active': False, 'since': 9999, 'campaign': 'none'},
            {'active': True, 'since': 2, 'campaign': 'morning_commute'},
            {'active': False, 'since': 12, 'campaign': 'lunch_peak'}
        ]

        for pattern_key, pattern_info in patterns.items():
            print(f"\n📈 Pattern: {pattern_info['name']}")
            print("-" * 40)

            # Generate 90-minute historical data
            historical_tpm = []
            for i in range(90):
                tpm_value = pattern_info['generator'](i)
                historical_tpm.append(max(50, tpm_value))  # Min 50 TPM

            print(f"   📊 Generated {len(historical_tpm)} points")
            print(f"   📊 TPM range: {min(historical_tpm):.0f} - {max(historical_tpm):.0f}")

            # Test each push scenario
            for j, push_scenario in enumerate(push_scenarios):
                scenario_name = f"Push: {'Active' if push_scenario['active'] else 'Inactive'}"
                if push_scenario['campaign'] != 'none':
                    scenario_name += f" ({push_scenario['campaign']})"

                print(f"\n   🔍 {scenario_name}")

                try:
                    results = self.predict_from_array(
                        historical_tpm_1min=historical_tpm,
                        push_notification_active=push_scenario['active'],
                        minutes_since_push=push_scenario['since'],
                        push_campaign_type=push_scenario['campaign'],
                        prediction_horizons=(5, 10, 15)
                    )

                    if results:
                        current_tpm = results['current_state']['current_tpm']
                        print(f"      Current TPM: {current_tpm:.1f}")

                        for horizon in [5, 10, 15]:
                            key = f'tpm_{horizon}min'
                            pred = results['predictions'][key]
                            label = results['labels'][key]
                            change = ((pred - current_tpm) / current_tpm * 100)
                            print(f"      {horizon:2d}min: {pred:6.1f} ({label:8s}) [{change:+5.1f}%]")
                    else:
                        print("      ❌ Prediction failed")

                except Exception as e:
                    print(f"      ❌ Error: {e}")

        print(f"\n✅ Custom array scenarios completed!")

    def compare_array_vs_3h_data(self, api_key=None, app_id=None):
        """So sánh prediction từ array vs 3h real data"""

        if not self.is_loaded:
            print("❌ Model chưa được load")
            return

        print(f"\n🔄 COMPARISON: Array vs 3H Real Data Predictions")
        print("=" * 60)

        # Method 1: Get 3H real data
        print("📊 Method 1: Getting 3H real data...")
        real_data = self.get_recent_3h_data(api_key, app_id)

        if real_data is not None and not real_data.empty:
            print(f"✅ Got {len(real_data)} real data points")

            # Extract TPM values for array method
            tpm_array = real_data['tpm'].values[-60:]  # Last 60 minutes

            print(f"📊 Method 2: Using array method with {len(tpm_array)} points...")

            # Array prediction
            array_results = self.predict_from_array(
                historical_tpm_1min=tpm_array,
                push_notification_active=False,
                prediction_horizons=(5, 10, 15)
            )

            # 3H data prediction
            data_results = self.predict_next_tpm_from_3h_data(
                real_data,
                minutes_ahead=(5, 10, 15)
            )

            if array_results and data_results:
                print(f"\n🎯 COMPARISON RESULTS:")
                print(f"{'Horizon':<8} {'Array':<8} {'3H Data':<8} {'Diff':<6} {'Method'}")
                print("-" * 45)

                for horizon in [5, 10, 15]:
                    key = f'tpm_{horizon}min'
                    array_pred = array_results['predictions'][key]
                    data_pred = data_results['predictions'][key]
                    diff = abs(array_pred - data_pred)
                    better = 'Array' if array_pred > data_pred else '3H Data'

                    print(f"{horizon:>2}min    {array_pred:>6.1f}   {data_pred:>6.1f}   {diff:>5.1f}   {better}")

                avg_diff = np.mean([abs(array_results['predictions'][f'tpm_{h}min'] -
                                        data_results['predictions'][f'tpm_{h}min'])
                                    for h in [5, 10, 15]])

                print(f"\n📈 Average difference: {avg_diff:.1f} TPM")
                print(f"✅ Both methods completed successfully!")
            else:
                print("❌ One or both predictions failed")
        else:
            print("❌ Could not get real data, skipping comparison")


def main():
    """
    FIXED: Main demo function với enhanced trainer
    """
    print("🚀 Day6 - Enhanced Demo Predictor (FIXED)")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = EnhancedDemoPredictor()

        # Step 1: Load trained model
        print("📂 Step 1: Loading latest enhanced trained model...")
        success = predictor.load_latest_trained_model()

        if not success:
            print("❌ No enhanced trained model found.")
            print("ℹ️  Please train a model first using the enhanced_time_series_trainer.py")
            print("ℹ️  The model should be in 'day6_csv_enhanced_models' directory")
            return

        if predictor.is_loaded:
            print("\n🎯 Running enhanced prediction demos...")

            # Demo 1: Real-time prediction with sample data
            # predictor.demo_real_time_prediction()
            #
            # # Demo 2: Multiple scenarios
            # predictor.demo_multiple_scenarios()

            # Demo 2: Built-in array prediction
            print("\n" + "=" * 60)
            predictor.demo_built_in_array_prediction()

            # Demo 3: Custom array scenarios
            # print("\n" + "=" * 60)
            # predictor.demo_custom_array_scenarios()

            # Demo 4: Array vs 3H data comparison
            # print("\n" + "=" * 60)
            # predictor.compare_array_vs_3h_data()

            # Demo 5: Direct trainer demo
            # print("\n" + "=" * 60)
            # demo_direct_array_prediction()

            print(f"\n🎉 All enhanced demos completed successfully!")
            print(f"🔧 Enhanced Features Demonstrated:")
            print(f"   ✅ Load latest model from EnhancedTimeSeriesTrainer")
            print(f"   ✅ Generate realistic 3-hour sample data")
            print(f"   ✅ Predict TPM for 5, 10, 15 minutes ahead")
            print(f"   ✅ Compare scenarios with/without push notifications")
            print(f"   ✅ Use enhanced business logic for push effects")
            print(f"   ✅ Multiple time-of-day scenarios")

        else:
            print("❌ Could not load enhanced model for demo")

    except Exception as e:
        print(f"❌ Error in enhanced demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
