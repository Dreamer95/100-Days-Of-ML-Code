import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedTrafficPredictionModel:
    """
    Advanced Traffic Prediction Model với multiple algorithms và short-term prediction
    """

    def __init__(self, model_dir="models"):
        """
        Initialize the advanced traffic prediction model

        Parameters:
        -----------
        model_dir : str
            Directory để lưu models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.target_columns = ['target_5min', 'target_10min', 'target_20min', 'target_30min']
        self.short_term_targets = ['target_5min', 'target_10min']  # For short-term prediction

        # Model configurations for comparison
        self.model_configs = {
            'RandomForest': {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Ridge': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                }
            },
            'KNN': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                }
            },
            'DecisionTree': {
                'model': DecisionTreeRegressor(),
                'params': {
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'MLP': {
                'model': MLPRegressor(max_iter=500, random_state=42),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }

        # Scaler options
        self.scaler_options = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        self.best_models = {}
        self.model_performances = {}

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

    def load_recent_data_for_short_term_prediction(self, api_key, app_id, hours_back=3):
        """
        Load recent high-granularity data for short-term prediction

        Parameters:
        -----------
        api_key : str
            New Relic API key
        app_id : str
            Application ID
        hours_back : int
            Hours back to collect data (max 3 for minute-level granularity)

        Returns:
        --------
        pandas.DataFrame
            Recent minute-level data
        """
        from datetime import datetime, timedelta, timezone
        from newrelic_traffic_collector import collect_newrelic_data_with_optimal_granularity, process_newrelic_data_with_enhanced_metadata

        # Calculate time range for recent data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)

        print(f"🔍 Loading recent {hours_back}h data for short-term prediction...")
        print(f"   Time range: {start_time} to {end_time} UTC")

        # Collect recent data with minute-level granularity
        data = collect_newrelic_data_with_optimal_granularity(
            api_key=api_key,
            app_id=app_id,
            metrics=["HttpDispatcher"],
            start_time=start_time,
            end_time=end_time,
            week_priority=10,  # Highest priority for recent data
            data_age_days=0
        )

        if not data:
            print("❌ Failed to collect recent data")
            return None

        # Process data
        df = process_newrelic_data_with_enhanced_metadata(data)

        if df is None or df.empty:
            print("❌ No processed data available")
            return None

        print(f"✅ Loaded {len(df)} recent data points with avg interval: {df['interval_minutes'].mean():.1f} min")
        return df

    def load_data(self, csv_file_path=None):
        """
        Load training data with enhanced features for short-term prediction
        """
        if csv_file_path is None:
            # Look for the most recent dataset
            dataset_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if 'newrelic_weekend_traffic' in file and file.endswith('.csv'):
                        dataset_files.append(os.path.join(root, file))

            if not dataset_files:
                print("❌ No dataset files found")
                return None

            csv_file_path = max(dataset_files, key=os.path.getmtime)
            print(f"📊 Using dataset: {csv_file_path}")

        try:
            df = pd.read_csv(csv_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"✅ Loaded {len(df)} records from {os.path.basename(csv_file_path)}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def create_enhanced_features(self, df):
        """
        Create enhanced features for better prediction including short-term patterns
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        print("🔧 Creating enhanced features...")

        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month

        # Circular encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Enhanced lag features for short-term prediction
        lag_windows = [1, 2, 3, 5, 10, 15, 20, 30]  # More granular lags
        for lag in lag_windows:
            df[f'tpm_lag_{lag}'] = df['tpm'].shift(lag)
            if 'response_time' in df.columns:
                df[f'response_time_lag_{lag}'] = df['response_time'].shift(lag)

        # Rolling statistics with multiple windows
        rolling_windows = [3, 5, 10, 15, 30, 60]
        for window in rolling_windows:
            if len(df) >= window:
                df[f'tpm_ma_{window}'] = df['tpm'].rolling(window=window, min_periods=1).mean()
                df[f'tpm_std_{window}'] = df['tpm'].rolling(window=window, min_periods=1).std()
                df[f'tpm_min_{window}'] = df['tpm'].rolling(window=window, min_periods=1).min()
                df[f'tpm_max_{window}'] = df['tpm'].rolling(window=window, min_periods=1).max()

                if 'response_time' in df.columns:
                    df[f'response_time_ma_{window}'] = df['response_time'].rolling(window=window, min_periods=1).mean()

        # Trend and momentum features
        df['tpm_change'] = df['tpm'].diff()
        df['tpm_change_rate'] = df['tpm'].pct_change()
        df['tpm_acceleration'] = df['tpm_change'].diff()

        # Velocity over different periods
        for period in [3, 5, 10]:
            if len(df) > period:
                df[f'tpm_velocity_{period}'] = (df['tpm'] - df['tpm'].shift(period)) / period

        # Push notification enhanced features
        if 'push_notification_active' in df.columns:
            df['push_notification_active'] = df['push_notification_active'].fillna(0)
            df['minutes_since_push'] = df['minutes_since_push'].fillna(999999)

            # Push notification effect decay
            df['push_effect_strength'] = np.where(
                df['minutes_since_push'] <= 15,
                np.exp(-df['minutes_since_push'] / 10),  # Exponential decay
                0
            )
        else:
            df['push_notification_active'] = 0
            df['minutes_since_push'] = 999999
            df['push_effect_strength'] = 0

        # Traffic pattern recognition
        df['is_peak_hour'] = df['hour'].between(10, 16).astype(int)
        df['is_lunch_time'] = df['hour'].between(11, 14).astype(int)
        df['is_evening'] = df['hour'].between(17, 21).astype(int)
        df['is_night'] = (df['hour'].between(22, 23) | df['hour'].between(0, 6)).astype(int)

        # Traffic volatility
        for window in [5, 10, 15]:
            if len(df) >= window:
                df[f'tpm_volatility_{window}'] = df['tpm'].rolling(window=window, min_periods=1).std() / df['tpm'].rolling(window=window, min_periods=1).mean()

        # Granularity and data quality features
        if 'granularity_minutes' in df.columns:
            df['is_minute_level'] = (df['granularity_minutes'] <= 1).astype(int)
            df['is_high_granularity'] = (df['granularity_minutes'] <= 5).astype(int)
        else:
            df['granularity_minutes'] = 1
            df['is_minute_level'] = 1
            df['is_high_granularity'] = 1

        # Fill missing values
        df = df.fillna(0)

        print(f"✅ Created {len([col for col in df.columns if col not in ['timestamp', 'tpm']])} features")

        return df

    def create_targets(self, df):
        """
        Create target variables for prediction with enhanced short-term targets
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        print("🎯 Creating prediction targets...")

        # Standard targets
        target_minutes = [5, 10, 20, 30]

        for minutes in target_minutes:
            target_col = f'target_{minutes}min'

            # For minute-level data, shift by exact minutes
            if 'granularity_minutes' in df.columns:
                # Calculate shift periods based on granularity
                shifts = []
                for idx, row in df.iterrows():
                    granularity = row['granularity_minutes']
                    if granularity <= 1:  # Minute-level data
                        shift = minutes
                    elif granularity <= 5:  # 5-minute data
                        shift = max(1, minutes // int(granularity))
                    else:  # Hourly or coarser
                        shift = max(1, minutes // 60)
                    shifts.append(shift)

                # Apply variable shifts
                df[target_col] = np.nan
                for shift in set(shifts):
                    mask = np.array(shifts) == shift
                    df.loc[mask, target_col] = df.loc[mask, 'tpm'].shift(-shift)
            else:
                # Default: assume minute-level data
                df[target_col] = df['tpm'].shift(-minutes)

        # Drop rows with missing targets
        before_drop = len(df)
        df = df.dropna(subset=[f'target_{m}min' for m in target_minutes])
        after_drop = len(df)
        print(f"📉 Dropped {before_drop - after_drop} rows with missing targets, {after_drop} rows remaining")

        return df

    def prepare_features_and_targets(self, df):
        """
        Prepare features and targets for training
        """
        # Identify feature columns (exclude metadata and targets)
        exclude_cols = [
                           'timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc',
                           'metric_name', 'sequence_id', 'day_name', 'collection_strategy',
                           'data_quality', 'granularity_category'
                       ] + [col for col in df.columns if col.startswith('target_')]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols

        # Prepare features and targets
        X = df[feature_cols].copy()
        y_dict = {}

        for target in self.target_columns:
            if target in df.columns:
                y_dict[target] = df[target].copy()

        print(f"🎯 Prepared {len(feature_cols)} features and {len(y_dict)} targets")
        print(f"📊 Feature columns: {feature_cols[:5]}... (showing first 5)")

        return X, y_dict

    def evaluate_model_performance(self, model, X_train, X_test, y_train, y_test, model_name, target):
        """
        Comprehensive model evaluation
        """
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        return {
            'model_name': model_name,
            'target': target,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'model': model
        }

    def train_and_compare_models(self, X, y_dict, test_size=0.2):
        """
        Train and compare multiple models for each target
        """
        print("🏃‍♂️ Training and comparing multiple models...")

        # Time series split to respect temporal order
        tscv = TimeSeriesSplit(n_splits=3)

        all_results = []
        self.best_models = {}

        for target in self.target_columns:
            if target not in y_dict:
                continue

            print(f"\n🎯 Training models for {target}...")

            y = y_dict[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )

            # Try different scalers
            scaler_results = {}

            for scaler_name, scaler in self.scaler_options.items():
                print(f"  📏 Testing with {scaler_name} scaler...")

                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train models with this scaler
                for model_name, config in self.model_configs.items():
                    try:
                        # Grid search for best parameters
                        grid_search = GridSearchCV(
                            config['model'],
                            config['params'],
                            cv=tscv,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1,
                            verbose=0
                        )

                        grid_search.fit(X_train_scaled, y_train)
                        best_model = grid_search.best_estimator_

                        # Evaluate
                        result = self.evaluate_model_performance(
                            best_model, X_train_scaled, X_test_scaled,
                            y_train, y_test, model_name, target
                        )
                        result['scaler'] = scaler_name
                        result['best_params'] = grid_search.best_params_
                        result['scaler_object'] = scaler

                        all_results.append(result)

                        print(f"    {model_name}: RMSE={result['rmse']:.2f}, R²={result['r2']:.3f}, MAE={result['mae']:.2f}")

                    except Exception as e:
                        print(f"    ❌ {model_name} failed: {e}")
                        continue

            # Select best model for this target
            target_results = [r for r in all_results if r['target'] == target]
            if target_results:
                best_result = min(target_results, key=lambda x: x['rmse'])
                self.best_models[target] = best_result
                self.scalers[target] = best_result['scaler_object']

                print(f"  🏆 Best model for {target}: {best_result['model_name']} with {best_result['scaler']} scaler")
                print(f"      RMSE: {best_result['rmse']:.2f}, R²: {best_result['r2']:.3f}, MAE: {best_result['mae']:.2f}")

        # Store performance comparison
        self.model_performances = pd.DataFrame(all_results)

        return all_results

    def predict_short_term_with_push_notification(self, current_data, push_notification_sent=False,
                                                  push_send_time=None):
        """
        Short-term prediction (5-10 minutes) with push notification effect

        Parameters:
        -----------
        current_data : dict or pandas.Series
            Current traffic data including recent TPM values
        push_notification_sent : bool
            Whether a push notification was just sent
        push_send_time : datetime, optional
            When the push notification was sent

        Returns:
        --------
        dict
            Predictions for next 5 and 10 minutes with confidence intervals
        """
        if not self.best_models:
            raise ValueError("Models not trained. Call train_and_compare_models first.")

        # Prepare input features
        if isinstance(current_data, dict):
            input_df = pd.DataFrame([current_data])
        else:
            input_df = pd.DataFrame([current_data])

        # Add push notification features
        if push_notification_sent:
            input_df['push_notification_active'] = 1
            input_df['minutes_since_push'] = 0
            input_df['push_effect_strength'] = 1.0
        else:
            input_df['push_notification_active'] = 0
            input_df['minutes_since_push'] = input_df.get('minutes_since_push', 999999)
            input_df['push_effect_strength'] = 0

        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        predictions = {}
        confidence_intervals = {}

        for target in self.short_term_targets:
            if target in self.best_models:
                model_info = self.best_models[target]
                model = model_info['model']
                scaler = self.scalers[target]

                # Scale features
                X_scaled = scaler.transform(input_df[self.feature_columns])

                # Make prediction
                pred = model.predict(X_scaled)[0]

                # Apply push notification boost if applicable
                if push_notification_sent:
                    if target == 'target_5min':
                        boost_factor = 1.15  # 15% boost for 5min
                    elif target == 'target_10min':
                        boost_factor = 1.25  # 25% boost for 10min
                    else:
                        boost_factor = 1.10

                    pred *= boost_factor

                predictions[target] = pred

                # Calculate confidence interval (if model supports it)
                if hasattr(model, 'predict_proba') or hasattr(model, 'std_'):
                    # For models with uncertainty estimates
                    mae = model_info['mae']
                    confidence_intervals[target] = {
                        'lower': pred - 1.96 * mae,
                        'upper': pred + 1.96 * mae
                    }
                else:
                    # Use historical MAE as confidence bound
                    mae = model_info['mae']
                    confidence_intervals[target] = {
                        'lower': pred - mae,
                        'upper': pred + mae
                    }

        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'push_notification_applied': push_notification_sent,
            'model_info': {target: self.best_models[target]['model_name']
                           for target in predictions.keys()}
        }

    def predict_high_tpm_periods(self, hours_ahead=24, threshold_percentile=75):
        """
        Predict periods with high TPM in the next N hours

        Parameters:
        -----------
        hours_ahead : int
            How many hours ahead to predict
        threshold_percentile : int
            Percentile threshold for "high" TPM

        Returns:
        --------
        dict
            Predicted high TPM periods with timestamps and expected values
        """
        from datetime import datetime, timedelta

        if not self.best_models:
            raise ValueError("Models not trained. Call train_and_compare_models first.")

        # Load recent data to establish current state
        df = self.load_data()
        if df is None:
            return {}

        # Calculate threshold
        threshold = np.percentile(df['tpm'], threshold_percentile)
        print(f"🎯 High TPM threshold ({threshold_percentile}th percentile): {threshold:.1f}")

        # Generate future time points
        current_time = datetime.now()
        future_times = [current_time + timedelta(minutes=10*i) for i in range(hours_ahead*6)]

        high_tpm_predictions = []

        # Get most recent data point as baseline
        latest_data = df.iloc[-1].to_dict()

        for future_time in future_times:
            # Update time features
            future_data = latest_data.copy()
            future_data['hour'] = future_time.hour
            future_data['minute'] = future_time.minute
            future_data['day_of_week'] = future_time.weekday()
            future_data['is_weekend'] = 1 if future_time.weekday() >= 5 else 0

            # Circular encoding
            future_data['hour_sin'] = np.sin(2 * np.pi * future_time.hour / 24)
            future_data['hour_cos'] = np.cos(2 * np.pi * future_time.hour / 24)
            future_data['minute_sin'] = np.sin(2 * np.pi * future_time.minute / 60)
            future_data['minute_cos'] = np.cos(2 * np.pi * future_time.minute / 60)

            # Predict
            prediction_result = self.predict_short_term_with_push_notification(
                future_data, push_notification_sent=False
            )

            # Check if any prediction exceeds threshold
            for target, pred_value in prediction_result['predictions'].items():
                if pred_value > threshold:
                    high_tpm_predictions.append({
                        'timestamp': future_time,
                        'predicted_tpm': pred_value,
                        'target_period': target,
                        'confidence_lower': prediction_result['confidence_intervals'][target]['lower'],
                        'confidence_upper': prediction_result['confidence_intervals'][target]['upper'],
                        'threshold_exceeded': (pred_value - threshold) / threshold * 100
                    })

        return {
            'threshold': threshold,
            'high_tpm_periods': sorted(high_tpm_predictions, key=lambda x: x['timestamp']),
            'total_periods_found': len(high_tpm_predictions)
        }

    def save_models(self):
        """Save trained models and scalers"""
        print("💾 Saving models and scalers...")

        for target, model_info in self.best_models.items():
            # Save model
            model_path = os.path.join(self.model_dir, f'{target}_best_model.pkl')
            joblib.dump(model_info['model'], model_path)

            # Save scaler
            scaler_path = os.path.join(self.model_dir, f'{target}_scaler.pkl')
            joblib.dump(self.scalers[target], scaler_path)

        # Save metadata
        metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'best_models_info': {k: {key: val for key, val in v.items() if key != 'model'}
                                 for k, v in self.best_models.items()},
            'model_performances': self.model_performances.to_dict() if hasattr(self, 'model_performances') else {}
        }
        joblib.dump(metadata, metadata_path)

        # Save performance summary
        if hasattr(self, 'model_performances'):
            performance_path = os.path.join(self.model_dir, 'model_performance_comparison.csv')
            self.model_performances.to_csv(performance_path, index=False)

        print(f"✅ Models saved to {self.model_dir}/")

    def load_models(self):
        """Load trained models and scalers"""
        print("📂 Loading saved models...")

        try:
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
            metadata = joblib.load(metadata_path)

            self.feature_columns = metadata['feature_columns']
            self.target_columns = metadata['target_columns']

            # Load models and scalers
            for target in self.target_columns:
                model_path = os.path.join(self.model_dir, f'{target}_best_model.pkl')
                scaler_path = os.path.join(self.model_dir, f'{target}_scaler.pkl')

                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)

                    self.best_models[target] = {'model': model}
                    self.scalers[target] = scaler

            print(f"✅ Loaded models for {len(self.best_models)} targets")
            return True

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def create_model_comparison_report(self):
        """Create comprehensive model comparison report"""
        if not hasattr(self, 'model_performances') or self.model_performances.empty:
            print("❌ No model performance data available")
            return

        print("\n" + "="*80)
        print("🏆 MODEL PERFORMANCE COMPARISON REPORT")
        print("="*80)

        # Summary by target
        for target in self.target_columns:
            target_results = self.model_performances[self.model_performances['target'] == target]
            if target_results.empty:
                continue

            print(f"\n📊 {target.upper()} PREDICTIONS:")
            print("-" * 60)

            # Sort by RMSE
            target_results_sorted = target_results.sort_values('rmse')

            for _, row in target_results_sorted.head(5).iterrows():  # Top 5 models
                print(f"  {row['model_name']:15} ({row['scaler']:8}): "
                      f"RMSE={row['rmse']:6.2f} | MAE={row['mae']:6.2f} | "
                      f"R²={row['r2']:5.3f} | CV-RMSE={row['cv_rmse']:6.2f}")

            # Best model summary
            best = target_results_sorted.iloc[0]
            print(f"  🏆 BEST: {best['model_name']} with {best['scaler']} scaling")
            print(f"      Performance: RMSE={best['rmse']:.2f}, MAE={best['mae']:.2f}, R²={best['r2']:.3f}")

        # Overall summary
        print(f"\n📈 OVERALL SUMMARY:")
        print("-" * 40)
        print(f"  Total models trained: {len(self.model_performances)}")
        print(f"  Average R² score: {self.model_performances['r2'].mean():.3f}")
        print(f"  Best overall R²: {self.model_performances['r2'].max():.3f}")

        # Model type performance
        print(f"\n🔍 PERFORMANCE BY MODEL TYPE:")
        print("-" * 50)
        model_avg = self.model_performances.groupby('model_name')[['rmse', 'r2', 'mae']].mean()
        model_avg_sorted = model_avg.sort_values('rmse')

        for model_name, stats in model_avg_sorted.iterrows():
            print(f"  {model_name:15}: RMSE={stats['rmse']:6.2f} | R²={stats['r2']:5.3f} | MAE={stats['mae']:6.2f}")

        print("="*80)

    def visualize_model_comparison(self, save_path=None):
        """Create comprehensive visualization of model performance"""
        if not hasattr(self, 'model_performances') or self.model_performances.empty:
            print("❌ No model performance data for visualization")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🏆 Advanced Traffic Prediction Model Comparison', fontsize=16, fontweight='bold')

        # 1. RMSE comparison by model type
        ax1 = axes[0, 0]
        model_rmse = self.model_performances.groupby('model_name')['rmse'].mean().sort_values()
        model_rmse.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Average RMSE by Model Type')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)

        # 2. R² score comparison
        ax2 = axes[0, 1]
        model_r2 = self.model_performances.groupby('model_name')['r2'].mean().sort_values(ascending=False)
        model_r2.plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Average R² Score by Model Type')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Performance by target period
        ax3 = axes[0, 2]
        target_performance = self.model_performances.groupby('target')['rmse'].mean()
        target_performance.plot(kind='bar', ax=ax3, color='orange')
        ax3.set_title('RMSE by Prediction Period')
        ax3.set_ylabel('RMSE')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Scaler effect
        ax4 = axes[1, 0]
        scaler_performance = self.model_performances.groupby('scaler')['rmse'].mean()
        scaler_performance.plot(kind='bar', ax=ax4, color='pink')
        ax4.set_title('RMSE by Scaler Type')
        ax4.set_ylabel('RMSE')
        ax4.tick_params(axis='x', rotation=45)

        # 5. Scatter plot: RMSE vs R²
        ax5 = axes[1, 1]
        scatter = ax5.scatter(self.model_performances['rmse'],
                              self.model_performances['r2'],
                              c=self.model_performances['mae'],
                              cmap='viridis', alpha=0.6)
        ax5.set_xlabel('RMSE')
        ax5.set_ylabel('R² Score')
        ax5.set_title('RMSE vs R² (Color = MAE)')
        plt.colorbar(scatter, ax=ax5)

        # 6. Best models summary
        ax6 = axes[1, 2]
        best_models_data = []
        for target in self.target_columns:
            if target in self.best_models:
                best_models_data.append({
                    'Target': target.replace('target_', '').replace('min', ' min'),
                    'RMSE': self.best_models[target]['rmse']
                })

        if best_models_data:
            best_df = pd.DataFrame(best_models_data)
            best_df.plot(x='Target', y='RMSE', kind='bar', ax=ax6, color='gold')
            ax6.set_title('Best Model RMSE by Target')
            ax6.set_ylabel('RMSE')
            ax6.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")

        plt.show()

    def train_and_save_complete_system(self, csv_file_path=None):
        """Complete training pipeline with all features"""
        print("🚀 Starting Advanced Traffic Prediction Model Training...")
        print("="*70)

        # 1. Load data
        df = self.load_data(csv_file_path)
        if df is None:
            return False

        # 2. Create enhanced features
        df = self.create_enhanced_features(df)

        # 3. Create targets
        df = self.create_targets(df)

        # 4. Prepare features and targets
        X, y_dict = self.prepare_features_and_targets(df)

        # 5. Train and compare models
        results = self.train_and_compare_models(X, y_dict)

        # 6. Create comparison report
        self.create_model_comparison_report()

        # 7. Save models
        self.save_models()

        # 8. Create visualization
        chart_path = os.path.join(self.model_dir, 'model_comparison_chart.png')
        self.visualize_model_comparison(save_path=chart_path)

        print("\n🎉 Complete training system finished!")
        print(f"📂 Models and reports saved to: {self.model_dir}/")

        return True

# Example usage and demonstration
def example_advanced_usage():
    """Demonstrate the advanced traffic prediction system"""
    print("🚀 Advanced Traffic Prediction System Demo")
    print("="*50)

    # Create model instance
    model = AdvancedTrafficPredictionModel()

    # Train complete system
    success = model.train_and_save_complete_system()

    if success:
        print("\n🔮 Testing short-term predictions...")

        # Example 1: Normal prediction
        current_data = {
            'tpm': 450,
            'hour': 14,
            'minute': 30,
            'day_of_week': 4,
            'response_time': 120,
            'tpm_lag_1': 440,
            'tpm_lag_2': 435,
            'tpm_lag_3': 450,
            'is_weekend': 0,
            'is_peak_hour': 1
        }

        result = model.predict_short_term_with_push_notification(current_data)
        print("\n📊 Normal Prediction:")
        for target, pred in result['predictions'].items():
            ci = result['confidence_intervals'][target]
            print(f"  {target}: {pred:.1f} TPM (CI: {ci['lower']:.1f} - {ci['upper']:.1f})")

        # Example 2: Prediction with push notification
        result_push = model.predict_short_term_with_push_notification(
            current_data, push_notification_sent=True
        )
        print("\n🔔 With Push Notification:")
        for target, pred in result_push['predictions'].items():
            ci = result_push['confidence_intervals'][target]
            boost = ((pred / result['predictions'][target]) - 1) * 100
            print(f"  {target}: {pred:.1f} TPM (CI: {ci['lower']:.1f} - {ci['upper']:.1f}) [+{boost:.1f}%]")

        # Example 3: Predict high TPM periods
        print("\n🎯 Predicting high TPM periods for next 6 hours...")
        high_periods = model.predict_high_tpm_periods(hours_ahead=6)

        print(f"High TPM threshold: {high_periods['threshold']:.1f}")
        print(f"Found {high_periods['total_periods_found']} high TPM periods")

        for period in high_periods['high_tpm_periods'][:5]:  # Show first 5
            print(f"  {period['timestamp'].strftime('%H:%M')}: "
                  f"{period['predicted_tpm']:.1f} TPM "
                  f"(+{period['threshold_exceeded']:.1f}% above threshold)")

if __name__ == "__main__":
    example_advanced_usage()