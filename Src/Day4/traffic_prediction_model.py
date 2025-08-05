import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


class TrafficPredictionModel:
    """
    A comprehensive traffic prediction model that can predict future traffic metrics
    based on current conditions and historical patterns.
    """

    def __init__(self,
                 data_path="/Users/dongdinh/Documents/Learning/100-Days-Of-ML-Code/Src/Day4/datasets/newrelic_weekend_traffic_enhanced.csv"):
        """
        Initialize the Traffic Prediction Model.

        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV data file. If None, synthetic data will be generated.
        """
        self.data_path = data_path
        self.models = {}
        self.feature_columns = []
        self.target_columns = ['target_5min', 'target_10min', 'target_20min', 'target_30min']
        self.model_dir = "models"
        self.feature_mapping = {}

        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def generate_synthetic_data(self, num_records=10000):
        """
        Generate synthetic traffic data for demonstration purposes.

        Parameters:
        -----------
        num_records : int
            Number of records to generate

        Returns:
        --------
        pandas.DataFrame
            Synthetic traffic data
        """
        print(f"Generating {num_records} synthetic traffic records...")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Generate timestamps (last 30 days, every 5 minutes)
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [start_date + timedelta(minutes=5 * i) for i in range(num_records)]

        data = []
        base_tpm = 300  # Base traffic per minute

        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            # Create realistic traffic patterns
            # Higher traffic during business hours (9-17)
            hour_factor = 1.0
            if 9 <= hour <= 17:
                hour_factor = 1.5 + 0.3 * np.sin((hour - 9) * np.pi / 8)
            elif 18 <= hour <= 22:
                hour_factor = 1.2
            else:
                hour_factor = 0.6

            # Weekend patterns (lower overall traffic)
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0

            # Add some randomness and trends
            random_factor = np.random.normal(1.0, 0.2)
            trend_factor = 1.0 + (i / num_records) * 0.1  # Slight upward trend

            # Calculate current TPM
            current_tpm = base_tpm * hour_factor * weekend_factor * random_factor * trend_factor
            current_tpm = max(50, current_tpm)  # Minimum traffic

            # Response time (correlated with traffic)
            response_time = 80 + (current_tpm / 10) + np.random.normal(0, 10)
            response_time = max(50, response_time)

            # Push notification simulation (random events)
            push_active = 1 if np.random.random() < 0.05 else 0
            minutes_since_push = np.random.randint(0, 1440) if not push_active else 999999

            record = {
                'timestamp': timestamp,
                'tpm': current_tpm,
                'hour': hour,
                'day_of_week': day_of_week,
                'response_time': response_time,  # Matching NewRelic structure
                'push_notification_active': push_active,
                'minutes_since_push': minutes_since_push,
                'granularity_minutes': 5
            }

            data.append(record)

        df = pd.DataFrame(data)

        # Add previous TPM values (rolling features)
        for i in range(1, 6):
            df[f'tpm_lag_{i}'] = df['tpm'].shift(i)

        # Fill NaN values for the first few rows
        df = df.fillna(method='bfill')

        # Generate target columns (future TPM values)
        for minutes in [5, 10, 20, 30]:
            target_col = f'target_{minutes}min'
            df[target_col] = df['tpm'].shift(-int(minutes / 5))

        # Handle missing values in target columns (at the end of the dataset)
        for col in df.columns:
            if col.startswith('target_'):
                df[col] = df[col].fillna(df['tpm'])

        print(f"Generated {len(df)} synthetic records.")
        return df

    def load_data(self):
        """
        Load and preprocess the traffic data from CSV or generate synthetic data.

        Returns:
        --------
        pandas.DataFrame
            Preprocessed traffic data
        """
        if self.data_path and os.path.exists(self.data_path):
            print(f"Loading data from {self.data_path}...")
            df = pd.read_csv(self.data_path)

            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Create target columns if they don't exist (based on actual granularity)
            if 'granularity_minutes' in df.columns:
                granularity = df['granularity_minutes'].min()
            else:
                granularity = 5  # Default granularity

            for minutes in [5, 10, 20, 30]:
                target_col = f'target_{minutes}min'
                if target_col not in df.columns:
                    # Calculate future TPM values based on actual granularity
                    shift_periods = int(minutes / granularity)
                    df[target_col] = df['tpm'].shift(-shift_periods)

            # Handle missing values in target columns (at the end of the dataset)
            for col in df.columns:
                if col.startswith('target_'):
                    df[col] = df[col].fillna(df['tpm'])

            # Add push notification features if they don't exist
            if 'push_notification_active' not in df.columns:
                df['push_notification_active'] = 0

            if 'minutes_since_push' not in df.columns:
                df['minutes_since_push'] = 999999  # Large default value

            print(f"Loaded {len(df)} records from file.")
            print(f"Available columns: {list(df.columns)}")
        else:
            print("Data file not found or not specified. Generating synthetic data...")
            df = self.generate_synthetic_data()

        return df

    def preprocess_data(self, df):
        """
        Preprocess the data for training with enhanced feature detection.

        Parameters:
        -----------
        df : pandas.DataFrame
            Raw traffic data

        Returns:
        --------
        tuple
            (X, y_dict, feature_names) where:
            - X: feature matrix
            - y_dict: dictionary of target arrays
            - feature_names: list of feature column names
        """
        print("Preprocessing data...")
        print(f"Available columns: {list(df.columns)}")

        # Define potential feature columns and their alternatives based on NewRelic data structure
        potential_features = {
            'tpm': ['tpm', 'transactions_per_minute', 'traffic'],
            'hour': ['hour', 'time_hour', 'hr'],
            'day_of_week': ['day_of_week', 'dow', 'weekday'],
            'response_time': ['response_time', 'response_time_ms', 'average_response_time', 'latency'],
            'push_notification_active': ['push_notification_active', 'push_active', 'notification_active'],
            'minutes_since_push': ['minutes_since_push', 'minutes_after_push', 'time_since_push'],
            # Additional NewRelic specific features
            'call_count': ['call_count', 'request_count', 'total_calls'],
            'min_response_time': ['min_response_time', 'minimum_response_time'],
            'max_response_time': ['max_response_time', 'maximum_response_time'],
            'standard_deviation': ['standard_deviation', 'std_dev', 'response_time_std'],
            'is_weekend': ['is_weekend', 'weekend_flag'],
            'ml_weight': ['ml_weight', 'data_weight', 'quality_weight'],
            'is_high_granularity': ['is_high_granularity', 'high_granularity'],
            'granularity_minutes': ['granularity_minutes', 'interval_minutes']
        }

        # Find available feature columns
        feature_columns = []
        feature_mapping = {}

        for feature_name, alternatives in potential_features.items():
            found_column = None
            for alt in alternatives:
                if alt in df.columns:
                    found_column = alt
                    break

            if found_column:
                feature_columns.append(found_column)
                feature_mapping[feature_name] = found_column
                print(f"✓ Using '{found_column}' for feature '{feature_name}'")

        # Add lagged TPM features if they exist
        for i in range(1, 6):
            lag_col = f'tpm_lag_{i}'
            if lag_col in df.columns:
                feature_columns.append(lag_col)
                print(f"✓ Found lagged feature: {lag_col}")

        # Add rolling statistics features if they exist
        for window in [3, 5]:
            for metric in ['tpm', 'response_time']:
                for stat in ['ma', 'std']:
                    stat_col = f'{metric}_{stat}_{window}'
                    if stat_col in df.columns:
                        feature_columns.append(stat_col)
                        print(f"✓ Found rolling feature: {stat_col}")

        # Add change features if they exist
        change_features = ['tpm_change', 'tpm_change_rate', 'response_time_change']
        for change_col in change_features:
            if change_col in df.columns:
                feature_columns.append(change_col)
                print(f"✓ Found change feature: {change_col}")

        # Add circular time features if they exist
        circular_features = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']
        for circ_col in circular_features:
            if circ_col in df.columns:
                feature_columns.append(circ_col)
                print(f"✓ Found circular feature: {circ_col}")

        # If no features found, use all numeric columns except target columns
        if not feature_columns:
            print("Warning: No predefined features found, using all numeric columns...")
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            # Remove target columns and timestamp-related columns from features
            excluded_patterns = ['target_', 'timestamp', 'minute', 'metric_name']
            feature_columns = [col for col in numeric_columns
                               if not any(pattern in col for pattern in excluded_patterns)]

        # Ensure we have at least some basic features
        essential_features = ['tpm', 'hour', 'day_of_week']
        for essential in essential_features:
            if essential in df.columns and essential not in feature_columns:
                feature_columns.append(essential)
                print(f"✓ Added essential feature: {essential}")

        # Store feature columns and mapping for later use
        self.feature_columns = feature_columns
        self.feature_mapping = feature_mapping

        # Create feature matrix with only existing columns
        available_features = [col for col in feature_columns if col in df.columns]
        if not available_features:
            raise ValueError("No valid feature columns found in the dataset")

        X = df[available_features].copy()

        # Handle any remaining missing values
        X = X.fillna(X.mean())

        # Create target dictionary with quality weighting
        y_dict = {}
        base_tpm_column = feature_mapping.get('tpm', 'tpm') if 'tpm' in feature_mapping else 'tpm'

        # Use ml_weight for sample weighting if available
        sample_weights = None
        if 'ml_weight' in df.columns:
            sample_weights = df['ml_weight'].values
            print(
                f"✓ Using ml_weight for sample weighting (range: {sample_weights.min():.1f}-{sample_weights.max():.1f})")

        for target_col in self.target_columns:
            if target_col in df.columns:
                y_dict[target_col] = df[target_col].fillna(df[base_tpm_column])
            else:
                # Create synthetic targets with realistic variations
                if base_tpm_column in df.columns:
                    base_values = df[base_tpm_column].values
                    # Create time-based variations
                    if 'target_5min' in target_col:
                        # Short-term: small variations
                        noise = np.random.normal(0, 0.02, len(base_values))
                        y_dict[target_col] = base_values * (1 + noise)
                    elif 'target_10min' in target_col:
                        noise = np.random.normal(0, 0.03, len(base_values))
                        y_dict[target_col] = base_values * (1 + noise)
                    elif 'target_20min' in target_col:
                        noise = np.random.normal(0, 0.05, len(base_values))
                        y_dict[target_col] = base_values * (1 + noise)
                    elif 'target_30min' in target_col:
                        # Longer-term: larger variations with trend
                        noise = np.random.normal(0, 0.08, len(base_values))
                        trend = np.sin(np.arange(len(base_values)) * 0.1) * 0.05
                        y_dict[target_col] = base_values * (1 + noise + trend)
                    else:
                        y_dict[target_col] = base_values
                else:
                    print(f"Warning: Cannot create target '{target_col}', using zeros")
                    y_dict[target_col] = np.zeros(len(df))

        print(f"✅ Final features ({len(available_features)}): {available_features}")
        print(f"✅ Samples: {len(X)}")
        print(f"✅ Targets: {list(y_dict.keys())}")

        if sample_weights is not None:
            print(
                f"✅ Sample weights: min={sample_weights.min():.1f}, max={sample_weights.max():.1f}, mean={sample_weights.mean():.1f}")

        return X, y_dict, available_features

    def train_models(self, X, y_dict):
        """
        Train separate models for each prediction horizon with enhanced configuration.

        Parameters:
        -----------
        X : pandas.DataFrame
            Feature matrix
        y_dict : dict
            Dictionary of target arrays

        Returns:
        --------
        dict
            Dictionary containing trained models and evaluation metrics
        """
        print("Training models...")

        results = {}

        for target_name, y in y_dict.items():
            print(f"\n🎯 Training model for {target_name}...")

            # Split data with stratification based on TPM levels if possible
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Enhanced Random Forest model with better parameters for time series
            model = RandomForestRegressor(
                n_estimators=200,  # More trees for better performance
                max_depth=15,  # Deeper trees for complex patterns
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',  # Better for high-dimensional data
                bootstrap=True,
                oob_score=True,  # Out-of-bag score for model validation
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate comprehensive metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Additional metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
            oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None

            # Store model and results
            self.models[target_name] = model

            results[target_name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'oob_score': oob_score,
                'feature_importance': dict(zip(self.feature_columns, model.feature_importances_))
            }

            print(f"  📊 RMSE: {rmse:.2f}")
            print(f"  📊 MAE: {mae:.2f}")
            print(f"  📊 R²: {r2:.3f}")
            print(f"  📊 MAPE: {mape:.2f}%")
            if oob_score:
                print(f"  📊 OOB Score: {oob_score:.3f}")

        return results

    def save_models(self):
        """
        Save trained models and metadata to disk.
        """
        print("Saving models...")

        for target_name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{target_name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Save feature columns and mapping
        feature_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)

        mapping_path = os.path.join(self.model_dir, 'feature_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.feature_mapping, f)

        print(f"✅ Models saved to {self.model_dir}/")

    def load_models(self):
        """
        Load pre-trained models from disk.
        """
        print("Loading models...")

        # Load feature columns
        feature_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                self.feature_columns = pickle.load(f)

        # Load feature mapping
        mapping_path = os.path.join(self.model_dir, 'feature_mapping.pkl')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.feature_mapping = pickle.load(f)

        # Load models
        for target_name in self.target_columns:
            model_path = os.path.join(self.model_dir, f'{target_name}_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[target_name] = pickle.load(f)
                print(f"  ✅ Loaded {target_name} model")
            else:
                print(f"  ⚠️ Warning: {target_name} model not found")

        print("Models loaded successfully!")

    def predict(self, current_tpm, hour, day_of_week, response_time,
                previous_tpm_values=None, push_notification_active=0,
                minutes_since_push=999999, **kwargs):
        """
        Make traffic predictions for different time horizons.

        Parameters:
        -----------
        current_tpm : float
            Current traffic per minute
        hour : int
            Hour of day (0-23)
        day_of_week : int
            Day of week (0=Monday, 6=Sunday)
        response_time : float
            Current response time in milliseconds
        previous_tpm_values : list, optional
            List of previous TPM values (up to 5 values)
        push_notification_active : int
            Whether push notification is currently active (0 or 1)
        minutes_since_push : int
            Minutes since last push notification
        **kwargs : dict
            Additional features that may be available

        Returns:
        --------
        dict
            Dictionary with predictions for each time horizon
        """
        if not self.models:
            raise ValueError("No models loaded. Please train or load models first.")

        # Prepare feature vector with flexible mapping
        features = {
            'tpm': current_tpm,
            'hour': hour,
            'day_of_week': day_of_week,
            'response_time': response_time,
            'push_notification_active': push_notification_active,
            'minutes_since_push': minutes_since_push,
            'is_weekend': 1 if day_of_week >= 5 else 0,
        }

        # Add any additional features from kwargs
        features.update(kwargs)

        # Add lagged TPM features
        if previous_tpm_values:
            for i, tpm_val in enumerate(previous_tpm_values[:5], 1):
                features[f'tpm_lag_{i}'] = tpm_val
        else:
            # Use current TPM as default for all lags
            for i in range(1, 6):
                features[f'tpm_lag_{i}'] = current_tpm

        # Add circular time encoding
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Create feature vector in the correct order
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Make predictions
        predictions = {}
        for target_name, model in self.models.items():
            prediction = model.predict(feature_vector)[0]
            predictions[target_name] = max(0, prediction)  # Ensure non-negative

        return predictions

    def get_feature_importance(self):
        """
        Get feature importance for all models.

        Returns:
        --------
        dict
            Dictionary with feature importance for each model
        """
        importance_dict = {}

        for target_name, model in self.models.items():
            importance_dict[target_name] = dict(
                zip(self.feature_columns, model.feature_importances_)
            )

        return importance_dict

    def classify_tpm_values(self, df=None, num_categories=3):
        """
        Classify TPM values into categories (low, medium, high).

        Parameters:
        -----------
        df : pandas.DataFrame, optional
            Data containing TPM values. If None, loads data from self.data_path.
        num_categories : int, optional
            Number of categories to create (default: 3 for low, medium, high)

        Returns:
        --------
        tuple
            (df_with_categories, thresholds, category_stats)
            - df_with_categories: DataFrame with added 'tpm_category' column
            - thresholds: Dictionary with threshold values for each category
            - category_stats: Dictionary with statistics for each category
        """
        print("Classifying TPM values into categories...")

        if df is None:
            df = self.load_data()

        # Make a copy to avoid modifying the original
        df_copy = df.copy()

        # Get the TPM column name
        tpm_col = self.feature_mapping.get('tpm', 'tpm')

        # Calculate thresholds using quantiles for equal-sized groups
        quantiles = np.linspace(0, 1, num_categories + 1)
        thresholds = np.quantile(df_copy[tpm_col], quantiles)

        # Create category labels
        if num_categories == 3:
            category_labels = ['low', 'medium', 'high']
        else:
            category_labels = [f'category_{i+1}' for i in range(num_categories)]

        # Create a new column with categories
        df_copy['tpm_category'] = pd.cut(
            df_copy[tpm_col], 
            bins=thresholds, 
            labels=category_labels,
            include_lowest=True
        )

        # Calculate statistics for each category
        category_stats = {}
        for category in category_labels:
            category_data = df_copy[df_copy['tpm_category'] == category]
            category_stats[category] = {
                'count': len(category_data),
                'percentage': len(category_data) / len(df_copy) * 100,
                'mean_tpm': category_data[tpm_col].mean(),
                'min_tpm': category_data[tpm_col].min(),
                'max_tpm': category_data[tpm_col].max()
            }

        # Create threshold dictionary
        threshold_dict = {
            'min': thresholds[0],
            'max': thresholds[-1]
        }
        for i in range(1, len(thresholds) - 1):
            threshold_dict[f'{category_labels[i-1]}_to_{category_labels[i]}'] = thresholds[i]

        print(f"✅ Classified TPM values into {num_categories} categories")
        for category, stats in category_stats.items():
            print(f"  - {category.upper()}: {stats['count']} samples ({stats['percentage']:.1f}%), " 
                  f"TPM range: {stats['min_tpm']:.1f} - {stats['max_tpm']:.1f}, mean: {stats['mean_tpm']:.1f}")

        return df_copy, threshold_dict, category_stats

    def analyze_tpm_categories_by_time(self, df_with_categories=None):
        """
        Analyze TPM categories by time periods (hour, day of week).

        Parameters:
        -----------
        df_with_categories : pandas.DataFrame, optional
            DataFrame with 'tpm_category' column. If None, calls classify_tpm_values().

        Returns:
        --------
        dict
            Dictionary with time period analysis for each category
        """
        print("Analyzing TPM categories by time periods...")

        if df_with_categories is None or 'tpm_category' not in df_with_categories.columns:
            df_with_categories, _, _ = self.classify_tpm_values()

        # Get hour and day_of_week column names
        hour_col = self.feature_mapping.get('hour', 'hour')
        day_col = self.feature_mapping.get('day_of_week', 'day_of_week')

        # Ensure the required columns exist
        if hour_col not in df_with_categories.columns or day_col not in df_with_categories.columns:
            print(f"⚠️ Warning: Required columns not found. Using default column names.")
            hour_col = 'hour'
            day_col = 'day_of_week'

        # Get unique categories
        categories = df_with_categories['tpm_category'].unique()

        # Initialize results dictionary
        time_analysis = {
            'hour_distribution': {},
            'day_distribution': {},
            'hour_day_heatmap': {}
        }

        # Analyze distribution by hour
        hour_distribution = df_with_categories.groupby([hour_col, 'tpm_category']).size().unstack(fill_value=0)
        time_analysis['hour_distribution'] = hour_distribution

        # Analyze distribution by day of week
        day_distribution = df_with_categories.groupby([day_col, 'tpm_category']).size().unstack(fill_value=0)
        time_analysis['day_distribution'] = day_distribution

        # Create hour-day heatmap data
        heatmap_data = {}
        for category in categories:
            category_data = df_with_categories[df_with_categories['tpm_category'] == category]
            pivot_table = pd.crosstab(
                index=category_data[hour_col], 
                columns=category_data[day_col],
                normalize='all'
            ) * 100  # Convert to percentage
            heatmap_data[category] = pivot_table

        time_analysis['hour_day_heatmap'] = heatmap_data

        # Find peak hours and days for each category
        peak_times = {}
        for category in categories:
            # Peak hours (top 3)
            if len(hour_distribution.columns) > 0 and category in hour_distribution.columns:
                peak_hours = hour_distribution[category].nlargest(3)
                peak_hours_dict = {str(hour): count for hour, count in peak_hours.items()}
            else:
                peak_hours_dict = {}

            # Peak days (top 2)
            if len(day_distribution.columns) > 0 and category in day_distribution.columns:
                peak_days = day_distribution[category].nlargest(2)
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                peak_days_dict = {day_names[day]: count for day, count in peak_days.items()}
            else:
                peak_days_dict = {}

            peak_times[category] = {
                'peak_hours': peak_hours_dict,
                'peak_days': peak_days_dict
            }

        time_analysis['peak_times'] = peak_times

        # Print summary
        print("✅ Time period analysis completed")
        for category in categories:
            print(f"\n  📊 {category.upper()} TPM category:")
            if category in peak_times and 'peak_hours' in peak_times[category]:
                print(f"    - Peak hours: {', '.join([f'{h}:00 ({c} samples)' for h, c in peak_times[category]['peak_hours'].items()])}")
            if category in peak_times and 'peak_days' in peak_times[category]:
                print(f"    - Peak days: {', '.join([f'{d} ({c} samples)' for d, c in peak_times[category]['peak_days'].items()])}")

        return time_analysis

    def predict_with_category_awareness(self, current_tpm, hour, day_of_week, response_time,
                                       previous_tpm_values=None, push_notification_active=0,
                                       minutes_since_push=999999, **kwargs):
        """
        Make traffic predictions with awareness of TPM categories.

        This method enhances the standard prediction by:
        1. Determining the current TPM category
        2. Checking if the current time period is typical for this category
        3. Adjusting predictions based on category-time period match

        Parameters:
        -----------
        current_tpm : float
            Current traffic per minute
        hour : int
            Hour of day (0-23)
        day_of_week : int
            Day of week (0=Monday, 6=Sunday)
        response_time : float
            Current response time in milliseconds
        previous_tpm_values : list, optional
            List of previous TPM values (up to 5 values)
        push_notification_active : int
            Whether push notification is currently active (0 or 1)
        minutes_since_push : int
            Minutes since last push notification
        **kwargs : dict
            Additional features that may be available

        Returns:
        --------
        dict
            Dictionary with predictions and category information
        """
        # First, get standard predictions
        standard_predictions = self.predict(
            current_tpm, hour, day_of_week, response_time,
            previous_tpm_values, push_notification_active,
            minutes_since_push, **kwargs
        )

        # Load and classify data if not done already
        if not hasattr(self, 'category_time_analysis'):
            df = self.load_data()
            df_with_categories, self.tpm_thresholds, self.category_stats = self.classify_tpm_values(df)
            self.category_time_analysis = self.analyze_tpm_categories_by_time(df_with_categories)

        # Determine current TPM category
        if current_tpm <= self.tpm_thresholds.get('low_to_medium', 0):
            current_category = 'low'
        elif current_tpm <= self.tpm_thresholds.get('medium_to_high', float('inf')):
            current_category = 'medium'
        else:
            current_category = 'high'

        # Check if current time period is typical for this category
        peak_times = self.category_time_analysis['peak_times'].get(current_category, {})
        peak_hours = peak_times.get('peak_hours', {})
        peak_days = peak_times.get('peak_days', {})

        is_peak_hour = str(hour) in peak_hours
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        is_peak_day = day_names[day_of_week] in peak_days

        # Adjust predictions based on category-time period match
        adjusted_predictions = {}
        for target, prediction in standard_predictions.items():
            adjustment_factor = 1.0

            # If current TPM category doesn't match typical time period, adjust prediction
            if current_category == 'high' and not (is_peak_hour or is_peak_day):
                # High TPM in non-peak time: likely to decrease
                adjustment_factor = 0.9
            elif current_category == 'low' and (is_peak_hour and is_peak_day):
                # Low TPM in peak time: likely to increase
                adjustment_factor = 1.15
            elif current_category == 'medium':
                # Medium TPM: smaller adjustments
                if is_peak_hour and is_peak_day:
                    adjustment_factor = 1.05
                elif not (is_peak_hour or is_peak_day):
                    adjustment_factor = 0.95

            adjusted_predictions[target] = prediction * adjustment_factor

        # Prepare result with both predictions and category information
        result = {
            'standard_predictions': standard_predictions,
            'category_aware_predictions': adjusted_predictions,
            'tpm_category': current_category,
            'is_peak_hour': is_peak_hour,
            'is_peak_day': is_peak_day,
            'adjustment_applied': any(standard_predictions[k] != adjusted_predictions[k] for k in standard_predictions)
        }

        return result

    def evaluate_category_based_prediction(self, test_data=None):
        """
        Evaluate if category-based prediction improves accuracy.

        Parameters:
        -----------
        test_data : pandas.DataFrame, optional
            Test data to evaluate on. If None, uses 20% of loaded data.

        Returns:
        --------
        dict
            Dictionary with evaluation metrics for standard and category-aware predictions
        """
        print("Evaluating category-based prediction improvement...")

        if test_data is None:
            # Load data and split for evaluation
            df = self.load_data()
            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            # Train models if not already trained
            if not self.models:
                X, y_dict, _ = self.preprocess_data(train_data)
                self.train_models(X, y_dict)

        # Classify data and analyze time periods
        df_with_categories, self.tpm_thresholds, self.category_stats = self.classify_tpm_values(df)
        self.category_time_analysis = self.analyze_tpm_categories_by_time(df_with_categories)

        # Get required column names
        tpm_col = self.feature_mapping.get('tpm', 'tpm')
        hour_col = self.feature_mapping.get('hour', 'hour')
        day_col = self.feature_mapping.get('day_of_week', 'day_of_week')
        response_time_col = self.feature_mapping.get('response_time', 'response_time')

        # Prepare for evaluation
        standard_errors = {target: [] for target in self.target_columns}
        category_errors = {target: [] for target in self.target_columns}

        # Evaluate on test data
        for _, row in test_data.iterrows():
            # Get actual values
            actual_values = {target: row[target] if target in row else row[tpm_col] for target in self.target_columns}

            # Get previous TPM values if available
            previous_tpm_values = []
            for i in range(1, 6):
                lag_col = f'tpm_lag_{i}'
                if lag_col in row:
                    previous_tpm_values.append(row[lag_col])

            # Make standard predictions
            standard_pred = self.predict(
                current_tpm=row[tpm_col],
                hour=row[hour_col],
                day_of_week=row[day_col],
                response_time=row.get(response_time_col, 100),
                previous_tpm_values=previous_tpm_values,
                push_notification_active=row.get('push_notification_active', 0),
                minutes_since_push=row.get('minutes_since_push', 999999)
            )

            # Make category-aware predictions
            category_pred_result = self.predict_with_category_awareness(
                current_tpm=row[tpm_col],
                hour=row[hour_col],
                day_of_week=row[day_col],
                response_time=row.get(response_time_col, 100),
                previous_tpm_values=previous_tpm_values,
                push_notification_active=row.get('push_notification_active', 0),
                minutes_since_push=row.get('minutes_since_push', 999999)
            )
            category_pred = category_pred_result['category_aware_predictions']

            # Calculate errors
            for target in self.target_columns:
                if target in actual_values:
                    standard_errors[target].append(abs(standard_pred[target] - actual_values[target]))
                    category_errors[target].append(abs(category_pred[target] - actual_values[target]))

        # Calculate metrics
        evaluation = {
            'standard_mae': {target: np.mean(errors) for target, errors in standard_errors.items()},
            'category_mae': {target: np.mean(errors) for target, errors in category_errors.items()},
            'improvement': {}
        }

        # Calculate improvement percentage
        for target in self.target_columns:
            std_mae = evaluation['standard_mae'][target]
            cat_mae = evaluation['category_mae'][target]
            improvement = ((std_mae - cat_mae) / std_mae) * 100 if std_mae > 0 else 0
            evaluation['improvement'][target] = improvement

        # Print summary
        print("\n✅ Evaluation completed")
        print("\nMean Absolute Error (MAE) comparison:")
        for target in self.target_columns:
            std_mae = evaluation['standard_mae'][target]
            cat_mae = evaluation['category_mae'][target]
            improvement = evaluation['improvement'][target]

            print(f"  {target}:")
            print(f"    - Standard prediction: {std_mae:.2f}")
            print(f"    - Category-aware prediction: {cat_mae:.2f}")

            if improvement > 0:
                print(f"    - Improvement: {improvement:.2f}% ✓")
            else:
                print(f"    - Change: {improvement:.2f}% ✗")

        return evaluation

    def visualize_predictions(self):
        """
        Create enhanced visualizations of model predictions and feature importance.
        """
        if not self.models:
            print("No models available for visualization.")
            return

        # Create subplots for feature importance
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Traffic Prediction Model - Feature Importance Analysis', fontsize=16)

        # Feature importance for each model
        importance_data = self.get_feature_importance()

        for i, (target_name, importance) in enumerate(importance_data.items()):
            row = i // 2
            col = i % 2

            # Sort features by importance and take top 10
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, values = zip(*sorted_features)

            bars = axes[row, col].barh(features, values)
            axes[row, col].set_title(f'Top Features - {target_name}')
            axes[row, col].set_xlabel('Importance')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col].text(value + 0.001, bar.get_y() + bar.get_height() / 2,
                                    f'{value:.3f}', ha='left', va='center', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Example prediction visualization over 24 hours
        print("\nGenerating example predictions over time...")

        hours = range(24)
        prediction_data = {target: [] for target in self.target_columns}

        for hour in hours:
            pred = self.predict(
                current_tpm=400,
                hour=hour,
                day_of_week=2,  # Wednesday
                response_time=100,
                previous_tpm_values=[390, 380, 370, 360, 350]
            )
            for target in self.target_columns:
                prediction_data[target].append(pred[target])

        # Plot predictions over 24 hours
        plt.figure(figsize=(14, 8))
        colors = ['blue', 'green', 'red', 'purple']

        for i, (target, predictions) in enumerate(prediction_data.items()):
            plt.plot(hours, predictions, label=f'{target.replace("target_", "").replace("min", " min")}',
                     marker='o', color=colors[i % len(colors)], linewidth=2, markersize=4)

        plt.xlabel('Hour of Day')
        plt.ylabel('Predicted TPM')
        plt.title('Traffic Predictions Over 24 Hours\n(Example: Wednesday, Current TPM=400)')
        plt.legend(title='Prediction Horizon')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))

        # Add business hours shading
        plt.axvspan(9, 17, alpha=0.2, color='yellow', label='Business Hours')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def train_and_save(self):
        """
        Convenience method to load data, train models, and save them.

        Returns:
        --------
        dict
            Dictionary of trained models and evaluation metrics
        """
        # Load and preprocess data
        df = self.load_data()
        X, y_dict, _ = self.preprocess_data(df)

        # Train models
        results = self.train_models(X, y_dict)

        # Save models
        self.save_models()

        return results


def example_usage():
    """
    Example usage of the TrafficPredictionModel class.
    """
    # Create model
    model = TrafficPredictionModel()

    # Option 1: Train new models
    print("\n=== Training new models ===")
    results = model.train_and_save()

    # Display training results
    print("\n=== Training Results Summary ===")
    for target_name, metrics in results.items():
        print(f"\n{target_name.upper()}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

    # Option 2: Load pre-trained models
    # print("\n=== Loading pre-trained models ===")
    # model.load_models()

    # Make predictions
    print("\n=== Making predictions ===")

    # Example 1: Current traffic with no push notification
    prediction1 = model.predict(
        current_tpm=500,
        hour=14,
        day_of_week=4,  # Friday
        response_time=100,
        previous_tpm_values=[480, 450, 420, 400, 380],
        push_notification_active=0
    )

    print("\nPrediction for current traffic (no push notification):")
    for target, value in prediction1.items():
        print(f"  {target}: {value:.2f} TPM")

    # Example 2: With active push notification
    prediction2 = model.predict(
        current_tpm=500,
        hour=14,
        day_of_week=4,  # Friday
        response_time=100,
        previous_tpm_values=[480, 450, 420, 400, 380],
        push_notification_active=1,
        minutes_since_push=0
    )

    print("\nPrediction for current traffic (with active push notification):")
    for target, value in prediction2.items():
        print(f"  {target}: {value:.2f} TPM")

    # Example 3: 5 minutes after push notification
    prediction3 = model.predict(
        current_tpm=550,  # Increased due to push notification
        hour=14,
        day_of_week=4,  # Friday
        response_time=110,
        previous_tpm_values=[500, 480, 450, 420, 400],
        push_notification_active=0,
        minutes_since_push=5
    )

    print("\nPrediction for traffic 5 minutes after push notification:")
    for target, value in prediction3.items():
        print(f"  {target}: {value:.2f} TPM")

    # Demonstrate TPM classification features
    print("\n=== TPM Classification Analysis ===")

    # Classify TPM values
    df_with_categories, thresholds, category_stats = model.classify_tpm_values()

    # Print threshold values
    print("\nTPM Category Thresholds:")
    for threshold_name, value in thresholds.items():
        if threshold_name not in ['min', 'max']:
            print(f"  {threshold_name}: {value:.2f}")

    # Analyze time periods for each category
    print("\n=== Time Period Analysis for TPM Categories ===")
    time_analysis = model.analyze_tpm_categories_by_time(df_with_categories)

    # Make category-aware predictions
    print("\n=== Category-Aware Predictions ===")

    # Example: Medium TPM during peak hour
    category_prediction1 = model.predict_with_category_awareness(
        current_tpm=thresholds.get('low_to_medium', 300) + 50,  # Medium TPM
        hour=11,  # Assuming this is a peak hour based on analysis
        day_of_week=5,  # Saturday
        response_time=100,
        previous_tpm_values=[480, 450, 420, 400, 380]
    )

    print("\nCategory-aware prediction for medium TPM during likely peak time:")
    print(f"TPM Category: {category_prediction1['tpm_category']}")
    print(f"Is peak hour: {category_prediction1['is_peak_hour']}")
    print(f"Is peak day: {category_prediction1['is_peak_day']}")
    print(f"Adjustment applied: {category_prediction1['adjustment_applied']}")

    print("\nStandard vs. Category-Aware Predictions:")
    for target in model.target_columns:
        std_pred = category_prediction1['standard_predictions'][target]
        cat_pred = category_prediction1['category_aware_predictions'][target]
        diff_pct = ((cat_pred - std_pred) / std_pred) * 100 if std_pred > 0 else 0

        print(f"  {target}:")
        print(f"    - Standard: {std_pred:.2f} TPM")
        print(f"    - Category-aware: {cat_pred:.2f} TPM")
        print(f"    - Difference: {diff_pct:.2f}%")

    # Evaluate if category-based prediction improves accuracy
    print("\n=== Evaluating Category-Based Prediction Improvement ===")
    evaluation = model.evaluate_category_based_prediction()

    # Visualize predictions
    print("\n=== Visualizing predictions ===")
    model.visualize_predictions()


if __name__ == "__main__":
    example_usage()
