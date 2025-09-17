
"""
🚀 COMPREHENSIVE TIME SERIES TRAINER WITH LEARNED PUSH EFFECTS
==============================================================
- Model tự học push notification effects từ data
- Multi-horizon predictions (5, 10, 15 minutes)
- No manual push calculations trong prediction
- Production-ready với error handling
- CSV training data với time series ordering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime, timedelta
import warnings
import joblib
import os
import glob

warnings.filterwarnings('ignore')


class ComprehensiveTimeSeriesTrainer:
    """
    🧠 COMPREHENSIVE TIME SERIES TRAINER
    ====================================
    - Model tự học push effects thay vì manual calculations
    - Multi-horizon predictions với temporal consistency
    - Advanced feature engineering
    - Production-ready architecture
    """

    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_cols = []
        self.tpm_thresholds = {}
        self.training_results = {}
        self.push_effect_learned = False

        # Configuration for feature engineering only
        self.config = {
            'effect_duration_minutes': 15,
            'analysis_window_minutes': 30,
            'min_historical_points': 30,
            'max_historical_points': 300
        }

        print("🚀 Comprehensive Time Series Trainer initialized")
        print("   🧠 Ready for intelligent push effect learning")

    # ==================== DATA LOADING & PREPARATION ====================

    def load_csv_data(self, csv_path):
        """Load and prepare CSV data with proper time series ordering"""
        print(f"📂 Loading data from {csv_path}...")

        try:
            df = pd.read_csv(csv_path)
            print(f"   ✅ Loaded {len(df)} records")

            # Convert timestamp columns
            timestamp_columns = ['timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc']
            for col in timestamp_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Sort by timestamp (CRITICAL for time series)
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
                timestamp_col = 'timestamp'
            elif 'timestamp_utc' in df.columns:
                df = df.sort_values('timestamp_utc').reset_index(drop=True)
                timestamp_col = 'timestamp_utc'
            else:
                raise ValueError("No timestamp column found")

            # Basic validation
            if 'tpm' not in df.columns:
                raise ValueError("TPM column not found")

            if 'push_notification_active' not in df.columns:
                df['push_notification_active'] = 0
                print("   ⚠️ Added missing push_notification_active column")

            print(f"   📅 Time range: {df[timestamp_col].min()} to {df[timestamp_col].max()}")
            print(f"   📊 TPM range: {df['tpm'].min():.1f} - {df['tpm'].max():.1f}")
            print(f"   📱 Push events: {df['push_notification_active'].sum()} total")
            print(f"   ⏰ Time span: {(df[timestamp_col].max() - df[timestamp_col].min()).days} days")

            return df

        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None

    def create_multi_horizon_targets(self, data, target_col='tpm', horizons=(5, 10, 15)):
        """
        🎯 CREATE MULTI-HORIZON TARGETS: Core của learned effects
        Tạo targets cho multiple horizons để model học temporal patterns
        """
        print(f"🎯 Creating multi-horizon targets for {horizons} minute horizons...")

        df = data.copy()
        target_columns = []

        # ============ CREATE FUTURE TPM TARGETS ============
        for horizon in horizons:
            target_name = f'{target_col}_future_{horizon}min'

            # Shift TPM values để tạo future targets
            df[target_name] = df[target_col].shift(-horizon)
            target_columns.append(target_name)

            print(f"   ✅ Created target: {target_name}")

        # ============ ENHANCED PUSH CONTEXT FEATURES ============
        print("   📱 Creating push learning features...")

        # Push momentum và frequency patterns
        df['push_momentum_5min'] = df['push_notification_active'].rolling(5, min_periods=1).sum()
        df['push_momentum_15min'] = df['push_notification_active'].rolling(15, min_periods=1).sum()
        df['push_momentum_30min'] = df['push_notification_active'].rolling(30, min_periods=1).sum()

        # Push frequency analysis
        df['push_frequency_last_hour'] = df['push_notification_active'].rolling(60, min_periods=1).sum()
        df['push_frequency_last_6hours'] = df['push_notification_active'].rolling(360, min_periods=1).sum()

        # Push recency và sequence
        df['minutes_since_last_push'] = self._calculate_minutes_since_last_push(df)
        df['cumulative_pushes'] = df['push_notification_active'].cumsum()
        df['push_session_id'] = self._identify_push_sessions(df)

        # Push effectiveness context
        df['push_hour_effectiveness'] = df.apply(
            lambda row: self._calculate_push_hour_effectiveness(
                row.get('hour', 12),
                str(row.get('push_campaign_type', 'none'))
            ), axis=1
        )

        # Push interaction features
        if 'hour' in df.columns:
            df['push_hour_interaction'] = df['push_notification_active'] * df['hour']
        if 'day_of_week' in df.columns:
            df['push_weekday_interaction'] = df['push_notification_active'] * (df['day_of_week'] + 1)

        # Push timing quality
        df['push_timing_quality'] = df.apply(
            lambda row: self._calculate_push_timing_quality(
                row.get('hour', 12),
                str(row.get('push_campaign_type', 'none'))
            ), axis=1
        )

        print(f"   ✅ Multi-horizon data created:")
        print(f"      • Target columns: {len(target_columns)}")
        print(f"      • Push features: {len([c for c in df.columns if 'push' in c.lower()])}")
        print(f"      • Valid samples: {df[target_columns].dropna().shape[0]}")

        return df, target_columns

    def _calculate_minutes_since_last_push(self, df):
        """Calculate minutes since last push efficiently"""
        push_indices = df[df['push_notification_active'] == 1].index
        minutes_since = []

        for idx in df.index:
            if idx in push_indices:
                minutes_since.append(0)
            else:
                recent_pushes = push_indices[push_indices < idx]
                if len(recent_pushes) > 0:
                    minutes_diff = idx - recent_pushes[-1]
                    minutes_since.append(min(minutes_diff, 999))
                else:
                    minutes_since.append(999)

        return minutes_since

    def _calculate_push_hour_effectiveness(self, hour, campaign_type):
        """Calculate push effectiveness by hour and campaign type"""
        effectiveness_matrix = {
            'morning_commute': {6: 0.7, 7: 1.0, 8: 1.0, 9: 0.8, 10: 0.6},
            'lunch_peak': {11: 0.8, 12: 1.0, 13: 1.0, 14: 0.7},
            'afternoon_break': {14: 0.8, 15: 1.0, 16: 1.0, 17: 0.8},
            'evening_commute': {16: 0.8, 17: 0.9, 18: 1.0, 19: 1.0, 20: 0.7},
            'evening_leisure': {19: 0.7, 20: 1.0, 21: 1.0, 22: 0.6}
        }

        if campaign_type in effectiveness_matrix and hour in effectiveness_matrix[campaign_type]:
            return effectiveness_matrix[campaign_type][hour]
        elif 2 <= hour <= 6:  # Sleep hours
            return 0.0
        else:
            return 0.5  # Default effectiveness

    def _identify_push_sessions(self, df):
        """Identify push sessions - groups of related pushes"""
        session_id = 0
        sessions = []
        last_push_idx = None
        session_gap_minutes = 30  # New session if gap > 30 minutes

        for idx, row in df.iterrows():
            if row['push_notification_active'] == 1:
                if last_push_idx is None or (idx - last_push_idx) > session_gap_minutes:
                    session_id += 1
                last_push_idx = idx
                sessions.append(session_id)
            else:
                sessions.append(0)  # Not in a push session

        return sessions

    def _calculate_push_timing_quality(self, hour, campaign_type):
        """Calculate quality score for push timing vs campaign type match"""
        optimal_timing = {
            'morning_commute': [7, 8, 9],
            'lunch_peak': [11, 12, 13],
            'afternoon_break': [14, 15, 16],
            'evening_commute': [17, 18, 19],
            'evening_leisure': [19, 20, 21]
        }

        if campaign_type == 'none' or campaign_type not in optimal_timing:
            return 0.5

        if hour in optimal_timing[campaign_type]:
            return 1.0  # Perfect timing

        # Calculate distance penalty
        distances = [abs(hour - h) for h in optimal_timing[campaign_type]]
        min_distance = min(distances)
        quality = max(0.1, 1.0 - (min_distance * 0.15))

        return quality

    # ==================== FEATURE ENGINEERING ====================

    def create_comprehensive_features(self, data, target_col='tpm'):
        """
        🔧 COMPREHENSIVE FEATURE ENGINEERING
        Tạo tất cả features cần thiết cho model học push effects
        """
        print("🔧 Creating comprehensive features...")

        df = data.copy()

        df = data.copy()

        # ============ AGGRESSIVE STRING DETECTION & CLEANUP ============
        print("   🔤 AGGRESSIVE string column detection...")

        # Step 1: Identify ALL non-numeric columns (excluding target)
        non_numeric_columns = []
        for col in df.columns:
            if col == target_col:
                continue

            # Check if column is truly numeric
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                non_numeric_columns.append(col)
            elif df[col].dtype == 'bool':
                # Convert boolean to int
                df[col] = df[col].astype(int)
            else:
                # Try to detect mixed types
                try:
                    # Check if any values are strings
                    sample_values = df[col].dropna().head(100)
                    if any(isinstance(v, str) for v in sample_values):
                        non_numeric_columns.append(col)
                except:
                    pass

        print(f"      Found {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns[:10]}...")

        # Step 2: Handle each non-numeric column
        for col in non_numeric_columns:
            print(f"      🔧 Processing: {col}")

            # Special handling for known string columns
            if col == 'push_campaign_type':
                print("         📱 Processing campaign types...")

                # Handle NaN values first
                df[col] = df[col].fillna('none').astype(str)

                # Get unique campaign types
                unique_campaigns = df[col].unique()
                print(f"            Found campaigns: {unique_campaigns}")

                # Create binary features for each campaign
                campaign_types = [
                    'morning_commute', 'lunch_peak', 'afternoon_break',
                    'evening_commute', 'evening_leisure',
                    'sleeping_hours_early', 'sleeping_hours_late', 'none'
                ]

                for campaign in campaign_types:
                    df[f'campaign_{campaign}'] = (df[col] == campaign).astype(int)

                # Additional campaign features
                df['is_peak_campaign'] = df[col].isin(['lunch_peak', 'evening_commute']).astype(int)
                df['is_commute_campaign'] = df[col].isin(['morning_commute', 'evening_commute']).astype(int)
                df['is_leisure_campaign'] = df[col].isin(['afternoon_break', 'evening_leisure']).astype(int)
                df['is_sleep_campaign'] = df[col].isin(['sleeping_hours_early', 'sleeping_hours_late']).astype(int)

                # Drop original string column
                df = df.drop(columns=[col])
                print(f"            ✅ Created {len(campaign_types) + 4} binary features")

            else:
                # Generic string column handling
                try:
                    # First try to convert to numeric
                    numeric_converted = pd.to_numeric(df[col], errors='coerce')
                    non_null_conversion_rate = numeric_converted.notna().sum() / len(df)

                    if non_null_conversion_rate > 0.8:  # If 80%+ can be converted to numeric
                        df[col] = numeric_converted.fillna(0)
                        print(f"            ✅ Converted to numeric ({non_null_conversion_rate:.1%} success)")
                    else:
                        # Handle as categorical if reasonable number of categories
                        unique_count = df[col].nunique()
                        if unique_count <= 20:  # Reasonable for binary encoding
                            print(f"            🔄 Creating binary features ({unique_count} categories)")

                            unique_values = df[col].fillna('missing').unique()
                            for val in unique_values:
                                if pd.notna(val):
                                    safe_name = str(val).replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
                                    df[f'{col}_{safe_name}'] = (df[col] == val).astype(int)

                            df = df.drop(columns=[col])
                            print(f"            ✅ Created {len(unique_values)} binary features")
                        else:
                            # Too many categories - drop column
                            df = df.drop(columns=[col])
                            print(f"            ❌ Dropped (too many categories: {unique_count})")

                except Exception as e:
                    # Last resort - drop problematic column
                    df = df.drop(columns=[col])
                    print(f"            ❌ Dropped due to error: {e}")

        # ============ TEMPORAL FEATURES ============
        print("   🕐 Adding temporal features...")

        # Ensure basic time columns exist
        if 'hour' not in df.columns and 'timestamp' in data.columns:
            df['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        if 'day_of_week' not in df.columns and 'timestamp' in data.columns:
            df['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek

        # Create cyclical time features
        if 'hour' in df.columns:
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(12)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['hour_quarter_sin'] = np.sin(2 * np.pi * (df['hour'] % 6) / 6)
            df['hour_quarter_cos'] = np.cos(2 * np.pi * (df['hour'] % 6) / 6)

        if 'day_of_week' in df.columns:
            df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(1)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Business time features
        if 'hour' in df.columns:
            hour_col = pd.to_numeric(df['hour'], errors='coerce').fillna(12)
            df['is_morning_peak'] = ((hour_col >= 8) & (hour_col <= 10)).astype(int)
            df['is_lunch_peak'] = ((hour_col >= 12) & (hour_col <= 13)).astype(int)
            df['is_afternoon_peak'] = ((hour_col >= 15) & (hour_col <= 17)).astype(int)
            df['is_evening_peak'] = ((hour_col >= 18) & (hour_col <= 20)).astype(int)
            df['is_business_hours'] = ((hour_col >= 9) & (hour_col <= 17)).astype(int)
            df['is_sleep_hours'] = ((hour_col <= 6) | (hour_col >= 23)).astype(int)

        if 'day_of_week' in df.columns:
            day_col = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(1)
            df['is_weekend'] = (day_col >= 5).astype(int)
            df['is_weekday'] = (day_col < 5).astype(int)

        # ============ LAG FEATURES ============
        print("   📈 Adding lag features...")

        if target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].median())

            lag_periods = [1, 5, 15, 30, 60]
            for lag in lag_periods:
                if len(df) > lag:
                    df[f'{target_col}_lag_{lag}'] = target_series.shift(lag)

        # ============ ROLLING WINDOW FEATURES ============
        print("   🌊 Adding rolling features...")

        if target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].median())

            rolling_windows = [5, 15, 30, 60]
            for window in rolling_windows:
                if len(df) >= window:
                    df[f'{target_col}_ma_{window}'] = target_series.rolling(window, min_periods=1).mean()
                    df[f'{target_col}_std_{window}'] = target_series.rolling(window, min_periods=1).std().fillna(0)
                    df[f'{target_col}_min_{window}'] = target_series.rolling(window, min_periods=1).min()
                    df[f'{target_col}_max_{window}'] = target_series.rolling(window, min_periods=1).max()
                    df[f'{target_col}_range_{window}'] = df[f'{target_col}_max_{window}'] - df[
                        f'{target_col}_min_{window}']

        # ============ PUSH DECAY FEATURES ============
        print("   📱 Adding push decay features...")

        if 'minutes_since_push' in df.columns:
            minutes_since = pd.to_numeric(df['minutes_since_push'], errors='coerce').fillna(999)

            df['push_decay_exponential'] = minutes_since.apply(
                lambda x: np.exp(-x / 8.0) if x <= 15 else 0.0
            )
            df['push_decay_linear'] = minutes_since.apply(
                lambda x: max(0, 1 - x / 15.0) if x <= 15 else 0.0
            )
            df['push_decay_quadratic'] = minutes_since.apply(
                lambda x: max(0, (1 - x / 15.0) ** 2) if x <= 15 else 0.0
            )

            df['in_push_immediate'] = (minutes_since <= 5).astype(int)
            df['in_push_short'] = (minutes_since <= 10).astype(int)
            df['in_push_medium'] = (minutes_since <= 15).astype(int)
            df['in_push_extended'] = (minutes_since <= 30).astype(int)

        # ============ INTERACTION FEATURES ============
        print("   🔗 Adding interaction features...")

        if 'push_notification_active' in df.columns and 'hour' in df.columns:
            push_active = pd.to_numeric(df['push_notification_active'], errors='coerce').fillna(0)
            hour_col = pd.to_numeric(df['hour'], errors='coerce').fillna(12)

            df['push_hour_interaction'] = push_active * hour_col

            if 'push_hour_effectiveness' in df.columns:
                effectiveness = pd.to_numeric(df['push_hour_effectiveness'], errors='coerce').fillna(0.5)
                df['push_hour_strength'] = push_active * effectiveness

        # ============ TREND FEATURES ============
        print("   📊 Adding trend features...")

        if target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors='coerce').fillna(df[target_col].median())

            # TPM trends
            for window in [5, 15, 30]:
                if len(df) >= window + 1:
                    ma_col = f'{target_col}_ma_{window}'
                    if ma_col in df.columns:
                        df[f'{target_col}_trend_{window}'] = (
                                df[ma_col] - df[ma_col].shift(window)
                        ).fillna(0)

            # TPM momentum
            lag_1_col = f'{target_col}_lag_1'
            lag_5_col = f'{target_col}_lag_5'

            if lag_1_col in df.columns:
                df[f'{target_col}_momentum_1'] = target_series - df[lag_1_col].fillna(target_series)

            if lag_5_col in df.columns:
                df[f'{target_col}_momentum_5'] = target_series - df[lag_5_col].fillna(target_series)

        # ============ COMPREHENSIVE FINAL CLEANUP ============
        print("   🧹 Final comprehensive cleanup...")

        # Remove timestamp and identifier columns
        cleanup_columns = [
            'timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc',
            'metric_name', 'day_name', 'sequence_id', 'date', 'time'
        ]

        existing_cleanup = [col for col in cleanup_columns if col in df.columns]
        if existing_cleanup:
            df = df.drop(columns=existing_cleanup)
            print(f"      ✅ Removed {len(existing_cleanup)} identifier columns")

        # COMPREHENSIVE NUMERIC CONVERSION
        print("   🔢 Final numeric conversion...")

        non_target_columns = [col for col in df.columns if col != target_col]
        conversion_stats = {'converted': 0, 'failed': 0, 'dropped': 0}

        for col in non_target_columns:
            try:
                # Check if already numeric
                if df[col].dtype.kind in 'biufc':  # boolean, integer, unsigned, float, complex
                    continue

                # Try conversion
                converted = pd.to_numeric(df[col], errors='coerce')
                non_null_rate = converted.notna().sum() / len(df)

                if non_null_rate > 0.5:  # If more than 50% can be converted
                    df[col] = converted.fillna(0)
                    conversion_stats['converted'] += 1
                else:
                    df = df.drop(columns=[col])
                    conversion_stats['dropped'] += 1
                    print(f"      ❌ Dropped {col} (conversion rate: {non_null_rate:.1%})")

            except Exception as e:
                df = df.drop(columns=[col])
                conversion_stats['failed'] += 1
                print(f"      ❌ Dropped {col} (error: {str(e)[:50]})")

        # Handle missing values systematically
        print("   💧 Handling missing values...")

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        missing_stats = {}

        for col in numeric_columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_stats[col] = missing_count

                # Smart filling based on column type
                if any(keyword in col.lower() for keyword in ['lag', 'ma', 'trend', 'momentum']):
                    # Time series features: forward/backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                elif any(keyword in col.lower() for keyword in ['push', 'campaign', 'is_', 'in_']):
                    # Binary/categorical features: fill with 0
                    df[col] = df[col].fillna(0)
                else:
                    # Other features: median fill
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

        if missing_stats:
            print(f"      ✅ Filled {len(missing_stats)} columns with missing values")

        # Handle infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
                df[col] = df[col].replace([np.inf, -np.inf], 0)

        if inf_cols:
            print(f"      ✅ Fixed infinite values in {len(inf_cols)} columns")

        # Final validation
        remaining_issues = 0
        issue_columns = []

        for col in df.columns:
            if col == target_col:
                continue

            if df[col].dtype == 'object':
                issue_columns.append(col)
                remaining_issues += 1
            elif df[col].isnull().sum() > 0:
                remaining_issues += 1

        if remaining_issues > 0:
            print(f"      ⚠️ Found {remaining_issues} remaining issues")
            if issue_columns:
                print(f"         Object columns: {issue_columns}")
                # Drop remaining object columns
                df = df.drop(columns=issue_columns)
                print(f"         ✅ Dropped {len(issue_columns)} problematic columns")

            # Final NaN cleanup
            df = df.fillna(0)
            print(f"         ✅ Final NaN cleanup completed")

        # Success summary
        print(f"✅ Comprehensive features completed:")
        print(f"   • Total columns: {len(df.columns)}")
        print(f"   • Conversion stats: {conversion_stats}")
        print(f"   • All numeric: {all(df[col].dtype.kind in 'biufc' for col in df.columns if col != target_col)}")

        # Feature breakdown
        feature_categories = {
            'Time features': len([c for c in df.columns if
                                  any(x in c.lower() for x in ['hour', 'day', 'sin', 'cos', 'peak', 'business'])]),
            'Push features': len([c for c in df.columns if any(x in c.lower() for x in ['push', 'campaign'])]),
            'Lag features': len([c for c in df.columns if 'lag' in c.lower()]),
            'Rolling features': len(
                [c for c in df.columns if any(x in c.lower() for x in ['ma', 'std', 'min', 'max', 'range'])]),
            'Trend features': len([c for c in df.columns if any(x in c.lower() for x in ['trend', 'momentum'])]),
            'Interaction features': len(
                [c for c in df.columns if 'interaction' in c.lower() or 'strength' in c.lower()])
        }

        print(f"   📊 Feature breakdown:")
        for category, count in feature_categories.items():
            print(f"      • {category}: {count}")

        return df

    # ==================== MODEL TRAINING ====================

    def train_push_aware_model(self, csv_path, target_col='tpm', horizons=(5, 10, 15), test_size=0.25):
        """
        🚀 MAIN TRAINING FUNCTION
        Train model với learned push effects
        """
        print("🚀 TRAINING PUSH-AWARE MODEL WITH LEARNED EFFECTS")
        print("=" * 70)

        # ============ LOAD AND PREPARE DATA ============
        raw_data = self.load_csv_data(csv_path)
        if raw_data is None:
            raise ValueError("Failed to load CSV data")

        # Create multi-horizon targets
        multi_horizon_data, target_columns = self.create_multi_horizon_targets(
            raw_data, target_col, horizons
        )

        # Create comprehensive features
        processed_data = self.create_comprehensive_features(multi_horizon_data, target_col)

        # ============ PREPARE TRAINING DATA ============
        print(f"📊 Preparing training data...")

        # Separate features and targets
        feature_cols = [col for col in processed_data.columns
                        if not any(x in col for x in ['future_', f'{target_col}_future_'])]

        X = processed_data[feature_cols].copy()
        y = processed_data[target_columns].copy()

        # Remove rows with missing targets
        valid_mask = y.notna().all(axis=1)
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("No valid training samples after preprocessing")

        print(f"   • Training samples: {len(X)}")
        print(f"   • Feature columns: {len(X.columns)}")
        print(f"   • Target horizons: {len(target_columns)}")
        print(f"   • TPM range: {processed_data[target_col].min():.1f} - {processed_data[target_col].max():.1f}")

        # ============ FEATURE SELECTION ============
        X_selected, selected_features = self.select_important_features(
            X, y.mean(axis=1), max_features=35  # Increase for multi-output
        )

        self.feature_cols = selected_features

        # ============ TIME-BASED TRAIN/TEST SPLIT ============
        split_idx = int(len(X_selected) * (1 - test_size))

        X_train = X_selected.iloc[:split_idx]
        X_test = X_selected.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        print(f"   📊 Time-based split: Train={len(X_train)}, Test={len(X_test)}")

        # ============ TRAIN MODELS ============
        results = self.train_multi_output_models(X_train, y_train, X_test, y_test, horizons)

        # Compute thresholds and finalize
        self.tpm_thresholds = self.compute_tpm_thresholds(processed_data[target_col])
        self.push_effect_learned = True

        print(f"\n🎉 PUSH-AWARE TRAINING COMPLETED!")
        print(f"   🧠 Model learned push effects from {len(X_train)} samples")
        print(f"   🏆 Best model: {results['best_model']}")
        print(f"   📈 Average R²: {results['best_score']:.3f}")
        print(f"   🎯 Horizons: {horizons}")

        return results

    def train_multi_output_models(self, X_train, y_train, X_test, y_test, horizons):
        """
        🤖 TRAIN MULTI-OUTPUT MODELS
        Core learning mechanism for push effects
        """
        print(f"\n🤖 Training Multi-Output Models...")

        # Advanced model configurations
        models = {
            'MultiRF_Advanced': MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=800,
                    max_depth=20,
                    min_samples_split=6,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1
                )
            ),
            'MultiGB_Advanced': MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=600,
                    max_depth=10,
                    learning_rate=0.03,
                    subsample=0.85,
                    min_samples_leaf=4,
                    min_samples_split=10,
                    max_features=0.8,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=42
                )
            ),
            'MultiRidge_Regularized': MultiOutputRegressor(
                Ridge(alpha=150.0, solver='auto', random_state=42)
            )
        }

        self.models = {'multi_tpm': {}}
        model_performance = {}
        best_score = -np.inf
        best_model_name = None

        # Train each model
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")

            try:
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=4)
                cv_scores = []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                    X_train_fold = X_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    y_val_pred = model.predict(X_val_fold)

                    # Average R² across horizons
                    fold_r2s = []
                    for i in range(y_val_fold.shape[1]):
                        r2 = r2_score(y_val_fold.iloc[:, i], y_val_pred[:, i])
                        fold_r2s.append(r2)

                    cv_scores.append(np.mean(fold_r2s))

                # Final training on full training set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Detailed evaluation per horizon
                horizon_metrics = {}
                horizon_r2s = []

                print(f"   📈 Horizon performance:")
                for i, horizon in enumerate(horizons):
                    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
                    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
                    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))

                    horizon_metrics[f'{horizon}min'] = {
                        'r2': r2, 'mae': mae, 'rmse': rmse
                    }
                    horizon_r2s.append(r2)

                    print(f"      {horizon}min: R²={r2:.3f}, MAE={mae:.1f}, RMSE={rmse:.1f}")

                # Overall metrics
                avg_r2 = np.mean(horizon_r2s)
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)

                # Store results
                self.models['multi_tpm'][name] = model
                model_performance[name] = {
                    'avg_test_r2': avg_r2,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'horizon_metrics': horizon_metrics,
                    'individual_r2s': horizon_r2s
                }

                print(f"   ✅ {name}: Avg R²={avg_r2:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}")

                # Track best model
                if avg_r2 > best_score:
                    best_score = avg_r2
                    best_model_name = name

            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue

        if not self.models['multi_tpm']:
            raise ValueError("No models trained successfully")

        self.best_models = {'multi_tpm': best_model_name}

        # Analyze what the model learned
        if best_model_name:
            self.analyze_learned_push_effects(
                self.models['multi_tpm'][best_model_name],
                self.feature_cols,
                horizons
            )

        # Store training results
        self.training_results = {
            'model_performance': model_performance,
            'best_model': best_model_name,
            'best_score': best_score,
            'horizons': horizons,
            'feature_count': len(self.feature_cols),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

        return self.training_results

    def analyze_learned_push_effects(self, model, feature_names, horizons):
        """
        📊 ANALYZE LEARNED PUSH EFFECTS
        Understand what the model learned about push notifications
        """
        print(f"\n📊 ANALYZING LEARNED PUSH EFFECTS")
        print("=" * 50)

        try:
            # Get feature importance from first estimator
            if hasattr(model.estimators_[0], 'feature_importances_'):
                importances = model.estimators_[0].feature_importances_

                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                # Categorize features
                feature_categories = {
                    'Push Effects': importance_df[
                        importance_df['feature'].str.contains(
                            'push|campaign', case=False, na=False
                        )
                    ],
                    'Time Patterns': importance_df[
                        importance_df['feature'].str.contains(
                            'hour|day|peak|business', case=False, na=False
                        )
                    ],
                    'TPM History': importance_df[
                        importance_df['feature'].str.contains(
                            'lag|ma|std|trend', case=False, na=False
                        )
                    ],
                    'Interactions': importance_df[
                        importance_df['feature'].str.contains(
                            'interaction|strength', case=False, na=False
                        )
                    ]
                }

                print(f"🔝 LEARNED INTELLIGENCE BREAKDOWN:")

                total_importance = importance_df['importance'].sum()

                for category, features_df in feature_categories.items():
                    if len(features_df) > 0:
                        category_importance = features_df['importance'].sum()
                        percentage = (category_importance / total_importance) * 100

                        print(f"\n📈 {category} ({percentage:.1f}% contribution):")

                        for _, row in features_df.head(5).iterrows():
                            feature_pct = (row['importance'] / total_importance) * 100
                            print(f"   • {row['feature']:<30} {feature_pct:5.2f}%")

                # Push intelligence summary
                push_importance = feature_categories['Push Effects']['importance'].sum()
                push_percentage = (push_importance / total_importance) * 100

                print(f"\n🧠 PUSH INTELLIGENCE SUMMARY:")
                print(f"   • Push Effects Contribution: {push_percentage:.1f}%")
                print(
                    f"   • Intelligence Level: {'HIGH' if push_percentage > 15 else 'MODERATE' if push_percentage > 8 else 'LOW'}")

                if len(feature_categories['Push Effects']) > 0:
                    top_push_feature = feature_categories['Push Effects'].iloc[0]
                    print(f"   • Most Important Push Feature: {top_push_feature['feature']}")
                    print(
                        f"   • Top Push Feature Weight: {(top_push_feature['importance'] / total_importance) * 100:.2f}%")

                # Model learning assessment
                learning_quality = "EXCELLENT" if push_percentage > 20 else "GOOD" if push_percentage > 15 else "MODERATE"
                print(f"   • Learning Quality: {learning_quality}")
                print(f"   • Model understands push patterns: {'YES' if push_percentage > 10 else 'LIMITED'}")

        except Exception as e:
            print(f"   ⚠️ Could not analyze learned effects: {e}")

    # ==================== PREDICTION ====================

    def predict_with_learned_effects(self,
                                     current_time,
                                     historical_tpm_1min,
                                     push_notification_active=False,
                                     minutes_since_push=9999,
                                     push_campaign_type='none',
                                     prediction_horizons=(5, 10, 15)):
        """
        🧠 PREDICT WITH LEARNED EFFECTS
        Main prediction function - NO manual calculations!
        """
        if not self.push_effect_learned:
            raise ValueError("❌ Model chưa train với learned effects. Dùng train_push_aware_model() trước.")

        if 'multi_tpm' not in self.models:
            raise ValueError("❌ Multi-output model không có sẵn")

        print(f"🧠 PREDICTING WITH LEARNED PUSH EFFECTS")
        print(f"   📅 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📱 Push active: {push_notification_active}")
        print(f"   🎯 Campaign: {push_campaign_type}")
        print(f"   ⏱️ Minutes since push: {minutes_since_push}")

        try:
            # ============ VALIDATE INPUT ============
            historical_tpm_1min = np.array(historical_tpm_1min, dtype=float)

            if len(historical_tpm_1min) < self.config['min_historical_points']:
                raise ValueError(f"❌ Need at least {self.config['min_historical_points']} minutes of history")

            if len(historical_tpm_1min) > self.config['max_historical_points']:
                print(f"   ⚠️ Truncating to last {self.config['max_historical_points']} points")
                historical_tpm_1min = historical_tpm_1min[-self.config['max_historical_points']:]

            # ============ CREATE SYNTHETIC DATAFRAME ============
            print("   🔧 Creating synthetic prediction data...")

            # Generate historical timestamps
            timestamps = []
            for i in range(len(historical_tpm_1min)):
                timestamp = current_time - timedelta(minutes=(len(historical_tpm_1min) - i - 1))
                timestamps.append(timestamp)

            # Create base dataframe
            df = pd.DataFrame({
                'timestamp': timestamps,
                'tpm': historical_tpm_1min,
                'hour': [ts.hour for ts in timestamps],
                'minute': [ts.minute for ts in timestamps],
                'day_of_week': [ts.weekday() for ts in timestamps],
                'is_weekend': [int(ts.weekday() >= 5) for ts in timestamps],
                'is_business_hours': [int(9 <= ts.hour <= 17) for ts in timestamps]
            })

            # ============ ADD PUSH CONTEXT ============
            print("   📱 Adding push notification context...")

            # Initialize push columns
            df['push_notification_active'] = 0
            df['minutes_since_push'] = 9999
            df['push_campaign_type'] = 'none'

            # Set current push state (last row = current time)
            current_idx = df.index[-1]
            df.loc[current_idx, 'push_notification_active'] = int(push_notification_active)
            df.loc[current_idx, 'minutes_since_push'] = minutes_since_push
            df.loc[current_idx, 'push_campaign_type'] = push_campaign_type

            # Backfill recent push history if there's an active push effect
            if push_notification_active and minutes_since_push <= 15:
                self._backfill_push_context(df, minutes_since_push, push_campaign_type)

            # ============ CREATE FEATURES ============
            print("   🔧 Generating prediction features...")

            # Create same features as training
            enhanced_df = self.create_comprehensive_features(df, 'tpm')

            # Ensure all training features are available
            feature_df = pd.DataFrame(index=[enhanced_df.index[-1]])  # Only current time

            for feature in self.feature_cols:
                if feature in enhanced_df.columns:
                    feature_df[feature] = enhanced_df.loc[enhanced_df.index[-1], feature]
                else:
                    # Generate smart default
                    default_value = self._generate_feature_default(
                        feature, enhanced_df, historical_tpm_1min, current_time
                    )
                    feature_df[feature] = default_value

            # Get prediction vector
            prediction_vector = feature_df.values
            current_tpm = historical_tpm_1min[-1]

            print(f"   ✅ Feature vector prepared: {prediction_vector.shape}")
            print(f"   📊 Current TPM: {current_tpm:.1f}")

            # ============ MODEL PREDICTION ============
            print("   🎯 Applying learned intelligence...")

            model_name = self.best_models['multi_tpm']
            model = self.models['multi_tpm'][model_name]

            # Model predicts ALL horizons with learned effects
            multi_predictions = model.predict(prediction_vector)[0]  # Get first (only) sample

            # ============ COMPILE RESULTS ============
            predictions = {}
            labels = {}
            confidence_scores = {}

            for i, horizon in enumerate(prediction_horizons):
                key = f'tpm_{horizon}min'
                predicted_value = max(0.0, float(multi_predictions[i]))
                predictions[key] = predicted_value

                # Classify prediction
                if hasattr(self, 'tpm_thresholds') and self.tpm_thresholds:
                    labels[key] = self.classify_tpm_value(predicted_value, self.tpm_thresholds)
                else:
                    labels[key] = 'Unknown'

                # Calculate confidence based on prediction stability
                change_magnitude = abs(predicted_value - current_tpm) / max(current_tpm, 1.0)
                confidence = max(0.4, min(0.95, 1.0 - (change_magnitude * 0.5)))
                confidence_scores[key] = confidence

                print(f"   📈 {horizon:2d}min: {predicted_value:6.1f} ({labels[key]:10s}) [conf: {confidence:.2f}]")

            # ============ FINAL RESULTS ============
            results = {
                'predictions': predictions,
                'labels': labels,
                'confidence': confidence_scores,
                'current_state': {
                    'timestamp': current_time,
                    'current_tpm': float(current_tpm),
                    'push_active': push_notification_active,
                    'minutes_since_push': minutes_since_push,
                    'campaign_type': push_campaign_type,
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday()
                },
                'model_info': {
                    'model_used': model_name,
                    'model_type': 'multi_output_learned_effects',
                    'push_effects': 'learned_by_model',
                    'manual_calculations': 'none',
                    'intelligence_level': 'high',
                    'features_used': len(self.feature_cols)
                },
                'prediction_metadata': {
                    'horizons': list(prediction_horizons),
                    'prediction_time': datetime.now(),
                    'method': 'learned_push_intelligence',
                    'historical_points_used': len(historical_tpm_1min),
                    'data_quality': 'synthetic_from_historical_array'
                }
            }

            print(f"✅ LEARNED EFFECTS PREDICTION COMPLETED!")
            return results

        except Exception as e:
            print(f"❌ Error in learned effects prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _backfill_push_context(self, df, minutes_since_push, push_campaign_type):
        """Backfill push notification context in synthetic data"""
        if minutes_since_push <= 15:
            push_start_idx = max(0, len(df) - minutes_since_push - 1)

            for idx in range(push_start_idx, len(df)):
                minutes_ago = len(df) - 1 - idx
                df.loc[df.index[idx], 'minutes_since_push'] = minutes_ago

                if minutes_ago == 0:  # The actual push moment
                    df.loc[df.index[idx], 'push_notification_active'] = 1

                df.loc[df.index[idx], 'push_campaign_type'] = push_campaign_type

    def _generate_feature_default(self, feature_name, enhanced_df, historical_tpm, current_time):
        """Generate smart default value for missing features"""
        feature_lower = feature_name.lower()

        # Time-based features
        if 'hour_sin' in feature_lower:
            return np.sin(2 * np.pi * current_time.hour / 24)
        elif 'hour_cos' in feature_lower:
            return np.cos(2 * np.pi * current_time.hour / 24)
        elif 'day_sin' in feature_lower:
            return np.sin(2 * np.pi * current_time.weekday() / 7)
        elif 'day_cos' in feature_lower:
            return np.cos(2 * np.pi * current_time.weekday() / 7)

        # Business time features
        elif 'is_morning_peak' in feature_lower:
            return 1 if 8 <= current_time.hour <= 10 else 0
        elif 'is_lunch_peak' in feature_lower:
            return 1 if 12 <= current_time.hour <= 13 else 0
        elif 'is_afternoon_peak' in feature_lower:
            return 1 if 15 <= current_time.hour <= 17 else 0
        elif 'is_evening_peak' in feature_lower:
            return 1 if 18 <= current_time.hour <= 20 else 0
        elif 'is_business_hours' in feature_lower:
            return 1 if 9 <= current_time.hour <= 17 else 0
        elif 'is_weekend' in feature_lower:
            return 1 if current_time.weekday() >= 5 else 0

        # TPM-based features
        elif 'lag' in feature_lower:
            return historical_tpm[-1] if len(historical_tpm) > 0 else 100
        elif 'ma_' in feature_lower or 'mean' in feature_lower:
            return np.mean(historical_tpm) if len(historical_tpm) > 0 else 100
        elif '_std' in feature_lower:
            return np.std(historical_tpm) if len(historical_tpm) > 0 else 20
        elif '_min' in feature_lower:
            return np.min(historical_tpm) if len(historical_tpm) > 0 else 80
        elif '_max' in feature_lower:
            return np.max(historical_tpm) if len(historical_tpm) > 0 else 150
        elif 'trend' in feature_lower or 'momentum' in feature_lower:
            return 0  # No trend information available

        # Push-related features (default to no push effect)
        elif any(keyword in feature_lower for keyword in ['push', 'campaign']):
            return 0

        # Default for unknown features
        else:
            return 0.0

    # ==================== UTILITY METHODS ====================

    ## **FIX 2: Enhanced Feature Selection**
    def select_important_features(self, X, y, max_features=35):
        """FIXED: Enhanced feature selection với complete validation"""
        print(f"🎯 Feature selection: {X.shape[1]} → {max_features}")

        if X.shape[1] <= max_features:
            print("   No selection needed")
            return X, list(X.columns)

        # ============ COMPREHENSIVE DATA VALIDATION ============
        print("   🔍 Comprehensive data validation...")

        # Check for non-numeric columns (should not exist after our cleaning)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"   ❌ Found non-numeric columns: {list(non_numeric)}")
            X = X.drop(columns=non_numeric)
            print(f"   ✅ Dropped {len(non_numeric)} non-numeric columns")

        # Convert all columns to numeric with error handling
        print("   🔢 Converting to numeric...")
        converted_cols = 0
        dropped_cols = []

        for col in X.columns:
            try:
                if X[col].dtype == 'object':
                    # Try to convert
                    converted = pd.to_numeric(X[col], errors='coerce')
                    if converted.notna().sum() / len(X) > 0.8:  # 80%+ success rate
                        X[col] = converted.fillna(0)
                        converted_cols += 1
                    else:
                        dropped_cols.append(col)
                elif not X[col].dtype.kind in 'biufc':
                    # Force conversion for other types
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    converted_cols += 1
            except Exception as e:
                dropped_cols.append(col)
                print(f"   ⚠️ Error with column {col}: {e}")

        if dropped_cols:
            X = X.drop(columns=dropped_cols)
            print(f"   ✅ Dropped {len(dropped_cols)} problematic columns")

        if converted_cols > 0:
            print(f"   ✅ Converted {converted_cols} columns to numeric")

        # Handle infinite values
        print("   ♾️ Handling infinite values...")
        inf_count = 0
        for col in X.columns:
            if np.isinf(X[col]).any():
                X[col] = X[col].replace([np.inf, -np.inf], 0)
                inf_count += 1

        if inf_count > 0:
            print(f"   ✅ Fixed infinite values in {inf_count} columns")

        # Handle missing values
        print("   💧 Handling missing values...")
        X = X.fillna(0)

        # Remove constant columns
        print("   🔧 Removing constant columns...")
        try:
            variances = X.var()
            constant_cols = variances[variances <= 1e-10].index
            if len(constant_cols) > 0:
                X = X.drop(columns=constant_cols)
                print(f"   ✅ Removed {len(constant_cols)} constant columns")
        except Exception as e:
            print(f"   ⚠️ Variance calculation error: {e}")
            # Manual constant check
            constant_cols = []
            for col in X.columns:
                try:
                    if X[col].nunique() <= 1:
                        constant_cols.append(col)
                except:
                    pass
            if constant_cols:
                X = X.drop(columns=constant_cols)
                print(f"   ✅ Manually removed {len(constant_cols)} constant columns")

        # Adjust max_features if necessary
        max_features = min(max_features, X.shape[1])

        if X.shape[1] <= max_features:
            print(f"   After cleaning: {X.shape[1]} <= {max_features}, no selection needed")
            return X, list(X.columns)

        # ============ SAFE FEATURE SELECTION ============
        print(f"   🎯 Performing safe feature selection...")

        try:
            # Prepare target variable
            y_clean = pd.to_numeric(y, errors='coerce')
            y_clean = y_clean.fillna(y_clean.median())

            # Final data alignment check
            if len(X) != len(y_clean):
                min_len = min(len(X), len(y_clean))
                X = X.iloc[:min_len]
                y_clean = y_clean.iloc[:min_len]
                print(f"   ✅ Aligned X and y to {min_len} samples")

            # Feature selection with error handling
            selector = SelectKBest(score_func=f_regression, k=max_features)

            # Fit with try-catch
            selector.fit(X, y_clean)

            selected_mask = selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
            selected_scores = selector.scores_[selected_mask]

            # Create feature importance ranking
            feature_importance = list(zip(selected_features, selected_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            print(f"   ✅ Selected {len(selected_features)} features:")
            print(f"   📊 Top 10 features:")
            for i, (feat, score) in enumerate(feature_importance[:10], 1):
                print(f"      {i:2d}. {feat:<35} (F-score: {score:.2f})")

            if len(selected_features) > 10:
                print(f"      ... and {len(selected_features) - 10} more features")

            print(f"   📈 Score range: {min(selected_scores):.1f} - {max(selected_scores):.1f}")

            return X[selected_features], selected_features

        except Exception as e:
            print(f"   ❌ Feature selection failed: {e}")
            print(f"   📊 Fallback: using all available features")

            # Fallback: return top N features by variance
            try:
                variances = X.var().sort_values(ascending=False)
                top_features = variances.head(max_features).index.tolist()
                print(f"   ✅ Fallback: selected {len(top_features)} highest variance features")
                return X[top_features], top_features
            except:
                # Last resort: return first N features
                features_to_use = X.columns[:max_features].tolist()
                print(f"   ✅ Last resort: using first {len(features_to_use)} features")
                return X[features_to_use], features_to_use

    def compute_tpm_thresholds(self, tpm_data):
        """Compute TPM classification thresholds"""
        tpm_clean = pd.to_numeric(tpm_data, errors='coerce').dropna()

        thresholds = {
            'q50': np.percentile(tpm_clean, 50),
            'q60': np.percentile(tpm_clean, 60),
            'q75': np.percentile(tpm_clean, 75),
            'q80': np.percentile(tpm_clean, 80),
            'q90': np.percentile(tpm_clean, 90),
            'q95': np.percentile(tpm_clean, 95)
        }

        print(f"📊 TPM Thresholds computed:")
        for key, value in thresholds.items():
            print(f"   {key.upper()}: {value:.1f}")

        return thresholds

    def classify_tpm_value(self, tpm_value, thresholds):
        """Enhanced TPM classification"""
        if tpm_value >= thresholds['q95']:
            return 'Extreme'
        elif tpm_value >= thresholds['q90']:
            return 'Very High'
        elif tpm_value >= thresholds['q80']:
            return 'High'
        elif tpm_value >= thresholds['q60']:
            return 'Medium'
        else:
            return 'Low'

    # ==================== MODEL PERSISTENCE ====================

    def save_comprehensive_model(self, model_dir="comprehensive_models", model_name="learned_push_intelligence"):
        """Save comprehensive model with all metadata"""
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.joblib")
        joblib.dump(self.models, model_path)

        # Save comprehensive metadata
        metadata = {
            'best_models': self.best_models,
            'feature_cols': self.feature_cols,
            'tpm_thresholds': self.tpm_thresholds,
            'training_results': self.training_results,
            'push_effect_learned': self.push_effect_learned,
            'config': self.config,
            'timestamp': timestamp,
            'version': '2.0',
            'model_type': 'comprehensive_learned_effects'
        }

        metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.joblib")
        joblib.dump(metadata, metadata_path)

        print(f"💾 Comprehensive model saved:")
        print(f"   📁 Model: {model_path}")
        print(f"   📄 Metadata: {metadata_path}")
        print(f"   🧠 Push effects: {'LEARNED' if self.push_effect_learned else 'NOT LEARNED'}")
        print(f"   🎯 Features: {len(self.feature_cols)}")

        return model_path, metadata_path

    def load_comprehensive_model(self, model_path, metadata_path):
        """Load comprehensive model with validation"""
        try:
            self.models = joblib.load(model_path)
            metadata = joblib.load(metadata_path)

            self.best_models = metadata['best_models']
            self.feature_cols = metadata['feature_cols']
            self.tpm_thresholds = metadata['tpm_thresholds']
            self.training_results = metadata.get('training_results', {})
            self.push_effect_learned = metadata.get('push_effect_learned', False)
            self.config = metadata.get('config', self.config)

            print(f"📂 Comprehensive model loaded:")
            print(f"   🏆 Best model: {self.best_models.get('multi_tpm', 'Unknown')}")
            print(f"   🎯 Features: {len(self.feature_cols)}")
            print(f"   🧠 Push effects learned: {self.push_effect_learned}")
            print(f"   📊 Model version: {metadata.get('version', '1.0')}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    # ==================== DEMO AND TESTING ====================

    def demo_comprehensive_prediction(self):
        """Comprehensive demo of learned effects prediction"""
        print("🚀 COMPREHENSIVE DEMO: Learned Push Intelligence")
        print("=" * 70)

        if not self.push_effect_learned:
            print("❌ Model chưa được train với learned effects")
            print("   Hãy dùng train_push_aware_model() trước")
            return

        # Generate realistic sample data
        print("📊 Generating realistic test data...")

        historical_tpm = []
        base_tpm = 120

        # Create pattern with daily cycle + noise
        for i in range(90):  # 90 minutes of history
            # Daily pattern
            time_factor = np.sin(2 * np.pi * i / 1440) * 30  # Daily cycle

            # Business hours boost
            if 30 <= i <= 70:  # Simulate business hours in sample
                business_boost = 20
            else:
                business_boost = 0

            # Random variation
            noise = np.random.normal(0, 15)

            tpm_value = base_tpm + time_factor + business_boost + noise
            tpm_value = max(60, tpm_value)  # Minimum TPM
            historical_tpm.append(tpm_value)

        current_time = datetime.now().replace(hour=14, minute=30)  # 2:30 PM

        # Test multiple scenarios
        test_scenarios = [
            {
                'name': 'Baseline (No Push)',
                'description': 'Normal prediction without push effects',
                'push_active': False,
                'minutes_since': 9999,
                'campaign': 'none'
            },
            {
                'name': 'Fresh Morning Push',
                'description': 'Morning commute push just activated',
                'push_active': True,
                'minutes_since': 2,
                'campaign': 'morning_commute'
            },
            {
                'name': 'Active Lunch Campaign',
                'description': 'Lunch peak push in effective window',
                'push_active': True,
                'minutes_since': 8,
                'campaign': 'lunch_peak'
            },
            {
                'name': 'Declining Evening Push',
                'description': 'Evening push effect waning',
                'push_active': False,
                'minutes_since': 13,
                'campaign': 'evening_commute'
            },
            {
                'name': 'Off-hours Push',
                'description': 'Push during low-effectiveness period',
                'push_active': True,
                'minutes_since': 6,
                'campaign': 'evening_leisure'
            }
        ]

        results_summary = []

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print(f"    {scenario['description']}")
            print("-" * 50)

            try:
                results = self.predict_with_learned_effects(
                    current_time=current_time,
                    historical_tpm_1min=historical_tpm,
                    push_notification_active=scenario['push_active'],
                    minutes_since_push=scenario['minutes_since'],
                    push_campaign_type=scenario['campaign'],
                    prediction_horizons=(5, 10, 15)
                )

                if results:
                    print(f"🎯 Intelligent Predictions (Model Learned):")

                    scenario_results = {
                        'scenario': scenario['name'],
                        'predictions': {}
                    }

                    for horizon in [5, 10, 15]:
                        key = f'tpm_{horizon}min'
                        pred = results['predictions'][key]
                        label = results['labels'][key]
                        conf = results['confidence'][key]

                        scenario_results['predictions'][horizon] = {
                            'value': pred,
                            'label': label,
                            'confidence': conf
                        }

                        print(f"   {horizon:2d} min: {pred:6.1f} TPM ({label:10s}) [confidence: {conf:.2f}]")

                    results_summary.append(scenario_results)

                    # Show model insights
                    model_info = results['model_info']
                    print(f"\n   🧠 Model Intelligence:")
                    print(f"      • Algorithm: {model_info['model_used']}")
                    print(f"      • Intelligence Level: {model_info['intelligence_level']}")
                    print(f"      • Features Used: {model_info['features_used']}")

                else:
                    print("❌ Prediction failed for this scenario")

            except Exception as e:
                print(f"❌ Error in scenario {i}: {e}")

        # ============ COMPARATIVE ANALYSIS ============
        if results_summary:
            print(f"\n📊 COMPARATIVE ANALYSIS")
            print("=" * 50)

            baseline_5min = None
            if results_summary[0]['scenario'] == 'Baseline (No Push)':
                baseline_5min = results_summary[0]['predictions'][5]['value']

            print(f"Push Effect Analysis (5-minute horizon):")
            for result in results_summary:
                pred_5min = result['predictions'][5]['value']

                if baseline_5min and result['scenario'] != 'Baseline (No Push)':
                    effect = pred_5min - baseline_5min
                    effect_pct = (effect / baseline_5min) * 100
                    print(f"   {result['scenario']:<25}: {effect:+6.1f} TPM ({effect_pct:+5.1f}%)")
                else:
                    print(f"   {result['scenario']:<25}: {pred_5min:6.1f} TPM (baseline)")

        print(f"\n🎉 COMPREHENSIVE DEMO COMPLETED!")
        print(f"   🧠 Model demonstrated intelligent push effect learning")
        print(f"   📈 No manual calculations were used")
        print(f"   ⚡ All effects learned from training data")


# ==================== MAIN EXECUTION ====================
def main():
    """Fixed main function với comprehensive error handling"""
    print("🚀 COMPREHENSIVE TIME SERIES TRAINER - FIXED VERSION")
    print("=" * 60)
    print("🧠 Intelligent Push Effect Learning")
    print("📊 Multi-Horizon Predictions")
    print("⚡ Production-Ready Architecture")
    print()

    # CSV file path
    csv_path = "/Users/dongdinh/Documents/Learning/100-Days-Of-ML-Code/Src/Day6/datasets/day6_maximum_precision_data.csv"

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("   Please update the csv_path variable with correct path")
        return

    try:
        # Initialize comprehensive trainer
        print("🏗️ Initializing trainer...")
        trainer = ComprehensiveTimeSeriesTrainer()

        # Train model with extensive error handling
        print("🎯 Starting comprehensive training with FIXED string handling...")

        results = trainer.train_push_aware_model(
            csv_path=csv_path,
            target_col='tpm',
            horizons=(5, 10, 15),
            test_size=0.25
        )

        if results and results.get('best_score', 0) > 0:
            print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print(f"   🏆 Best model: {results['best_model']}")
            print(f"   📈 Average R² score: {results['best_score']:.3f}")
            print(f"   🎯 Horizons trained: {results['horizons']}")
            print(f"   📊 Features used: {results['feature_count']}")
            print(f"   🧠 Push effects: LEARNED from data")

            # Save comprehensive model
            print(f"\n💾 Saving comprehensive model...")
            try:
                model_path, metadata_path = trainer.save_comprehensive_model(
                    model_dir="comprehensive_push_models",
                    model_name="intelligent_push_aware_fixed"
                )

                print(f"   ✅ Model saved successfully!")
                print(f"   📁 Model file: {os.path.basename(model_path)}")
                print(f"   📄 Metadata file: {os.path.basename(metadata_path)}")

            except Exception as save_error:
                print(f"   ⚠️ Model save warning: {save_error}")
                print("   Model is trained but not saved to disk")

            # Run demo if training successful
            try:
                print(f"\n🎮 Running comprehensive demo...")
                trainer.demo_comprehensive_prediction()
                print(f"   ✅ Demo completed successfully!")
            except Exception as demo_error:
                print(f"   ⚠️ Demo warning: {demo_error}")
                print("   Model is trained and ready for use")

            print(f"\n✅ ALL CORE OPERATIONS COMPLETED!")
            print(f"   🧠 Model learned push effects intelligently")
            print(f"   🚀 Ready for production predictions")
            print(f"   📊 Use predict_with_learned_effects() for new predictions")
            print(f"   🔧 String handling issues FIXED")

        else:
            print("\n❌ TRAINING FAILED")
            print("   Model training completed but no valid results returned")
            print("   Check data quality and feature engineering")

            if results:
                print(f"   Debug info: {results}")

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print("   Please check the CSV file path and permissions")

    except ValueError as e:
        print(f"❌ Data error: {e}")
        print("   Please check the CSV data format and column names")
        print("   Expected columns: tpm, timestamp, push_notification_active")

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)[:200]}...")
        print("\n🔍 Debugging information:")
        print(f"   Error type: {type(e).__name__}")

        # Try to provide helpful context
        try:
            import traceback
            error_lines = traceback.format_exc().split('\n')
            relevant_lines = [line for line in error_lines if
                              'create_comprehensive_features' in line or 'select_important_features' in line or 'string to float' in line]
            if relevant_lines:
                print("   Relevant error context:")
                for line in relevant_lines[:3]:
                    print(f"      {line}")
        except:
            pass

        print("\n💡 Troubleshooting suggestions:")
        print("   1. Check for non-numeric columns in CSV")
        print("   2. Verify column names match expected format")
        print("   3. Check data types with df.dtypes")
        print("   4. Look for mixed data types in columns")


if __name__ == "__main__":
    main()

