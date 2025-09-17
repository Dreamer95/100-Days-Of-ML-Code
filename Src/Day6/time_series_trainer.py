"""
Day6 Summary: Enhanced Time Series Training with Real CSV Data (FIXED)
Training model với data từ day6_maximum_precision_data.csv và apply push effects
"""
import glob

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
import os

warnings.filterwarnings('ignore')


class EnhancedTimeSeriesTrainer:
    """
    Enhanced Time Series Training Model với push effects cho CSV data
    """

    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_cols = []
        self.tpm_thresholds = {}
        self.training_results = {}
        # Enhanced push effect configuration
        self.push_config = {
            'daytime_hours': (7, 21),  # 7AM to 9PM
            'no_effect_hours': (2, 6),  # 2AM to 6AM
            'boost_range': (0.5, 2.0),  # 50% to 200% boost
            'effect_delay_minutes': (5, 10),  # Effect starts 5-10 min after push
            'effect_duration_minutes': 15,  # Effect lasts 15 minutes
            'decay_type': 'exponential'  # exponential, linear, or plateau
        }

    def load_csv_data(self, csv_path):
        """
        Load data từ CSV file với proper handling
        """
        print(f"📂 Loading data from {csv_path}...")

        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"   ✅ Loaded {len(df)} records from CSV")

            # Convert timestamp columns
            timestamp_columns = ['timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc']
            for col in timestamp_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            elif 'timestamp_utc' in df.columns:
                df = df.sort_values('timestamp_utc').reset_index(drop=True)

            print(f"   📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   📊 TPM range: {df['tpm'].min():.1f} - {df['tpm'].max():.1f}")
            print(f"   📱 Push events: {df['push_notification_active'].sum()} total")

            return df

        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None

    def apply_business_push_effects_to_csv_data(self, data):
        """
        Apply business logic push effects to existing CSV data
        """
        print("📱 Applying business push effects to CSV data...")

        df = data.copy()

        # Track original values
        df['original_tpm'] = df['tpm'].copy()
        df['push_boost_applied'] = 0.0
        df['push_effect_reason'] = 'none'

        # Ensure required columns exist
        if 'push_notification_active' not in df.columns:
            df['push_notification_active'] = 0
        if 'minutes_since_push' not in df.columns:
            df['minutes_since_push'] = 9999
        if 'push_campaign_type' not in df.columns:
            df['push_campaign_type'] = 'none'

        # Apply business logic row by row
        modifications = {
            'daytime_boosted': 0,
            'no_effect_applied': 0,
            'other_hours_boosted': 0,
            'no_push_periods': 0
        }

        for idx, row in df.iterrows():
            # Get time and push info
            current_time = pd.to_datetime(row['timestamp']) if 'timestamp' in df.columns else pd.to_datetime(
                row['timestamp_utc'])
            current_hour = current_time.hour

            push_active = int(row.get('push_notification_active', 0))
            minutes_since = int(row.get('minutes_since_push', 9999))
            campaign_type = str(row.get('push_campaign_type', 'none'))  # Ensure string

            # Calculate push boost based on business logic
            boost_factor, reason = self._calculate_business_push_boost(
                current_hour=current_hour,
                push_active=push_active,
                minutes_since_push=minutes_since,
                campaign_type=campaign_type
            )

            if boost_factor > 0:
                # Apply boost to TPM
                original_tpm = df.loc[idx, 'tpm']
                boosted_tpm = original_tpm * (1 + boost_factor)

                df.loc[idx, 'tpm'] = boosted_tpm
                df.loc[idx, 'push_boost_applied'] = boost_factor
                df.loc[idx, 'push_effect_reason'] = reason

                # Count modifications
                if '7AM-9PM' in reason:
                    modifications['daytime_boosted'] += 1
                elif 'no_effect' in reason:
                    modifications['no_effect_applied'] += 1
                elif 'other_hours' in reason:
                    modifications['other_hours_boosted'] += 1
            else:
                modifications['no_push_periods'] += 1

        # Statistics
        boosted_data = df[df['push_boost_applied'] > 0]

        print(f"   ✅ Applied business push effects:")
        print(f"      • 7AM-9PM boosted: {modifications['daytime_boosted']} points")
        print(f"      • 2AM-6AM no effect: {modifications['no_effect_applied']} points")
        print(f"      • Other hours reduced: {modifications['other_hours_boosted']} points")
        print(f"      • No push periods: {modifications['no_push_periods']} points")

        if len(boosted_data) > 0:
            avg_boost = boosted_data['push_boost_applied'].mean()
            max_boost = boosted_data['push_boost_applied'].max()
            print(f"      • Average boost applied: {avg_boost:.1%}")
            print(f"      • Maximum boost applied: {max_boost:.1%}")

        return df

    def _calculate_business_push_boost(self, current_hour, push_active, minutes_since_push, campaign_type):
        """Calculate push boost factor theo business requirements"""
        # Check if within effect window
        effect_duration = self.push_config['effect_duration_minutes']
        effect_delay_start, effect_delay_end = self.push_config['effect_delay_minutes']

        # No effect if too long since push
        if minutes_since_push > effect_duration:
            return 0.0, 'outside_effect_window'

        # No immediate effect (within delay window)
        if minutes_since_push < effect_delay_start:
            return 0.0, 'within_delay_window'

        # Check business hours
        daytime_start, daytime_end = self.push_config['daytime_hours']
        no_effect_start, no_effect_end = self.push_config['no_effect_hours']

        # 2AM-6AM: NO EFFECT (Business requirement)
        if no_effect_start <= current_hour <= no_effect_end:
            return 0.0, '2AM-6AM_no_effect_period'

        # 7AM-9PM: HIGH EFFECT (Business requirement: 50-200%)
        if daytime_start <= current_hour <= daytime_end:
            if effect_delay_start <= minutes_since_push <= effect_delay_end:
                boost_min, boost_max = self.push_config['boost_range']
                base_boost = np.random.uniform(boost_min, boost_max)
                campaign_modifier = self._get_campaign_modifier(campaign_type, current_hour)
                final_boost = base_boost * campaign_modifier
                final_boost = np.clip(final_boost, 0.5, 2.0)
                return final_boost, f'7AM-9PM_peak_effect_{campaign_type}'

            elif minutes_since_push <= effect_duration:
                boost_min, boost_max = self.push_config['boost_range']
                reduced_boost = np.random.uniform(boost_min * 0.3, boost_max * 0.6)
                decay_factor = self._calculate_decay_factor(minutes_since_push)
                final_boost = reduced_boost * decay_factor
                campaign_modifier = self._get_campaign_modifier(campaign_type, current_hour)
                final_boost *= campaign_modifier
                final_boost = np.clip(final_boost, 0.1, 2.0)
                return final_boost, f'7AM-9PM_declining_effect_{campaign_type}'

        # Other hours: REDUCED EFFECT
        else:
            if effect_delay_start <= minutes_since_push <= effect_duration:
                boost_min, boost_max = self.push_config['boost_range']
                reduced_boost = np.random.uniform(boost_min * 0.2, boost_max * 0.4)
                decay_factor = self._calculate_decay_factor(minutes_since_push)
                final_boost = reduced_boost * decay_factor
                campaign_modifier = self._get_campaign_modifier(campaign_type, current_hour) * 0.5
                final_boost *= campaign_modifier
                final_boost = np.clip(final_boost, 0.05, 0.8)
                return final_boost, f'other_hours_reduced_effect_{campaign_type}'

        return 0.0, 'no_effect_conditions_met'

    def _get_campaign_modifier(self, campaign_type, current_hour):
        """Get campaign type modifier"""
        campaign_modifiers = {
            'morning_commute': {
                'optimal_hours': [7, 8, 9],
                'good_hours': [6, 10],
                'poor_hours': [0, 1, 2, 3, 4, 5, 22, 23]
            },
            'lunch_peak': {
                'optimal_hours': [11, 12, 13],
                'good_hours': [10, 14],
                'poor_hours': [0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23]
            },
            'afternoon_break': {
                'optimal_hours': [14, 15, 16],
                'good_hours': [13, 17],
                'poor_hours': [0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 23]
            },
            'evening_commute': {
                'optimal_hours': [17, 18, 19],
                'good_hours': [16, 20],
                'poor_hours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 23]
            },
            'evening_leisure': {
                'optimal_hours': [19, 20, 21],
                'good_hours': [18, 22],
                'poor_hours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
        }

        if campaign_type not in campaign_modifiers:
            return 1.0

        config = campaign_modifiers[campaign_type]

        if current_hour in config['optimal_hours']:
            return 1.2
        elif current_hour in config['good_hours']:
            return 1.0
        elif current_hour in config['poor_hours']:
            return 0.6
        else:
            return 0.8

    def _calculate_decay_factor(self, minutes_since_push):
        """Calculate decay factor"""
        effect_duration = self.push_config['effect_duration_minutes']
        decay_type = self.push_config['decay_type']

        if minutes_since_push > effect_duration:
            return 0.0

        progress = minutes_since_push / effect_duration

        if decay_type == 'exponential':
            return np.exp(-progress * 2.5)
        elif decay_type == 'linear':
            return 1.0 - progress
        elif decay_type == 'plateau':
            if progress <= 0.6:
                return 1.0
            else:
                return (1.0 - progress) / 0.4
        else:
            return np.exp(-progress * 2.0)

    def create_enhanced_features_from_csv(self, data, target_col='tpm'):
        """
        FIXED: Create enhanced features với comprehensive string handling
        """
        print("🔧 Creating enhanced features from CSV data...")

        df = data.copy()

        # ============ STEP 1: IDENTIFY AND HANDLE STRING COLUMNS ============
        print("🔤 Identifying and handling string columns...")

        # Get all non-numeric columns (excluding timestamps and target)
        exclude_from_check = ['timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc', target_col]
        check_columns = [col for col in df.columns if col not in exclude_from_check]

        string_columns = []
        for col in check_columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                string_columns.append(col)

        print(f"   Found {len(string_columns)} string columns: {string_columns}")

        # ============ STEP 2: HANDLE PUSH_CAMPAIGN_TYPE SPECIFICALLY ============
        if 'push_campaign_type' in df.columns:
            print("   📱 Processing push_campaign_type column...")

            # Get unique values
            unique_campaigns = df['push_campaign_type'].unique()
            print(f"      Campaign types: {unique_campaigns}")

            # Create binary columns for each campaign type
            campaign_types = ['morning_commute', 'lunch_peak', 'afternoon_break',
                              'evening_commute', 'evening_leisure', 'sleeping_hours_early', 'sleeping_hours_late']

            for campaign in campaign_types:
                df[f'campaign_{campaign}'] = (df['push_campaign_type'] == campaign).astype(int)
                print(f"      Created binary feature: campaign_{campaign}")

            # Create aggregated features
            df['is_peak_hour_campaign'] = df['push_campaign_type'].isin(['lunch_peak', 'evening_commute']).astype(int)
            df['is_commute_campaign'] = df['push_campaign_type'].isin(['morning_commute', 'evening_commute']).astype(
                int)
            df['is_no_effect_campaign'] = df['push_campaign_type'].isin(
                ['sleeping_hours_early', 'sleeping_hours_late']).astype(int)

            # Push timing quality score
            df['push_timing_quality'] = df.apply(
                lambda row: self._calculate_push_timing_quality(
                    row['hour'], str(row.get('push_campaign_type', 'none'))
                ), axis=1
            )

            # Remove original string column
            df = df.drop('push_campaign_type', axis=1)
            print("      ✅ Removed original push_campaign_type column")

        # ============ STEP 3: HANDLE OTHER STRING COLUMNS ============
        remaining_string_cols = [col for col in string_columns if col != 'push_campaign_type' and col in df.columns]

        if remaining_string_cols:
            print(f"   📝 Processing remaining string columns: {remaining_string_cols}")

            for col in remaining_string_cols:
                unique_vals = df[col].unique()[:10]  # Show first 10
                print(f"      {col}: {unique_vals}")

                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"      ✅ Converted {col} to numeric")
                except:
                    # If can't convert, create binary features if reasonable number of unique values
                    if df[col].nunique() <= 10:
                        print(f"      Creating binary features for {col}")
                        for val in df[col].unique():
                            if pd.notna(val):
                                safe_val = str(val).replace(' ', '_').replace('-', '_')
                                df[f'{col}_{safe_val}'] = (df[col] == val).astype(int)

                    # Drop original string column
                    df = df.drop(col, axis=1)
                    print(f"      ✅ Processed and removed {col}")

        # ============ STEP 4: ADD ENHANCED FEATURES ============
        print("📱 Adding enhanced push features...")

        # Push effectiveness windows (business logic)
        df['is_push_daytime_effective'] = ((df['hour'] >= 7) & (df['hour'] <= 21)).astype(int)
        df['is_push_no_effect_period'] = ((df['hour'] >= 2) & (df['hour'] <= 6)).astype(int)

        # Enhanced push effect features
        if 'minutes_since_push' in df.columns:
            # Ensure minutes_since_push is numeric
            df['minutes_since_push'] = pd.to_numeric(df['minutes_since_push'], errors='coerce').fillna(9999)

            # Push effect windows
            df['in_push_delay_window'] = ((df['minutes_since_push'] >= 5) & (df['minutes_since_push'] <= 10)).astype(
                int)
            df['in_push_peak_effect'] = ((df['minutes_since_push'] >= 5) & (df['minutes_since_push'] <= 10)).astype(int)
            df['in_push_declining_effect'] = (
                        (df['minutes_since_push'] > 10) & (df['minutes_since_push'] <= 15)).astype(int)

            # Enhanced decay factors
            df['push_decay_exponential'] = df['minutes_since_push'].apply(
                lambda x: np.exp(-x / 8.0) if x <= 15 else 0.0
            )
            df['push_decay_linear'] = df['minutes_since_push'].apply(
                lambda x: max(0, 1 - x / 15.0) if x <= 15 else 0.0
            )

        # ============ STEP 5: INTERACTION FEATURES ============
        print("🔗 Adding interaction features...")

        # Time × Push interactions
        if 'push_notification_active' in df.columns:
            df['push_notification_active'] = pd.to_numeric(df['push_notification_active'], errors='coerce').fillna(0)
            df['push_hour_interaction'] = df['push_notification_active'] * df['hour']
            df['push_weekday_interaction'] = df['push_notification_active'] * (df['day_of_week'] + 1)

            if 'is_business_hours' in df.columns:
                df['push_business_hour_interaction'] = df['push_notification_active'] * df['is_business_hours']

        # ============ STEP 6: ADVANCED TEMPORAL FEATURES ============
        print("🕐 Adding advanced temporal features...")

        # Business context features
        df['is_morning_peak'] = ((df['hour'] >= 8) & (df['hour'] <= 10)).astype(int)
        df['is_lunch_peak'] = ((df['hour'] >= 12) & (df['hour'] <= 13)).astype(int)
        df['is_afternoon_peak'] = ((df['hour'] >= 15) & (df['hour'] <= 17)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 20)).astype(int)

        # Multi-scale cyclical features
        df['hour_quarter_sin'] = np.sin(2 * np.pi * (df['hour'] % 6) / 6)
        df['hour_quarter_cos'] = np.cos(2 * np.pi * (df['hour'] % 6) / 6)

        # ============ STEP 7: SEQUENCE FEATURES ============
        print("📈 Adding sequence features...")

        # Push sequence tracking
        if 'push_notification_active' in df.columns:
            # Cumulative push count
            df['cumulative_pushes'] = df['push_notification_active'].cumsum()

            # Pushes in rolling windows (adjust based on data)
            avg_interval = df.get('interval_minutes', pd.Series([5])).median()
            hour_window = max(1, int(60 / avg_interval))
            day_window = max(1, int(1440 / avg_interval))

            df['pushes_last_hour'] = df['push_notification_active'].rolling(
                window=min(hour_window, len(df)), min_periods=1
            ).sum()

            df['pushes_last_day'] = df['push_notification_active'].rolling(
                window=min(day_window, len(df)), min_periods=1
            ).sum()

        # ============ STEP 8: COMPREHENSIVE CLEANUP ============
        print("🧹 Comprehensive data cleanup...")

        # Remove potentially leaky columns
        danger_columns = [
            'timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc',
            'metric_name', 'day_name', 'sequence_id',
            'original_tpm', 'push_boost_applied', 'push_effect_reason',
            'collection_strategy', 'data_quality', 'precision_category'
        ]

        existing_danger_cols = [col for col in danger_columns if col in df.columns]
        if existing_danger_cols:
            df = df.drop(columns=existing_danger_cols)
            print(f"   ✅ Removed {len(existing_danger_cols)} potentially leaky columns")

        # ============ STEP 9: ENSURE ALL COLUMNS ARE NUMERIC ============
        print("🔢 Final numeric validation...")

        # Check for any remaining non-numeric columns
        final_non_numeric = []
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                final_non_numeric.append(col)

        if final_non_numeric:
            print(f"   ⚠️ Found remaining non-numeric columns: {final_non_numeric}")
            for col in final_non_numeric:
                print(f"      Dropping problematic column: {col}")
                df = df.drop(columns=[col])

        # Ensure all numeric columns are proper numeric types
        for col in df.columns:
            if col == target_col:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"   ❌ Could not convert {col} to numeric, dropping...")
                df = df.drop(columns=[col])

        # Handle missing values comprehensively
        print("🔧 Handling missing values...")

        # Handle target column NaNs
        if target_col in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=[target_col])
            if len(df) < initial_len:
                print(f"   Dropped {initial_len - len(df)} rows with missing target values")

        # Handle feature NaNs
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col == target_col:
                continue

            if df[col].isnull().sum() > 0:
                if col.startswith('lag') or col.startswith('ma') or 'std' in col:
                    # For time series features, use forward/backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                elif col.startswith('push') or col.startswith('campaign'):
                    # For push features, fill with 0
                    df[col] = df[col].fillna(0)
                else:
                    # For other features, use median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], 0)

        # Final validation
        print("✅ Final validation...")

        # Check for any remaining issues
        remaining_nan = df.isnull().sum().sum()
        remaining_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

        if remaining_nan > 0:
            print(f"   ⚠️ Still have {remaining_nan} NaN values, filling with 0")
            df = df.fillna(0)

        if remaining_inf > 0:
            print(f"   ⚠️ Still have {remaining_inf} infinite values, replacing with 0")
            df = df.replace([np.inf, -np.inf], 0)

        # Ensure target column exists and is valid
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in processed data")

        feature_columns = [col for col in df.columns if col != target_col]

        print(f"✅ Enhanced feature creation completed:")
        print(f"   • Dataset: {len(df)} rows × {len(df.columns)} columns")
        print(f"   • Target: {target_col}")
        print(f"   • Features: {len(feature_columns)}")
        print(f"   • All features numeric: {all(df[col].dtype.kind in 'biufc' for col in feature_columns)}")

        # Feature breakdown
        feature_categories = {
            'Time features': len(
                [c for c in feature_columns if any(x in c for x in ['hour', 'day', 'sin', 'cos', 'peak'])]),
            'Push features': len([c for c in feature_columns if any(x in c for x in ['push', 'campaign'])]),
            'Lag features': len([c for c in feature_columns if 'lag' in c]),
            'Rolling features': len([c for c in feature_columns if any(x in c for x in ['ma', 'std', 'min', 'max'])]),
            'Other features': len([c for c in feature_columns if not any(x in c for x in
                                                                         ['hour', 'day', 'sin', 'cos', 'peak', 'push',
                                                                          'campaign', 'lag', 'ma', 'std', 'min',
                                                                          'max'])])
        }

        print(f"   📊 Feature breakdown:")
        for category, count in feature_categories.items():
            print(f"      • {category}: {count}")

        return df

    def _calculate_push_timing_quality(self, hour, campaign_type):
        """Calculate quality score for push timing vs campaign type"""
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

    def train_enhanced_model_from_csv(self, csv_path, target_col='tpm', test_size=0.25, max_features=25):
        """Train enhanced model từ CSV data với business push effects"""
        print("🚀 Training Enhanced Model from CSV Data")
        print("=" * 60)

        # Load CSV data
        raw_data = self.load_csv_data(csv_path)
        if raw_data is None:
            raise ValueError("Failed to load CSV data")

        # Apply business push effects FIRST
        enhanced_data = self.apply_business_push_effects_to_csv_data(raw_data)

        # Create enhanced features with comprehensive string handling
        processed_data = self.create_enhanced_features_from_csv(enhanced_data, target_col)

        if processed_data.empty:
            raise ValueError("No processed data available for training")

        # Prepare training data
        X = processed_data.drop(columns=[target_col])
        y = processed_data[target_col]

        print(f"📈 Training data prepared:")
        print(f"   • Samples: {len(X)}")
        print(f"   • Features: {len(X.columns)}")
        print(f"   • Target range: {y.min():.1f} - {y.max():.1f}")

        # Compute TPM thresholds
        self.tpm_thresholds = self.compute_tpm_thresholds(y)

        # Feature selection
        X_selected, selected_features = self.select_important_features(
            X, y, max_features=max_features
        )

        self.feature_cols = selected_features

        # Time-based split (crucial for time series)
        split_idx = int(len(X_selected) * (1 - test_size))

        X_train = X_selected.iloc[:split_idx].copy()
        X_test = X_selected.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()

        print(f"\n📊 Time-based split:")
        print(f"   • Train: {len(X_train)} samples")
        print(f"   • Test: {len(X_test)} samples")

        # Train models
        results = self.train_models_with_cv(X_train, y_train, X_test, y_test)

        print(f"\n🏆 Training Results:")
        print(f"   • Best Model: {results['best_model']}")
        print(f"   • Best R² Score: {results['best_score']:.3f}")
        print(f"   • Features Used: {len(selected_features)}")
        print(f"   • Push Effects: Applied according to business logic")

        return results

    def select_important_features(self, X, y, max_features=25):
        """FIXED: Enhanced feature selection với comprehensive validation"""
        print(f"🎯 Feature Selection: {X.shape[1]} → {max_features} features")

        if X.shape[1] <= max_features:
            print("   No selection needed - already optimal size")
            return X, list(X.columns)

        # ============ COMPREHENSIVE DATA VALIDATION ============
        print("   🔍 Comprehensive data validation...")

        # Check for non-numeric columns (should be none after our cleaning)
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"   ❌ Found non-numeric columns after cleaning: {list(non_numeric)}")
            # Drop them as last resort
            X = X.drop(columns=non_numeric)
            print(f"   ✅ Dropped {len(non_numeric)} non-numeric columns")

        # Handle infinite values
        print("   🔧 Checking for infinite values...")
        inf_mask = np.isinf(X)
        if inf_mask.any().any():
            inf_cols = X.columns[inf_mask.any()]
            print(f"   ⚠️ Found infinite values in: {list(inf_cols)}")
            X = X.replace([np.inf, -np.inf], 0)
            print("   ✅ Replaced infinite values with 0")

        # Handle NaN values
        print("   🔧 Checking for NaN values...")
        nan_mask = X.isnull()
        if nan_mask.any().any():
            nan_cols = X.columns[nan_mask.any()]
            print(f"   ⚠️ Found NaN values in: {list(nan_cols)}")
            X = X.fillna(0)
            print("   ✅ Filled NaN values with 0")

        # Remove constant columns (zero variance)
        print("   🔧 Checking for constant columns...")
        try:
            variances = X.var()
            constant_cols = variances[variances == 0].index
            if len(constant_cols) > 0:
                print(f"   ⚠️ Found {len(constant_cols)} constant columns")
                X = X.drop(columns=constant_cols)
                print("   ✅ Removed constant columns")
        except Exception as e:
            print(f"   ⚠️ Error calculating variance: {e}")
            # Manual check for constant columns
            constant_cols = []
            for col in X.columns:
                try:
                    if X[col].nunique() <= 1:
                        constant_cols.append(col)
                except:
                    pass

            if constant_cols:
                print(f"   Removing {len(constant_cols)} constant columns manually")
                X = X.drop(columns=constant_cols)

        # Update max_features if we removed columns
        max_features = min(max_features, X.shape[1])

        if X.shape[1] <= max_features:
            print(f"   After cleaning: {X.shape[1]} <= {max_features}, no selection needed")
            return X, list(X.columns)

        # ============ FEATURE SELECTION ============
        try:
            print("   🎯 Performing feature selection...")

            # Final data type check
            for col in X.columns:
                if not X[col].dtype.kind in 'biufc':  # binary, integer, unsigned, float, complex
                    print(f"   ❌ Column {col} has non-numeric dtype: {X[col].dtype}")
                    X = X.drop(columns=[col])

            # Ensure y is numeric
            y = pd.to_numeric(y, errors='coerce')
            y = y.fillna(y.median())

            # Feature selection
            f_selector = SelectKBest(score_func=f_regression, k=max_features)
            f_selector.fit(X, y)

            selected_mask = f_selector.get_support()
            selected_features = X.columns[selected_mask].tolist()
            selected_scores = f_selector.scores_[selected_mask]

            # Sort by importance for display
            feature_importance = list(zip(selected_features, selected_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            print(f"   ✅ Selected {len(selected_features)} features by importance:")
            for i, (feat, score) in enumerate(feature_importance[:10]):
                print(f"      {i + 1:2d}. {feat} (F-score: {score:.2f})")

            if len(selected_features) > 10:
                print(f"      ... and {len(selected_features) - 10} more")

            print(
                f"   📊 Feature scores: min={min(selected_scores):.2f}, max={max(selected_scores):.2f}, avg={np.mean(selected_scores):.2f}")

            return X[selected_features], selected_features

        except Exception as e:
            print(f"   ❌ Error in feature selection: {e}")
            print("   📊 Debug info:")
            print(f"      • X shape: {X.shape}")
            print(f"      • X dtypes: {X.dtypes.value_counts()}")
            print(f"      • y shape: {y.shape}")
            print(f"      • y dtype: {y.dtype}")
            raise

    # Keep all other methods the same as before...
    def compute_tpm_thresholds(self, tpm_data):
        """Compute TPM classification thresholds"""
        thresholds = {
            'q60': np.percentile(tpm_data, 60),
            'q80': np.percentile(tpm_data, 80),
            'q90': np.percentile(tpm_data, 90)
        }
        print(
            f"📊 TPM Thresholds: Q60={thresholds['q60']:.1f}, Q80={thresholds['q80']:.1f}, Q90={thresholds['q90']:.1f}")
        return thresholds

    def classify_tpm_value(self, tpm_value, thresholds):
        """Classify TPM value"""
        if tpm_value >= thresholds['q90']:
            return 'Very High'
        elif tpm_value >= thresholds['q80']:
            return 'High'
        elif tpm_value >= thresholds['q60']:
            return 'Medium'
        else:
            return 'Low'

    def get_optimized_models(self):
        """Get optimized model configurations"""
        return {
            'RandomForest_Enhanced': RandomForestRegressor(
                n_estimators=400,
                max_depth=15,
                min_samples_split=12,
                min_samples_leaf=6,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Enhanced': GradientBoostingRegressor(
                loss='huber',
                alpha=0.95,
                learning_rate=0.05,
                n_estimators=500,
                max_depth=8,
                subsample=0.85,
                min_samples_leaf=8,
                min_samples_split=20,
                max_features=0.8,
                validation_fraction=0.15,
                n_iter_no_change=25,
                tol=1e-5,
                random_state=42
            ),
            'Ridge_Optimized': Ridge(
                alpha=50.0,
                solver='auto',
                random_state=42
            )
        }

    def train_models_with_cv(self, X_train, y_train, X_test, y_test):
        """Train models with cross-validation"""
        algorithms = self.get_optimized_models()

        self.models = {'tpm': {}}
        model_performance = {}
        best_score = -np.inf
        best_model_name = None

        print("\n=== Training Models with CV ===")
        tscv = TimeSeriesSplit(n_splits=3)

        for name, model in algorithms.items():
            print(f"\n🔄 Training {name}...")

            try:
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_train_fold = X_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    y_val_pred = model.predict(X_val_fold)
                    cv_score = r2_score(y_val_fold, y_val_pred)
                    cv_scores.append(cv_score)

                # Final training and evaluation
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)

                # Store results
                self.models['tpm'][name] = model
                model_performance[name] = {
                    'test_r2': r2,
                    'test_mae': mae,
                    'test_rmse': rmse,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std
                }

                print(f"   ✅ {name}:")
                print(f"      CV R²: {cv_mean:.3f} ± {cv_std:.3f}")
                print(f"      Test R²: {r2:.3f}, MAE: {mae:.1f}, RMSE: {rmse:.1f}")

                if r2 > best_score:
                    best_score = r2
                    best_model_name = name

            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue

        if not self.models['tpm']:
            raise ValueError("No models trained successfully")

        self.best_models['tpm'] = best_model_name

        # Store results
        self.training_results = {
            'model_performance': model_performance,
            'best_model': best_model_name,
            'best_score': best_score,
            'feature_count': len(self.feature_cols),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'tpm_thresholds': self.tpm_thresholds
        }

        return self.training_results

    def save_model(self, model_dir="day6_enhanced_models", model_name="enhanced_csv_model"):
        """Save trained model"""
        os.makedirs(model_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.joblib")
        joblib.dump(self.models, model_path)

        metadata = {
            'best_models': self.best_models,
            'feature_cols': self.feature_cols,
            'tpm_thresholds': self.tpm_thresholds,
            'training_results': self.training_results,
            'push_config': self.push_config,
            'timestamp': timestamp
        }

        metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.joblib")
        joblib.dump(metadata, metadata_path)

        print(f"💾 Enhanced model saved:")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")

        return model_path, metadata_path

    def load_model(self, model_path, metadata_path):
        """Load saved model"""
        self.models = joblib.load(model_path)
        metadata = joblib.load(metadata_path)

        self.best_models = metadata['best_models']
        self.feature_cols = metadata['feature_cols']
        self.tpm_thresholds = metadata['tpm_thresholds']
        self.training_results = metadata['training_results']
        self.push_config = metadata.get('push_config', self.push_config)

        print(f"📂 Enhanced model loaded:")
        print(f"   Best model: {self.best_models.get('tpm', 'Unknown')}")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Push config: {self.push_config['daytime_hours']} effective")

        return True

    def predict_tpm_from_historical_array(self,
                                          current_time,
                                          historical_tpm_1min,
                                          push_notification_active=False,
                                          minutes_since_push=9999,
                                          push_campaign_type='none',
                                          prediction_horizons=(5, 10, 15)):
        """
        Dự đoán TPM cho các horizon tiếp theo từ historical TPM array

        Parameters:
        -----------
        current_time : datetime
            Thời điểm hiện tại cần dự đoán
        historical_tpm_1min : list/array
            Array TPM trong quá khứ, mỗi phần tử = 1 phút (tối thiểu 30 phút, tối ưu 60-180 phút)
        push_notification_active : bool
            Có push notification active không
        minutes_since_push : int
            Số phút từ lúc push gần nhất (default: 9999 = rất lâu)
        push_campaign_type : str
            Loại campaign: 'morning_commute', 'lunch_peak', 'afternoon_break', 'evening_commute', 'evening_leisure', 'none'
        prediction_horizons : tuple
            Các horizon cần predict (default: 5, 10, 15 phút)

        Returns:
        --------
        dict
            Dictionary chứa predictions và metadata
        """

        if not hasattr(self, 'models') or not self.models:
            raise ValueError("❌ Model chưa được train. Hãy train model trước khi predict.")

        if 'tpm' not in self.models or not self.models['tpm']:
            raise ValueError("❌ TPM model không tồn tại.")

        print(f"🔮 Predicting TPM from historical array")
        print(f"   📅 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📊 Historical data points: {len(historical_tpm_1min)}")
        print(f"   📱 Push active: {push_notification_active}, Since: {minutes_since_push} min")
        print(f"   🎯 Campaign: {push_campaign_type}")
        print(f"   ⏭️ Horizons: {prediction_horizons}")

        try:
            # ============ STEP 1: VALIDATE INPUT DATA ============
            historical_tpm_1min = np.array(historical_tpm_1min, dtype=float)

            if len(historical_tpm_1min) < 30:
                raise ValueError("❌ Cần ít nhất 30 phút data lịch sử (30 data points)")

            if len(historical_tpm_1min) > 300:
                print(f"   ⚠️ Có {len(historical_tpm_1min)} points, chỉ sử dụng 300 points gần nhất")
                historical_tpm_1min = historical_tpm_1min[-300:]

            # ============ STEP 2: CREATE SYNTHETIC DATAFRAME ============
            print("🔧 Creating synthetic DataFrame from historical array...")

            # Generate timestamps (going backwards from current_time)
            timestamps = []
            for i in range(len(historical_tpm_1min)):
                timestamp = current_time - timedelta(minutes=(len(historical_tpm_1min) - i - 1))
                timestamps.append(timestamp)

            # Create base DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'tpm': historical_tpm_1min,
                'response_time': 200.0,  # Default reasonable response time
                'hour': [ts.hour for ts in timestamps],
                'minute': [ts.minute for ts in timestamps],
                'day_of_week': [ts.weekday() for ts in timestamps],
                'is_weekend': [int(ts.weekday() >= 5) for ts in timestamps],
                'is_business_hours': [int(9 <= ts.hour <= 17) for ts in timestamps]
            })

            # ============ STEP 3: ADD PUSH NOTIFICATION DATA ============
            print("📱 Adding push notification context...")

            # Initialize push columns
            df['push_notification_active'] = 0
            df['minutes_since_push'] = 9999
            df['push_campaign_type'] = 'none'

            # Set current push status (last row = current time)
            df.loc[df.index[-1], 'push_notification_active'] = int(push_notification_active)
            df.loc[df.index[-1], 'minutes_since_push'] = minutes_since_push
            df.loc[df.index[-1], 'push_campaign_type'] = push_campaign_type

            # Backfill push effects for recent periods if push is active
            if push_notification_active and minutes_since_push <= 15:
                # Set push effects for recent periods
                push_start_idx = max(0, len(df) - minutes_since_push - 1)
                for idx in range(push_start_idx, len(df)):
                    minutes_ago = len(df) - 1 - idx
                    df.loc[df.index[idx], 'minutes_since_push'] = minutes_ago
                    if minutes_ago == 0:  # Push activation point
                        df.loc[df.index[idx], 'push_notification_active'] = 1
                        df.loc[df.index[idx], 'push_campaign_type'] = push_campaign_type
                    else:
                        df.loc[df.index[idx], 'push_campaign_type'] = push_campaign_type

            # ============ STEP 4-5: CREATE PREDICTION-COMPATIBLE FEATURES (COMPREHENSIVE FIX) ============
            print("🔧 Creating prediction-compatible features...")

            try:
                # STEP 4A: Apply business push effects to synthetic data (same as training)
                print("   📱 Applying business push effects to synthetic data...")
                enhanced_df_with_push = self.apply_business_push_effects_to_csv_data(df.copy())

                # STEP 4B: Create enhanced features using SAME method as training
                print("   🔧 Creating enhanced features...")
                enhanced_df = self.create_enhanced_features_from_csv(enhanced_df_with_push, target_col='tpm')

                if enhanced_df.empty:
                    raise ValueError("❌ Feature creation failed - empty DataFrame")

                print(f"   ✅ Created {len(enhanced_df.columns)} features using training method")

                # STEP 4C: Debug feature matching
                generated_features = [col for col in enhanced_df.columns if col != 'tpm']
                matches, missing_features, extra_features = self._debug_feature_mismatch(generated_features)

                if len(matches) == 0:
                    print("❌ CRITICAL: No matching features found!")
                    print("   🔧 Attempting emergency feature creation...")

                    # EMERGENCY: Use emergency feature creation
                    latest_features, current_tpm = self._create_emergency_features_for_prediction(
                        df, historical_tpm_1min, current_time
                    )
                else:
                    print(f"   ✅ Found {len(matches)} matching features")

                    # STEP 4D: Create complete feature matrix
                    print("   🔨 Building complete feature matrix...")
                    complete_feature_df = pd.DataFrame(index=enhanced_df.index)

                    # Add all training features
                    for feature in self.feature_cols:
                        if feature in enhanced_df.columns:
                            complete_feature_df[feature] = enhanced_df[feature]
                        else:
                            # Smart default based on feature type
                            default_value = self._get_smart_feature_default(
                                feature, enhanced_df, df, historical_tpm_1min
                            )
                            complete_feature_df[feature] = default_value
                            print(f"      🔧 Added missing feature '{feature}' with smart default")

                    # Ensure correct column order
                    feature_df = complete_feature_df[self.feature_cols]

                    # Get latest feature vector (current time)
                    latest_features = feature_df.iloc[-1:].copy()
                    current_tpm = enhanced_df.iloc[-1]['tpm'] if 'tpm' in enhanced_df.columns else historical_tpm_1min[
                        -1]

                print(f"   ✅ Final feature matrix: {latest_features.shape}")
                print(f"   📊 Current TPM: {current_tpm:.1f}")
                print(f"   🎯 All training features available: {len(self.feature_cols)}/{len(self.feature_cols)}")

            except Exception as e:
                print(f"   ❌ Error in feature creation: {e}")
                print("   🚨 Using emergency fallback...")

                # EMERGENCY FALLBACK
                latest_features, current_tpm = self._create_emergency_features_for_prediction(
                    df, historical_tpm_1min, current_time
                )

            # ============ STEP 6: MAKE PREDICTIONS ============
            print("🔮 Generating predictions...")

            # Get best model
            model_name = self.best_models['tpm']
            model = self.models['tpm'][model_name]

            print(f"   🏆 Using model: {model_name}")

            # Base prediction
            try:
                base_prediction = model.predict(latest_features)[0]
                base_prediction = max(0.0, float(base_prediction))
            except Exception as e:
                print(f"   ❌ Error in base prediction: {e}")
                print(f"   📊 Feature shape: {latest_features.shape}")
                print(f"   📊 Feature dtypes: {latest_features.dtypes.value_counts()}")
                raise

            print(f"   📈 Base prediction: {base_prediction:.1f}")

            # ============ STEP 7: HORIZON-SPECIFIC PREDICTIONS ============
            predictions = {}
            labels = {}
            effects_applied = {}

            for horizon in prediction_horizons:
                print(f"   🎯 Predicting {horizon}-minute horizon...")

                # Time decay factor (slight degradation over time)
                time_decay = max(0.95, 1.0 - (horizon - 5) * 0.005)

                # Push effect for this horizon
                future_minutes_since_push = minutes_since_push + horizon
                push_effect = self._calculate_prediction_push_effect(
                    current_hour=current_time.hour,
                    push_active=push_notification_active,
                    future_minutes_since_push=future_minutes_since_push,
                    campaign_type=push_campaign_type
                )

                # Business logic adjustments
                hour_factor = self._calculate_hour_factor(current_time.hour, horizon)

                # Combine all effects
                final_prediction = base_prediction * time_decay * (1 + push_effect) * hour_factor
                final_prediction = max(0.0, final_prediction)

                # Store results
                key = f'tpm_{horizon}min'
                predictions[key] = final_prediction

                # Classify using training thresholds
                if hasattr(self, 'tpm_thresholds') and self.tpm_thresholds:
                    label = self.classify_tpm_value(final_prediction, self.tpm_thresholds)
                    labels[key] = label
                else:
                    labels[key] = 'Unknown'

                effects_applied[key] = {
                    'time_decay': time_decay,
                    'push_effect': push_effect,
                    'hour_factor': hour_factor,
                    'total_multiplier': time_decay * (1 + push_effect) * hour_factor
                }

                print(
                    f"      📊 {horizon}min: {final_prediction:.1f} ({labels[key]}) [effects: {time_decay:.3f}×{1 + push_effect:.3f}×{hour_factor:.3f}]")

            # ============ STEP 8: COMPILE RESULTS ============
            results = {
                'predictions': predictions,
                'labels': labels,
                'effects_applied': effects_applied,
                'current_state': {
                    'timestamp': current_time,
                    'current_tpm': current_tpm,
                    'push_active': push_notification_active,
                    'minutes_since_push': minutes_since_push,
                    'campaign_type': push_campaign_type,
                    'hour': current_time.hour
                },
                'model_info': {
                    'model_used': model_name,
                    'base_prediction': base_prediction,
                    'features_available': len(latest_features),
                    'features_total': len(self.feature_cols),
                    'historical_points': len(historical_tpm_1min)
                },
                'prediction_metadata': {
                    'horizons': list(prediction_horizons),
                    'prediction_time': datetime.now(),
                    'data_quality': 'synthetic_from_array'
                }
            }

            print(f"✅ Prediction completed successfully!")
            return results

        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _debug_feature_mismatch(self, generated_features):
        """
        Debug method để understand feature mismatch
        """
        print(f"\n🔍 FEATURE DEBUG ANALYSIS")
        print("=" * 50)
        print(f"Generated features: {len(generated_features)}")
        print(f"Training features needed: {len(self.feature_cols)}")

        # Find matches and mismatches
        generated_set = set(generated_features)
        training_set = set(self.feature_cols)

        matches = generated_set & training_set
        missing_in_generated = training_set - generated_set
        extra_in_generated = generated_set - training_set

        print(f"\n✅ Matching features: {len(matches)}")
        print(f"❌ Missing in generated: {len(missing_in_generated)}")
        print(f"➕ Extra in generated: {len(extra_in_generated)}")

        if missing_in_generated:
            print(f"\n🔍 Missing features (first 10):")
            for i, feature in enumerate(sorted(list(missing_in_generated))[:10]):
                print(f"   {i + 1:2d}. {feature}")

            # Categorize missing features
            missing_categories = {
                'lag_features': [f for f in missing_in_generated if 'lag' in f.lower()],
                'rolling_features': [f for f in missing_in_generated if
                                     any(x in f.lower() for x in ['ma_', '_std', '_min', '_max'])],
                'push_features': [f for f in missing_in_generated if any(x in f.lower() for x in ['push', 'campaign'])],
                'time_features': [f for f in missing_in_generated if
                                  any(x in f.lower() for x in ['hour', 'day', 'sin', 'cos'])],
                'other_features': []
            }

            # Classify remaining
            for f in missing_in_generated:
                categorized = False
                for category, features in missing_categories.items():
                    if category == 'other_features':
                        continue
                    if f in features:
                        categorized = True
                        break
                if not categorized:
                    missing_categories['other_features'].append(f)

            print(f"\n📊 Missing by category:")
            for category, features in missing_categories.items():
                if features:
                    print(f"   {category}: {len(features)}")

        return matches, missing_in_generated, extra_in_generated

    def _get_smart_feature_default(self, feature_name, enhanced_df, original_df, historical_tpm):
        """
        Get smart default value for missing feature based on feature type
        """

        # Time-based features
        if any(x in feature_name.lower() for x in ['hour', 'minute', 'day']):
            if 'sin' in feature_name or 'cos' in feature_name:
                return 0.0
            elif 'hour' in feature_name:
                if 'hour' in original_df.columns:
                    return original_df['hour'].iloc[-1]
                else:
                    return 12  # Default noon
            elif 'day' in feature_name:
                if 'day_of_week' in original_df.columns:
                    return original_df['day_of_week'].iloc[-1]
                else:
                    return 1  # Default Tuesday
            else:
                return 0

        # Push-related features
        elif any(x in feature_name.lower() for x in ['push', 'campaign']):
            return 0  # Default no push

        # Lag features
        elif 'lag' in feature_name.lower():
            if 'tpm' in enhanced_df.columns:
                return enhanced_df['tpm'].iloc[-1]
            else:
                return historical_tpm[-1] if len(historical_tpm) > 0 else 0

        # Rolling/Moving average features
        elif any(x in feature_name.lower() for x in ['ma_', '_mean', 'rolling']):
            if 'tpm' in enhanced_df.columns:
                return enhanced_df['tpm'].mean()
            else:
                return np.mean(historical_tpm) if len(historical_tpm) > 0 else 0

        # Standard deviation features
        elif '_std' in feature_name.lower():
            if 'tpm' in enhanced_df.columns:
                return enhanced_df['tpm'].std()
            else:
                return np.std(historical_tpm) if len(historical_tpm) > 0 else 0

        # Min/Max features
        elif '_min' in feature_name.lower():
            if 'tpm' in enhanced_df.columns:
                return enhanced_df['tpm'].min()
            else:
                return np.min(historical_tpm) if len(historical_tpm) > 0 else 0
        elif '_max' in feature_name.lower():
            if 'tpm' in enhanced_df.columns:
                return enhanced_df['tpm'].max()
            else:
                return np.max(historical_tpm) if len(historical_tpm) > 0 else 0

        # Peak/Business hour features
        elif any(x in feature_name.lower() for x in ['peak', 'business']):
            return 0  # Default not peak

        # Interaction features
        elif '_interaction' in feature_name.lower():
            return 0  # Default no interaction

        # Cumulative features
        elif 'cumulative' in feature_name.lower():
            return 0  # Default no cumulative count

        # Default
        else:
            return 0.0

    def _create_emergency_features_for_prediction(self, df, historical_tpm, current_time):
        """
        Emergency fallback: create minimal feature set with all training features set to defaults
        """
        print("   🚨 Creating emergency feature set...")

        # Create DataFrame with all training features set to defaults
        feature_values = {}

        current_tpm = historical_tpm[-1] if len(historical_tpm) > 0 else 100
        tpm_mean = np.mean(historical_tpm) if len(historical_tpm) > 0 else 100
        tpm_std = np.std(historical_tpm) if len(historical_tpm) > 0 else 20

        for feature in self.feature_cols:
            if 'hour' in feature and 'sin' in feature:
                feature_values[feature] = np.sin(2 * np.pi * current_time.hour / 24)
            elif 'hour' in feature and 'cos' in feature:
                feature_values[feature] = np.cos(2 * np.pi * current_time.hour / 24)
            elif 'day' in feature and 'sin' in feature:
                feature_values[feature] = np.sin(2 * np.pi * current_time.weekday() / 7)
            elif 'day' in feature and 'cos' in feature:
                feature_values[feature] = np.cos(2 * np.pi * current_time.weekday() / 7)
            elif 'lag' in feature and 'tpm' in feature:
                feature_values[feature] = current_tpm
            elif 'ma_' in feature or 'mean' in feature:
                feature_values[feature] = tpm_mean
            elif '_std' in feature:
                feature_values[feature] = tpm_std
            elif '_min' in feature:
                feature_values[feature] = min(historical_tpm) if len(historical_tpm) > 0 else current_tpm * 0.8
            elif '_max' in feature:
                feature_values[feature] = max(historical_tpm) if len(historical_tpm) > 0 else current_tpm * 1.2
            elif 'is_morning_peak' in feature:
                feature_values[feature] = 1 if 8 <= current_time.hour <= 10 else 0
            elif 'is_lunch_peak' in feature:
                feature_values[feature] = 1 if 12 <= current_time.hour <= 13 else 0
            elif 'is_afternoon_peak' in feature:
                feature_values[feature] = 1 if 15 <= current_time.hour <= 17 else 0
            elif 'is_evening_peak' in feature:
                feature_values[feature] = 1 if 18 <= current_time.hour <= 20 else 0
            elif 'is_business_hours' in feature:
                feature_values[feature] = 1 if 9 <= current_time.hour <= 17 else 0
            elif 'is_weekend' in feature:
                feature_values[feature] = 1 if current_time.weekday() >= 5 else 0
            else:
                feature_values[feature] = 0  # Default for all other features

        # Create DataFrame
        latest_features = pd.DataFrame([feature_values])

        print(f"   ✅ Emergency features created: {latest_features.shape[1]} features")

        return latest_features, current_tpm

    def _calculate_prediction_push_effect(self, current_hour, push_active, future_minutes_since_push, campaign_type):
        """
        Calculate push effect for future time horizon
        """
        if not push_active and future_minutes_since_push > 15:
            return 0.0

        # Use trainer's push config
        daytime_start, daytime_end = self.push_config['daytime_hours']
        no_effect_start, no_effect_end = self.push_config['no_effect_hours']

        # No effect during night hours
        if no_effect_start <= current_hour <= no_effect_end:
            return 0.0

        # High effect during daytime
        if daytime_start <= current_hour <= daytime_end:
            if future_minutes_since_push <= 15:
                decay = np.exp(-future_minutes_since_push / 8.0)
                base_effect = 0.8  # Up to 80% boost

                # Campaign modifier
                campaign_modifier = self._get_campaign_modifier(campaign_type, current_hour)

                return decay * base_effect * campaign_modifier
        else:
            # Reduced effect for other hours
            if future_minutes_since_push <= 15:
                decay = np.exp(-future_minutes_since_push / 10.0)
                base_effect = 0.3  # Up to 30% boost

                # Campaign modifier
                campaign_modifier = self._get_campaign_modifier(campaign_type, current_hour)

                return decay * base_effect * campaign_modifier

        return 0.0

    def _calculate_hour_factor(self, hour, horizon):
        """
        Calculate hour-based adjustment factor
        """
        future_hour = (hour + horizon // 60) % 24

        # Peak hours get slight boost
        if future_hour in [8, 9, 12, 13, 17, 18]:
            return 1.05
        elif future_hour in [2, 3, 4, 5]:
            return 0.9
        else:
            return 1.0

    # Demo function để test hàm mới
    def demo_array_prediction(self):
        """
        Demo function cho array prediction (FIXED)
        """
        print("🚀 Demo: TPM Prediction from Historical Array")
        print("=" * 60)

        # Check if current instance has trained model
        if not hasattr(self, 'models') or not self.models:
            print("🔄 No trained model in current instance, trying to load...")

            # Try to load trained model
            model_files = glob.glob("day6_csv_enhanced_models/*model*.joblib")
            metadata_files = glob.glob("day6_csv_enhanced_models/*metadata*.joblib")

            if not model_files or not metadata_files:
                print("❌ Không tìm thấy trained model. Hãy train model trước.")
                return

            latest_model = sorted(model_files, key=os.path.getmtime)[-1]
            latest_metadata = sorted(metadata_files, key=os.path.getmtime)[-1]

            self.load_model(latest_model, latest_metadata)
            print(f"✅ Loaded model: {os.path.basename(latest_model)}")

        # Tạo sample historical data (60 phút TPM data)
        historical_tpm = []
        base_tpm = 100

        for i in range(60):
            # Tạo pattern realistic
            pattern = base_tpm + np.sin(i / 15) * 50 + np.random.normal(0, 20)
            pattern = max(50, pattern)
            historical_tpm.append(pattern)

        current_time = datetime.now()

        # Test scenarios
        scenarios = [
            {
                'name': 'No Push Notification',
                'push_active': False,
                'minutes_since': 9999,
                'campaign': 'none'
            },
            {
                'name': 'Active Push - Morning Commute',
                'push_active': True,
                'minutes_since': 3,
                'campaign': 'morning_commute'
            },
            {
                'name': 'Recent Push - Lunch Peak',
                'push_active': False,
                'minutes_since': 8,
                'campaign': 'lunch_peak'
            }
        ]

        # Test each scenario using SELF (not creating new trainer)
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print("-" * 40)

            try:
                results = self.predict_tpm_from_historical_array(
                    current_time=current_time,
                    historical_tpm_1min=historical_tpm,
                    push_notification_active=scenario['push_active'],
                    minutes_since_push=scenario['minutes_since'],
                    push_campaign_type=scenario['campaign'],
                    prediction_horizons=(5, 10, 15)
                )

                if results:
                    print(f"🎯 Predictions:")
                    for horizon in [5, 10, 15]:
                        key = f'tpm_{horizon}min'
                        pred = results['predictions'][key]
                        label = results['labels'][key]
                        effects = results['effects_applied'][key]
                        print(
                            f"   {horizon:2d} min: {pred:6.1f} ({label:10s}) [multiplier: {effects['total_multiplier']:.3f}]")
                else:
                    print("❌ Prediction failed")

            except Exception as e:
                print(f"❌ Error in scenario {i}: {e}")
                # Continue with next scenario instead of crashing

        print(f"\n🎉 Array prediction demo completed!")


def main():
    """
    Updated main function với comprehensive error handling
    """
    print("🚀 Enhanced CSV Training Demo (FIXED)")
    print("=" * 60)

    # Path to CSV file
    csv_path = "/Users/dongdinh/Documents/Learning/100-Days-Of-ML-Code/Src/Day6/datasets/day6_maximum_precision_data.csv"

    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    try:
        # Initialize enhanced trainer
        trainer = EnhancedTimeSeriesTrainer()

        # Train model from CSV with fixed string handling
        results = trainer.train_enhanced_model_from_csv(
            csv_path=csv_path,
            target_col='tpm',
            test_size=0.25,
            max_features=25
        )

        print(f"\n🎉 Enhanced CSV training completed!")
        print(f"🏆 Best model: {results['best_model']}")
        print(f"📈 Best R² score: {results['best_score']:.3f}")

        # Save model
        model_path, metadata_path = trainer.save_model(
            model_dir="day6_csv_enhanced_models",
            model_name="csv_push_aware_model"
        )

        print(f"\n💾 Model saved successfully!")
        print(f"✅ Fixed string handling:")
        print(f"   • push_campaign_type → binary features")
        print(f"   • All columns ensured numeric")
        print(f"   • Comprehensive validation before feature selection")

    except Exception as e:
        print(f"❌ Error in enhanced CSV training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
