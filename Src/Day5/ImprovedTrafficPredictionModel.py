import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os

# Load .env file manually
def load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded .env from {env_path}")
    else:
        print(f"⚠️ .env file not found at {env_path}")

# Load environment variables
load_env_file()

warnings.filterwarnings('ignore')


class ImprovedTrafficPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_models = {}
        self.feature_importance = {}
        self.data_file_path = "/Users/dongdinh/Documents/Learning/100-Days-Of-ML-Code/Src/Day5/datasets/newrelic_weekend_traffic_enhanced.csv"

    def load_real_data(self):
        """Load New Relic traffic data từ CSV file thật"""
        try:
            print(f"📂 Loading data from: {self.data_file_path}")
            df = pd.read_csv(self.data_file_path)

            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'timestamp_to' in df.columns:
                df['timestamp_to'] = pd.to_datetime(df['timestamp_to'], errors='coerce')

            # Remove rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            print(f"✅ Loaded {len(df)} records")
            print(f"📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Analyze granularity distribution
            self.analyze_granularity_distribution(df)

            # Store enhanced data for later use
            self.enhanced_data = df

            return df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def analyze_granularity_distribution(self, df):
        """Phân tích distribution của granularity trong data"""
        if 'interval_minutes' in df.columns:
            granularity_stats = df['interval_minutes'].value_counts().sort_index()
            print(f"\n📊 Granularity Distribution:")
            for interval, count in granularity_stats.items():
                pct = (count / len(df)) * 100
                quality = "🟢 High-value" if interval <= 5 else "🟡 Medium" if interval <= 15 else "🔴 Low-detail"
                print(f"   {interval:2.0f} min intervals: {count:4d} records ({pct:5.1f}%) {quality}")

        # Analyze collection strategies
        if 'collection_strategy' in df.columns:
            strategy_stats = df['collection_strategy'].value_counts()
            print(f"\n🛠️ Collection Strategy Distribution:")
            for strategy, count in strategy_stats.items():
                pct = (count / len(df)) * 100
                print(f"   {strategy}: {count:4d} records ({pct:5.1f}%)")

    def create_granularity_adaptive_features(self, data, target_col='tpm'):
        """Tạo features thích ứng với granularity khác nhau - LEAK-SAFE VERSION"""
        df = data.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        print("🔒 Creating leak-safe temporal and lag features...")

        # ============ SAFE TEMPORAL FEATURES ============
        print("🕐 Creating temporal features...")

        # Basic temporal features (safe)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['timestamp'].dt.quarter

        # Business logic features
        df['is_working_hour'] = ((df['day_of_week'] < 5) &
                                 (df['hour'].between(8, 18))).astype(int)
        df['is_peak_hour'] = df['hour'].isin([9, 10, 11, 14, 15, 16]).astype(int)
        df['is_night_hour'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        df['is_lunch_hour'] = df['hour'].isin([12, 13]).astype(int)

        # IMPORTANT: Cyclical encoding để model hiểu patterns
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # ============ SAFE LAG FEATURES (NO LEAKAGE) ============
        print("📊 Creating safe lag features...")

        # CHỈ sử dụng lag >= 1 để tránh data leakage
        # Giảm số lượng lag features để tránh overfitting
        safe_lags = [1, 2, 3, 6, 12, 24]  # Reduced from many lags

        for lag in safe_lags:
            # TPM lag features
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

            # Response time lag features (nếu có)
            if 'response_time' in df.columns:
                df[f'response_time_lag_{lag}'] = df['response_time'].shift(lag)

        # ============ SAFE ROLLING FEATURES ============
        print("📈 Creating safe rolling statistics...")

        # CHỈ sử dụng data từ TRƯỚC thời điểm hiện tại
        safe_windows = [3, 6, 12, 24]  # Reduced windows

        for window in safe_windows:
            # Moving averages (shifted to avoid leakage)
            df[f'{target_col}_ma_{window}'] = df[target_col].shift(1).rolling(
                window=window, min_periods=1).mean()

            # Rolling standard deviation
            df[f'{target_col}_std_{window}'] = df[target_col].shift(1).rolling(
                window=window, min_periods=1).std()

        # ============ SAFE CHANGE FEATURES ============
        print("📉 Creating safe trend features...")

        # Change features (safe với shift)
        df[f'{target_col}_change_1'] = df[target_col].diff(1)
        df[f'{target_col}_change_3'] = df[target_col].diff(3)
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_3'] = df[target_col].pct_change(3)

        # ============ DOMAIN-SPECIFIC SAFE FEATURES ============
        print("🎯 Adding domain-specific features...")

        # Push notification features (safe)
        if 'push_notification_active' in df.columns:
            df['push_active'] = df['push_notification_active'].fillna(0)

        if 'minutes_since_push' in df.columns:
            df['minutes_since_push_safe'] = df['minutes_since_push'].fillna(9999)
            df['recent_push'] = (df['minutes_since_push_safe'] <= 60).astype(int)

        # Granularity features (safe)
        if 'interval_minutes' in df.columns:
            df['is_high_freq'] = (df['interval_minutes'] <= 5).astype(int)
            df['is_medium_freq'] = ((df['interval_minutes'] > 5) &
                                    (df['interval_minutes'] <= 15)).astype(int)

        # Data quality indicators (safe)
        if 'data_age_days' in df.columns:
            df['is_fresh_data'] = (df['data_age_days'] <= 1).astype(int)

        if 'ml_weight' in df.columns:
            df['high_weight'] = (df['ml_weight'] >= 4).astype(int)

        # ============ CLEAN UP - REMOVE DANGEROUS COLUMNS ============
        print("🧹 Removing potentially leaky columns...")

        # Columns that can cause data leakage
        danger_columns = [
            # Raw timestamps - đã extract features rồi
            'timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc',

            # String columns - không cần cho ML
            'metric_name', 'day_name', 'data_quality', 'collection_strategy',
            'granularity_category',

            # ID columns có thể leak information
            'sequence_id',

            # Raw push notification columns (đã tạo safe features)
            'push_notification_active', 'minutes_since_push',

            # Other potentially dangerous columns
            'data_age_days', 'ml_weight', 'week_priority'  # Đã tạo binary features
        ]

        existing_danger_cols = [col for col in danger_columns if col in df.columns]
        if existing_danger_cols:
            df = df.drop(columns=existing_danger_cols)
            print(f"   ✅ Removed {len(existing_danger_cols)} potentially leaky columns")

        # ============ HANDLE NaN AND INFINITE VALUES ============
        print("🔧 Cleaning data...")

        # Drop rows with too many NaN (from lag features)
        initial_len = len(df)
        df = df.dropna(thresh=len(df.columns) * 0.7)  # Keep rows with at least 70% non-NaN
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            print(f"   Dropped {dropped_rows} rows with too many NaN values")

        # Fill remaining NaN
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], 0)

        # ============ CONVERT ALL TO NUMERIC ============
        print("🔢 Ensuring all features are numeric...")

        for col in df.columns:
            if col == target_col:
                continue

            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                print(f"   Converting {col} to numeric")
                # Try direct conversion first
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Final cleanup
        df = df.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"✅ Safe feature engineering completed:")
        print(f"   • Final dataset: {len(df)} rows × {len(df.columns)} columns")
        print(f"   • Target: {target_col}")
        print(f"   • Features: {len(df.columns) - 1}")
        print(f"   • No data leakage risk!")

        return df

    def select_important_features(self, X, y, max_features=15):
        """Chọn features quan trọng nhất để tránh overfitting"""
        from sklearn.feature_selection import SelectKBest, f_regression, RFE
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV

        print(f"🎯 Feature Selection: {X.shape[1]} → {max_features} features")

        if X.shape[1] <= max_features:
            print("   No selection needed - already optimal size")
            return X, list(X.columns)

        # Method 1: Statistical F-test
        print("   🔬 Statistical F-test selection...")
        f_selector = SelectKBest(score_func=f_regression, k=max_features)
        X_f = f_selector.fit_transform(X, y)
        f_features = X.columns[f_selector.get_support()].tolist()

        # Method 2: Lasso-based selection
        print("   🔍 Lasso-based selection...")
        try:
            lasso = LassoCV(cv=3, random_state=42, max_iter=1000)
            lasso.fit(X, y)

            # Get features with non-zero coefficients
            lasso_features = X.columns[np.abs(lasso.coef_) > 0.01].tolist()

            if len(lasso_features) > max_features:
                # Sort by absolute coefficient value
                feature_importance = [(feat, abs(coef)) for feat, coef in
                                      zip(X.columns, lasso.coef_) if abs(coef) > 0.01]
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                lasso_features = [feat for feat, _ in feature_importance[:max_features]]

        except Exception as e:
            print(f"   ⚠️ Lasso selection failed: {e}")
            lasso_features = f_features  # Fallback to F-test

        # Method 3: Random Forest-based importance
        print("   🌲 Random Forest importance...")
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            feature_importance = [(feat, imp) for feat, imp in
                                  zip(X.columns, rf.feature_importances_)]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            rf_features = [feat for feat, _ in feature_importance[:max_features]]

        except Exception as e:
            print(f"   ⚠️ RandomForest selection failed: {e}")
            rf_features = f_features  # Fallback

        # Combine selections - vote-based approach
        print("   🗳️ Combining selection methods...")
        feature_votes = {}

        # Add votes from each method
        for feat in f_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1
        for feat in lasso_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1
        for feat in rf_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1

        # Sort by votes, then by F-statistic as tiebreaker
        f_scores = dict(zip(X.columns[f_selector.get_support()],
                            f_selector.scores_[f_selector.get_support()]))

        selected_features = sorted(feature_votes.keys(),
                                   key=lambda x: (feature_votes[x], f_scores.get(x, 0)),
                                   reverse=True)[:max_features]

        print(f"   ✅ Selected {len(selected_features)} features:")
        for i, feat in enumerate(selected_features[:10]):  # Show top 10
            votes = feature_votes[feat]
            print(f"      {i + 1:2d}. {feat} (votes: {votes})")

        if len(selected_features) > 10:
            print(f"      ... and {len(selected_features) - 10} more")

        # Return selected features
        X_selected = X[selected_features]
        return X_selected, selected_features

    def get_optimized_models_for_mixed_granularity(self):
        """Improved models with better hyperparameters for enhanced performance"""
        return {
            'RandomForest_Enhanced': RandomForestRegressor(
                n_estimators=300,  # Increased for better performance
                max_depth=12,  # Balanced depth for complexity vs overfitting
                min_samples_split=15,  # Optimized split threshold
                min_samples_leaf=8,  # Balanced leaf size
                max_features='sqrt',  # Optimal feature sampling
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Enhanced': GradientBoostingRegressor(
                loss='huber',  # Robust to outliers
                alpha=0.95,  # Slightly more conservative
                learning_rate=0.08,  # Improved learning rate
                n_estimators=400,  # Increased for better learning
                max_depth=6,  # Optimal depth
                subsample=0.8,  # Good sampling ratio
                min_samples_leaf=12,  # Balanced leaf constraint
                min_samples_split=25,  # Optimal split constraint
                max_features=0.7,  # Better feature sampling
                validation_fraction=0.15,
                n_iter_no_change=20,  # More patience for convergence
                tol=1e-5,  # Tighter tolerance
                random_state=42
            ),
            'HistGradientBoosting_Enhanced': HistGradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.1,  # Optimal learning rate
                max_iter=300,  # Increased iterations
                max_depth=8,  # Better depth for complex patterns
                min_samples_leaf=8,  # Balanced constraint
                l2_regularization=0.5,  # Moderate regularization
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,  # More patience
                random_state=42
            ),
            'XGBoost_Enhanced': GradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.1,
                n_estimators=350,
                max_depth=7,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.85,
                max_features=0.8,
                random_state=42
            ),
            'Ridge_Optimized': Ridge(
                alpha=100.0,  # Optimized regularization
                solver='auto',
                random_state=42
            )
        }

    def train_and_evaluate_optimized(self):
        """
        Consolidated optimized training with feature selection and time series validation
        """
        print("🚀 Starting optimized training with real New Relic data...")

        # Load and process data
        if not hasattr(self, 'enhanced_data') or self.enhanced_data is None:
            print("📂 Loading and processing data...")
            raw_data = self.load_real_data()
            processed_data = self.create_granularity_adaptive_features(raw_data)

            if processed_data.empty:
                raise ValueError("No processed data available for training")

            # Prepare processed data
            self.X_processed = processed_data.drop(columns=['tpm'])
            self.y_processed = processed_data['tpm']

        # Compute TPM thresholds from training data
        print("\n📊 Computing TPM classification thresholds...")
        if hasattr(self, 'enhanced_data') and self.enhanced_data is not None:
            training_thresholds = self.compute_tpm_thresholds_from_df(self.enhanced_data)
        else:
            temp_df = pd.DataFrame({'tpm': self.y_processed})
            training_thresholds = self.compute_tpm_thresholds_from_df(temp_df)

        # Feature selection for optimal performance
        print("\n🎯 Applying feature selection...")
        n_samples = len(self.X_processed)
        optimal_features = min(20, max(8, n_samples // 150))  # Improved ratio

        print(f"   Data size: {n_samples} samples")
        print(f"   Optimal features: {optimal_features}")

        X_selected, selected_features = self.select_important_features(
            self.X_processed, self.y_processed, max_features=optimal_features
        )

        # Update processed data
        self.X_processed = X_selected
        self.feature_cols = selected_features

        # Ensure all data is numeric
        print("🔢 Final data validation...")
        for col in self.X_processed.columns:
            if not pd.api.types.is_numeric_dtype(self.X_processed[col]):
                self.X_processed[col] = pd.to_numeric(self.X_processed[col], errors='coerce').fillna(0)

        self.X_processed = self.X_processed.fillna(0).replace([np.inf, -np.inf], 0)

        # Time-based split for better validation
        n_samples = len(self.X_processed)
        split_idx = int(n_samples * 0.75)  # 75% train, 25% test

        X_train = self.X_processed.iloc[:split_idx].copy()
        X_test = self.X_processed.iloc[split_idx:].copy()
        y_train = self.y_processed.iloc[:split_idx].copy()
        y_test = self.y_processed.iloc[split_idx:].copy()

        print(f"\nTime-based split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Get enhanced models
        algorithms = self.get_optimized_models_for_mixed_granularity()

        # Initialize storage
        self.models = {'tpm': {}}
        self.scalers = {}
        model_performance = {}
        best_score = -np.inf
        best_model_name = None

        # Train models with cross-validation
        print("\n=== Training Enhanced Models ===")
        tscv = TimeSeriesSplit(n_splits=3)

        for name, model in algorithms.items():
            print(f"\n🔄 Training {name}...")
            cv_scores = []

            try:
                # Cross-validation
                for train_idx, val_idx in tscv.split(X_train):
                    X_train_fold = X_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    y_val_pred = model.predict(X_val_fold)
                    cv_score = r2_score(y_val_fold, y_val_pred)
                    cv_scores.append(cv_score)

                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)

                # Final training on full training set
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Store model and performance
                self.models['tpm'][name] = model
                model_performance[name] = {
                    'test_r2': r2,
                    'test_mae': mae,
                    'test_rmse': rmse,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std
                }

                print(f"   ✅ {name}:")
                print(f"      Test R² = {r2:.3f}, MAE = {mae:.1f}, RMSE = {rmse:.1f}")
                print(f"      CV R² = {cv_mean:.3f} ± {cv_std:.3f}")

                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name

            except Exception as e:
                print(f"   ❌ {name} failed: {e}")
                continue

        if best_model_name:
            print(f"\n🏆 Best model: {best_model_name} (R² = {best_score:.3f})")

            # Set best models
            self.best_models = {'tpm': best_model_name}
            self.model_performance = {'tpm': model_performance}

            # Store training thresholds
            self.tpm_thresholds_stats = training_thresholds

            # Prepare results
            results = {
                'best_models': self.best_models,
                'model_performance': self.model_performance,
                'feature_columns': self.feature_cols,
                'target_columns': ['tpm'],
                'total_records': len(self.X_processed),
                'training_info': {
                    'data_source': self.data_file_path,
                    'num_features': len(self.feature_cols),
                    'tpm_thresholds': training_thresholds
                }
            }

            return results
        else:
            print("❌ No models trained successfully")
            return None

    # ============ REQUIRED HELPER METHODS ============

    def compute_tpm_thresholds_from_df(self, df):
        """
        Tính các ngưỡng quantile để phân loại TPM theo yêu cầu
        và in ra phân phối số lượng bản ghi theo từng nhóm.

        - 0-60%: low
        - 60-80%: medium
        - 80-90%: high
        - 90-100%: very_high
        """
        series = df['tpm'].dropna()
        total = int(len(series))
        if total == 0:
            print("⚠️ No TPM data available to compute thresholds.")
            self.tpm_thresholds_stats = None
            return {'q60': np.nan, 'q80': np.nan, 'q90': np.nan}

        q60 = float(series.quantile(0.60))
        q80 = float(series.quantile(0.80))
        q90 = float(series.quantile(0.90))

        thresholds = {'q60': q60, 'q80': q80, 'q90': q90}

        # Phân loại theo bins dựa trên thresholds
        labels = ['low', 'medium', 'high', 'very_high']
        bins = [-np.inf, q60, q80, q90, np.inf]

        cat = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
        counts = cat.value_counts().reindex(labels, fill_value=0).astype(int)
        percents = (counts / total * 100).round(2)

        # Range hiển thị cho từng nhóm
        s_min = float(series.min())
        s_max = float(series.max())
        ranges = {
            'low': f"[{s_min:.2f}, {q60:.2f}]",
            'medium': f"({q60:.2f}, {q80:.2f}]",
            'high': f"({q80:.2f}, {q90:.2f}]",
            'very_high': f"({q90:.2f}, {s_max:.2f}]",
        }

        # Lưu stats vào class để dùng sau
        stats = {
            k: {'count': int(counts[k]), 'percent': float(percents[k]), 'range': ranges[k]}
            for k in labels
        }
        self.tpm_thresholds_stats = {
            'total_records': total,
            'thresholds': thresholds,
            'distribution': stats
        }

        # In ra tổng hợp để đánh giá
        print("\n📊 TPM Thresholds (training-based):")
        print(f"  - q60: {q60:.2f}, q80: {q80:.2f}, q90: {q90:.2f}")
        print(f"  - Total training records: {total}")
        print("  - Distribution by category:")
        for k in labels:
            info = stats[k]
            print(f"    • {k:10s} | {info['count']:6d} recs "
                  f"({info['percent']:6.2f}%) | range {info['range']}")

        return thresholds


    def classify_tpm_value(self, tpm_value, thresholds=None):
        """Phân loại giá trị TPM thành các levels dựa trên training data thresholds"""

        # Sử dụng thresholds từ training data nếu có
        if thresholds is None:
            if hasattr(self, 'tpm_thresholds_stats') and self.tpm_thresholds_stats:
                thresholds = self.tpm_thresholds_stats['thresholds']
            else:
                # Fallback to default thresholds nếu không có training stats
                print("⚠️ No training-based thresholds available, using defaults")
                thresholds = {'q60': 200, 'q80': 500, 'q90': 1000}

        # Phân loại dựa trên quantile thresholds
        if isinstance(thresholds, dict) and 'q60' in thresholds:
            # New quantile-based classification
            q60 = thresholds['q60']
            q80 = thresholds['q80']
            q90 = thresholds['q90']

            if tpm_value <= q60:
                return 'low'
            elif tpm_value <= q80:
                return 'medium'
            elif tpm_value <= q90:
                return 'high'
            else:
                return 'very_high'
        else:
            # Legacy classification for backward compatibility
            if tpm_value <= thresholds.get('very_low', 50):
                return 'very_low'
            elif tpm_value <= thresholds.get('low', 200):
                return 'low'
            elif tpm_value <= thresholds.get('normal', 500):
                return 'normal'
            elif tpm_value <= thresholds.get('high', 1000):
                return 'high'
            else:
                return 'very_high'

    @property
    def tpm_thresholds(self):
        """Property để truy cập thresholds với lazy loading"""
        if not hasattr(self, '_tpm_thresholds') or self._tpm_thresholds is None:
            if hasattr(self, 'tpm_thresholds_stats') and self.tpm_thresholds_stats:
                self._tpm_thresholds = self.tpm_thresholds_stats['thresholds']
            else:
                self._tpm_thresholds = {'q60': 200, 'q80': 500, 'q90': 1000}
        return self._tpm_thresholds

    @property
    def feature_columns(self):
        """Compatibility property"""
        return getattr(self, 'feature_cols', [])

    def walk_forward_validation(self, train_ratio=0.7, step_size=1):
        """Walk-forward validation để mô phỏng real-time prediction"""
        print("\n=== WALK-FORWARD VALIDATION ===")

        # Đảm bảo data được sort theo thời gian
        if 'timestamp' in self.enhanced_data.columns:
            self.enhanced_data = self.enhanced_data.sort_values('timestamp').reset_index(drop=True)

        n_samples = len(self.X_processed)
        initial_train_size = int(n_samples * train_ratio)

        walk_forward_results = {}

        for name, model in self.models.items():
            print(f"\nRunning walk-forward validation for {name}...")

            predictions = []
            actuals = []

            for i in range(initial_train_size, n_samples, step_size):
                # Training data: từ đầu đến thời điểm i
                X_train = self.X_processed.iloc[:i]
                y_train = self.y_processed.iloc[:i]

                # Test data: thời điểm tiếp theo
                X_test = self.X_processed.iloc[i:i + step_size]
                y_test = self.y_processed.iloc[i:i + step_size]

                if len(X_test) == 0:
                    break

                # Train và predict
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                predictions.extend(pred)
                actuals.extend(y_test.values)

            # Calculate performance
            if len(predictions) > 0:
                mae = mean_absolute_error(actuals, predictions)
                mse = mean_squared_error(actuals, predictions)
                r2 = r2_score(actuals, predictions)

                walk_forward_results[name] = {
                    'mae': mae,
                    'mse': mse,
                    'r2': r2,
                    'predictions': predictions,
                    'actuals': actuals
                }

                print(f"  Walk-forward R²: {r2:.4f}")
                print(f"  Walk-forward MAE: {mae:.2f}")
                print(f"  Number of predictions: {len(predictions)}")

        return walk_forward_results

    def debug_data_info(self):
        """Debug thông tin data để tìm lỗi"""
        print("\n🔍 DEBUG DATA INFORMATION")
        print("=" * 50)

        if hasattr(self, 'X_processed') and self.X_processed is not None:
            print(f"X_processed shape: {self.X_processed.shape}")
            print(f"X_processed dtypes:")
            dtype_counts = self.X_processed.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                print(f"  {dtype}: {count} columns")

            print(f"\nColumns with non-numeric data:")
            for col in self.X_processed.columns:
                if not pd.api.types.is_numeric_dtype(self.X_processed[col]):
                    unique_vals = self.X_processed[col].unique()[:5]  # First 5 unique values
                    print(f"  {col}: {self.X_processed[col].dtype} - samples: {unique_vals}")

            print(f"\nNaN counts:")
            nan_counts = self.X_processed.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if len(nan_cols) > 0:
                print(f"  Columns with NaN: {dict(nan_cols)}")
            else:
                print(f"  No NaN values found")

        if hasattr(self, 'y_processed') and self.y_processed is not None:
            print(f"\ny_processed shape: {self.y_processed.shape}")
            print(f"y_processed dtype: {self.y_processed.dtype}")
            print(f"y_processed NaN count: {self.y_processed.isnull().sum()}")
            print(f"y_processed range: [{self.y_processed.min():.2f}, {self.y_processed.max():.2f}]")

    def save_models(self):
        """Save trained models using new consistent format"""
        import joblib
        from datetime import datetime

        if not hasattr(self, 'models') or not self.models:
            print("⚠️ No models to save")
            return

        if not hasattr(self, 'best_models') or not self.best_models:
            print("⚠️ No best models identified")
            return

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)

        print(f"💾 Saving models with timestamp: {timestamp}")
        print(f"📁 Directory: {models_dir}")

        # Save individual models and scalers
        for target, model_name in self.best_models.items():
            if target in self.models and model_name in self.models[target]:
                # Save model
                model_path = os.path.join(models_dir, f'{target}_{model_name}_{timestamp}.joblib')
                joblib.dump(self.models[target][model_name], model_path)
                print(f"✅ Saved model: {os.path.basename(model_path)}")

                # Save scaler if exists
                if hasattr(self, 'scalers') and self.scalers and target in self.scalers:
                    scaler_path = os.path.join(models_dir, f'scaler_{target}_{timestamp}.joblib')
                    joblib.dump(self.scalers[target], scaler_path)
                    print(f"✅ Saved scaler: {os.path.basename(scaler_path)}")

        # Prepare metadata
        metadata = {
            'timestamp': timestamp,
            'best_models': self.best_models,
            'feature_columns': self.feature_cols if hasattr(self, 'feature_cols') else [],
            'target_columns': list(self.best_models.keys()),
            'model_performance': getattr(self, 'model_performance', {}),
            'data_source': getattr(self, 'data_file_path', None),
            'total_samples': len(self.X_processed) if hasattr(self, 'X_processed') else 0,
            'num_features': len(self.feature_cols) if hasattr(self, 'feature_cols') else 0
        }

        # Save metadata
        metadata_path = os.path.join(models_dir, f'metadata_{timestamp}.joblib')
        joblib.dump(metadata, metadata_path)
        print(f"✅ Saved metadata: {os.path.basename(metadata_path)}")

        print(f"   • Timestamp: {timestamp}")
        print(f"   • Models: {len(self.best_models)}")
        print(f"   • Features: {metadata['num_features']}")

    def demonstrate_predictions(self, results=None):
        """Demo predictions với data có sẵn"""
        print("\n🎯 DEMO: Dự đoán TPM từ dữ liệu có sẵn")
        print("=" * 60)

        if not hasattr(self, 'models') or not self.models:
            print("❌ No trained models available")
            return

        if not hasattr(self, 'X_processed') or self.X_processed is None or self.X_processed.empty:
            print("❌ No processed data available")
            return

        # Use last 10 samples for demonstration
        X_demo = self.X_processed.tail(10)
        y_demo = self.y_processed.tail(10)

        print(f"📊 Using last {len(X_demo)} samples for demonstration")

        # Make predictions with best model
        for target, model_name in self.best_models.items():
            if target in self.models and model_name in self.models[target]:
                model = self.models[target][model_name]
                predictions = model.predict(X_demo)

                print(f"\n🔮 Predictions using {model_name}:")
                print("-" * 50)

                for i, (actual, pred) in enumerate(zip(y_demo.values, predictions)):
                    error = actual - pred
                    error_pct = abs(error / (actual + 1e-6)) * 100
                    print(f"Sample {i+1:2d}: Actual={actual:6.1f}, Predicted={pred:6.1f}, Error={error:+6.1f} ({error_pct:4.1f}%)")

                # Summary statistics
                mae = np.mean(np.abs(y_demo.values - predictions))
                rmse = np.sqrt(np.mean((y_demo.values - predictions) ** 2))
                mape = np.mean(np.abs((y_demo.values - predictions) / (y_demo.values + 1e-6))) * 100

                print(f"\n📈 Performance Summary:")
                print(f"   MAE:  {mae:.2f}")
                print(f"   RMSE: {rmse:.2f}")
                print(f"   MAPE: {mape:.1f}%")

    def plot_learning_curves(self):
        """Plot learning curves for each trained model"""
        print("\n📈 CREATING LEARNING CURVES...")
        print("=" * 60)

        # Kiểm tra prerequisites
        if not hasattr(self, 'models') or not self.models:
            print("❌ No trained models found. Please train models first.")
            return

        if not hasattr(self, 'best_models') or not self.best_models:
            print("❌ No best models identified. Please train models first.")
            return

        if not hasattr(self, 'X_processed') or not hasattr(self, 'y_processed'):
            print("❌ No processed data found. Please train models first.")
            return

        # Tạo thư mục charts nếu chưa có
        charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        # Prepare data - đảm bảo all numeric
        print("🔧 Preparing data for learning curves...")
        X = self.X_processed.copy()

        # Ensure all columns are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')

        X = X.fillna(0)

        print(f"   Data shape: {X.shape}")
        print(f"   Features: {len(X.columns)}")

        # Determine số lượng subplots cần thiết
        num_targets = len(self.best_models)
        if num_targets == 1:
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]  # Make it iterable
        else:
            cols = min(3, num_targets)
            rows = (num_targets + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
            if num_targets == 2:
                axes = axes.flatten()
            elif num_targets > 2:
                axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        fig.suptitle('Learning Curves for Best Models (Real New Relic Data)',
                     fontsize=16, fontweight='bold')

        plot_idx = 0
        for target, best_model_name in self.best_models.items():
            print(f"📊 Creating learning curve for {target} ({best_model_name})...")

            try:
                # Get target values
                if target == 'tpm':
                    y = self.y_processed.copy()
                else:
                    # Fallback nếu có multiple targets
                    y = getattr(self, f'y_{target}', self.y_processed)

                # Get best model
                if target in self.models and best_model_name in self.models[target]:
                    model = self.models[target][best_model_name]
                else:
                    print(f"   ⚠️ Model {best_model_name} not found for {target}, skipping...")
                    continue

                # Prepare model cho learning curve
                from sklearn.base import clone
                model_for_lc = clone(model)

                # Handle scaling nếu cần (cho SVR)
                X_processed = X.copy()
                if 'SVR' in best_model_name:
                    if hasattr(self, 'scalers') and target in self.scalers:
                        scaler = self.scalers[target]
                        X_processed = pd.DataFrame(
                            scaler.transform(X_processed),
                            columns=X_processed.columns,
                            index=X_processed.index
                        )
                    else:
                        # Create scaler on the fly
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_processed = pd.DataFrame(
                            scaler.fit_transform(X_processed),
                            columns=X_processed.columns,
                            index=X_processed.index
                        )

                # Calculate learning curve với robust settings
                print(f"   🔄 Computing learning curve...")

                # Adjust train_sizes based on data size
                data_size = len(X_processed)
                if data_size < 100:
                    train_sizes = np.linspace(0.3, 1.0, 5)
                elif data_size < 1000:
                    train_sizes = np.linspace(0.2, 1.0, 8)
                else:
                    train_sizes = np.linspace(0.1, 1.0, 10)

                train_sizes, train_scores, val_scores = learning_curve(
                    model_for_lc, X_processed, y,
                    train_sizes=train_sizes,
                    cv=min(5, len(X_processed) // 20),  # Adaptive CV folds
                    scoring='r2',
                    n_jobs=1,  # Tránh memory issues
                    random_state=42
                )

                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)

                # Plot on appropriate axis
                ax = axes[plot_idx] if len(axes) > plot_idx else axes[0]

                # Training score
                ax.plot(train_sizes, train_mean, 'o-', color='blue',
                        label=f'Training Score (μ={train_mean[-1]:.3f})', linewidth=2)
                ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                                alpha=0.15, color='blue')

                # Validation score
                ax.plot(train_sizes, val_mean, 'o-', color='red',
                        label=f'Validation Score (μ={val_mean[-1]:.3f})', linewidth=2)
                ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                                alpha=0.15, color='red')

                # Formatting
                ax.set_title(f'{target.upper()}: {best_model_name}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Training Set Size', fontsize=10)
                ax.set_ylabel('R² Score', fontsize=10)
                ax.legend(loc='lower right', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Set y-axis limits for better visualization
                y_min = min(np.min(val_mean - val_std), np.min(train_mean - train_std))
                y_max = max(np.max(val_mean + val_std), np.max(train_mean + train_std))
                ax.set_ylim(max(-1, y_min - 0.05), min(1.05, y_max + 0.05))

                # Add model performance info
                final_train_score = train_mean[-1]
                final_val_score = val_mean[-1]
                gap = final_train_score - final_val_score

                # Color code the gap
                gap_color = 'green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red'
                ax.text(0.02, 0.98, f'Overfitting Gap: {gap:.3f}',
                        transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=gap_color, alpha=0.3))

                print(f"   ✅ Learning curve for {target} completed")
                print(f"      Final Training R²: {final_train_score:.3f}")
                print(f"      Final Validation R²: {final_val_score:.3f}")
                print(f"      Overfitting Gap: {gap:.3f}")

                plot_idx += 1

            except Exception as e:
                print(f"   ❌ Error creating learning curve for {target}: {e}")
                continue

        # Hide unused subplots
        if len(axes) > plot_idx:
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)

        # Adjust layout and save
        plt.tight_layout()

        # Save plot với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(charts_dir, f'learning_curves_newrelic_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n💾 Learning curves saved to: {plot_path}")

        # Show plot
        plt.show()

        print("✅ Learning curves analysis completed!")

        return plot_path

    def create_comprehensive_plots(self):
        """Create comprehensive visualization suite"""
        print("\n📊 CREATING COMPREHENSIVE VISUALIZATION SUITE...")
        print("=" * 70)

        plot_paths = []

        try:
            # 1. Learning Curves
            lc_path = self.plot_learning_curves()
            if lc_path:
                plot_paths.append(lc_path)

            # 2. Feature Importance (if available)
            fi_path = self.plot_feature_importance()
            if fi_path:
                plot_paths.append(fi_path)

            # 3. Prediction vs Actual
            pa_path = self.plot_prediction_vs_actual()
            if pa_path:
                plot_paths.append(pa_path)

            print(f"\n✅ Created {len(plot_paths)} visualization files:")
            for path in plot_paths:
                print(f"   📈 {os.path.basename(path)}")

            return plot_paths

        except Exception as e:
            print(f"❌ Error in comprehensive plots: {e}")
            return plot_paths

    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("🎯 Creating feature importance plots...")

        if not hasattr(self, 'models') or not self.models:
            print("❌ No models available for feature importance")
            return None

        # Tạo thư mục charts
        charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        importance_data = {}

        for target, best_model_name in self.best_models.items():
            if target in self.models and best_model_name in self.models[target]:
                model = self.models[target][best_model_name]

                # Chỉ plot cho models có feature_importances_
                if hasattr(model, 'feature_importances_'):
                    importance_data[f"{target}_{best_model_name}"] = {
                        'importances': model.feature_importances_,
                        'features': self.feature_cols
                    }

        if not importance_data:
            print("⚠️ No tree-based models found for feature importance")
            return None

        # Create subplots
        n_plots = len(importance_data)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 6 * n_plots))
        if n_plots == 1:
            axes = [axes]

        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')

        for idx, (model_name, data) in enumerate(importance_data.items()):
            ax = axes[idx]

            # Sort features by importance
            feature_importance = list(zip(data['features'], data['importances']))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            # Get top 20 features
            top_features = feature_importance[:20]
            features, importances = zip(*top_features)

            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, alpha=0.8)

            # Color bars by importance
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top 20 Features: {model_name}')
            ax.grid(axis='x', alpha=0.3)

            # Add importance values on bars
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{imp:.3f}', va='center', ha='left', fontsize=8)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(charts_dir, f'feature_importance_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"✅ Feature importance plot saved: {os.path.basename(plot_path)}")
        return plot_path

    def plot_prediction_vs_actual(self):
        """Plot prediction vs actual values"""
        print("📊 Creating prediction vs actual plots...")

        if not hasattr(self, 'X_processed') or not hasattr(self, 'y_processed'):
            print("❌ No processed data available")
            return None

        # Tạo thư mục charts
        charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
        os.makedirs(charts_dir, exist_ok=True)

        # Use last 20% of data for visualization
        split_idx = int(len(self.X_processed) * 0.8)
        X_test = self.X_processed.iloc[split_idx:]
        y_test = self.y_processed.iloc[split_idx:]

        fig, axes = plt.subplots(1, len(self.best_models), figsize=(8 * len(self.best_models), 6))
        if len(self.best_models) == 1:
            axes = [axes]

        fig.suptitle('Prediction vs Actual Values', fontsize=16, fontweight='bold')

        for idx, (target, best_model_name) in enumerate(self.best_models.items()):
            if target in self.models and best_model_name in self.models[target]:
                model = self.models[target][best_model_name]

                # Make predictions
                predictions = model.predict(X_test)
                actual = y_test.values if target == 'tpm' else getattr(self, f'y_{target}').iloc[split_idx:].values

                ax = axes[idx]

                # Scatter plot
                ax.scatter(actual, predictions, alpha=0.6, s=30)

                # Perfect prediction line
                min_val = min(actual.min(), predictions.min())
                max_val = max(actual.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

                # Calculate metrics
                r2 = r2_score(actual, predictions)
                mae = mean_absolute_error(actual, predictions)
                rmse = np.sqrt(mean_squared_error(actual, predictions))

                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{target.upper()}: {best_model_name}\nR²={r2:.3f}, MAE={mae:.1f}, RMSE={rmse:.1f}')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(charts_dir, f'prediction_vs_actual_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"✅ Prediction vs actual plot saved: {os.path.basename(plot_path)}")
        return plot_path

    def get_tpm_thresholds(self):
        """Lấy thresholds đã tính toán từ training data"""

        # Ưu tiên sử dụng thresholds từ training data
        if hasattr(self, 'tpm_thresholds_stats') and self.tpm_thresholds_stats:
            return self.tpm_thresholds_stats['thresholds']

        # Nếu chưa có, tính toán từ processed data hiện tại
        if hasattr(self, 'enhanced_data') and self.enhanced_data is not None and 'tpm' in self.enhanced_data.columns:
            print("🔄 Computing TPM thresholds from training data...")
            return self.compute_tpm_thresholds_from_df(self.enhanced_data)

        # Fallback cuối cùng
        print("⚠️ No training data available, using default quantile thresholds")
        return {'q60': 200, 'q80': 500, 'q90': 1000}

    @property
    def tpm_thresholds(self):
        """Property để truy cập thresholds với lazy loading"""
        if not hasattr(self, '_tpm_thresholds') or self._tpm_thresholds is None:
            self._tpm_thresholds = self.get_tpm_thresholds()
        return self._tpm_thresholds

    def predict_from_inputs(self,
                            when: datetime,
                            current_tpm: float,
                            previous_tpms: list = None,
                            response_time: float = None,
                            push_notification_active: int = 0,
                            minutes_since_push: int = 999999,
                            minutes_ahead: tuple = (5, 10, 15),
                            extra_features: dict = None):
        """
        Predict TPM for specific future horizons given a point-in-time context.

        Args:
            when: Target prediction time
            current_tpm: Current TPM value
            previous_tpms: List of previous TPM values [1min_ago, 2min_ago, ...]
            response_time: Current response time
            push_notification_active: 1 if push active, 0 if not
            minutes_since_push: Minutes since last push notification
            minutes_ahead: Tuple of minutes to predict ahead
            extra_features: Additional features dict

        Returns:
            dict: {'predictions': {...}, 'labels': {...}, 'thresholds': {...}}
        """

        if not hasattr(self, 'models') or not self.models or not hasattr(self, 'best_models') or not self.best_models:
            raise ValueError("❌ Models are not loaded/trained. Please train or load models first.")

        if not hasattr(self, 'feature_cols') or not self.feature_cols:
            raise ValueError("❌ Feature columns not available. Please train or load models first.")

        print(f"🔮 Predicting TPM for {when.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Current TPM: {current_tpm}")
        print(f"   Previous TPMs: {previous_tpms}")

        # ============ CHUẨN BỊ TEMPORAL FEATURES ============
        hour = when.hour
        minute = when.minute
        day_of_week = when.weekday()  # Monday = 0
        day_of_month = when.day
        month = when.month
        quarter = (month - 1) // 3 + 1
        is_weekend = int(day_of_week >= 5)

        # Business logic features
        is_working_hour = int((day_of_week < 5) and (8 <= hour <= 18))
        is_peak_hour = int(hour in [9, 10, 11, 14, 15, 16])
        is_night_hour = int(hour in [22, 23, 0, 1, 2, 3, 4, 5])
        is_lunch_hour = int(hour in [12, 13])

        # Cyclical encodings - QUAN TRỌNG cho model
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        minute_sin = np.sin(2 * np.pi * minute / 60)
        minute_cos = np.cos(2 * np.pi * minute / 60)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        # ============ BASE FEATURE DICT ============
        base_features = {
            # Temporal features
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'quarter': quarter,
            'is_weekend': is_weekend,

            # Business features
            'is_working_hour': is_working_hour,
            'is_peak_hour': is_peak_hour,
            'is_night_hour': is_night_hour,
            'is_lunch_hour': is_lunch_hour,

            # Cyclical encodings
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'minute_sin': minute_sin,
            'minute_cos': minute_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'month_sin': month_sin,
            'month_cos': month_cos,
        }

        # ============ PUSH NOTIFICATION FEATURES ============
        # Giới hạn trong khoảng hợp lý
        ms_push = max(0, min(int(minutes_since_push), 10000))

        base_features.update({
            'push_active': int(push_notification_active),
            'minutes_since_push_safe': ms_push,
            'recent_push': int(ms_push <= 60),
        })

        # Response time features
        if response_time is not None:
            base_features['response_time_lag_1'] = float(response_time)

        # ============ LAG FEATURES ============
        # Tạo history array từ previous_tpms + current_tpm
        history = []
        if previous_tpms and len(previous_tpms) > 0:
            # previous_tpms = [1min_ago, 2min_ago, ...]
            # Reverse để có thứ tự thời gian tăng dần
            history = list(reversed(previous_tpms))

        history.append(current_tpm)  # Thêm giá trị hiện tại

        print(f"   History length: {len(history)}")

        # Map lag features
        lag_features = {}
        for col in self.feature_cols:
            if col.startswith('tpm_lag_'):
                try:
                    lag_num = int(col.split('_')[-1])
                    if previous_tpms and len(previous_tpms) >= lag_num:
                        lag_features[col] = float(previous_tpms[lag_num - 1])
                    else:
                        # Fallback to current TPM if not enough history
                        lag_features[col] = float(current_tpm)
                except (ValueError, IndexError):
                    lag_features[col] = float(current_tpm)

        # ============ ROLLING STATISTICS ============
        def safe_rolling_mean(values, window):
            if len(values) == 0:
                return float(current_tpm)
            w = min(window, len(values))
            return float(np.mean(values[-w:]))

        def safe_rolling_std(values, window):
            if len(values) <= 1:
                return 0.0
            w = min(window, len(values))
            return float(np.std(values[-w:], ddof=0))

        # Calculate rolling features for available windows
        rolling_features = {}
        for col in self.feature_cols:
            if '_ma_' in col or '_std_' in col:
                try:
                    # Extract window size from column name
                    if '_ma_' in col:
                        window = int(col.split('_ma_')[-1])
                        rolling_features[col] = safe_rolling_mean(history, window)
                    elif '_std_' in col:
                        window = int(col.split('_std_')[-1])
                        rolling_features[col] = safe_rolling_std(history, window)
                except (ValueError, IndexError):
                    rolling_features[col] = 0.0

        # ============ CHANGE FEATURES ============
        change_features = {}
        for col in self.feature_cols:
            if col.startswith('tpm_change_') or col.startswith('tpm_pct_change_'):
                try:
                    if 'pct_change' in col:
                        period = int(col.split('_')[-1])
                        if len(history) > period:
                            old_val = history[-(period + 1)]
                            new_val = history[-1]
                            if old_val != 0:
                                change_features[col] = (new_val - old_val) / old_val
                            else:
                                change_features[col] = 0.0
                        else:
                            change_features[col] = 0.0
                    else:  # regular change
                        period = int(col.split('_')[-1])
                        if len(history) > period:
                            change_features[col] = history[-1] - history[-(period + 1)]
                        else:
                            change_features[col] = 0.0
                except (ValueError, IndexError):
                    change_features[col] = 0.0

        # ============ COMBINE ALL FEATURES ============
        row_features = {}
        row_features.update(base_features)
        row_features.update(lag_features)
        row_features.update(rolling_features)
        row_features.update(change_features)

        # Add extra features if provided
        if extra_features:
            for k, v in extra_features.items():
                if k in self.feature_cols:
                    row_features[k] = v

        # ============ CREATE FEATURE VECTOR ============
        # Tạo feature vector theo đúng thứ tự của training
        feature_vector = []
        missing_features = []

        for col in self.feature_cols:
            if col in row_features:
                feature_vector.append(row_features[col])
            else:
                feature_vector.append(0.0)  # Default value for missing features
                missing_features.append(col)

        if missing_features:
            print(f"   ⚠️ Missing features filled with 0: {len(missing_features)}")
            if len(missing_features) <= 5:
                print(f"      {missing_features}")

        # Convert to DataFrame
        X_pred = pd.DataFrame([feature_vector], columns=self.feature_cols)

        print(f"   Feature vector shape: {X_pred.shape}")
        print(f"   Sample values: {dict(list(X_pred.iloc[0].items())[:5])}")

        # ============ MAKE PREDICTIONS ============
        predictions = {}
        prediction_details = {}

        # Current model structure - single TPM prediction
        for target in self.best_models.keys():
            best_model_name = self.best_models[target]

            if target in self.models and best_model_name in self.models[target]:
                model = self.models[target][best_model_name]

                try:
                    # Handle SVR scaling
                    if 'SVR' in best_model_name and hasattr(self, 'scalers') and target in self.scalers:
                        X_scaled = self.scalers[target].transform(X_pred)
                        pred = model.predict(X_scaled)[0]
                    else:
                        pred = model.predict(X_pred)[0]

                    # Ensure non-negative prediction
                    pred = max(0.0, float(pred))

                    # Create predictions for each horizon (simulation)
                    # Since we only have one model, we'll simulate different horizons
                    for minutes in minutes_ahead:
                        # Apply time decay factor for longer horizons
                        decay_factor = 1.0 - (minutes - 5) * 0.02  # 2% decay per minute after 5min
                        decay_factor = max(0.8, decay_factor)  # Minimum 80% confidence

                        horizon_pred = pred * decay_factor
                        predictions[f'tpm_{minutes}min'] = horizon_pred

                    prediction_details[target] = {
                        'model': best_model_name,
                        'base_prediction': pred,
                        'horizons': predictions
                    }

                    print(f"   ✅ Prediction using {best_model_name}: {pred:.1f}")

                except Exception as e:
                    print(f"   ❌ Prediction failed for {target}: {e}")
                    continue

        if not predictions:
            raise ValueError("❌ No predictions could be made")

        # ============ CLASSIFY PREDICTIONS ============
        thresholds = self.tpm_thresholds
        labels = {}

        for pred_key, pred_value in predictions.items():
            labels[pred_key] = self.classify_tpm_value(pred_value, thresholds)

        # ============ PREPARE RESULTS ============
        results = {
            'predictions': predictions,
            'labels': labels,
            'thresholds': thresholds,
            'details': {
                'timestamp': when.strftime('%Y-%m-%d %H:%M:%S'),
                'input_tpm': current_tpm,
                'features_used': len(self.feature_cols),
                'model_details': prediction_details
            }
        }

        print(f"   🎯 Predictions: {predictions}")
        print(f"   🏷️ Labels: {labels}")

        return results

    def demo_manual_predictions(self):
        """Demo predictions với input thủ công"""
        print("\n🎯 DEMO: Dự đoán TPM từ input thủ công")
        print("=" * 60)

        if not hasattr(self, 'models') or not self.models:
            print("❌ No trained models available for demo")
            return

        # Demo scenarios
        demo_scenarios = [
            {
                'name': 'Peak Hour - Monday Morning',
                'when': datetime(2024, 12, 23, 10, 30),  # Monday 10:30 AM
                'current_tpm': 850.0,
                'previous_tpms': [800.0, 780.0, 760.0, 720.0, 650.0],
                'response_time': 145.0,
                'push_notification_active': 0,
                'minutes_since_push': 120,
            },
            {
                'name': 'Night Hour - Low Traffic',
                'when': datetime(2024, 12, 23, 2, 15),  # Monday 2:15 AM
                'current_tpm': 45.0,
                'previous_tpms': [52.0, 48.0, 39.0, 44.0, 55.0],
                'response_time': 89.0,
                'push_notification_active': 0,
                'minutes_since_push': 480,
            },
            {
                'name': 'Weekend Peak - Saturday Afternoon',
                'when': datetime(2024, 12, 21, 15, 45),  # Saturday 3:45 PM
                'current_tpm': 1200.0,
                'previous_tpms': [1180.0, 1150.0, 1100.0, 1050.0, 980.0],
                'response_time': 210.0,
                'push_notification_active': 1,
                'minutes_since_push': 5,
            },
            {
                'name': 'Lunch Hour - Working Day',
                'when': datetime(2024, 12, 24, 12, 30),  # Tuesday 12:30 PM
                'current_tpm': 650.0,
                'previous_tpms': [620.0, 600.0, 580.0, 590.0, 610.0],
                'response_time': 165.0,
                'push_notification_active': 0,
                'minutes_since_push': 45,
            }
        ]

        for i, scenario in enumerate(demo_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print("-" * 50)

            try:
                # Make prediction
                result = self.predict_from_inputs(
                    when=scenario['when'],
                    current_tpm=scenario['current_tpm'],
                    previous_tpms=scenario['previous_tpms'],
                    response_time=scenario['response_time'],
                    push_notification_active=scenario['push_notification_active'],
                    minutes_since_push=scenario['minutes_since_push'],
                    minutes_ahead=(5, 10, 15, 30)
                )

                # Display results
                print(f"\n📊 PREDICTION RESULTS:")
                print(f"   Context: {scenario['when'].strftime('%A %H:%M')} | Current TPM: {scenario['current_tpm']}")

                print(f"\n🔮 Predictions:")
                for horizon, pred in result['predictions'].items():
                    label = result['labels'][horizon]
                    print(f"   • {horizon}: {pred:6.1f} TPM ({label})")

                print(f"\n🏷️ Classification Thresholds:")
                for level, threshold in result['thresholds'].items():
                    print(f"   • {level}: ≤ {threshold:.0f} TPM")

                # Analysis
                current = scenario['current_tpm']
                pred_5min = result['predictions'].get('tpm_5min', current)
                trend = "📈 Rising" if pred_5min > current else "📉 Falling" if pred_5min < current else "➡️ Stable"
                change_pct = ((pred_5min - current) / current) * 100

                print(f"\n📈 Analysis:")
                print(f"   • Trend: {trend}")
                print(f"   • 5min Change: {change_pct:+.1f}%")
                print(f"   • Traffic Level: {result['labels'].get('tpm_5min', 'unknown')}")

            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                continue

        print(f"\n✅ Manual prediction demo completed!")

    def interactive_prediction_demo(self):
        """Interactive demo cho phép user nhập input"""
        print("\n🎮 INTERACTIVE PREDICTION DEMO")
        print("=" * 50)
        print("Nhập thông tin để dự đoán TPM:")

        try:
            # Get user inputs
            print(f"\n📅 Thời gian dự đoán:")
            year = int(input("   Năm (2024): ") or "2024")
            month = int(input("   Tháng (1-12): ") or "12")
            day = int(input("   Ngày (1-31): ") or "23")
            hour = int(input("   Giờ (0-23): ") or "14")
            minute = int(input("   Phút (0-59): ") or "30")

            when = datetime(year, month, day, hour, minute)

            print(f"\n📊 Traffic hiện tại:")
            current_tpm = float(input("   Current TPM: ") or "500")

            print(f"\n📈 Lịch sử TPM (tùy chọn, bỏ trống để skip):")
            prev_input = input("   Previous TPMs [1min_ago,2min_ago,...]: ").strip()
            previous_tpms = None
            if prev_input:
                try:
                    previous_tpms = [float(x.strip()) for x in prev_input.split(',')]
                    print(f"   ✅ Loaded {len(previous_tpms)} historical values")
                except:
                    print(f"   ⚠️ Invalid format, using current TPM as fallback")

            print(f"\n⚡ Response time:")
            rt_input = input("   Response time (ms, optional): ").strip()
            response_time = float(rt_input) if rt_input else None

            print(f"\n📱 Push notifications:")
            push_active = int(input("   Push active (0/1): ") or "0")
            minutes_since_push = int(input("   Minutes since last push: ") or "999")

            # Make prediction
            print(f"\n🔮 Making prediction...")
            result = self.predict_from_inputs(
                when=when,
                current_tpm=current_tpm,
                previous_tpms=previous_tpms,
                response_time=response_time,
                push_notification_active=push_active,
                minutes_since_push=minutes_since_push,
                minutes_ahead=(5, 10, 15, 30, 60)
            )

            # Display detailed results
            print(f"\n🎯 DETAILED PREDICTION RESULTS")
            print(f"=" * 50)
            print(f"📅 Prediction Time: {when.strftime('%A, %B %d, %Y at %H:%M')}")
            print(f"📊 Current TPM: {current_tpm}")

            print(f"\n🔮 Future TPM Predictions:")
            print(f"{'Horizon':<10} {'TPM':<10} {'Level':<12} {'Change':<10}")
            print(f"-" * 45)

            for horizon, pred in result['predictions'].items():
                label = result['labels'][horizon]
                change = ((pred - current_tpm) / current_tpm) * 100
                minutes = horizon.replace('tpm_', '').replace('min', '')

                print(f"{minutes + 'min':<10} {pred:<10.1f} {label:<12} {change:+.1f}%")

            print(f"\n🏷️ Classification System:")
            for level, threshold in result['thresholds'].items():
                print(f"   • {level.replace('_', ' ').title()}: ≤ {threshold:.0f} TPM")

            print(f"\n📋 Model Details:")
            if 'details' in result:
                for target, details in result['details'].get('model_details', {}).items():
                    model_name = details.get('model', 'Unknown')
                    base_pred = details.get('base_prediction', 0)
                    print(f"   • {target}: {model_name} (base: {base_pred:.1f})")

            print(f"\n✅ Interactive prediction completed!")

        except KeyboardInterrupt:
            print(f"\n⏹️ Demo cancelled by user")
        except Exception as e:
            print(f"\n❌ Error in interactive demo: {e}")

    def predict_realtime_from_newrelic(self, minutes_ahead=(5, 10, 15, 30)):
        """
        Real-time prediction using the last 3 hours of data from New Relic

        Args:
            minutes_ahead: Tuple of minutes to predict ahead

        Returns:
            dict: Prediction results with TPM forecasts and confidence levels
        """
        print("🔴 REAL-TIME PREDICTION FROM NEW RELIC")
        print("=" * 50)

        if not hasattr(self, 'models') or not self.models or not hasattr(self, 'best_models') or not self.best_models:
            raise ValueError("❌ Models are not loaded/trained. Please train or load models first.")

        try:
            # Import the function from newrelic_traffic_collector
            from newrelic_traffic_collector import get_recent_3h_1min_dataframe

            print("📡 Fetching recent 3-hour data from New Relic...")
            recent_data = get_recent_3h_1min_dataframe()

            if recent_data is None or recent_data.empty:
                raise ValueError("❌ No recent data available from New Relic")

            print(f"✅ Retrieved {len(recent_data)} data points from New Relic")
            print(f"📅 Time range: {recent_data['timestamp'].min()} to {recent_data['timestamp'].max()}")

            # Get the most recent data point
            latest_data = recent_data.iloc[-1]
            current_time = latest_data['timestamp']
            current_tpm = latest_data['tpm']

            print(f"🕐 Current time: {current_time}")
            print(f"📊 Current TPM: {current_tpm:.1f}")

            # Extract previous TPM values (last 10 minutes for context)
            if len(recent_data) >= 10:
                previous_tpms = recent_data['tpm'].iloc[-10:-1].tolist()  # Last 9 values before current
                previous_tpms.reverse()  # Reverse to get [1min_ago, 2min_ago, ...]
            else:
                previous_tpms = recent_data['tpm'].iloc[:-1].tolist()
                previous_tpms.reverse()

            print(f"📈 Using {len(previous_tpms)} previous TPM values for context")

            # Calculate response time trend (if we have enough data)
            response_time = None
            if len(recent_data) >= 5:
                # Use TPM as a proxy for response time trend (higher TPM might indicate higher load)
                recent_tpms = recent_data['tpm'].iloc[-5:].values
                response_time = np.mean(recent_tpms) * 0.1  # Simple heuristic

            # Detect if there might be a push notification effect
            # Look for sudden TPM spikes in the last 30 minutes
            push_notification_active = 0
            minutes_since_push = 999999

            if len(recent_data) >= 30:
                last_30_min = recent_data['tpm'].iloc[-30:].values
                current_avg = np.mean(last_30_min[-5:])  # Last 5 minutes average
                baseline_avg = np.mean(last_30_min[:10])  # First 10 minutes average

                # If current is significantly higher than baseline, might be push effect
                if current_avg > baseline_avg * 1.5:  # 50% increase threshold
                    push_notification_active = 1
                    # Find the spike point
                    for i in range(len(last_30_min) - 5, 0, -1):
                        if last_30_min[i] > baseline_avg * 1.3:
                            minutes_since_push = len(last_30_min) - i
                            break

            print(f"🔍 Push detection: Active={push_notification_active}, Minutes since={minutes_since_push}")

            # Make prediction using the existing predict_from_inputs method
            print(f"\n🔮 Making real-time predictions...")

            prediction_time = pd.to_datetime(current_time)
            if prediction_time.tzinfo is None:
                # If timezone-naive, assume UTC
                prediction_time = prediction_time.tz_localize('UTC')

            # Convert to datetime object for prediction
            prediction_datetime = prediction_time.to_pydatetime()

            result = self.predict_from_inputs(
                when=prediction_datetime,
                current_tpm=float(current_tpm),
                previous_tpms=previous_tpms,
                response_time=response_time,
                push_notification_active=push_notification_active,
                minutes_since_push=minutes_since_push,
                minutes_ahead=minutes_ahead
            )

            # Add real-time specific information
            result['realtime_info'] = {
                'data_source': 'New Relic API',
                'data_points_used': len(recent_data),
                'current_time': current_time.isoformat(),
                'current_tpm': float(current_tpm),
                'previous_tpms_count': len(previous_tpms),
                'push_detection': {
                    'active': bool(push_notification_active),
                    'minutes_since': minutes_since_push
                },
                'data_freshness': 'Real-time (last 3 hours)'
            }

            # Display results
            print(f"\n🎯 REAL-TIME PREDICTION RESULTS")
            print(f"=" * 50)
            print(f"📅 Prediction Time: {prediction_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"📊 Current TPM: {current_tpm:.1f}")

            print(f"\n🔮 Future TPM Predictions:")
            print(f"{'Horizon':<10} {'TPM':<10} {'Level':<12} {'Change':<10}")
            print(f"-" * 45)

            for horizon, pred in result['predictions'].items():
                label = result['labels'][horizon]
                change = ((pred - current_tpm) / current_tpm) * 100 if current_tpm > 0 else 0
                minutes = horizon.replace('tpm_', '').replace('min', '')

                print(f"{minutes + 'min':<10} {pred:<10.1f} {label:<12} {change:+.1f}%")

            # Traffic trend analysis
            if len(previous_tpms) >= 3:
                recent_trend = np.mean(previous_tpms[:3]) - np.mean(previous_tpms[-3:]) if len(previous_tpms) >= 6 else 0
                trend_direction = "📈 Increasing" if recent_trend > 0 else "📉 Decreasing" if recent_trend < 0 else "➡️ Stable"
                print(f"\n📊 Traffic Trend: {trend_direction}")

            print(f"\n✅ Real-time prediction completed!")
            return result

        except ImportError:
            raise ValueError("❌ Cannot import get_recent_3h_1min_dataframe. Make sure newrelic_traffic_collector.py is available.")
        except Exception as e:
            print(f"❌ Real-time prediction failed: {e}")
            raise


def train_model():
    """Huấn luyện và lưu model với format mới"""
    print("🚀 TRAINING MODE - HUẤN LUYỆN MÔ HÌNH MỚI")
    print("=" * 70)

    # Khởi tạo predictor
    predictor = ImprovedTrafficPredictor()

    try:
        print("📊 Bước 1: Load dữ liệu và huấn luyện mô hình...")

        results = predictor.train_and_evaluate_optimized()

        if results is None:
            print("❌ Training failed - no results returned")
            return None, None

        print(f"\n🏆 Training Summary:")
        if 'best_models' in results:
            for target, model_name in results['best_models'].items():
                print(f"   • {target}: {model_name}")
        else:
            print("   • No best models found")

        print("\n📁 Bước 2: Lưu mô hình...")
        predictor.save_models()

        # ============ THÊM: TẠO LEARNING CURVES VÀ VISUALIZATIONS ============
        print("\n📈 Bước 3: Tạo learning curves và visualizations...")
        try:
            plot_paths = predictor.create_comprehensive_plots()
            print(f"✅ Created {len(plot_paths)} visualization files")
        except Exception as e:
            print(f"⚠️ Warning: Could not create all visualizations: {e}")
            # Still try to create learning curves only
            try:
                predictor.plot_learning_curves()
            except Exception as e2:
                print(f"⚠️ Could not create learning curves: {e2}")

        print("\n✅ HUẤN LUYỆN VÀ VISUALIZATION HOÀN THÀNH!")
        return predictor, results

    except Exception as e:
        print(f"❌ Lỗi trong quá trình huấn luyện: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_demo_predictions(predictor, results):
    """Chạy demo predictions với xử lý lỗi"""
    print("\n" + "=" * 70)
    print("🎯 BƯỚC: Demo dự đoán từ input thủ công...")

    try:
        # Đảm bảo predictor có đầy đủ thông tin
        if not hasattr(predictor, 'models') or not predictor.models:
            print("⚠️ Models not available - skipping demonstration")
            return

        if not hasattr(predictor, 'X_processed') or predictor.X_processed is None or predictor.X_processed.empty:
            print("⚠️ No processed data available - skipping demonstration")
            return

        predictor.demonstrate_predictions(results)
        print("✅ Demo predictions hoàn thành")

    except Exception as e:
        print(f"❌ Lỗi trong demo predictions: {e}")

def run_future_predictions(predictor):
    """Chạy dự đoán tương lai với New Relic API"""
    api_key = os.getenv("NEWRELIC_API_KEY")
    app_id = os.getenv("NEW_RELIC_APP_ID")

    if api_key and app_id:
        print("\n" + "=" * 70)
        print("🔮 BƯỚC: Demo dự đoán TPM tương lai từ dữ liệu thật...")
        try:
            # Kiểm tra models trước khi gọi future predictions
            if not hasattr(predictor, 'models') or not predictor.models:
                print("⚠️ Models not available - skipping future predictions")
                return

            # Note: This would call the future predictions method if it exists
            # predictor.demo_future_predictions(api_key=api_key, app_id=app_id)
            print("⚠️ Future predictions method not implemented in this version")
            print("✅ Future predictions hoàn thành")
        except Exception as e:
            print(f"❌ Lỗi trong future predictions: {e}")
    else:
        print("\n⚠️ Bỏ qua demo dự đoán tương lai - thiếu New Relic credentials")
        print("💡 Để sử dụng tính năng này, hãy set biến môi trường:")
        print("   - NEWRELIC_API_KEY: Your New Relic API key")
        print("   - NEW_RELIC_APP_ID: Your New Relic Application ID")


def load_trained_model(models_dir=None):
    """Load model đã được huấn luyện từ thư mục models"""
    import glob
    import joblib

    # Xác định thư mục models
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')

    if not os.path.exists(models_dir):
        print(f"❌ Models directory không tồn tại: {models_dir}")
        return None, None

    # Tìm metadata file mới nhất
    metadata_files = glob.glob(os.path.join(models_dir, 'metadata_*.joblib'))
    if not metadata_files:
        print(f"❌ Không tìm thấy metadata files trong {models_dir}")
        return None, None

    latest_metadata = max(metadata_files, key=os.path.getctime)
    print(f"📁 Loading from: {os.path.basename(latest_metadata)}")

    # Khởi tạo predictor
    predictor = ImprovedTrafficPredictor()

    try:
        # Load metadata
        metadata = joblib.load(latest_metadata)

        # Khôi phục thông tin cơ bản
        predictor.best_models = metadata.get('best_models', {})
        predictor.feature_cols = metadata.get('feature_columns', [])

        if not predictor.best_models:
            raise ValueError("❌ Không tìm thấy thông tin best_models trong metadata")

        print("✅ Thông tin model:")
        for target, model_name in predictor.best_models.items():
            print(f"   • {target}: {model_name}")

        # Khởi tạo containers
        predictor.models = {}
        predictor.scalers = {}

        # Extract timestamp from metadata filename
        timestamp = os.path.basename(latest_metadata).replace('metadata_', '').replace('.joblib', '')

        # Load từng model và scaler
        for target, model_name in predictor.best_models.items():
            # Load model
            model_path = os.path.join(models_dir, f'{target}_{model_name}_{timestamp}.joblib')
            if os.path.exists(model_path):
                if target not in predictor.models:
                    predictor.models[target] = {}
                predictor.models[target][model_name] = joblib.load(model_path)
                print(f"✅ Loaded model: {target}_{model_name}")
            else:
                print(f"⚠️ Model file không tồn tại: {model_path}")

            # Load scaler
            scaler_path = os.path.join(models_dir, f'scaler_{target}_{timestamp}.joblib')
            if os.path.exists(scaler_path):
                predictor.scalers[target] = joblib.load(scaler_path)
                print(f"✅ Loaded scaler: {target}")
            else:
                print(f"⚠️ Scaler file không tồn tại: {scaler_path}")

        # Load dữ liệu và tạo features (cần cho demonstrate_predictions và reports)
        print("📊 Đang load và xử lý dữ liệu huấn luyện...")
        raw_data = predictor.load_real_data()  # Load raw data vào predictor.enhanced_data

        # Tạo features từ raw data và lưu vào processed data
        if hasattr(predictor, 'enhanced_data') and predictor.enhanced_data is not None:
            processed_data = predictor.create_granularity_adaptive_features(predictor.enhanced_data)
            print(f"✅ Đã tạo {len(processed_data)} samples với features")

            # Lưu processed data cho các functions khác sử dụng
            predictor.X_processed = processed_data.drop(columns=['tpm'] if 'tpm' in processed_data.columns else [])
            predictor.y_processed = processed_data['tpm'] if 'tpm' in processed_data.columns else None

        else:
            print("⚠️ Không thể load dữ liệu huấn luyện")
            # Tạo dummy data để tránh crash
            predictor.X_processed = pd.DataFrame()
            predictor.y_processed = pd.Series()

        # Tạo results với thông tin đầy đủ từ metadata và dữ liệu đã load
        results = {
            'best_models': predictor.best_models,
            'timestamp': timestamp,
            'feature_columns': predictor.feature_cols,
            'total_records': len(predictor.enhanced_data) if hasattr(predictor, 'enhanced_data') and predictor.enhanced_data is not None else 0,
            'processed_records': len(predictor.X_processed) if hasattr(predictor, 'X_processed') else 0,
            'model_performance': metadata.get('model_performance', {}),
            'training_info': {
                'data_source': predictor.data_file_path,
                'timestamp': timestamp,
                'num_features': len(predictor.feature_cols) if predictor.feature_cols else 0
            }
        }

        print(f"✅ LOAD MODEL THÀNH CÔNG - Timestamp: {timestamp}")
        print(f"📊 Dữ liệu: {results['total_records']} raw records, {results['processed_records']} processed samples")
        return predictor, results

    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None, None


def run_comprehensive_report(predictor, results):
    """Tạo báo cáo tổng hợp với xử lý lỗi"""
    print("\n" + "=" * 70)
    print("📋 BƯỚC: Tạo báo cáo tổng hợp...")

    try:
        # Kiểm tra dữ liệu có sẵn
        if not results or not results.get('best_models'):
            print("⚠️ Không có dữ liệu để tạo báo cáo - tạo báo cáo cơ bản")

            # Báo cáo cơ bản khi không có dữ liệu đầy đủ
            print("\n📊 BÁO CÁO CƠ BẢN:")
            print("=" * 50)
            print(f"   • Timestamp: {results.get('timestamp', 'Unknown') if results else 'No results'}")
            print(f"   • Total records: {results.get('total_records', 0) if results else 0}")
            print(f"   • Processed records: {results.get('processed_records', 0) if results else 0}")

            if results and results.get('best_models'):
                print(f"   • Best models:")
                for target, model_name in results['best_models'].items():
                    print(f"     - {target}: {model_name}")

            # In performance nếu có
            if results and results.get('model_performance'):
                print(f"   • Model performance:")
                models = results['model_performance']
                for target, target_models in models.items():
                    print(f"     {target}:")
                    if isinstance(target_models, dict):
                        for model_name, metrics in target_models.items():
                            if isinstance(metrics, dict):
                                r2 = metrics.get('test_r2', 'N/A')
                                mae = metrics.get('test_mae', 'N/A')
                                print(f"       • {model_name}: R² = {r2}, MAE = {mae}")

            print("\n✅ Summary report hoàn thành")

        else:
            # Có dữ liệu - tạo báo cáo đơn giản
            print("\n📊 BÁO CÁO TỔNG HỢP:")
            print("=" * 50)
            print(f"   • Timestamp: {results.get('timestamp', 'Unknown')}")
            print(f"   • Data source: {results.get('training_info', {}).get('data_source', 'Unknown')}")
            print(f"   • Total records: {results.get('total_records', 0)}")
            print(f"   • Processed records: {results.get('processed_records', 0)}")
            print(f"   • Features: {results.get('training_info', {}).get('num_features', 0)}")

            print(f"\n🏆 Best Models:")
            for target, model_name in results['best_models'].items():
                print(f"   • {target}: {model_name}")

            # Performance summary
            if results.get('model_performance'):
                print(f"\n📈 Model Performance:")
                models = results['model_performance']
                for target, target_models in models.items():
                    print(f"   {target}:")
                    if isinstance(target_models, dict):
                        for model_name, metrics in target_models.items():
                            if isinstance(metrics, dict):
                                r2 = metrics.get('test_r2', 'N/A')
                                mae = metrics.get('test_mae', 'N/A')
                                print(f"     • {model_name}: R² = {r2}, MAE = {mae}")

            # THÊM: Thông tin về visualizations
            charts_dir = os.path.join(os.path.dirname(__file__), 'charts')
            if os.path.exists(charts_dir):
                chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
                if chart_files:
                    print(f"\n📈 Generated Visualizations: {len(chart_files)} files")
                    recent_charts = sorted(chart_files, reverse=True)[:5]  # 5 most recent
                    for chart in recent_charts:
                        print(f"   • {chart}")

            print("\n✅ Comprehensive report hoàn thành")

    except Exception as e:
        print(f"❌ Lỗi trong comprehensive report: {e}")
        # In thông tin cơ bản thay vì crash
        print("\n⚠️ Tạo báo cáo cơ bản thay thế:")
        print(f"   • Models loaded: {len(results.get('best_models', {})) if results else 0}")
        print(f"   • Timestamp: {results.get('timestamp', 'Unknown') if results else 'Unknown'}")
        print(f"   • Data records: {results.get('total_records', 0) if results else 0}")


def run_manual_prediction_demos(predictor):
    """Chạy demos prediction với xử lý lỗi"""
    print("\n" + "=" * 70)
    print("🎯 BƯỚC: Demo dự đoán từ input thủ công...")

    try:
        if not hasattr(predictor, 'models') or not predictor.models:
            print("⚠️ Models not available - skipping prediction demos")
            return

            # Demo 1: Automated scenarios
        predictor.demo_manual_predictions()

        # Demo 2: Interactive (optional)
        print("\n" + "=" * 50)
        user_input = input("🎮 Muốn thử Interactive Demo? (y/N): ").strip().lower()
        if user_input in ['y', 'yes']:
            predictor.interactive_prediction_demo()
        else:
            print("⏭️ Bỏ qua Interactive Demo")

        print("✅ Manual prediction demos hoàn thành")

    except Exception as e:
        print(f"❌ Lỗi trong prediction demos: {e}")


def main():
    """Main function với lựa chọn train mới hoặc load model có sẵn"""
    import sys

    # Kiểm tra arguments
    mode = "train"  # Mặc định là train model mới
    if len(sys.argv) > 1:
        if sys.argv[1] in ["train", "--train", "-t"]:
            mode = "train"
        elif sys.argv[1] in ["load", "--load", "-l"]:
            mode = "load"
        elif sys.argv[1] in ["help", "--help", "-h"]:
            print("🚀 IMPROVED TRAFFIC PREDICTION MODEL")
            print("=" * 50)
            print("Cách sử dụng:")
            print("  python ImprovedTrafficPredictionModel.py [mode]")
            print("")
            print("Modes:")
            print("  train   : Huấn luyện model mới (mặc định)")
            print("  load    : Load model đã huấn luyện và chạy demo")
            print("  help    : Hiển thị trợ giúp")
            print("")
            print("Ví dụ:")
            print("  python ImprovedTrafficPredictionModel.py train")
            print("  python ImprovedTrafficPredictionModel.py load")
            return

    try:
        # Chọn mode
        if mode == "train":
            predictor, results = train_model()
        else:  # mode == "load"
            predictor, results = load_trained_model()

        if predictor is None or results is None:
            print("❌ Failed to initialize predictor or results")
            return

        # Chạy các demo
        run_demo_predictions(predictor, results)
        run_manual_prediction_demos(predictor)
        run_future_predictions(predictor)
        run_comprehensive_report(predictor, results)

        print("\n✅ HOÀN THÀNH TẤT CẢ CÁC BƯỚC!")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình thực thi: {e}")
        raise


if __name__ == "__main__":
    main()
