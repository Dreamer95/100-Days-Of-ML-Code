"""
Day6 Summary: Time Series Training Model
Hàm training model theo time base series
"""

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

class TimeSeriesTrainer:
    """
    Time Series Training Model cho TPM prediction
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.feature_cols = []
        self.tpm_thresholds = {}
        self.training_results = {}
        
    def create_time_series_features(self, data, target_col='tpm'):
        """
        Tạo features cho time series prediction
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw data với columns ['timestamp', 'tpm', ...]
        target_col : str
            Tên column target (mặc định 'tpm')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame với features đã được tạo
        """
        
        print("🔧 Creating time series features...")
        
        df = data.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # ============ TEMPORAL FEATURES ============
        print("🕐 Creating temporal features...")
        
        # Basic temporal features
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
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ============ LAG FEATURES ============
        print("📊 Creating lag features...")
        
        # Safe lag features (no data leakage)
        safe_lags = [1, 2, 3, 6, 12, 24]
        
        for lag in safe_lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
            # Response time lag features (if available)
            if 'response_time' in df.columns:
                df[f'response_time_lag_{lag}'] = df['response_time'].shift(lag)
        
        # ============ ROLLING FEATURES ============
        print("📈 Creating rolling statistics...")
        
        safe_windows = [3, 6, 12, 24]
        
        for window in safe_windows:
            # Moving averages (shifted to avoid leakage)
            df[f'{target_col}_ma_{window}'] = df[target_col].shift(1).rolling(
                window=window, min_periods=1).mean()
            
            # Rolling standard deviation
            df[f'{target_col}_std_{window}'] = df[target_col].shift(1).rolling(
                window=window, min_periods=1).std()
        
        # ============ CHANGE FEATURES ============
        print("📉 Creating trend features...")
        
        df[f'{target_col}_change_1'] = df[target_col].diff(1)
        df[f'{target_col}_change_3'] = df[target_col].diff(3)
        df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
        df[f'{target_col}_pct_change_3'] = df[target_col].pct_change(3)
        
        # ============ PUSH NOTIFICATION FEATURES ============
        print("🔔 Creating push notification features...")
        
        if 'push_notification_active' in df.columns:
            df['push_active'] = df['push_notification_active'].fillna(0)
        
        if 'minutes_since_push' in df.columns:
            df['minutes_since_push_safe'] = df['minutes_since_push'].fillna(9999)
            df['recent_push'] = (df['minutes_since_push_safe'] <= 60).astype(int)
            
            # Enhanced push notification features
            df['is_daytime_effect_window'] = ((df['hour'] >= 7) & (df['hour'] <= 21)).astype(int)
            df['is_nighttime_no_effect'] = ((df['hour'] >= 1) & (df['hour'] <= 6)).astype(int)
            
            # Push decay factor
            df['push_decay_factor'] = np.where(
                df['minutes_since_push_safe'] <= 15,
                np.exp(-df['minutes_since_push_safe'] / 5.0),
                0.0
            )
            
            # Push effect multiplier
            df['push_effect_multiplier'] = np.where(
                (df['push_active'] == 1) | (df['minutes_since_push_safe'] <= 15),
                np.where(
                    df['is_daytime_effect_window'] == 1,
                    np.where(
                        df['minutes_since_push_safe'] <= 15,
                        np.exp(-df['minutes_since_push_safe'] / 5.0),
                        0.0
                    ),
                    np.where(
                        df['is_nighttime_no_effect'] == 1,
                        0.0,
                        np.where(
                            df['minutes_since_push_safe'] <= 15,
                            np.exp(-df['minutes_since_push_safe'] / 10.0) * 0.3,
                            0.0
                        )
                    )
                ),
                0.0
            )
        
        # ============ CLEAN UP ============
        print("🧹 Cleaning data...")
        
        # Remove dangerous columns that can cause data leakage
        danger_columns = [
            'timestamp', 'timestamp_to', 'timestamp_utc', 'timestamp_to_utc',
            'metric_name', 'day_name', 'data_quality', 'collection_strategy',
            'granularity_category', 'sequence_id',
            'push_notification_active', 'minutes_since_push',
            'data_age_days', 'ml_weight', 'week_priority'
        ]
        
        existing_danger_cols = [col for col in danger_columns if col in df.columns]
        if existing_danger_cols:
            df = df.drop(columns=existing_danger_cols)
            print(f"   ✅ Removed {len(existing_danger_cols)} potentially leaky columns")
        
        # Handle NaN and infinite values
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
        
        # Convert all to numeric
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Final cleanup
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        print(f"✅ Feature engineering completed:")
        print(f"   • Final dataset: {len(df)} rows × {len(df.columns)} columns")
        print(f"   • Target: {target_col}")
        print(f"   • Features: {len(df.columns) - 1}")
        
        return df
    
    def select_important_features(self, X, y, max_features=15):
        """
        Chọn features quan trọng nhất để tránh overfitting
        """
        print(f"🎯 Feature Selection: {X.shape[1]} → {max_features} features")
        
        if X.shape[1] <= max_features:
            print("   No selection needed - already optimal size")
            return X, list(X.columns)
        
        # Statistical F-test selection
        f_selector = SelectKBest(score_func=f_regression, k=max_features)
        X_selected = f_selector.fit_transform(X, y)
        selected_features = X.columns[f_selector.get_support()].tolist()
        
        print(f"   ✅ Selected {len(selected_features)} features:")
        for i, feat in enumerate(selected_features[:10]):
            print(f"      {i + 1:2d}. {feat}")
        
        if len(selected_features) > 10:
            print(f"      ... and {len(selected_features) - 10} more")
        
        return X[selected_features], selected_features
    
    def get_optimized_models(self):
        """
        Lấy các models đã được tối ưu cho time series
        """
        return {
            'RandomForest_Enhanced': RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting_Enhanced': GradientBoostingRegressor(
                loss='huber',
                alpha=0.95,
                learning_rate=0.08,
                n_estimators=400,
                max_depth=6,
                subsample=0.8,
                min_samples_leaf=12,
                min_samples_split=25,
                max_features=0.7,
                validation_fraction=0.15,
                n_iter_no_change=20,
                tol=1e-5,
                random_state=42
            ),
            'Ridge_Optimized': Ridge(
                alpha=100.0,
                solver='auto',
                random_state=42
            )
        }
    
    def compute_tpm_thresholds(self, tpm_data):
        """
        Tính toán thresholds cho classification TPM
        """
        thresholds = {
            'q60': np.percentile(tpm_data, 60),
            'q80': np.percentile(tpm_data, 80),
            'q90': np.percentile(tpm_data, 90)
        }
        
        print(f"📊 TPM Thresholds computed:")
        for k, v in thresholds.items():
            print(f"   {k}: {v:.1f}")
        
        return thresholds
    
    def classify_tpm_value(self, tpm_value, thresholds):
        """
        Phân loại giá trị TPM
        """
        if tpm_value >= thresholds['q90']:
            return 'Very High'
        elif tpm_value >= thresholds['q80']:
            return 'High'
        elif tpm_value >= thresholds['q60']:
            return 'Medium'
        else:
            return 'Low'
    
    def train_time_series_model(self, data, target_col='tpm', test_size=0.25, max_features=20):
        """
        Training time series model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw data với timestamp và target column
        target_col : str
            Tên column target
        test_size : float
            Tỷ lệ test set
        max_features : int
            Số features tối đa
            
        Returns:
        --------
        dict
            Kết quả training
        """
        
        print("🚀 Starting Time Series Model Training...")
        print(f"📊 Input data shape: {data.shape}")
        
        # Create features
        processed_data = self.create_time_series_features(data, target_col)
        
        if processed_data.empty:
            raise ValueError("No processed data available for training")
        
        # Prepare X and y
        X = processed_data.drop(columns=[target_col])
        y = processed_data[target_col]
        
        print(f"📈 Processed data shape: {X.shape}")
        
        # Compute TPM thresholds
        self.tpm_thresholds = self.compute_tpm_thresholds(y)
        
        # Feature selection
        n_samples = len(X)
        optimal_features = min(max_features, max(8, n_samples // 150))
        
        X_selected, selected_features = self.select_important_features(
            X, y, max_features=optimal_features
        )
        
        self.feature_cols = selected_features
        
        # Time-based split
        split_idx = int(len(X_selected) * (1 - test_size))
        
        X_train = X_selected.iloc[:split_idx].copy()
        X_test = X_selected.iloc[split_idx:].copy()
        y_train = y.iloc[:split_idx].copy()
        y_test = y.iloc[split_idx:].copy()
        
        print(f"\n📊 Time-based split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Get models
        algorithms = self.get_optimized_models()
        
        # Initialize storage
        self.models = {'tpm': {}}
        model_performance = {}
        best_score = -np.inf
        best_model_name = None
        
        # Train models with cross-validation
        print("\n=== Training Models ===")
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
                print(f"      CV R²: {cv_mean:.3f} ± {cv_std:.3f}")
                print(f"      Test R²: {r2:.3f}")
                print(f"      Test MAE: {mae:.1f}")
                print(f"      Test RMSE: {rmse:.1f}")
                
                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
        
        if not self.models['tpm']:
            raise ValueError("❌ No models were successfully trained")
        
        # Set best model
        self.best_models['tpm'] = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name} (R² = {best_score:.3f})")
        
        # Store training results
        self.training_results = {
            'model_performance': model_performance,
            'best_model': best_model_name,
            'best_score': best_score,
            'feature_count': len(selected_features),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'tpm_thresholds': self.tpm_thresholds
        }
        
        return self.training_results
    
    def save_model(self, model_dir="models", model_name="time_series_model"):
        """
        Lưu model đã train
        """
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(model_dir, f"{model_name}_{timestamp}.joblib")
        joblib.dump(self.models, model_path)
        
        # Save metadata
        metadata = {
            'best_models': self.best_models,
            'feature_cols': self.feature_cols,
            'tpm_thresholds': self.tpm_thresholds,
            'training_results': self.training_results,
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.joblib")
        joblib.dump(metadata, metadata_path)
        
        print(f"💾 Model saved:")
        print(f"   Model: {model_path}")
        print(f"   Metadata: {metadata_path}")
        
        return model_path, metadata_path
    
    def load_model(self, model_path, metadata_path):
        """
        Load model đã lưu
        """
        self.models = joblib.load(model_path)
        metadata = joblib.load(metadata_path)
        
        self.best_models = metadata['best_models']
        self.feature_cols = metadata['feature_cols']
        self.tpm_thresholds = metadata['tpm_thresholds']
        self.training_results = metadata['training_results']
        
        print(f"📂 Model loaded successfully")
        print(f"   Best model: {self.best_models.get('tpm', 'Unknown')}")
        print(f"   Features: {len(self.feature_cols)}")
        
        return True
    
    def plot_training_results(self, save_plot=True, output_dir="charts"):
        """
        Vẽ biểu đồ kết quả training
        """
        if not self.training_results:
            print("❌ No training results available")
            return None
        
        model_performance = self.training_results['model_performance']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Time Series Model Training Results', fontsize=16, fontweight='bold')
        
        # Plot 1: R² scores
        ax1 = axes[0, 0]
        models = list(model_performance.keys())
        r2_scores = [model_performance[m]['test_r2'] for m in models]
        
        bars = ax1.bar(models, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Test R² Scores by Model')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: MAE scores
        ax2 = axes[0, 1]
        mae_scores = [model_performance[m]['test_mae'] for m in models]
        
        bars = ax2.bar(models, mae_scores, color=['orange', 'purple', 'brown'])
        ax2.set_title('Test MAE by Model')
        ax2.set_ylabel('MAE')
        
        # Add value labels on bars
        for bar, score in zip(bars, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_scores)*0.01,
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 3: CV vs Test R² comparison
        ax3 = axes[1, 0]
        cv_scores = [model_performance[m]['cv_r2_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax3.bar(x - width/2, cv_scores, width, label='CV R²', color='lightblue')
        ax3.bar(x + width/2, r2_scores, width, label='Test R²', color='lightcoral')
        
        ax3.set_title('CV vs Test R² Comparison')
        ax3.set_ylabel('R² Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Plot 4: Training summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""Training Summary:
        
Best Model: {self.training_results['best_model']}
Best R² Score: {self.training_results['best_score']:.3f}

Dataset Info:
• Training samples: {self.training_results['training_samples']:,}
• Test samples: {self.training_results['test_samples']:,}
• Features used: {self.training_results['feature_count']}

TPM Thresholds:
• Q60: {self.tpm_thresholds['q60']:.1f}
• Q80: {self.tpm_thresholds['q80']:.1f}
• Q90: {self.tpm_thresholds['q90']:.1f}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            try:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_results_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"💾 Training results plot saved to: {filepath}")
            except Exception as e:
                print(f"⚠️ Error saving plot: {e}")
        
        plt.show()
        return fig

def main():
    """
    Demo function để test time series training
    """
    print("🚀 Day6 - Time Series Trainer Demo")
    print("=" * 50)
    
    # Tạo sample data để demo
    print("📊 Creating sample time series data for demo...")
    
    from datetime import datetime, timedelta
    import random
    
    # Create 30 days of data with 1-hour intervals
    start_date = datetime.now() - timedelta(days=30)
    timestamps = []
    tpm_values = []
    response_times = []
    push_active_values = []
    minutes_since_push_values = []
    
    for i in range(30 * 24):  # 30 days * 24 hours
        timestamp = start_date + timedelta(hours=i)
        
        # Simulate realistic TPM pattern
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base TPM with patterns
        base_tpm = 150
        
        # Daily pattern
        if 8 <= hour <= 18:
            base_tpm += 300
        elif 19 <= hour <= 22:
            base_tpm += 150
        
        # Weekly pattern
        if day_of_week >= 5:  # Weekend
            base_tpm *= 0.7
        
        # Push notification effect (simulate push at 8h and 15h)
        if hour in [8, 15]:
            push_active = 1
            minutes_since = 0
            base_tpm += 200  # Push boost
        else:
            push_active = 0
            # Calculate minutes since last push
            if hour > 15:
                minutes_since = (hour - 15) * 60
            elif hour > 8:
                minutes_since = (hour - 8) * 60
            else:
                minutes_since = (24 - 15 + hour) * 60  # From yesterday 15h
        
        # Add randomness
        tpm = base_tpm + random.gauss(0, 50)
        tpm = max(10, tpm)
        
        response_time = 100 + random.gauss(0, 20)
        response_time = max(50, response_time)
        
        timestamps.append(timestamp)
        tpm_values.append(tpm)
        response_times.append(response_time)
        push_active_values.append(push_active)
        minutes_since_push_values.append(minutes_since)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'tpm': tpm_values,
        'response_time': response_times,
        'push_notification_active': push_active_values,
        'minutes_since_push': minutes_since_push_values
    })
    
    print(f"✅ Created sample data: {len(sample_data)} records")
    print(f"📅 Time range: {sample_data['timestamp'].min()} to {sample_data['timestamp'].max()}")
    print(f"📊 TPM range: {sample_data['tpm'].min():.1f} to {sample_data['tpm'].max():.1f}")
    
    try:
        # Initialize trainer
        trainer = TimeSeriesTrainer()
        
        # Train model
        results = trainer.train_time_series_model(
            data=sample_data,
            target_col='tpm',
            test_size=0.25,
            max_features=15
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"🏆 Best model: {results['best_model']}")
        print(f"📈 Best R² score: {results['best_score']:.3f}")
        
        # Plot results
        trainer.plot_training_results(output_dir="day6_charts")
        
        # Save model
        model_path, metadata_path = trainer.save_model(
            model_dir="day6_models",
            model_name="demo_time_series_model"
        )
        
        print(f"\n💾 Model saved successfully!")
        print(f"📂 Model files:")
        print(f"   • {model_path}")
        print(f"   • {metadata_path}")
        
    except Exception as e:
        print(f"❌ Lỗi trong demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()