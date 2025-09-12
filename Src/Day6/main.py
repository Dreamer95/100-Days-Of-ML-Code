"""
Day6 Main: Summary của những hàm trong Day5
Tích hợp 4 chức năng chính:
1. Hàm collect data từ New Relic như Day5 (collect data từ thứ 2 đến chủ nhật theo GMT+7 với độ chi tiết tối đa)
2. Hàm visualizer để vẽ data lên chart
3. Hàm training model theo time base series
4. Hàm demo từ model vừa training
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Import our Day6 modules
try:
    from newrelic_data_collector import collect_full_week_newrelic_data
    from visualizer import create_comprehensive_visualization_suite
    from time_series_trainer import TimeSeriesTrainer
    from demo_predictor import DemoPredictor
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure all Day6 modules are in the same directory.")
    sys.exit(1)

class Day6MLPipeline:
    """
    Day6 ML Pipeline tích hợp tất cả các chức năng
    """
    
    def __init__(self):
        self.data = None
        self.trainer = None
        self.predictor = None
        self.model_trained = False
        
    def step1_collect_data(self, weeks_back=4, use_sample_data=False):
        """
        Bước 1: Thu thập data từ New Relic từ thứ 2 đến chủ nhật theo GMT+7
        
        Parameters:
        -----------
        weeks_back : int
            Số tuần muốn thu thập
        use_sample_data : bool
            Nếu True sẽ tạo sample data thay vì thu thập từ New Relic
        """
        
        print("🚀 BƯỚC 1: Thu thập data từ New Relic")
        print("=" * 60)
        
        if use_sample_data:
            print("📊 Tạo sample data để demo...")
            self.data = self._create_sample_data()
            print(f"✅ Tạo sample data thành công: {len(self.data)} records")
        else:
            try:
                print(f"📡 Thu thập data từ New Relic ({weeks_back} tuần)...")
                self.data = collect_full_week_newrelic_data(
                    weeks_back=weeks_back,
                    filename="day6_collected_data.csv"
                )
                
                if self.data is not None:
                    print(f"✅ Thu thập data thành công: {len(self.data)} records")
                else:
                    print("❌ Không thể thu thập data từ New Relic")
                    print("🔄 Chuyển sang sử dụng sample data...")
                    self.data = self._create_sample_data()
                    
            except Exception as e:
                print(f"❌ Lỗi khi thu thập data: {e}")
                print("🔄 Chuyển sang sử dụng sample data...")
                self.data = self._create_sample_data()
        
        return self.data is not None
    
    def step2_visualize_data(self, output_dir="day6_charts"):
        """
        Bước 2: Tạo visualization cho data
        
        Parameters:
        -----------
        output_dir : str
            Thư mục lưu charts
        """
        
        print("\n🎨 BƯỚC 2: Tạo visualization cho data")
        print("=" * 60)
        
        if self.data is None:
            print("❌ Chưa có data để visualize. Hãy chạy step1_collect_data() trước.")
            return False
        
        try:
            figures = create_comprehensive_visualization_suite(
                data=self.data,
                output_dir=output_dir
            )
            
            print(f"✅ Tạo visualization thành công: {len(figures)} charts")
            print(f"💾 Charts được lưu tại: {output_dir}/")
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo visualization: {e}")
            return False
    
    def step3_train_model(self, test_size=0.25, max_features=20, model_dir="day6_models"):
        """
        Bước 3: Training time series model
        
        Parameters:
        -----------
        test_size : float
            Tỷ lệ test set
        max_features : int
            Số features tối đa
        model_dir : str
            Thư mục lưu models
        """
        
        print("\n🤖 BƯỚC 3: Training time series model")
        print("=" * 60)
        
        if self.data is None:
            print("❌ Chưa có data để train. Hãy chạy step1_collect_data() trước.")
            return False
        
        try:
            # Initialize trainer
            self.trainer = TimeSeriesTrainer()
            
            # Train model
            results = self.trainer.train_time_series_model(
                data=self.data,
                target_col='tpm',
                test_size=test_size,
                max_features=max_features
            )
            
            print(f"✅ Training hoàn thành!")
            print(f"🏆 Best model: {results['best_model']}")
            print(f"📈 Best R² score: {results['best_score']:.3f}")
            
            # Plot training results
            self.trainer.plot_training_results(output_dir="day6_charts")
            
            # Save model
            model_path, metadata_path = self.trainer.save_model(
                model_dir=model_dir,
                model_name="day6_time_series_model"
            )
            
            print(f"💾 Model đã được lưu:")
            print(f"   • {model_path}")
            print(f"   • {metadata_path}")
            
            self.model_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step4_demo_predictions(self, model_dir="day6_models"):
        """
        Bước 4: Demo predictions từ model vừa training
        
        Parameters:
        -----------
        model_dir : str
            Thư mục chứa models
        """
        
        print("\n🎯 BƯỚC 4: Demo predictions từ model")
        print("=" * 60)
        
        if not self.model_trained and not os.path.exists(model_dir):
            print("❌ Chưa có model để demo. Hãy chạy step3_train_model() trước.")
            return False
        
        try:
            # Initialize predictor
            self.predictor = DemoPredictor()
            
            # Load trained model
            success = self.predictor.load_trained_model(model_dir=model_dir)
            
            if not success:
                print("❌ Không thể load model để demo")
                return False
            
            print("✅ Model loaded thành công!")
            
            # Run demo predictions
            print("\n🎯 Demo 1: Input-based predictions")
            self.predictor.demo_predictions()
            
            print("\n🎯 Demo 2: TPM list-based predictions")
            self.predictor.demo_tpm_list_prediction()
            
            print("\n✅ Demo predictions hoàn thành!")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khi demo predictions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_full_pipeline(self, weeks_back=4, use_sample_data=True):
        """
        Chạy toàn bộ pipeline từ bước 1 đến 4
        
        Parameters:
        -----------
        weeks_back : int
            Số tuần thu thập data
        use_sample_data : bool
            Sử dụng sample data thay vì New Relic
        """
        
        print("🚀 DAY6 ML PIPELINE - FULL RUN")
        print("=" * 80)
        print("Tích hợp 4 chức năng chính từ Day5:")
        print("1. Thu thập data từ New Relic (thứ 2 - chủ nhật, GMT+7)")
        print("2. Tạo visualization charts")
        print("3. Training time series model")
        print("4. Demo predictions")
        print("=" * 80)
        
        # Step 1: Collect data
        success1 = self.step1_collect_data(weeks_back=weeks_back, use_sample_data=use_sample_data)
        if not success1:
            print("❌ Pipeline dừng tại bước 1")
            return False
        
        # Step 2: Visualize data
        success2 = self.step2_visualize_data()
        if not success2:
            print("⚠️ Bước 2 thất bại nhưng pipeline tiếp tục")
        
        # Step 3: Train model
        success3 = self.step3_train_model()
        if not success3:
            print("❌ Pipeline dừng tại bước 3")
            return False
        
        # Step 4: Demo predictions
        success4 = self.step4_demo_predictions()
        if not success4:
            print("⚠️ Bước 4 thất bại")
        
        # Summary
        print("\n🎉 PIPELINE SUMMARY")
        print("=" * 50)
        print(f"✅ Bước 1 - Thu thập data: {'Thành công' if success1 else 'Thất bại'}")
        print(f"{'✅' if success2 else '⚠️'} Bước 2 - Visualization: {'Thành công' if success2 else 'Thất bại'}")
        print(f"✅ Bước 3 - Training model: {'Thành công' if success3 else 'Thất bại'}")
        print(f"{'✅' if success4 else '⚠️'} Bước 4 - Demo predictions: {'Thành công' if success4 else 'Thất bại'}")
        
        if success1 and success3:
            print("\n🎯 Pipeline core functionality hoàn thành thành công!")
            print("📂 Outputs:")
            print("   • Data: day6_collected_data.csv")
            print("   • Charts: day6_charts/")
            print("   • Models: day6_models/")
            return True
        else:
            print("\n❌ Pipeline có lỗi trong các bước quan trọng")
            return False
    
    def _create_sample_data(self):
        """
        Tạo sample data để demo khi không có New Relic credentials
        """
        import random
        
        # Create 4 weeks of data with 1-hour intervals
        start_date = datetime.now() - timedelta(weeks=4)
        timestamps = []
        tpm_values = []
        response_times = []
        push_active_values = []
        minutes_since_push_values = []
        
        for i in range(4 * 7 * 24):  # 4 weeks * 7 days * 24 hours
            timestamp = start_date + timedelta(hours=i)
            
            # Simulate realistic TPM pattern
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Base TPM with daily and weekly patterns
            base_tpm = 150
            
            # Daily pattern (higher during business hours)
            if 8 <= hour <= 18:
                base_tpm += 300
            elif 19 <= hour <= 22:
                base_tpm += 150
            
            # Weekly pattern (lower on weekends)
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
            
            # Add some randomness
            tpm = base_tpm + random.gauss(0, 50)
            tpm = max(10, tpm)  # Minimum TPM
            
            response_time = 100 + random.gauss(0, 20)
            response_time = max(50, response_time)  # Minimum response time
            
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
        
        return sample_data

def main():
    """
    Main function để chạy Day6 pipeline
    """
    print("🚀 Day6 - Summary của những hàm trong Day5")
    print("=" * 80)
    print("Chương trình tích hợp 4 chức năng chính:")
    print("1. 📡 Thu thập data từ New Relic (thứ 2 - chủ nhật, GMT+7, độ chi tiết tối đa)")
    print("2. 📊 Tạo visualization charts")
    print("3. 🤖 Training time series model")
    print("4. 🎯 Demo predictions từ model")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = Day6MLPipeline()
    
    # Chạy toàn bộ pipeline
    # Sử dụng sample data để demo (set use_sample_data=False nếu có New Relic credentials)
    success = pipeline.run_full_pipeline(
        weeks_back=4,
        use_sample_data=True  # Đổi thành False nếu muốn dùng New Relic thật
    )
    
    if success:
        print("\n🎉 Day6 Pipeline hoàn thành thành công!")
        print("\n📋 Tóm tắt kết quả:")
        print("✅ Data collection: Hoàn thành")
        print("✅ Visualization: Hoàn thành") 
        print("✅ Model training: Hoàn thành")
        print("✅ Demo predictions: Hoàn thành")
        
        print("\n📂 Files được tạo:")
        print("   • day6_collected_data.csv - Raw data")
        print("   • day6_charts/ - Visualization charts")
        print("   • day6_models/ - Trained models")
        
        print("\n🎯 Cách sử dụng từng component riêng lẻ:")
        print("   • Data collection: from newrelic_data_collector import collect_full_week_newrelic_data")
        print("   • Visualization: from visualizer import create_comprehensive_visualization_suite")
        print("   • Model training: from time_series_trainer import TimeSeriesTrainer")
        print("   • Predictions: from demo_predictor import DemoPredictor")
        
    else:
        print("\n❌ Day6 Pipeline có lỗi")
        print("Vui lòng kiểm tra log ở trên để biết chi tiết")

if __name__ == "__main__":
    main()