Dưới đây là ý nghĩa và tác động của từng tham số trong RandomForestRegressor cấu hình bạn đưa ra:

- n_estimators=500
  - Ý nghĩa: Số lượng cây trong rừng.
  - Tăng: Giảm phương sai (ổn định hơn), thường cải thiện độ chính xác; tốn thời gian và RAM hơn, lợi ích giảm dần sau một ngưỡng.
  - Giảm: Nhanh hơn, nhưng dự đoán kém ổn định hơn.

- max_depth=16
  - Ý nghĩa: Độ sâu tối đa của mỗi cây.
  - Tăng (hoặc None): Cây phức tạp hơn, dễ fit sát dữ liệu huấn luyện → nguy cơ overfit tăng.
  - Giảm: Cây nông hơn, tăng bias nhưng giảm variance → tổng quan ổn hơn nếu dữ liệu nhiễu hoặc ít mẫu.

- min_samples_leaf=4
  - Ý nghĩa: Số mẫu tối thiểu ở lá.
  - Tăng: Lá “to” hơn, làm mượt dự đoán (ít cực trị) → giảm overfit, tăng bias.
  - Giảm: Lá nhỏ, cây nhạy với nhiễu → dễ overfit nhưng có thể bắt pattern nhỏ.

- min_samples_split=8
  - Ý nghĩa: Số mẫu tối thiểu để một node được chia.
  - Tăng: Ít phép chia hơn, cây đơn giản hơn → giảm overfit, tăng bias.
  - Giảm: Nhiều phép chia hơn, cây phức tạp hơn → nguy cơ overfit.

- max_features='sqrt'
  - Ý nghĩa: Số đặc trưng được xét tại mỗi split. 'sqrt' = căn bậc hai số feature.
  - Tăng (ví dụ 'auto' cho regression ~ n_features, hoặc số thập phân 0.8): Cây dễ giống nhau hơn, giảm tính ngẫu nhiên → có thể giảm đa dạng, tăng nguy cơ overfit.
  - Giảm (ví dụ 0.3): Tăng ngẫu nhiên, tăng đa dạng giữa cây → thường giảm overfit, đôi khi giảm hiệu năng nếu quá thấp.

- bootstrap=True
  - Ý nghĩa: Mỗi cây huấn luyện trên mẫu bootstrap (lấy mẫu có hoàn lại).
  - Bật: Tăng tính ngẫu nhiên, đa dạng mô hình; cho phép dùng oob_score.
  - Tắt: Dùng toàn bộ dữ liệu cho từng cây → cây giống nhau hơn, dễ overfit.

- oob_score=True
  - Ý nghĩa: Tính điểm Out-Of-Bag (xấp xỉ cross-validation) để ước lượng hiệu năng.
  - Bật: Có thêm chỉ số kiểm chứng nhanh mà không cần tách tập; tốn thêm thời gian tính.
  - Tắt: Huấn luyện nhanh hơn nhưng không có OOB estimate.

- random_state=42
  - Ý nghĩa: Seed ngẫu nhiên để tái lập kết quả.
  - Đổi giá trị: Kết quả thay đổi theo hạt giống; nên cố định để tái hiện.

- n_jobs=-1
  - Ý nghĩa: Số luồng CPU sử dụng.
  - -1: Dùng tối đa tài nguyên CPU → nhanh hơn.
  - Số nhỏ hơn: Tiết kiệm tài nguyên, chạy chậm hơn (có ích khi chạy cùng tác vụ khác).

Một vài lưu ý phối hợp tham số:
- Nếu dữ liệu nhiễu/spike lớn: tăng min_samples_leaf, min_samples_split; giữ max_depth vừa phải; max_features='sqrt' là hợp lý.
- Nếu underfit (sai số cao cả train và val): thử tăng max_depth hoặc giảm min_samples_leaf/split; tăng n_estimators (nếu còn tài nguyên).
- Nếu overfit (train tốt nhưng val kém): giảm max_depth hoặc tăng min_samples_leaf/split; giữ bootstrap=True; có thể giảm max_features. OOB score giúp phát hiện nhanh tình trạng này.



==============


Dưới đây là ý nghĩa và tác động của từng tham số trong GradientBoostingRegressor theo cấu hình bạn nêu:

- loss='huber' (hoặc 'absolute_error', 'squared_error', 'quantile')
  - Ý nghĩa: Hàm mất mát dùng để cập nhật cây.
  - huber: Robust với outlier (kết hợp MAE và MSE).
    - Ảnh hưởng: Giảm tác động của spike, dự đoán ổn định hơn. Có tham số alpha đi kèm.
  - absolute_error (MAE): Rất robust, nhưng tối ưu khó hơn → hội tụ chậm, có thể cần nhiều estimators hơn.
  - squared_error (MSE): Nhạy với outlier, dễ overfit nếu có spike.
  - quantile: Dùng cho dự báo phân vị (P90/P95…), kiểm soát thiên lệch rủi ro.

- alpha=0.9
  - Ý nghĩa: Ngưỡng chuyển giữa MAE và MSE của Huber; hoặc mức phân vị khi loss='quantile'.
  - Tăng (gần 1.0): Mạnh tay hơn với outlier (ít bị kéo bởi điểm cao), có thể hơi “bảo thủ”.
  - Giảm (gần 0.5–0.8): Gần MSE hơn, nhạy với outlier hơn, có thể chính xác hơn khi dữ liệu ít nhiễu.

- learning_rate=0.05
  - Ý nghĩa: Hệ số thu nhỏ đóng góp của mỗi cây (shrinkage).
  - Giảm: Học chậm hơn, cần tăng n_estimators để đạt chất lượng tương tự; thường tổng quát hóa tốt hơn.
  - Tăng: Học nhanh, ít cây hơn; dễ overfit nếu quá lớn.

- n_estimators=600
  - Ý nghĩa: Số lượng cây boosting.
  - Tăng: Mô hình mạnh hơn, rủi ro overfit nếu learning_rate quá lớn; tốn thời gian hơn.
  - Giảm: Nhanh hơn, nhưng có thể underfit nếu learning_rate nhỏ.

- max_depth=3
  - Ý nghĩa: Độ sâu của cây yếu (base learners).
  - Tăng: Mỗi cây phức tạp hơn, có thể học tương tác cao; tăng nguy cơ overfit.
  - Giảm: Đơn giản, thiên bias; phù hợp dữ liệu nhiễu để tổng quát tốt.

- subsample=0.8
  - Ý nghĩa: Tỷ lệ mẫu sử dụng mỗi bước boosting (stochastic gradient boosting).
  - Giảm (0.5–0.8): Tăng ngẫu nhiên, giảm phương sai/overfit; quá thấp có thể tăng bias.
  - Tăng (gần 1.0): Ổn định hơn, nhưng dễ overfit hơn trong dữ liệu nhiễu.

- max_features=None
  - Ý nghĩa: Số đặc trưng xét khi tìm split.
  - None: Dùng tất cả feature → cây “mạnh” hơn, nhưng dễ giống nhau → tăng risk overfit.
  - Giá trị khác ('sqrt', 'log2' hoặc tỷ lệ float): Tăng ngẫu nhiên, giảm overfit; nếu quá nhỏ có thể underfit.

- min_samples_leaf=10
  - Ý nghĩa: Số mẫu tối thiểu ở lá của mỗi cây yếu.
  - Tăng: Lá lớn, mượt hơn, ít nhạy outlier → giảm overfit, tăng bias.
  - Giảm: Bắt chi tiết nhỏ, có thể overfit nếu quá thấp.

- random_state=42
  - Ý nghĩa: Hạt giống ngẫu nhiên để tái lập kết quả.
  - Đổi seed: Kết quả có thể khác nhẹ.

- validation_fraction=0.1
  - Ý nghĩa: Tỷ lệ dữ liệu giữ làm validation bên trong để early stopping.
  - Tăng: Nhiều dữ liệu để dừng sớm chính xác hơn nhưng ít dữ liệu training hơn.
  - Giảm: Nhiều dữ liệu cho train nhưng early stopping kém tin cậy.

- n_iter_no_change=20
  - Ý nghĩa: Số vòng lặp không cải thiện trước khi dừng sớm.
  - Tăng: Chờ lâu hơn, ít rủi ro dừng quá sớm nhưng tốn thời gian; có thể overfit nhẹ.
  - Giảm: Dừng nhanh, tiết kiệm thời gian; có rủi ro underfit nếu tín hiệu nhiễu.

- tol=1e-4
  - Ý nghĩa: Ngưỡng cải thiện tối thiểu để coi là “có tiến bộ”.
  - Giảm (nhỏ hơn): Cần cải thiện nhỏ cũng tiếp tục huấn luyện → lâu hơn, có thể tốt hơn hoặc overfit.
  - Tăng: Dừng sớm hơn, nhanh hơn nhưng có thể underfit.

Cách điều chỉnh thực tế:
- Nếu thấy overfit: giảm max_depth, tăng min_samples_leaf, giảm max_features (ví dụ 'sqrt' hoặc 0.6), giảm learning_rate và tăng n_estimators, giảm subsample một chút (0.7–0.8), giữ loss='huber' và có thể tăng alpha (0.9→0.95).
- Nếu underfit: tăng n_estimators, tăng max_depth (3→4), giảm min_samples_leaf (10→5), tăng max_features, tăng nhẹ learning_rate (0.05→0.07), tăng subsample (0.8→0.9).
- Dữ liệu có nhiều spike/outlier: dùng loss='huber' hoặc 'absolute_error'; giữ learning_rate thấp; min_samples_leaf cao hơn; subsample < 1; alpha ~ 0.9–0.95.

=======================
Ý nghĩa và mục đích của hàm vẽ learning_curves

Mục đích
- Đánh giá khả năng học của “best model” cho từng mục tiêu dự báo (tpm_5min, tpm_10min, tpm_15min) theo kích thước dữ liệu huấn luyện.
- Chẩn đoán nhanh mô hình đang overfit hay underfit để quyết định điều chỉnh siêu tham số hoặc dữ liệu.

Hàm làm gì
- Chuẩn bị dữ liệu:
  - X: các feature trong feature_columns, điền thiếu bằng 0.
  - y: cột target tương ứng, nếu thiếu sẽ fallback về cột tpm.
- Chọn mô hình tốt nhất cho từng target từ best_models.
  - Nếu là SVR, chuẩn hóa X bằng scaler phù hợp trước khi vẽ.
- Tính learning curve:
  - Dùng sklearn.learning_curve với 5-fold CV, thước đo R², train_sizes từ 10% đến 100% (10 mốc).
  - Lấy trung bình và độ lệch chuẩn của điểm số train và validation.
- Vẽ biểu đồ:
  - Đường màu xanh: điểm số trên tập huấn luyện (training score) theo kích thước dữ liệu.
  - Đường màu đỏ: điểm số cross-validation (validation score) theo kích thước dữ liệu.
  - Vùng mờ là ±1 std cho mỗi đường, thể hiện độ ổn định giữa các fold.
- Lưu ảnh ra charts/learning_curves_real_data_<timestamp>.png và hiển thị.

Cách diễn giải
- Underfitting (bias cao):
  - Cả training và validation score đều thấp và hội tụ gần nhau khi tăng dữ liệu.
  - Hướng xử lý: tăng độ phức tạp mô hình (max_depth cao hơn, nhiều estimators hơn), thêm/đổi feature, tăng learning_rate (với GB) vừa phải, giảm regularization.
- Overfitting (variance cao):
  - Training score cao nhưng validation score thấp; khoảng cách giữa hai đường lớn, ít thu hẹp khi tăng dữ liệu.
  - Hướng xử lý: tăng regularization/giảm độ phức tạp (GB: giảm max_depth, tăng min_samples_leaf, giảm max_features, giảm learning_rate và tăng n_estimators); dùng subsample < 1; thêm dữ liệu/augmentation; loại outlier hoặc dùng loss robust.
- Dư liệu chưa đủ:
  - Validation score cải thiện rõ khi tăng train_size và khoảng cách với training score thu hẹp.
  - Hướng xử lý: thu thập thêm dữ liệu, hoặc dùng kỹ thuật giảm variance (subsample, regularization).
- R² âm hoặc dao động mạnh:
  - Mô hình kém trên CV hoặc không ổn định theo fold.
  - Hướng xử lý: xem lại pipeline feature (lag/rolling có rò rỉ?), scale (với SVR), chất lượng nhãn, phân phối outlier (cân nhắc loss='huber' hoặc robust scaler).

Vì sao quan trọng
- Cho thấy mối quan hệ “độ phức tạp mô hình – lượng dữ liệu – hiệu năng”, giúp chọn đúng hướng tối ưu: tăng dữ liệu hay tinh chỉnh siêu tham số/đặc trưng.
- Tránh tối ưu mù quáng; bạn thấy được khi nào mô hình đạt “điểm bão hòa” và khi nào cần đổi kiến trúc/đặc trưng.