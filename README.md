# Dự án: Ứng dụng Machine Learning dự đoán nguy cơ biến chứng tim mạch ở bệnh nhân cao huyết áp từ EMR

## 1. Giới thiệu tổng quan
Dự án nhằm mục tiêu xây dựng và đánh giá các mô hình học máy (Machine Learning) để dự đoán nguy cơ biến chứng tim mạch ở các bệnh nhân tăng huyết áp tại Bệnh viện Hữu Nghị. Dựa trên Hồ sơ bệnh án điện tử (EMR), dự án đề xuất một công cụ hỗ trợ quyết định lâm sàng cho các bác sĩ.

### Mục tiêu cụ thể:
- Phân tích đặc điểm dữ liệu EMR bao gồm: nhân khẩu học, lâm sàng, cận lâm sàng, và tiền sử bệnh.
- Tiền xử lý và làm sạch dữ liệu (xử lý dữ liệu khuyết thiếu, ngoại lai, cân bằng dữ liệu bằng SMOTE).
- Xây dựng và so sánh hiệu suất của 3 mô hình học máy: **Logistic Regression**, **Random Forest**, và **XGBoost**.
- Đánh giá mô hình dựa trên độ nhạy (Recall), F1-score, AUC và giải thích mô hình thông qua **SHAP**.
- Xây dựng giao diện Demo hỗ trợ người dùng cuối bằng **Streamlit**.

## 2. Cấu trúc thư mục dự án

Dự án được tổ chức như sau:

```text
ml-project/
│
├── data/                   # Chứa dữ liệu gốc và dữ liệu đã qua tiền xử lý (Lưu ý: `data/processed/*` được ignore trong Git)
├── models/                 # Chứa các mô hình đã được huấn luyện (Bị ignore trong Git)
├── outputs/                # Kết quả xuất ra như hình ảnh, báo cáo (Bị ignore trong Git)
├── docs/                   # Tài liệu chi tiết dự án (Bị ignore trong Git)
├── context/                # Các file bối cảnh, mô tả dự án và yêu cầu (Bị ignore trong Git)
│   └── context.txt
├── myEnv/                  # Môi trường ảo Python cục bộ (Bị ignore trong Git)
│
├── clean_data.py           # Script đọc dữ liệu gốc, lọc IQR, tổng hợp đặc trưng và trích xuất mã ICD
├── preprocess.py           # Script tiền xử lý (Train/Test split, StandardScaler, SMOTE, Missing value) trên dữ liệu đã tổng hợp
├── EDA_data.py             # Script Phân tích khám phá dữ liệu (EDA) phiên bản 1
├── EDA_data_v2.py          # Script EDA cải tiến với các kiểm định thống kê và trực quan hóa chi tiết
├── create_table_img.py     # Script chuyển đổi bảng thống kê mô tả (CSV) sang hình ảnh
├── train_models.py         # Script huấn luyện, đánh giá và chọn mô hình tối ưu (champion_model.pkl)
├── explain_shap.py         # Script phân tích Explainable AI (SHAP) cho mô hình XGBoost
├── init_db.py              # Script khởi tạo cơ sở dữ liệu SQLite (clinic.db) và data mẫu
├── app.py                  # Mã nguồn ứng dụng Web App giao diện Streamlit
├── read_excel.ipynb        # Jupyter notebook dùng để đọc và khám phá nhanh dữ liệu đầu vào
│
├── requirements.txt        # Danh sách các thư viện cần thiết để chạy dự án
└── README.md               # File tài liệu dự án này
```

## 3. Quy trình kỹ thuật triển khai (CRISP-DM)

### 3.1. Thu thập dữ liệu
- Dữ liệu được trích xuất từ EMR Bệnh viện Hữu Nghị: thông tin hành chính, khám bệnh, xét nghiệm, chẩn đoán, đơn thuốc.
- Thời gian: 01/01/2023 - 28/02/2026. Bệnh nhân có theo dõi tối thiểu 12 tháng.

### 3.2. Làm sạch và tổng hợp dữ liệu (`clean_data.py`)
- **Lọc ngoại lai (IQR) trên dữ liệu thô**: Đọc dữ liệu từ nhiều sheet (`HBa1C`, `LDL`...). Lọc bỏ các giá trị bất thường (do gõ sai) bằng phương pháp IQR.
- **Tổng hợp dữ liệu (Feature Aggregation)**: Tính toán `mean, min, max, std, last` để gom cụm các kết quả xét nghiệm của từng bệnh nhân thành một dòng duy nhất.
- **Trích xuất nhãn mục tiêu (Target)**: Dùng Regex trên mã ICD để trích xuất các bệnh lý nền và gán nhãn mục tiêu. Tuyệt đối loại bỏ các biến gây rò rỉ dữ liệu (`suy_tim`, `dot_quy`, `rung_nhi`) khỏi tập đầu vào (Features) để đảm bảo mô hình dự báo hoàn toàn dựa trên sinh hóa và độ tuổi.

### 3.3. Phân tích khám phá dữ liệu - EDA (`EDA_data_v2.py`)
- Thực hiện trên dữ liệu đã tổng hợp, **trước khi** Scale và SMOTE để giữ nguyên đơn vị y khoa thực tế.
- **Thống kê mô tả**: Mean, median, std, min, max, tần số.
- **Trực quan hóa**: Histogram, boxplot, bar chart, heatmap tương quan (Pearson).
- **Kiểm định thống kê**: t-test / Mann-Whitney U (biến liên tục), Chi-square (biến phân loại).

### 3.4. Tiền xử lý dữ liệu chuẩn bị cho mô hình (`preprocess.py`)
- **Chia dữ liệu (Train/Test split)**: Áp dụng Stratified Split theo tỷ lệ 80/20. Bước này làm **trước tiên** để tránh rò rỉ dữ liệu (Data Leakage).
- **Dữ liệu khuyết thiếu (Missing values)**: Điền biến liên tục bằng trung vị (median) - chỉ fit trên tập Train và áp dụng sang tập Test.
- **Chuẩn hóa & IQR Capping**: Áp dụng Capping ngoại lai và `StandardScaler` cho biến liên tục. Fit trên tập Train, áp dụng sang Test.
- **Mất cân bằng dữ liệu**: Áp dụng **SMOTE** kết hợp Stratified k-fold cross-validation chỉ trên tập huấn luyện.

### 3.5. Lựa chọn đặc trưng (Feature Selection)
- Đảm bảo **Không có Target Leakage**: Không sử dụng các biến chẩn đoán biến chứng (`suy_tim`, `dot_quy`, `rung_nhi`) làm đầu vào.
- Loại biến có `p-value > 0.05` (từ bước EDA) và tương quan chéo `> 0.9`.
- Sử dụng *Feature Importance* từ Random Forest và XGBoost. Kiểm tra VIF cho Logistic Regression để tránh đa cộng tuyến.

### 3.6. Xây dựng & Huấn luyện mô hình (`train_models.py`)
- **Chia dữ liệu**: 80% Train, 20% Test (Stratified Split theo biến mục tiêu).
- **Mô hình**: Logistic Regression, Random Forest, XGBoost tối ưu qua `GridSearchCV` (5-fold stratified CV).
- Trọng số lớp (class weights) và scale_pos_weight được sử dụng kết hợp với SMOTE để xử lý mất cân bằng.

### 3.7. Đánh giá mô hình
- Các chỉ số: Confusion Matrix để lấy **Accuracy**, **Precision**, **Recall**, **F1-score**. Vẽ biểu đồ **ROC-AUC**, Calibration curve & Brier score.
- **Tiêu chí ưu tiên**: Sử dụng **F1-score** làm tiêu chí chọn mô hình tốt nhất (Champion Model) nhằm cân bằng giữa việc không bỏ sót bệnh nhân (Recall) và hạn chế báo động giả (Precision). Đề cao các thuật toán có khả năng giải thích tốt như **XGBoost**.

### 3.8. Giải thích mô hình (XAI)
- **SHAP (SHapley Additive exPlanations)**: Xác định mức độ đóng góp của từng đặc trưng lên kết quả dự đoán.
- Vẽ biểu đồ `SHAP summary plot`, `SHAP dependence plot` cho các đặc trưng quan trọng. Sử dụng 2-3 bệnh nhân làm minh họa (case study).

### 3.9. Ứng dụng Demo (Streamlit)
- Sử dụng **Streamlit** bằng Python để xây dựng giao diện demo.
- Cho phép nhập dữ liệu theo trường đã xác định -> hiển thị xác suất biến chứng tim mạch, mức độ rủi ro, và giải thích các yếu tố ảnh hưởng bằng SHAP.

## 4. Hướng dẫn chạy dự án

1. **Cài đặt thư viện** (nên sử dụng môi trường ảo như `myEnv/`):
   ```bash
   pip install -r requirements.txt
   ```

2. **Làm sạch và tổng hợp dữ liệu (Lọc IQR trên Raw Data)**:
   ```bash
   python clean_data.py --input docs/YCDL.xlsx --output data/preprocess/YCDL_Features_Mapped.xlsx
   ```
   *Script đọc dữ liệu gốc, lọc ngoại lai bằng NaN, tính toán `mean, min, max, std` và gom bệnh nhân thành 1 dòng duy nhất.*

3. **Phân tích khám phá dữ liệu (EDA)**:
   ```bash
   python EDA_data_v2.py --input data/preprocess/YCDL_Features_Mapped.xlsx --output_dir outputs
   ```
   *Thực hiện phân tích thống kê trên dữ liệu thực tế. Các hình ảnh biểu đồ, báo cáo thống kê sẽ được lưu vào thư mục `outputs/`.*

4. **Tiền xử lý chuẩn bị cho mô hình (Train/Test split, Chuẩn hóa, SMOTE)**:
   ```bash
   python preprocess.py --input data/preprocess/YCDL_Features_Mapped.xlsx --output_dir data/processed --model_dir models
   ```
   *Thực hiện chia tập Train/Test, Imputation, Capping, Scaling, và SMOTE. Artifacts xử lý sẽ được lưu vào `models/preprocessing_pipeline.pkl`.*

5. **Huấn luyện mô hình**:
   ```bash
   python train_models.py --processed_dir data/processed --stat_csv outputs/statistical_summary.csv --model_dir models --result_dir outputs/model_results
   ```
   *Tự động loại bỏ các biến không có ý nghĩa thống kê (p >= 0.05). Mô hình hiệu quả nhất (XGBoost) sẽ được lựa chọn và lưu vào thư mục `models/`.*

6. **Giải thích mô hình (Explainable AI - SHAP)**:
   ```bash
   python explain_shap.py
   ```
   *Xuất các biểu đồ SHAP Global và Local vào thư mục `outputs/shap_plots/`.*

7. **Khởi tạo Cơ sở Dữ liệu cho App**:
   ```bash
   python init_db.py
   ```
   *Tạo database SQLite `data/clinic.db` và thêm 3 bệnh nhân mẫu để test hệ thống.*

8. **Chạy ứng dụng Web Demo (Streamlit)**:
   ```bash
   streamlit run app.py --server.port 3636
   ```
   *Ứng dụng Web tải tự động mô hình và pipeline xử lý. Nhập liệu, tính toán features, và sinh báo cáo SHAP ngay lập tức.*

nvq
