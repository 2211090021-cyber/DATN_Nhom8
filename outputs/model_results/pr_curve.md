# Báo cáo Đánh giá Mô hình bằng Biểu đồ Precision-Recall (PR Curve)

Báo cáo này phân tích hiệu năng của các mô hình dự báo biến chứng tim mạch (**Logistic Regression, Random Forest, XGBoost**) dựa trên biểu đồ Precision-Recall (PR) thu được từ tập kiểm thử độc lập (Test Set).

---

## 1. Bối cảnh & Vai trò của PR Curve

Trong bài toán dự báo biến chứng tim mạch này, tập dữ liệu kiểm thử có đặc điểm **mất cân bằng lớp cực kỳ nghiêm trọng**:
* **Tổng số mẫu tập Test**: $2,471$ ca.
* **Số ca biến chứng (Complication - Lớp 1)**: $127$ ca (chiếm khoảng **$5.14\%$**).
* **Số ca không biến chứng (No Complication - Lớp 0)**: $2,344$ ca (chiếm **$94.86\%$**).

### Tại sao PR Curve quan trọng hơn ROC-AUC ở đây?
* **ROC-AUC** có thể cho kết quả quá lạc quan vì False Positive Rate (FPR) có mẫu số là số ca không biến chứng rất lớn ($2,344$ ca). Một số lượng lớn cảnh báo giả (False Positives) vẫn sẽ cho ra giá trị FPR nhỏ, làm tăng AUC một cách ảo tưởng.
* **PR Curve** tập trung trực tiếp vào lớp thiểu số (Complication) bằng cách so sánh **Precision** (Tỷ lệ cảnh báo đúng trên tổng số cảnh báo) và **Recall** (Tỷ lệ phát hiện được ca bệnh trên tổng số ca bệnh thực tế). Đường cơ sở ngẫu nhiên (No Skill Baseline) của PR Curve tương ứng với tỷ lệ mẫu dương tính trong tập dữ liệu: **$0.051$**.

---

## 2. Bảng Tổng hợp Kết quả Đánh giá trên Tập Test

Dưới đây là so sánh chi tiết giữa Average Precision (AP) từ PR Curve và các chỉ số phân lớp chính (ở ngưỡng mặc định $0.5$):

| Mô hình | Average Precision (AP) | F1-Score | Recall (Sensitivity) | Precision | ROC-AUC | Brier Score (Độ hiệu chuẩn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.1243** | **0.1698** | 0.2520 | **0.1280** | **0.6504** | **0.0940** (Tốt nhất) |
| **Logistic Regression** | 0.1144 | 0.1463 | **0.5197** | 0.0852 | 0.6440 | 0.1966 |
| **Random Forest** | 0.0919 | 0.1306 | 0.2756 | 0.0856 | 0.6443 | 0.1281 |
| *No Skill (Baseline)* | *0.0514* | *N/A* | *1.0000* | *0.0514* | *0.5000* | *N/A* |

---

## 3. Phân tích Chi tiết Đường biểu diễn PR Curve

Dựa trên hình ảnh [07_pr_curves.png](file:///d:/Working/Coding/ml-project/outputs/model_results/07_pr_curves.png), ta có các nhận định sau:

### 3.1. XGBoost (AP = 0.1243) — Mô hình tối ưu nhất
* **Nhận xét**: XGBoost nằm ở vị trí cao nhất trên đồ thị PR ở phần lớn các dải ngưỡng. Điều này được thể hiện qua chỉ số AP cao nhất ($0.1243$, gấp khoảng $2.4$ lần so với baseline ngẫu nhiên).
* **Đặc điểm**: Ở ngưỡng mặc định, mô hình đạt độ chính xác Precision tốt nhất ($12.8\%$), nghĩa là giảm bớt các cảnh báo giả cho bác sĩ lâm sàng, tuy nhiên khả năng bao phủ ca bệnh (Recall) chỉ đạt $25.2\%$.

### 3.2. Logistic Regression (AP = 0.1144) — Cân nhắc cho sàng lọc diện rộng
* **Nhận xét**: Mặc dù AP thấp hơn XGBoost một chút, Logistic Regression thể hiện ưu thế vượt trội về khả năng phát hiện bệnh ở các ngưỡng thấp hơn, đạt Recall cao nhất là **$51.97\%$**.
* **Đặc điểm**: Đường PR của Logistic Regression duy trì mức Precision ổn định tốt khi kéo Recall lên cao. Tuy nhiên, ở ngưỡng mặc định, Precision chỉ là $8.52\%$ (khoảng 12 ca cảnh báo mới có 1 ca đúng).

### 3.3. Random Forest (AP = 0.0919) — Hiệu năng kém nhất
* **Nhận xét**: Đường PR của Random Forest hầu như nằm dưới hai mô hình còn lại. Mô hình này gặp khó khăn trong việc phân biệt ranh giới xác suất trên dữ liệu mất cân bằng nặng này, dẫn tới AP thấp nhất.

---

## 4. Ý nghĩa Lâm sàng & Khuyến nghị Điều chỉnh Ngưỡng (Threshold Tuning)

Trong y tế, việc áp dụng mô hình cần cân đối giữa hai yếu tố:
1. **Bỏ sót bệnh nhân (False Negatives)**: Nguy hiểm tính mạng nếu bệnh nhân có nguy cơ cao biến chứng nhưng mô hình dự đoán là bình thường.
2. **Cảnh báo giả quá nhiều (False Positives)**: Gây quá tải cho hệ thống y tế và tâm lý lo lắng không đáng có cho bệnh nhân, giảm độ tin cậy của bác sĩ vào hệ thống hỗ trợ quyết định.

### Đề xuất giải pháp thực tế:
* **Không dùng ngưỡng mặc định $0.5$**: Cần thực hiện quét ngưỡng (Threshold Scanning) trên đường PR Curve để tìm điểm cắt tối ưu. 
  * Nếu ưu tiên **sàng lọc (Screening)** không bỏ sót ca bệnh, nên hạ ngưỡng quyết định của **Logistic Regression** hoặc **XGBoost** xuống để đạt Recall $\ge 70\%$, chấp nhận Precision giảm xuống mức $\approx 7-8\%$.
  * Nếu ưu tiên **can thiệp sâu chuyên khoa (Intervention)** với chi phí điều trị lớn, nên tăng ngưỡng của **XGBoost** để nâng Precision lên trên $20\%$, chấp nhận Recall thấp hơn.
