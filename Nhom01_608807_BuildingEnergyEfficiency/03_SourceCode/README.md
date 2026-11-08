# Dự đoán Hiệu quả Năng lượng của Tòa nhà bằng Học máy

Bài tập lớn học phần **Học máy — mã học phần 608807**, Nhóm 01, Trường Đại học Xây dựng Hà Nội.

Bài toán **hồi quy đa đầu ra** trên bộ dữ liệu [UCI Energy Efficiency (ENB2012)](https://archive.ics.uci.edu/dataset/242/energy+efficiency):
từ 8 thông số thiết kế (hình khối, diện tích, chiều cao, hướng, kính) dự đoán đồng thời

- **Y1 — Heating Load** (tải sưởi, kWh/m²)
- **Y2 — Cooling Load** (tải làm mát, kWh/m²)

---

## 1. Kết quả chính

Mô hình được chọn: **XGBoost** trong Pipeline (tiền xử lý + mô hình), huấn luyện trên 614 mẫu,
đánh giá **một lần** trên 154 mẫu test độc lập:

| Chỉ số | Y1 (Heating) | Y2 (Cooling) | Trung bình |
|---|---|---|---|
| R² | 0.9987 | 0.9883 | **0.9935** |
| RMSE (kWh/m²) | 0.373 | 1.039 | **0.706** |
| MAE (kWh/m²) | 0.274 | 0.649 | **0.462** |
| MAPE (%) | 1.21 | 2.28 | **1.74** |

- CV 5-fold trên train: **R² = 0.9929 ± 0.0007** (chọn mô hình chỉ dựa trên CV, không dùng test).
- So với hồi quy tuyến tính (R² 0.9707, RMSE 1.560): **RMSE giảm 55%**.
- Thời gian: fit < 1 s, dự đoán ~12 ms cho 154 mẫu.

> **Đọc kèm hạn chế:** dữ liệu chỉ có **12 cấu hình hình học**, mỗi cấu hình lặp 64 lần. Khi đánh giá bằng
> `GroupKFold` (dự đoán cho hình dạng chưa từng gặp), R² sụt mạnh: Random Forest 0.81, XGBoost 0.35,
> Ridge −2.37. Vì vậy R² ≈ 0.99 chỉ có nghĩa **nội suy trong miền 12 thiết kế đã biết**. Chi tiết ở
> notebook 02, mục 10 và báo cáo mục 4.7.

---

## 2. Cấu trúc thư mục

```
03_SourceCode/
├── data/
│   ├── ENB2012_data.xlsx          # dữ liệu gốc UCI (768 × 10)
│   └── README.md                  # nguồn, license, checksum
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb # EDA, chia train/test, 4 thực nghiệm kiểm chứng
│   └── 02_Modeling_Evaluation.ipynb # CV, tuning, test, sai số, SHAP, GroupKFold
├── src/
│   ├── features.py                # NGUỒN DUY NHẤT cho feature engineering + preprocessor
│   └── evaluation.py              # metric, CV helper, chọn mô hình
├── models/
│   ├── best_model.joblib          # fit trên 614 mẫu train — tái lập đúng số trong báo cáo
│   ├── best_model_full_data.joblib# fit trên 768 mẫu — dùng cho demo/triển khai
│   └── best_model_meta.json       # siêu tham số, metric test, MD5 tập test, phiên bản thư viện
├── results/                       # toàn bộ CSV/JSON làm nguồn cho báo cáo
├── figures/                       # fig01–fig15
├── predict.py                     # CLI + API predict()/predict_one()
├── app.py                         # demo Streamlit (3 tab)
├── requirements.txt
└── README.md
```

---

## 3. Cài đặt

Yêu cầu Python ≥ 3.11 (đã kiểm chứng trên 3.14.7).

```powershell
cd 03_SourceCode
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` ghim đúng phiên bản đã dùng khi tạo kết quả (scikit-learn 1.9.1, pandas 3.0.6,
numpy 2.5.3, xgboost 3.4.1, shap 0.54.1, streamlit 1.65.0).

---

## 4. Sử dụng

### 4.1. Dự đoán một công trình

```powershell
python predict.py --x1 0.79 --x2 637 --x3 343 --x4 147 --x5 7 --x6 5 --x7 0.25 --x8 4
```

### 4.2. Dự đoán từ JSON

```powershell
python predict.py --json "{\"X1\":0.79,\"X2\":637,\"X3\":343,\"X4\":147,\"X5\":7,\"X6\":5,\"X7\":0.25,\"X8\":4}"
```

### 4.3. Dự đoán hàng loạt từ CSV

```powershell
python predict.py --csv data/new_buildings.csv --out results/new_predictions.csv
```

File CSV cần đủ 8 cột `X1..X8`. Kết quả thêm hai cột `Y1_pred`, `Y2_pred`.

### 4.4. Không truyền tham số

```powershell
python predict.py
```

Chạy hai ví dụ mẫu để kiểm tra cài đặt.

### 4.5. Demo giao diện

```powershell
streamlit run app.py
```

Ba tab: (1) dự đoán một công trình kèm thanh trượt, (2) dự đoán hàng loạt bằng CSV, (3) thông tin mô hình
(metric, siêu tham số, cảnh báo miền áp dụng).

### 4.6. Dùng như thư viện

```python
from predict import predict_one

y1, y2 = predict_one({"X1": 0.79, "X2": 637, "X3": 343, "X4": 147,
                      "X5": 7, "X6": 5, "X7": 0.25, "X8": 4})
print(y1, y2)   # kWh/m2
```

---

## 5. Tái lập kết quả

Thứ tự bắt buộc (notebook 02 đọc `results/split_indices.json` do notebook 01 sinh ra):

```powershell
jupyter nbconvert --to notebook --execute notebooks/01_EDA_Preprocessing.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/02_Modeling_Evaluation.ipynb --inplace
```

- Toàn bộ seed = 42 (`RANDOM_STATE` trong `src/features.py`).
- Tập test được cố định bằng `results/split_indices.json`; MD5 của chỉ số tập test được lưu trong
  `models/best_model_meta.json` (`test_fingerprint_md5_10`).
- Chạy lại sẽ ghi đè `results/` và `figures/`; báo cáo đọc trực tiếp các file này.

Thời gian chạy tham khảo: notebook 01 ≈ 70 s, notebook 02 ≈ 5–6 phút (CPU 4 nhân).

---

## 6. Phương pháp tóm tắt

| Hạng mục | Lựa chọn |
|---|---|
| Chia dữ liệu | 80/20 stratified theo Y1, seed 42 → 614 train / 154 test |
| Chống rò rỉ | Mọi bước tiền xử lý nằm trong `Pipeline`; fit lại trong từng fold |
| Đặc trưng | 18 biến số (12 đặc trưng suy diễn + 6 gốc) + one-hot X8 → 23 cột |
| Mã hoá | X6 cyclic (sin/cos), X8 one-hot `drop='first'` (chọn theo ngữ nghĩa, không theo điểm) |
| Đa đầu ra | MultiOutputRegressor / native multi-output (thắng RegressorChain) |
| Mô hình | Dummy, Linear, Ridge, Lasso, k-NN, Decision Tree, Random Forest, GradientBoosting, HistGB, XGBoost, Stacking |
| CV | K-Fold 5, shuffle, seed 42, chỉ trên train |
| Tuning | GridSearchCV (lưới ≤ 48) + RandomizedSearchCV (20–30) |
| Chọn mô hình | CV R² cao nhất; nếu chênh < 0.002 thì ưu tiên mô hình đơn giản/nhanh hơn |
| Test | Dùng đúng một lần ở bước cuối |
| Diễn giải | Permutation Importance (20 lần lặp) + SHAP TreeExplainer |
| Độ bền vững | GroupKFold theo 12 cấu hình hình học |

---

## 7. Những điểm cần biết trước khi dùng

1. **Miền áp dụng:** mô hình chỉ đáng tin với công trình nằm trong miền giá trị đã học
   (`X1` 0.62–0.98; `X2` 514.5–808.5; `X3` 245–416.5; `X4` 110.25–220.5; `X5` 3.5–7.0; `X6` 2–5;
   `X7` 0–0.4; `X8` 0–5). `predict.py` và `app.py` sẽ **cảnh báo** khi đầu vào nằm ngoài miền này
   nhưng không chặn.
2. **Sai số không đều:** MAE ở nhóm tải cao (Q4) là 0.89 kWh/m², gấp ~5 lần nhóm tải thấp (Q1 = 0.17).
   Với công trình lớn/nhiều kính, nên cộng biên an toàn.
3. **Lỗi có hướng:** ở vùng `X1 = 0.79, X5 = 7, X7 = 0.25–0.40, X8 = 4`, mô hình **dự đoán thấp** Y2
   khoảng 3–5 kWh/m².
4. **Hai file model:** dùng `best_model.joblib` nếu cần tái lập số trong báo cáo;
   dùng `best_model_full_data.joblib` cho demo vì tận dụng thêm 154 mẫu.
5. **Phiên bản thư viện:** mô hình `joblib` được tạo với scikit-learn 1.9.1 / xgboost 3.4.1. Nếu nạp
   bằng phiên bản cũ hơn có thể bị cảnh báo `InconsistentVersionWarning`.

---

## 8. Tài liệu tham khảo

- Tsanas, A., & Xifara, A. (2012). *Energy and Buildings*, 49, 560–567.
- UCI ML Repository, dataset 242.
- scikit-learn, XGBoost, SHAP documentation.

## 9. Ghi chú học thuật

Mã nguồn được viết cho mục đích học tập. Nhóm không chỉnh sửa tay file kết quả; mọi bảng trong báo cáo
được sinh từ `results/` và `models/best_model_meta.json`.
