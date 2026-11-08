# Dự đoán Hiệu quả Năng lượng của Tòa nhà bằng Học máy
### (Building Energy Efficiency Prediction using Machine Learning)

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-red.svg)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Bài tập lớn môn:** Học máy — **Mã học phần:** 608807  
> **Đơn vị:** Khoa Công nghệ Thông tin — Trường Đại học Xây dựng Hà Nội (HUCE)  
> **Nhóm thực hiện:** Nhóm 01  

---

## 📌 Tổng quan Đề tài

Đồ án giải quyết bài toán **Hồi quy đa đầu ra (Multi-Output Regression)** trên bộ dữ liệu chuẩn mực [UCI Energy Efficiency (ENB2012)](https://archive.ics.uci.edu/dataset/242/energy+efficiency):
Từ 8 thông số thiết kế kiến trúc và hình học công trình ($X_1 \dots X_8$), dự đoán đồng thời hai biến mục tiêu:
- **$Y_1$ — Heating Load** (Tải sưởi ấm, $\text{kWh/m}^2/\text{năm}$)
- **$Y_2$ — Cooling Load** (Tải làm mát, $\text{kWh/m}^2/\text{năm}$)

Mô hình đóng vai trò **Surrogate Model** (mô hình thay thế), giúp các kiến trúc sư và kỹ sư đánh giá hiệu năng năng lượng tức thì trong **~12 ms**, thay thế cho các phần mềm mô phỏng nhiệt truyền thống (EnergyPlus, Ecotect) vốn mất hàng chục phút cho mỗi phương án.

---

## 🏆 Kết quả Thực nghiệm Chính

Mô hình **Champion** được lựa chọn: **XGBoost Multi-Output Pipeline** (đóng gói toàn bộ tiền xử lý + ColumnTransformer + XGBoost Tuned), huấn luyện trên 614 mẫu Train và đánh giá **duy nhất 1 lần** trên 154 mẫu Test độc lập:

| Chỉ số đánh giá | $Y_1$ (Heating Load) | $Y_2$ (Cooling Load) | Trung bình Đa mục tiêu |
|---|:---:|:---:|:---:|
| **$R^2$ Score** | **0.9987** | **0.9883** | **0.9935** |
| **RMSE** ($\text{kWh/m}^2$) | **0.373** | **1.039** | **0.706** |
| **MAE** ($\text{kWh/m}^2$) | **0.274** | **0.649** | **0.462** |
| **MAPE** (%) | **1.21%** | **2.28%** | **1.74%** |
| **Thời gian suy luận** | — | — | **~12 ms / 154 mẫu** |

> 🔍 **Điểm nhấn khoa học & phản biện:** Dữ liệu UCI thực chất được sinh từ **12 cấu hình hình học cơ sở**. Khi thử nghiệm độ bền vững bằng `GroupKFold` (ép mô hình ngoại suy sang hình dạng chưa từng thấy), $R^2$ sụt giảm mạnh (Random Forest còn 0.81, XGBoost 0.35, Ridge rơi xuống -2.37). Do đó, nhóm khẳng định $R^2 \approx 0.99$ là kết quả **nội suy** trong miền thiết kế đã học.

---

## 📂 Cấu trúc Thư mục Dự án

```text
Nhom01_608807_BuildingEnergyEfficiency/
├── 01_Report/
│   ├── Bao_cao_BTL_Nhom01.pdf            # Báo cáo kỹ thuật IEEE/A4 (30 trang có bìa & phân công)
│   ├── Bao_cao_BTL_Nhom01.docx           # File Word gốc để chỉnh sửa
│   ├── BaoCao_BTL_HocMay_608807.pdf
│   └── BaoCao_BTL_HocMay_608807.docx
├── 02_Slide/
│   ├── Slide_Thuyet_trinh_Nhom01.pdf     # Slide thuyết trình PDF 15 trang chuẩn 16:9
│   ├── Slide_Thuyet_trinh_Nhom01.pptx    # Slide PowerPoint có thể trình chiếu trực tiếp
│   ├── KichBan_ThuyetTrinh.md            # Kịch bản thuyết trình chi tiết 15 slide
│   └── BoCauHoi_PhanBien.md              # Bộ 20 câu hỏi phản biện chuyên sâu kèm đáp án
├── 03_SourceCode/
│   ├── data/
│   │   ├── ENB2012_data.xlsx             # Dữ liệu gốc UCI (768 mẫu × 10 cột)
│   │   └── README.md                     # Checksum SHA-256 & mô tả biến
│   ├── figures/                          # 15 biểu đồ báo cáo và slide (PNG độ phân giải cao)
│   ├── models/
│   │   ├── best_model.joblib             # Pipeline XGBoost tối ưu trên tập Train
│   │   ├── best_model_full_data.joblib   # Pipeline huấn luyện trên 100% dữ liệu
│   │   └── best_model_meta.json          # Metadata siêu tham số & chỉ số
│   ├── notebooks/
│   │   ├── 01_EDA_Preprocessing.ipynb    # EDA, Tiền xử lý, Feature Engineering & Baseline
│   │   └── 02_Modeling_Evaluation.ipynb  # Thử nghiệm 11 mô hình, CV, Tuning, SHAP, GroupKFold
│   ├── results/                          # Toàn bộ 17 file kết quả thực nghiệm CSV/JSON
│   ├── src/                              # Module tái sử dụng: features.py & evaluation.py
│   ├── app.py                            # Web Dashboard Streamlit 3 tab tương tác
│   ├── predict.py                        # Công cụ suy luận dòng lệnh CLI / JSON / CSV
│   ├── requirements.txt                  # Phiên bản thư viện chính xác
│   └── README.md                         # Hướng dẫn chi tiết mã nguồn
└── 04_Video_Demo/
    ├── KichBan_Quay_Video_3Phut.md       # Kịch bản quay video demo 180s từng giây
    ├── demo_walkthrough.py               # Script chạy mẫu tự động khi quay video
    └── README.md                         # Hướng dẫn thiết lập quay màn hình
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Thử

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
git clone https://github.com/xuandinh030705/DoAn-HocMay.git
cd DoAn-HocMay/Nhom01_608807_BuildingEnergyEfficiency/03_SourceCode
pip install -r requirements.txt
```

### 2. Dự đoán nhanh qua Dòng lệnh (CLI)
Mô hình tự động nạp pipeline từ `models/best_model.joblib`:
```bash
# Chạy ví dụ mặc định:
python predict.py

# Dự đoán cho một công trình cụ thể:
python predict.py --x1 0.98 --x2 514.5 --x3 294.0 --x4 110.25 --x5 7.0 --x6 2 --x7 0.10 --x8 1

# Dự đoán hàng loạt từ file CSV:
python predict.py --csv data/sample_input.csv --out results/predictions.csv
```

### 3. Khởi chạy Giao diện Web (Streamlit)
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại `http://localhost:8501` với 3 tab:
- **Tab 1:** Dự đoán đơn lẻ bằng thanh trượt trực quan kèm biểu đồ phân tích.
- **Tab 2:** Dự đoán hàng loạt bằng file CSV tải lên.
- **Tab 3:** Thông số mô hình, siêu tham số tối ưu và khuyến cáo miền áp dụng an toàn.

---

## 👥 Thành viên Thực hiện
- **Nhóm 01 — Lớp 608807**
- Trường Đại học Xây dựng Hà Nội (HUCE)
- GVHD: Giảng viên phụ trách học phần Học máy
