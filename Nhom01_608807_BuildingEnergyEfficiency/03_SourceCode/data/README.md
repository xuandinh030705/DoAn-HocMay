# Dữ liệu — UCI Energy Efficiency Dataset

## File trong thư mục này

| File | Mô tả |
|---|---|
| `ENB2012_data.xlsx` | Bộ dữ liệu gốc, tải nguyên bản từ UCI Machine Learning Repository (ID 242). 768 dòng × 10 cột, không chỉnh sửa. |
| `README.md` | File này — mô tả nguồn, cách tải lại, và từ điển biến. |

## Nguồn gốc

- **Bài báo gốc:** Athanasios Tsanas và Angeliki Xifara (2012).
  *"Accurate quantitative estimation of energy performance of residential buildings using
  statistical machine learning tools"*, **Energy and Buildings**, 49, 560–567.
  DOI: [10.1016/j.enbuild.2012.03.024](https://doi.org/10.1016/j.enbuild.2012.03.024)
- **Kho dữ liệu:** UCI Machine Learning Repository — *Energy Efficiency* (ID 242)
  <https://archive.ics.uci.edu/dataset/242/energy+efficiency>
- **Giấy phép:** CC BY 4.0 (theo công bố của UCI).

## Cách tải lại (nếu cần)

```bash
# Cách 1: tải trực tiếp file zip từ UCI
curl -L "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip" -o energy.zip
unzip energy.zip     # được ENB2012_data.xlsx, chép vào thư mục data/

# Cách 2: dùng gói ucimlrepo
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=242); d.data.features.to_csv('ENB2012_data.csv', index=False)"
```

## Từ điển biến

| Biến | Ý nghĩa | Đơn vị | Miền giá trị |
|---|---|---|---|
| X1 | Relative Compactness | – | 0.62 – 0.98 (12 mức) |
| X2 | Surface Area | m² | 514.5 – 808.5 (12 mức) |
| X3 | Wall Area | m² | 245.0 – 416.5 (7 mức) |
| X4 | Roof Area | m² | 110.25 – 220.5 (4 mức) |
| X5 | Overall Height | m | 3.5 hoặc 7.0 |
| X6 | Orientation | mã hướng | 2, 3, 4, 5 (Bắc/Đông/Nam/Tây) |
| X7 | Glazing Area | tỷ lệ | 0, 0.1, 0.25, 0.4 |
| X8 | Glazing Area Distribution | mã kiểu | 0 – 5 (0 = không kính) |
| **Y1** | **Heating Load** | kWh/m² | 6.01 – 43.10 |
| **Y2** | **Cooling Load** | kWh/m² | 10.90 – 48.03 |

## Đặc điểm cần lưu ý (đã phân tích trong notebook 01)

1. **Không có giá trị thiếu, không có dòng trùng lặp.**
2. **Dữ liệu mô phỏng** (sinh bằng phần mềm Ecotect), không phải đo thực tế → quan hệ gần tất định,
   R² thường rất cao (≈ 0.99) và **không nên** so sánh trực tiếp với bài toán dữ liệu thực.
3. **Chỉ có 12 cấu hình hình học** (X1..X5), mỗi cấu hình lặp 64 lần (4 hướng × 16 tổ hợp X7, X8).
   Vì vậy điểm số với random split bị **lạc quan** — notebook 02 có kiểm tra thêm bằng GroupKFold.
4. **Đa cộng tuyến chính xác:** `X2 = 2·X4 + X3` đúng trên toàn bộ 768 dòng.
5. **X8 = 0 ⟺ X7 = 0** (khi không có kính thì không có "kiểu phân bố").
