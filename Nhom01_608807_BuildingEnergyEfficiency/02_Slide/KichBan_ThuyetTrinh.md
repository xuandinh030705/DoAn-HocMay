# Kịch bản thuyết trình — 15 slide, khoảng 12–14 phút

> Cách dùng: mỗi mục là một slide. Phần **Nói** là lời thoại gợi ý (không đọc nguyên văn).
> Phần **Nếu bị hỏi** là điểm cần nhấn khi hội đồng đặt câu hỏi đúng chỗ đó.

---

## Slide 1 — Bìa (30 giây)
**Nói:** Kính thưa thầy/cô và các bạn, nhóm 01 trình bày bài tập lớn môn Học máy: dự đoán hiệu quả năng lượng
của tòa nhà trên bộ dữ liệu UCI ENB2012. Đây là bài toán hồi quy **đa đầu ra**: từ 8 thông số thiết kế,
dự đoán đồng thời tải sưởi và tải làm mát.

## Slide 2 — Bài toán & động lực (45 giây)
**Nói:** Trong thiết kế sơ bộ, kỹ sư phải duyệt rất nhiều phương án hình khối, kính, hướng nhà. Mô phỏng
năng lượng chính xác nhưng chậm. Mục tiêu của nhóm là một mô hình **thay thế** trả lời trong mili-giây,
đủ nhanh để duyệt hàng nghìn phương án. Đầu ra Y1, Y2 có tương quan 0,976 nên phải xử lý như bài toán
đa đầu ra, không phải hai bài toán rời.

## Slide 3 — Dữ liệu & EDA (60 giây)
**Nói:** 768 mẫu, 8 biến vào, 2 biến ra, không thiếu, không trùng. Ba phát hiện quan trọng: (1) X2 = 2·X4 + X3
đúng cho toàn bộ 768 dòng → đa cộng tuyến hoàn hảo, VIF ~10^15; (2) dữ liệu chỉ có **12 cấu hình hình học**,
mỗi cấu hình 64 mẫu — điều này quyết định toàn bộ phần đánh giá về sau; (3) X8 = 0 chỉ có 48 mẫu và luôn
đi với X7 = 0, tức "không kính", không phải một kiểu phân bố kính.
**Nếu bị hỏi:** đa cộng tuyến ảnh hưởng hệ số, không nhất thiết ảnh hưởng dự đoán.

## Slide 4 — Feature engineering có kiểm chứng (60 giây)
**Nói:** Nhóm tạo 12 đặc trưng suy diễn (diện tích kính, tỉ lệ kính/tường, thể tích, aspect ratio…). Nhưng
không đưa vào vì "nghe hợp lý" — nhóm làm ba thực nghiệm: so sánh 4 cách mã hoá (chênh < 0,001 → chọn theo
ngữ nghĩa), so sánh F0/F1 (tuyến tính +0,072; cây −0,004), và ablation (bỏ nhóm kính làm RF giảm 0,069).
**Nếu bị hỏi:** FE không giúp cây — nhóm báo cáo trung thực và giữ F1 vì pipeline thống nhất.

## Slide 5 — Pipeline & chống rò rỉ (45 giây)
**Nói:** Toàn bộ tiền xử lý nằm trong Pipeline: imputer, scaler, one-hot. Nhờ vậy khi predict trên dữ liệu
thô, không thể lệch tiền xử lý. Điểm mấu chốt trên sơ đồ: **tập test chỉ đi vào bước đánh giá cuối** —
không tham gia CV, không tham gia tuning. Nhóm còn lưu MD5 của tập test để chứng minh tính tái lập.

## Slide 6 — Chiến lược đa đầu ra (45 giây)
**Nói:** Nhóm so sánh 6 cấu hình. Với Ridge, ba cách cho kết quả giống hệt — đúng về toán học. Với XGBoost,
native và MultiOutput tương đương, còn RegressorChain **kém hơn** vì lỗi Y1 truyền sang Y2. Kết luận: dùng
native/MultiOutput, tránh lan truyền sai số.

## Slide 7 — 11 mô hình & protocol (40 giây)
**Nói:** Bốn họ mô hình cộng baseline: tuyến tính (Linear/Ridge/Lasso), láng giềng (k-NN), cây (DT, RF,
GradientBoosting, HistGB, XGBoost), và Stacking. Tất cả dùng chung 5-fold trên train, seed 42; lưới nhỏ
GridSearch, lưới lớn RandomizedSearch. Test chỉ dùng một lần.

## Slide 8 — Kết quả CV (50 giây)
**Nói:** Baseline Dummy R² ≈ 0. Tuyến tính đã đạt 0,972 — phần lớn tải năng lượng là quan hệ gần tuyến tính
với diện tích và chiều cao. Nhưng RMSE 1,52 xuống 0,79 khi sang boosting, tức **giảm gần một nửa sai số**
dù R² chỉ tăng 0,02. Nhóm nhấn vào RMSE vì R² đã cao.

## Slide 9 — Tuning & đường cong (40 giây)
**Nói:** Ridge/Lasso chọn alpha cực nhỏ (1e-5) — xác nhận dữ liệu gần như không nhiễu. Validation curve
cho thấy mô hình tuyến tính **underfit** chứ không overfit. Learning curve của XGBoost bão hoà → thêm dữ liệu
cùng 12 thiết kế không giúp; nút thắt là đa dạng hình học.

## Slide 10 — Kết quả test (50 giây)
**Nói:** Đây là lần duy nhất chạm test. XGBoost: R² 0,9935, RMSE 0,706, MAE 0,462 kWh/m², MAPE 1,74%.
So với tuyến tính, RMSE giảm 55%. Y2 khó hơn Y1 (R² 0,988 so với 0,999) vì tải làm mát phụ thuộc bức xạ
qua kính, tương tác với hướng và phân bố kính.

## Slide 11 — Phân tích sai số (50 giây)
**Nói:** Residual đối xứng quanh 0, không có xu hướng. Nhưng sai số tăng theo tải: nhóm Q1 MAE 0,17, nhóm Q4
0,89 — gấp 5 lần. Sáu mẫu sai nhất đều cùng vùng thiết kế (nhà cao, kính 0,25–0,40, X8 = 4) và Y2 bị dự
đoán **thấp** hệ thống 3–5 kWh/m². Đây là lỗi có hướng, cần biên an toàn khi dùng.

## Slide 12 — Giải thích mô hình (50 giây)
**Nói:** Hai công cụ nhất quán: X1 chiếm ưu thế, GLAZING_AREA thứ hai, ASPECT thứ hai ở Y2. Điểm cần nói
rõ: ΔR² của X1 lớn hơn 1 **không phải lỗi**; X5 tương quan 0,89 nhưng permutation ≈ 0 là do hiệu ứng thay
thế. SHAP cho thấy chiều: X1 thấp (nhà kém gọn) đẩy tải lên; kính nhiều làm tăng cả hai tải.

## Slide 13 — Độ bền vững (70 giây) — điểm nhấn
**Nói:** Đây là phần nhóm tâm đắc nhất. Dữ liệu chỉ có 12 hình học, nên random K-Fold luôn có "anh em" trong
train. Nhóm dùng GroupKFold: giữ nguyên cả nhóm hình học ngoài train. Kết quả: Random Forest còn 0,81,
XGBoost 0,35, còn Ridge **−2,37**. Đi ngược trực giác: Ridge ngoại suy được nhưng ngoại suy **sai** vì quan
hệ thật phi tuyến; cây không ngoại suy nên sai số bị chặn. Kết luận: R² 0,99 chỉ đúng trong miền nội suy.
**Nếu bị hỏi:** đây là hạn chế của **dữ liệu**, không phải lỗi code; nhóm chủ động báo cáo.

## Slide 14 — Kết luận & hạn chế (45 giây)
**Nói:** Tóm lại: pipeline hoàn chỉnh, 11 mô hình, 6 chiến lược đa đầu ra, tuning đầy đủ, test một lần,
đóng gói joblib + CLI + Streamlit. Hạn chế: 12 hình học, Stacking chưa tuning, F1 chưa tối ưu cho từng họ,
5 fold chưa đủ lực thống kê. Hướng phát triển: thêm dữ liệu hình học đa dạng, mô hình lai vật lý + ML,
dự đoán khoảng tin cậy.

## Slide 15 — Demo & Q&A (30 giây)
**Nói:** Demo gồm CLI `predict.py` và giao diện Streamlit 3 tab. Mời thầy/cô và các bạn đặt câu hỏi.

---

### Checklist trước khi trình bày
- [ ] Mở sẵn notebook 02 ở phần 4.7 (GroupKFold) để minh hoạ nếu bị hỏi.
- [ ] Mở sẵn `results/final_model_comparison.csv` và `best_model_meta.json`.
- [ ] Chạy thử `streamlit run app.py` trước 10 phút.
- [ ] Không đọc slide; mỗi slide chỉ nói 2–3 ý.
- [ ] Câu chốt: *"Mô hình tốt trong miền dữ liệu đã học, và nhóm biết rõ giới hạn đó."*
