# Kịch Bản Quay Video Demo 3 Phút (180 Giây)
### Đề tài 2: Dự đoán Hiệu quả Năng lượng của Tòa nhà bằng Học máy
**Học phần:** Học máy — Mã học phần: 608807 | **Nhóm thực hiện:** Nhóm 01 (Trường ĐH Xây dựng Hà Nội)

---

## 1. Thiết lập trước khi bấm máy quay (Pre-recording Checklist)
- **Độ phân giải màn hình:** 1920 × 1080 (Full HD), tỷ lệ 16:9.
- **Phần mềm quay video gợi ý:** OBS Studio, Windows Game Bar (`Win + G`), hoặc Clipchamp.
- **Bố cục màn hình đề xuất:** 
  - Mở sẵn VS Code (chia đôi: 1 bên là terminal PowerShell, 1 bên là `predict.py` hoặc notebook).
  - Trình duyệt mở sẵn: Slide thuyết trình PDF và Tab Web Streamlit (`http://localhost:8501`).
- **Lệnh khởi động sẵn trên terminal phụ:**
  ```powershell
  cd D:\MCLN\Nhom01_608807_BuildingEnergyEfficiency\03_SourceCode
  streamlit run app.py
  ```

---

## 2. Kịch bản chi tiết từng giây (0:00 – 3:00)

### Phân cảnh 1: Đặt vấn đề & Mục tiêu đồ án (0:00 – 0:30 | 30 giây)
- **Hình ảnh hiển thị:** Slide 1 (Bìa) chuyển sang Slide 2 (Bài toán & Động lực).
- **Thao tác chuột:** Chỉ nhẹ vào sơ đồ 8 đặc trưng kiến trúc $X_1 \dots X_8$ dự đoán đồng thời 2 biến mục tiêu $Y_1$ (Heating Load) và $Y_2$ (Cooling Load).
- **Lời thoại gợi ý:**
  > *"Kính chào thầy cô và các bạn! Nhóm 01 xin trình bày sản phẩm đồ án Học máy: Dự đoán Hiệu quả Năng lượng Tòa nhà trên bộ dữ liệu UCI ENB2012.*  
  > *Trong thiết kế kiến trúc sơ bộ, các phần mềm mô phỏng nhiệt truyền thống mất hàng chục phút cho mỗi phương án. Mục tiêu của nhóm là phát triển một mô hình học máy thay thế (surrogate model) với tốc độ phản hồi tính bằng mili-giây, giải quyết bài toán hồi quy đa đầu ra với độ chính xác cao."*

---

### Phân cảnh 2: Điểm sáng EDA & Pipeline chống rò rỉ (0:30 – 1:15 | 45 giây)
- **Hình ảnh hiển thị:** Chuyển nhanh qua Slide 3 (Heatmap tương quan) và Slide 5 (Sơ đồ Pipeline `fig15_pipeline_diagram.png`).
- **Thao tác chuột:** Rê chuột qua mối liên hệ $X_2 = 2 \cdot X_4 + X_3$ và nhánh phân tách dữ liệu trong Pipeline.
- **Lời thoại gợi ý:**
  > *"Qua phân tích khám phá EDA, nhóm phát hiện 3 đặc trưng quan trọng: thứ nhất, hiện tượng đa cộng tuyến hoàn hảo giữa diện tích bề mặt, tường và mái; thứ hai, dữ liệu thực chất chỉ gồm 12 cấu hình hình học cơ sở; và thứ ba, biến X8 mang giá trị 0 chỉ khi công trình hoàn toàn không có kính.*  
  > *Để chống rò rỉ dữ liệu tuyệt đối, nhóm đóng gói toàn bộ quy trình từ tiền xử lý, chuẩn hóa, mã hóa One-hot vào Scikit-learn Pipeline. Tập Test độc lập 20% được khóa bằng mã băm MD5 và hoàn toàn không tham gia vào Cross-Validation hay tinh chỉnh siêu tham số."*

---

### Phân cảnh 3: Kết quả thực nghiệm & Phản biện ngoại suy (1:15 – 1:55 | 40 giây)
- **Hình ảnh hiển thị:** Slide 9 (Biểu đồ so sánh 11 mô hình) và Slide 12 (Biểu đồ GroupKFold `fig14_robustness_groupkfold.png`).
- **Thao tác chuột:** Nhấn vào cột RMSE của XGBoost và điểm sụt giảm của Ridge trong GroupKFold.
- **Lời thoại gợi ý:**
  > *"Nhóm đã thử nghiệm 11 mô hình thuộc 4 dòng thuật toán. Mô hình Champion là XGBoost Multi-Output Pipeline, đạt R² 0.9935 và RMSE chỉ 0.706 kWh/m² trên tập Test độc lập, giảm tới 55% sai số so với mô hình hồi quy tuyến tính chuẩn.*  
  > *Đặc biệt, nhóm thực hiện thử nghiệm phản biện bằng GroupKFold trên 12 nhóm hình học. Kết quả cho thấy hồi quy tuyến tính ngoại suy sai lệch hoàn toàn với R² âm, trong khi các mô hình cây giữ được sai số hữu hạn. Điều này chứng minh tính minh bạch khoa học của nhóm: R² xấp xỉ 0.99 chỉ có giá trị nội suy trong không gian hình khối đã biết."*

---

### Phân cảnh 4: Live Demo thực tế — CLI & Web Dashboard (1:55 – 2:45 | 50 giây)
- **Hình ảnh hiển thị:** Chuyển sang màn hình VS Code Terminal và Trình duyệt Web Streamlit.
- **Thao tác thực tế:**
  1. **Trên Terminal (20s):** Gõ lệnh dự đoán mẫu:
     ```powershell
     python predict.py
     ```
     Chỉ vào kết quả in ra màn hình: Y1 = 15.62 kWh/m², Y2 = 21.54 kWh/m² trong tích tắc ~12 ms.
  2. **Trên Giao diện Web (30s):** 
     - Tab 1: Kéo thanh trượt Relative Compactness ($X_1$) và Glazing Area ($X_7$), bấm nút **Dự đoán năng lượng**. Chỉ vào biểu đồ cột và đồng hồ gauge trực quan hóa kết quả.
     - Tab 2: Bấm tải file CSV mẫu và hiển thị bảng dự đoán hàng loạt tức thì.
     - Tab 3: Giới thiệu ngắn tab cảnh báo miền áp dụng hình khối.
- **Lời thoại gợi ý:**
  > *"Bây giờ, nhóm xin demo sản phẩm thực tế. Đầu tiên là công cụ dòng lệnh predict.py. Chỉ với một câu lệnh, mô hình nạp pipeline và trả về kết quả dự đoán Heating Load và Cooling Load trong chưa đầy 15 mili-giây.*  
  > *Tiếp theo là ứng dụng Web Streamlit trực quan. Kỹ sư kiến trúc có thể tùy chỉnh các thanh trượt như độ đặc chắc công trình, diện tích kính, hướng nhà. Khi nhấn dự đoán, hệ thống tính toán đồng thời 2 tải nhiệt, trực quan hóa biểu đồ và đưa ra khuyến cáo tiết kiệm năng lượng. Hệ thống cũng hỗ trợ dự đoán hàng loạt bằng file CSV trong tab 2."*

---

### Phân cảnh 5: Kết luận & Chào kết thúc (2:45 – 3:00 | 15 giây)
- **Hình ảnh hiển thị:** Chuyển về Slide 15 (Trình diễn Demo & Q&A).
- **Lời thoại gợi ý:**
  > *"Sản phẩm của nhóm 01 đã đáp ứng toàn diện 7 yêu cầu kỹ thuật, đạt tính minh bạch học thuật cao và sẵn sàng triển khai thực tiễn. Nhóm xin trân trọng cảm ơn quý thầy cô và các bạn đã theo dõi!"*

---

## 3. Hướng dẫn xuất và nộp video
1. Định dạng file: `.mp4` (chuẩn mã hóa H.264, âm thanh AAC).
2. Tên file khuyến nghị: `Video_Demo_Nhom01_608807.mp4` hoặc lưu trực tiếp vào thư mục `04_Video_Demo/`.
3. Kiểm tra âm thanh rõ ràng, không bị rè hoặc quá nhỏ trước khi nộp.

