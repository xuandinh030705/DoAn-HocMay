# Thư mục Video Demo (04_Video_Demo)

Thư mục này lưu trữ tài liệu chuẩn bị, kịch bản chi tiết và video demo sản phẩm của **Nhóm 01 (Mã HP: 608807)**.

## Danh mục tài liệu trong thư mục
1. `KichBan_Quay_Video_3Phut.md`: Kịch bản chi tiết từng giây (từ 0:00 đến 3:00) phân bổ lời thoại và thao tác quay màn hình tương ứng.
2. `demo_walkthrough.py`: Kịch bản lệnh chạy thử tự động để phục vụ quay demo nhanh gọn.
3. `Video_Demo_Nhom01_608807.mp4`: File video demo chính thức (thời lượng 3 phút, chuẩn Full HD 1080p).

## Hướng dẫn quay và chạy demo
Để thực hiện quay demo:
1. Mở terminal tại thư mục mã nguồn:
   ```powershell
   cd ../03_SourceCode
   streamlit run app.py
   ```
2. Mở file slide thuyết trình:
   ```powershell
   ../02_Slide/Slide_Thuyet_trinh_Nhom01.pdf
   ```
3. Chạy thử nghiệm CLI:
   ```powershell
   python predict.py
   ```
4. Bật phần mềm quay màn hình (OBS / Xbox Game Bar `Win + G`) và thực hiện theo kịch bản trong `KichBan_Quay_Video_3Phut.md`.

