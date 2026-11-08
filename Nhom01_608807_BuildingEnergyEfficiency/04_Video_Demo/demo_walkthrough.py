# -*- coding: utf-8 -*-
"""
demo_walkthrough.py — Kịch bản chạy mẫu tự động cho Video Demo (3 phút)
Học phần: Học máy (Mã HP: 608807) — Nhóm 01 — HUCE

Chạy script này khi quay màn hình terminal:
    python demo_walkthrough.py
"""
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "03_SourceCode"
sys.path.insert(0, str(ROOT))

import pandas as pd
from predict import predict, load_model

def print_slow(text, delay=0.015):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print("=" * 72)
    print("  BÀI TẬP LỚN HỌC MÁY (608807) - NHÓM 01 - ĐH XÂY DỰNG HÀ NỘI")
    print("  DEMO SUY LUẬN MÔ HÌNH DỰ ĐOÁN HIỆU QUẢ NĂNG LƯỢNG TÒA NHÀ")
    print("=" * 72)
    time.sleep(1)

    print("\n[BƯỚC 1] Nạp mô hình Champion (Pipeline Scikit-learn + XGBoost)...")
    t0 = time.time()
    pipe = load_model()
    load_time = (time.time() - t0) * 1000
    print(f" -> Nạp thành công artifact: models/best_model.joblib ({load_time:.1f} ms)")
    time.sleep(1)

    print("\n[BƯỚC 2] Dự đoán đơn lẻ cho Công trình Mẫu 1 (Nhà ở tiêu chuẩn):")
    sample_1 = {
        "X1": 0.98, "X2": 514.5, "X3": 294.0, "X4": 110.25,
        "X5": 7.0, "X6": 2, "X7": 0.10, "X8": 1
    }
    for k, v in sample_1.items():
        print(f"    {k} = {v:<8}", end="  ")
    print()

    t_pred0 = time.time()
    res1 = predict(pd.DataFrame([sample_1]))
    dt1 = (time.time() - t_pred0) * 1000
    y1_pred = res1.iloc[0]["Y1_pred"]
    y2_pred = res1.iloc[0]["Y2_pred"]
    print(f" -> KẾT QUẢ: Heating Load = {y1_pred:.2f} kWh/m² | Cooling Load = {y2_pred:.2f} kWh/m² (Thời gian: {dt1:.2f} ms)")
    time.sleep(1.5)

    print("\n[BƯỚC 3] Dự đoán đơn lẻ cho Công trình Mẫu 2 (Nhà kính diện tích lớn):")
    sample_2 = {
        "X1": 0.62, "X2": 808.5, "X3": 367.5, "X4": 220.5,
        "X5": 3.5, "X6": 4, "X7": 0.40, "X8": 3
    }
    for k, v in sample_2.items():
        print(f"    {k} = {v:<8}", end="  ")
    print()

    t_pred1 = time.time()
    res2 = predict(pd.DataFrame([sample_2]))
    dt2 = (time.time() - t_pred1) * 1000
    y1_pred2 = res2.iloc[0]["Y1_pred"]
    y2_pred2 = res2.iloc[0]["Y2_pred"]
    print(f" -> KẾT QUẢ: Heating Load = {y1_pred2:.2f} kWh/m² | Cooling Load = {y2_pred2:.2f} kWh/m² (Thời gian: {dt2:.2f} ms)")
    time.sleep(1.5)

    print("\n[BƯỚC 4] Khởi động Web Dashboard giao diện Streamlit:")
    print(" -> Lệnh thực thi: streamlit run app.py")
    print("=" * 72)

if __name__ == "__main__":
    main()

