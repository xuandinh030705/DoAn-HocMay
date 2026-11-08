"""
app.py — Demo web (Streamlit) cho bài toán dự đoán năng lượng công trình.

Chạy:
    streamlit run app.py

Ứng dụng gồm 3 tab:
    1. Dự đoán một công trình (slider + preset)
    2. Dự đoán hàng loạt (tải CSV)
    3. Thông tin mô hình (chỉ số, siêu tham số, khuyến cáo sử dụng)

Tác giả: Nhóm 01 - Học phần 608807 (HUCE)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
META_PATH = BASE_DIR / "models" / "best_model_meta.json"
DATA_PATH = BASE_DIR / "data" / "ENB2012_data.xlsx"

# predict.py nằm cùng thư mục -> import trực tiếp
import sys
sys.path.insert(0, str(BASE_DIR))
from predict import predict  # noqa: E402

st.set_page_config(page_title="Building Energy Predictor | Nhóm 01", page_icon="🏢", layout="wide")


# --------------------------------------------------------------------------- #
# Nạp tài nguyên (cache để không đọc lại mỗi lần bấm nút)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


# --------------------------------------------------------------------------- #
# Giao diện
# --------------------------------------------------------------------------- #
st.title("🏢 Dự đoán Tiêu thụ Năng lượng trong Tòa nhà")
st.caption(
    "Mô hình Machine Learning dự đoán **Heating Load (Y1)** và **Cooling Load (Y2)** "
    "từ 8 đặc trưng thiết kế · UCI Energy Efficiency Dataset · Nhóm 01 — Học phần 608807 (HUCE)"
)

if not MODEL_PATH.exists():
    st.error(
        f"Không tìm thấy mô hình tại `{MODEL_PATH}`.\n\n"
        "Hãy chạy `notebooks/02_Modeling_Evaluation.ipynb` để huấn luyện và lưu `best_model.joblib`."
    )
    st.stop()

model = load_model()
df_train = load_training_data()
meta = load_meta()

tab1, tab2, tab3 = st.tabs(["🔮 Dự đoán một công trình", "📄 Dự đoán hàng loạt (CSV)", "ℹ️ Về mô hình"])

# --------------------------------------------------------------------------- #
# TAB 1
# --------------------------------------------------------------------------- #
with tab1:
    col_left, col_right = st.columns([1.05, 1])

    with col_left:
        st.subheader("Thông số thiết kế")

        presets = {
            "Tuỳ chỉnh": None,
            "Nhà vuông, thấp, không kính": dict(X1=0.98, X2=514.5, X3=294.0, X4=110.25,
                                                X5=7.0, X6=2, X7=0.0, X8=0),
            "Nhà dài, thấp, nhiều kính": dict(X1=0.62, X2=808.5, X3=367.5, X4=220.5,
                                              X5=3.5, X6=4, X7=0.4, X8=3),
            "Nhà trung bình, kính vừa": dict(X1=0.76, X2=661.5, X3=416.5, X4=122.5,
                                             X5=7.0, X6=3, X7=0.25, X8=2),
        }
        preset_name = st.selectbox("Chọn cấu hình mẫu", list(presets.keys()), index=0)
        preset = presets[preset_name] or presets["Nhà vuông, thấp, không kính"]

        c1, c2 = st.columns(2)
        with c1:
            x1 = st.slider("X1 · Relative Compactness", 0.62, 0.98,
                           float(preset["X1"]), 0.02,
                           help="Độ 'đặc' tương đối của khối nhà (1 = vuông nhất)")
            x2 = st.slider("X2 · Surface Area (m²)", 514.5, 808.5,
                           float(preset["X2"]), 24.5)
            x3 = st.slider("X3 · Wall Area (m²)", 245.0, 416.5,
                           float(preset["X3"]), 24.5)
            x4 = st.slider("X4 · Roof Area (m²)", 110.25, 220.5,
                           float(preset["X4"]), 12.25)
        with c2:
            x5 = st.select_slider("X5 · Overall Height (m)", options=[3.5, 7.0],
                                  value=float(preset["X5"]))
            x6 = st.selectbox("X6 · Orientation", options=[2, 3, 4, 5],
                              index=[2, 3, 4, 5].index(int(preset["X6"])),
                              format_func=lambda v: {2: "2 — Bắc", 3: "3 — Đông",
                                                     4: "4 — Nam", 5: "5 — Tây"}[v])
            x7 = st.select_slider("X7 · Glazing Area ratio", options=[0.0, 0.1, 0.25, 0.4],
                                  value=float(preset["X7"]))
            x8 = st.selectbox("X8 · Glazing Distribution", options=[0, 1, 2, 3, 4, 5],
                              index=[0, 1, 2, 3, 4, 5].index(int(preset["X8"])),
                              format_func=lambda v: "0 — Không có kính" if v == 0 else f"{v} — Kiểu phân bố {v}")

        st.caption(
            "Ghi chú: X8 chỉ có ý nghĩa khi X7 > 0. Khi X7 = 0, X8 luôn bằng 0 trong dữ liệu huấn luyện."
        )

    # ---- Dự đoán ----
    payload = pd.DataFrame([dict(X1=x1, X2=x2, X3=x3, X4=x4, X5=x5, X6=x6, X7=x7, X8=x8)])
    result = predict(payload, model=model).iloc[0]
    y1, y2 = float(result["Y1_pred"]), float(result["Y2_pred"])

    with col_right:
        st.subheader("Kết quả dự đoán")
        m1, m2 = st.columns(2)
        m1.metric("Heating Load (Y1)", f"{y1:.2f}", help="kWh/m²/năm")
        m2.metric("Cooling Load (Y2)", f"{y2:.2f}", help="kWh/m²/năm")
        st.metric("Tổng nhu cầu năng lượng", f"{y1 + y2:.2f} kWh/m²/năm")

        # So sánh với phân phối dữ liệu huấn luyện
        pct1 = float((df_train["Y1"] < y1).mean() * 100)
        pct2 = float((df_train["Y2"] < y2).mean() * 100)
        st.write("**Vị trí so với 768 công trình trong dữ liệu huấn luyện**")
        st.progress(min(max(pct1 / 100, 0.0), 1.0), text=f"Y1 cao hơn {pct1:.0f}% công trình mẫu")
        st.progress(min(max(pct2 / 100, 0.0), 1.0), text=f"Y2 cao hơn {pct2:.0f}% công trình mẫu")

        # Cảnh báo ngoại suy đơn giản: kiểm tra tổ hợp hiếm gặp (X7=0 nhưng X8>0)
        if x7 == 0 and x8 != 0:
            st.warning(
                "Tổ hợp X7 = 0 (không kính) nhưng X8 > 0 **không tồn tại** trong dữ liệu huấn luyện. "
                "Kết quả chỉ mang tính tham khảo."
            )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        ax.hist(df_train["Y1"], bins=25, color="#c6dbef", label="Y1 (dữ liệu mẫu)")
        ax.hist(df_train["Y2"], bins=25, color="#fdd0a2", alpha=0.65, label="Y2 (dữ liệu mẫu)")
        ax.axvline(y1, color="#c53030", lw=2, ls="--", label=f"Y1 dự đoán = {y1:.1f}")
        ax.axvline(y2, color="#2b6cb0", lw=2, ls="--", label=f"Y2 dự đoán = {y2:.1f}")
        ax.set_xlabel("kWh/m²/năm"); ax.set_ylabel("Số công trình")
        ax.set_title("Vị trí dự đoán trong phân phối dữ liệu")
        ax.legend(fontsize=7, loc="upper right")
        st.pyplot(fig, use_container_width=True)

    st.divider()
    with st.expander("Xem vector đặc trưng sau Feature Engineering (dùng để kiểm tra)"):
        from src.features import build_features, FEATURE_COLS
        st.dataframe(build_features(payload)[FEATURE_COLS].T.rename(columns={0: "Giá trị"}).round(3),
                     use_container_width=True)

# --------------------------------------------------------------------------- #
# TAB 2
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Dự đoán hàng loạt từ file CSV")
    st.write(
        "File CSV cần có **8 cột**: `X1, X2, X3, X4, X5, X6, X7, X8` (đúng tên, kiểu số). "
        "Các cột khác sẽ được bỏ qua."
    )

    template = pd.DataFrame([
        dict(X1=0.98, X2=514.5, X3=294.0, X4=110.25, X5=7.0, X6=2, X7=0.0, X8=0),
        dict(X1=0.62, X2=808.5, X3=367.5, X4=220.5, X5=3.5, X6=4, X7=0.4, X8=3),
    ])
    st.download_button("⬇️ Tải file CSV mẫu",
                       template.to_csv(index=False).encode("utf-8-sig"),
                       file_name="template_input.csv", mime="text/csv")

    uploaded = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            df_result = predict(df_upload, model=model)
            st.success(f"Đã dự đoán {len(df_result)} công trình.")
            st.dataframe(df_result[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8",
                                    "Y1_pred", "Y2_pred"]].round(3),
                         use_container_width=True, height=380)
            st.download_button("⬇️ Tải kết quả",
                               df_result.to_csv(index=False).encode("utf-8-sig"),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Lỗi khi xử lý file: {exc}")

# --------------------------------------------------------------------------- #
# TAB 3
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Thông tin mô hình")
    if meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mô hình", meta.get("champion", "—"))
        c2.metric("R² (test, TB 2 mục tiêu)", f'{meta.get("test_metrics", {}).get("Test_R2_mean", float("nan")):.4f}')
        c3.metric("RMSE (test, TB)", f'{meta.get("test_metrics", {}).get("Test_RMSE_mean", float("nan")):.3f} kWh/m²')

        st.write("**Chỉ số chi tiết trên tập test**")
        st.json(meta.get("test_metrics", {}))

        st.write("**Siêu tham số tốt nhất**")
        st.json(meta.get("best_params", {}))

        st.write("**Đặc trưng đầu vào**")
        st.code(", ".join(meta.get("features", [])))

    st.warning(
        "**Khuyến cáo sử dụng (applicability domain):** mô hình được huấn luyện trên 768 mẫu mô phỏng "
        "với 12 cấu hình hình học. Với công trình có hình dạng nằm ngoài miền này, sai số có thể tăng đáng kể "
        "(xem phân tích GroupKFold trong notebook 02). Không nên dùng kết quả làm con số thiết kế cuối cùng "
        "mà chỉ dùng để so sánh phương án."
    )

    st.info(
        "**Cách mô hình hoạt động:** đầu vào thô X1..X8 → `build_features()` tạo 12 đặc trưng vật lý → "
        "`ColumnTransformer` (imputer + scaler + one-hot) → mô hình hồi quy đa đầu ra. "
        "Toàn bộ chuỗi này nằm trong một `Pipeline` duy nhất được lưu trong `best_model.joblib`."
    )
