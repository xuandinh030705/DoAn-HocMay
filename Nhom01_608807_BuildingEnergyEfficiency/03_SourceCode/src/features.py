"""
features.py
===========
Module dùng chung cho toàn bộ đồ án "Dự đoán Tiêu thụ Năng lượng trong Tòa nhà".

Mục đích: đảm bảo **công thức Feature Engineering** và **cấu hình tiền xử lý**
được định nghĩa DUY NHẤT MỘT LẦN, rồi được tái sử dụng ở:
    - notebooks/01_EDA_Preprocessing.ipynb
    - notebooks/02_Modeling_Evaluation.ipynb
    - predict.py
    - app.py (demo Streamlit)

Nếu notebook và predict.py dùng hai bản feature engineering khác nhau thì mô
hình đã lưu sẽ nhận sai đầu vào -> đây là lỗi rất hay gặp khi "đóng gói" mô
hình. Gom vào một module là cách phòng tránh lỗi này.

Tác giả: Nhóm 01 - Học phần 608807 (HUCE)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# 1. HẰNG SỐ CẤU HÌNH (đặt một chỗ để mọi nơi dùng giống nhau)
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42          # cố định "hạt giống" -> chạy lại ra cùng kết quả
TEST_SIZE: float = 0.20         # 20% cho tập test độc lập (154 mẫu)
CV_FOLDS: int = 5               # K-Fold = 5 theo yêu cầu đề bài

TARGETS: list[str] = ["Y1", "Y2"]
TARGET_NAMES: dict[str, str] = {
    "Y1": "Heating Load (tải sưởi)",
    "Y2": "Cooling Load (tải làm mát)",
}

# 8 đặc trưng gốc của bộ dữ liệu UCI
RAW_FEATURES: list[str] = [f"X{i}" for i in range(1, 9)]

# --------------------------------------------------------------------------- #
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
# 13 đặc trưng số "chính" = 6 đặc trưng gốc dạng liên tục + 7 đặc trưng vật lý mới
NUM_BASE: list[str] = ["X1", "X2", "X3", "X4", "X5", "X7"]

NUM_PHYSICAL: list[str] = [
    "GLAZING_AREA",        # diện tích kính tuyệt đối (m2)
    "GLAZING_WALL_RATIO",  # tỷ lệ kính / diện tích tường đặc
    "ENVELOPE_FLOOR",      # tỷ lệ diện tích vỏ bao / diện tích sàn (shape factor)
    "VOLUME",              # thể tích xấp xỉ (m3)
    "HEIGHT_COMPACT",      # tương tác chiều cao x độ đặc (X5 x X1)
    "WALL_ROOF",           # tỷ lệ diện tích tường / diện tích mái
    "ASPECT",              # tỷ lệ cạnh dài / cạnh ngắn (suy ra từ chu vi & diện tích)
]

NUM_INTERACTION: list[str] = [
    "GLASS_FLOOR",     # diện tích kính trên 1 m2 sàn
    "H_X_GLASS",       # tương tác chiều cao x tỷ lệ kính
    "H_X_SURF",        # tương tác chiều cao x diện tích vỏ bao (đại diện tổn thất nhiệt)
]

NUM_CYCLIC: list[str] = ["ORIENT_SIN", "ORIENT_COS"]  # mã hóa vòng (cyclic) cho X6

NUM_COLS: list[str] = NUM_BASE + NUM_PHYSICAL + NUM_INTERACTION + NUM_CYCLIC

# X8 (Glazing Area Distribution) là biến ĐỊNH DANH: 0 = không có kính,
# 1..5 = 5 kiểu phân bố kính khác nhau -> one-hot là cách mã hóa trung thực nhất.
CAT_COLS: list[str] = ["X8"]

FEATURE_COLS: list[str] = NUM_COLS + CAT_COLS

# Tập đặc trưng gốc, dùng cho thí nghiệm "trước vs sau Feature Engineering"
RAW_COLS: list[str] = RAW_FEATURES.copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng mới từ 8 đặc trưng gốc X1..X8.

    Tham số
    -------
    df : DataFrame chứa tối thiểu các cột X1..X8 (tên cột đúng như file UCI).

    Trả về
    ------
    DataFrame mới, giữ nguyên các cột gốc và bổ sung các cột đặc trưng.
    Hàm KHÔNG phụ thuộc vào tham số nào học từ dữ liệu (không fit gì cả),
    nên gọi trước hay sau khi chia train/test đều không gây rò rỉ dữ liệu.

    Ý nghĩa vật lý của từng đặc trưng được giải thích trong báo cáo, Chương 2.
    """
    d = df.copy()

    # --- (a) Năng lượng liên quan tới kính ---------------------------------
    # X7 là tỷ lệ kính trên diện tích vỏ bao (glazing area ratio).
    # Diện tích kính tuyệt đối = tỷ lệ kính x diện tích vỏ bao.
    d["GLAZING_AREA"] = d["X7"] * d["X2"]

    # Kính nằm trên tường -> tỷ lệ kính/tường đặc cho biết "mức trong suốt"
    d["GLAZING_WALL_RATIO"] = d["X7"] * d["X2"] / d["X3"]

    # Mật độ kính trên mỗi m2 sàn: chỉ số thường dùng trong thiết kế (WWR/floor)
    d["GLASS_FLOOR"] = d["GLAZING_AREA"] / d["X4"]

    # --- (b) Hình học khối nhà --------------------------------------------
    # Tỷ lệ diện tích vỏ bao / diện tích sàn: càng lớn -> càng nhiều bề mặt
    # trao đổi nhiệt với môi trường bên ngoài.
    d["ENVELOPE_FLOOR"] = d["X2"] / d["X4"]

    # Với nhà dạng hộp, diện tích mái xấp xỉ diện tích sàn -> V ~ A_mai x h
    d["VOLUME"] = d["X4"] * d["X5"]

    # Diện tích tường trên diện tích mái: phản ánh tỷ lệ bề mặt đứng/ngang
    d["WALL_ROOF"] = d["X3"] / d["X4"]

    # Tương tác: nhà cao + kín (compact) thì tổn thất nhiệt theo phương đứng
    # tăng, nhưng tổn thất qua mái trên một đơn vị sàn lại giảm.
    d["HEIGHT_COMPACT"] = d["X5"] * d["X1"]

    # Suy ra chiều dài/cạnh ngắn của mặt bằng từ hệ phương trình:
    #   chu vi  P = X3 / X5        (vì A_tuong = P * h)
    #   diện tích A = X4
    #   L + W = P/2 ; L*W = A  ->  L, W là nghiệm phương trình bậc 2
    half_perimeter = d["X3"] / (2.0 * d["X5"])
    discriminant = np.clip(half_perimeter**2 - 4.0 * d["X4"], 0.0, None)
    long_side = (half_perimeter + np.sqrt(discriminant)) / 2.0
    short_side = np.clip(half_perimeter - long_side, 1e-6, None)
    d["ASPECT"] = long_side / short_side  # = 1.0 với nhà vuông

    # --- (c) Tương tác phi tuyến ------------------------------------------
    # Tải sưởi/mát phụ thuộc vào CẢ chiều cao VÀ lượng kính (hiệu ứng solar gain
    # chỉ đáng kể khi có kính), nên cần số hạng tích để mô hình tuyến tính
    # biểu diễn được hiệu ứng này.
    d["H_X_GLASS"] = d["X5"] * d["X7"]
    d["H_X_SURF"] = d["X5"] * d["X2"]

    # --- (d) Mã hóa vòng cho Orientation ------------------------------------
    # X6 in {2,3,4,5} ứng với 4 hướng Bắc/Đông/Nam/Tây (cách nhau 90 độ).
    # Không phải biến định danh thông thường mà có tính chu kỳ: Tây và Bắc
    # "gần nhau" hơn Tây và Đông nếu xét góc. sin/cos giữ được tính chu kỳ đó.
    angle = np.pi / 2.0 * (d["X6"] - 2.0)
    d["ORIENT_SIN"] = np.sin(angle)
    d["ORIENT_COS"] = np.cos(angle)

    return d


# --------------------------------------------------------------------------- #
# 3. TIỀN XỬ LÝ (đóng gói trong ColumnTransformer để KHÔNG rò rỉ dữ liệu)
# --------------------------------------------------------------------------- #
def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Trả về ColumnTransformer chuẩn cho mọi mô hình.

    Tham số
    -------
    scale : bool
        - True  : chuẩn hóa (StandardScaler) các biến số -> cần cho Linear/Ridge/
                  Lasso/KNN/SVM vì các thuật toán này nhạy với thang đo.
        - False : bỏ qua bước scale -> dùng cho mô hình cây (DecisionTree,
                  RandomForest, GradientBoosting, XGBoost) vốn bất biến với
                  phép biến đổi đơn điệu. Bỏ scale giúp pipeline gọn và nhanh hơn.

    Ghi chú chống rò rỉ
    -------------------
    SimpleImputer / StandardScaler / OneHotEncoder đều là *estimator có học
    tham số*. Vì chúng nằm TRONG Pipeline, mỗi lần Cross-Validation chỉ được
    fit trên fold-train rồi transform fold-validation. Tuyệt đối không gọi
    .fit_transform() trên toàn bộ dữ liệu trước khi chia train/test.
    """
    num_steps: list[tuple[str, object]] = [
        # median an toàn hơn mean khi có ngoại lai; dữ liệu UCI không thiếu giá
        # trị nhưng giữ imputer để pipeline chịu được dữ liệu mới bị khuyết.
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",       # bỏ các cột không dùng (X6 gốc, Y1, Y2, ...)
        verbose_feature_names_out=True,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Lấy tên cột sau tiền xử lý (dùng để vẽ hệ số hồi quy / SHAP)."""
    return list(preprocessor.get_feature_names_out())
