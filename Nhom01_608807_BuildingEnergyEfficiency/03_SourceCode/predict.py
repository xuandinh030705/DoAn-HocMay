"""
predict.py
==========
Công cụ dự đoán Heating Load (Y1) và Cooling Load (Y2) cho công trình mới.

Mô hình được nạp từ `models/best_model.joblib` — đây là **Pipeline hoàn chỉnh**
(ColumnTransformer + mô hình), nên file này chỉ cần đưa vào 8 đặc trưng thô X1..X8.
Không được tự chuẩn hóa / mã hóa ở ngoài, nếu không sẽ lệch với lúc huấn luyện.

Cách dùng
---------
1) Chế độ dòng lệnh (một công trình):
    python predict.py --x1 0.98 --x2 514.5 --x3 294 --x4 110.25 \
                      --x5 7 --x6 2 --x7 0 --x8 0

2) Chế độ JSON:
    python predict.py --json "{\"X1\":0.98,\"X2\":514.5,\"X3\":294,\"X4\":110.25,\"X5\":7,\"X6\":2,\"X7\":0,\"X8\":0}"

3) Chế độ hàng loạt (CSV có các cột X1..X8):
    python predict.py --csv data/new_buildings.csv --out results/new_predictions.csv

4) Không truyền tham số -> chạy ví dụ mẫu (2 công trình).

Tác giả: Nhóm 01 - Học phần 608807 (HUCE)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Cho phép chạy `python predict.py` từ bất kỳ thư mục nào
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.features import FEATURE_COLS, TARGETS, build_features  # noqa: E402

MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "best_model.joblib"

# Miền giá trị hợp lệ của dữ liệu huấn luyện (dùng để cảnh báo ngoại suy)
TRAIN_RANGES = {
    "X1": (0.62, 0.98),
    "X2": (514.5, 808.5),
    "X3": (245.0, 416.5),
    "X4": (110.25, 220.5),
    "X5": (3.5, 7.0),
    "X6": (2, 5),
    "X7": (0.0, 0.4),
    "X8": (0, 5),
}

# Mô tả ngắn để in ra cho người dùng
FEATURE_LABELS = {
    "X1": "Relative Compactness",
    "X2": "Surface Area (m2)",
    "X3": "Wall Area (m2)",
    "X4": "Roof Area (m2)",
    "X5": "Overall Height (m)",
    "X6": "Orientation (2=Bắc, 3=Đông, 4=Nam, 5=Tây)",
    "X7": "Glazing Area ratio (0 - 0.4)",
    "X8": "Glazing Area Distribution (0 - 5)",
}

RAW_COLS = list(TRAIN_RANGES.keys())


def load_model(path: str | Path | None = None):
    """Nạp Pipeline đã huấn luyện."""
    model_path = Path(path) if path else DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình tại {model_path}.\n"
            "Hãy chạy notebooks/02_Modeling_Evaluation.ipynb trước để sinh ra best_model.joblib."
        )
    return joblib.load(model_path)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra đủ cột, đúng kiểu số, và cảnh báo nếu nằm ngoài miền huấn luyện."""
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}. Cần đủ 8 cột {RAW_COLS}.")

    out = df[RAW_COLS].copy()
    for col in RAW_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise")

    if out[RAW_COLS].isna().any().any():
        raise ValueError("Dữ liệu đầu vào có giá trị thiếu (NaN).")

    # Cảnh báo ngoại suy (không chặn, vì người dùng có thể cố ý thử)
    warnings = []
    for col, (lo, hi) in TRAIN_RANGES.items():
        n_out = int(((out[col] < lo) | (out[col] > hi)).sum())
        if n_out:
            warnings.append(f"  - {col} có {n_out} giá trị ngoài miền huấn luyện [{lo}, {hi}]")
    if warnings:
        print("⚠ CẢNH BÁO: dữ liệu nằm ngoài miền đã học, kết quả có thể kém tin cậy:")
        print("\n".join(warnings))
    return out


def predict(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Dự đoán Y1, Y2 cho DataFrame thô có các cột X1..X8.

    Trả về DataFrame gồm dữ liệu gốc + cột 'Y1_pred' (Heating Load) và 'Y2_pred' (Cooling Load),
    đơn vị kWh/m².
    """
    model = model if model is not None else load_model()

    raw = validate_input(df)
    featured = build_features(raw)              # cùng hàm dùng lúc huấn luyện
    preds = np.asarray(model.predict(featured[FEATURE_COLS]), dtype=float)

    result = raw.copy()
    result["Y1_pred"] = preds[:, 0]
    result["Y2_pred"] = preds[:, 1]
    return result


def predict_one(**kwargs) -> dict:
    """Hàm tiện dụng cho 1 công trình, trả về dict (dùng cho demo/Streamlit)."""
    model = kwargs.pop("model", None)
    row = pd.DataFrame([{k: kwargs[k] for k in RAW_COLS}])
    out = predict(row, model=model).iloc[0]
    return {"Y1": float(out["Y1_pred"]), "Y2": float(out["Y2_pred"])}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Dự đoán Heating Load (Y1) và Cooling Load (Y2) từ 8 đặc trưng công trình.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    for col in RAW_COLS:
        parser.add_argument(f"--{col.lower()}", type=float, default=None,
                            help=FEATURE_LABELS[col])
    parser.add_argument("--json", type=str, default=None,
                        help='Chuỗi JSON chứa X1..X8, ví dụ \'{"X1":0.98, ...}\'')
    parser.add_argument("--csv", type=str, default=None,
                        help="File CSV có các cột X1..X8 (dự đoán hàng loạt)")
    parser.add_argument("--out", type=str, default=None,
                        help="Đường dẫn lưu kết quả CSV (chỉ dùng với --csv)")
    parser.add_argument("--model", type=str, default=None,
                        help="Đường dẫn mô hình (mặc định models/best_model.joblib). "
                             "Dùng models/best_model_full_data.joblib cho bản huấn luyện trên toàn bộ dữ liệu.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    model = load_model(args.model)

    # ---- Chế độ 1: CSV hàng loạt ----
    if args.csv:
        df_in = pd.read_csv(args.csv)
        result = predict(df_in, model=model)
        out_path = Path(args.out) if args.out else Path(args.csv).with_name(
            Path(args.csv).stem + "_predictions.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Đã dự đoán {len(result)} công trình.")
        print(f"Kết quả lưu tại: {out_path}")
        print(result[["Y1_pred", "Y2_pred"]].describe().round(3).to_string())
        return 0

    # ---- Chế độ 2: JSON ----
    if args.json:
        payload = json.loads(args.json)
        result = predict(pd.DataFrame([payload]), model=model)
        _print_single(result.iloc[0])
        return 0

    # ---- Chế độ 3: tham số dòng lệnh ----
    values = {col: getattr(args, col.lower()) for col in RAW_COLS}
    if all(v is None for v in values.values()):
        print("Không có tham số -> chạy 2 ví dụ mẫu.\n")
        examples = pd.DataFrame([
            {  # ví dụ 1: nhà vuông, thấp, không kính (mẫu dòng 0 của UCI)
                "X1": 0.98, "X2": 514.5, "X3": 294.0, "X4": 110.25,
                "X5": 7.0, "X6": 2, "X7": 0.0, "X8": 0,
            },
            {  # ví dụ 2: nhà dài, thấp, nhiều kính (cấu hình "khó" nhất)
                "X1": 0.62, "X2": 808.5, "X3": 367.5, "X4": 220.5,
                "X5": 3.5, "X6": 4, "X7": 0.4, "X8": 3,
            },
        ])
        result = predict(examples, model=model)
        for i, row in result.iterrows():
            print(f"--- Ví dụ {i + 1} ---")
            _print_single(row)
            print()
        return 0

    missing = [col for col in RAW_COLS if values[col] is None]
    if missing:
        print(f"LỖI: thiếu tham số {', '.join('--' + c.lower() for c in missing)}")
        print("Gợi ý: truyền đủ 8 đặc trưng, hoặc dùng --json / --csv / không truyền gì để xem ví dụ.")
        return 2

    result = predict(pd.DataFrame([values]), model=model)
    _print_single(result.iloc[0])
    return 0


def _print_single(row: pd.Series) -> None:
    """In kết quả 1 công trình cho đẹp."""
    print("Đặc trưng đầu vào:")
    for col in RAW_COLS:
        print(f"  {col:<3} {FEATURE_LABELS[col]:<52} = {row[col]}")
    print("\nKẾT QUẢ DỰ ĐOÁN (kWh/m²/năm):")
    print(f"  Heating Load (Y1) = {row['Y1_pred']:6.2f}")
    print(f"  Cooling Load (Y2) = {row['Y2_pred']:6.2f}")
    print(f"  Tổng nhu cầu      = {row['Y1_pred'] + row['Y2_pred']:6.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
