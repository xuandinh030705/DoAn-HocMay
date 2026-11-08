"""
evaluation.py
=============
Các tiện ích đánh giá dùng chung: tính chỉ số hồi quy cho từng mục tiêu,
đo thời gian, in bảng so sánh đẹp.

Vì bài toán có HAI biến mục tiêu (Y1, Y2), mọi chỉ số đều được tính riêng cho
từng mục tiêu rồi lấy trung bình (macro average). Cách này trung thực hơn việc
"gộp" hai mục tiêu lại vì đơn vị của Y1 và Y2 là như nhau (kWh/m2) nhưng mức
độ khó thì khác nhau.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .features import TARGETS


def regression_metrics(y_true, y_pred, target_names: list[str] = TARGETS) -> dict:
    """Tính R2, RMSE, MAE, MAPE cho từng mục tiêu + trung bình.

    Quy ước đặt tên: 'R2_Y1', 'RMSE_Y1', ..., và 'R2_mean' (trung bình 2 mục tiêu).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    out: dict[str, float] = {}
    for j, name in enumerate(target_names):
        t, p = y_true[:, j], y_pred[:, j]
        out[f"R2_{name}"] = r2_score(t, p)
        out[f"RMSE_{name}"] = float(np.sqrt(mean_squared_error(t, p)))
        out[f"MAE_{name}"] = float(mean_absolute_error(t, p))
        # MAPE chỉ hợp lệ vì mọi giá trị Y1, Y2 trong bộ dữ liệu đều > 0 (min ~6)
        out[f"MAPE_{name}"] = float(mean_absolute_percentage_error(t, p) * 100.0)

    for metric in ["R2", "RMSE", "MAE", "MAPE"]:
        out[f"{metric}_mean"] = float(np.mean([out[f"{metric}_{n}"] for n in target_names]))
    return out


def timed_fit_predict(estimator, X_train, y_train, X_test, y_test) -> tuple[dict, object]:
    """Fit mô hình, đo thời gian huấn luyện / dự đoán, trả về (metrics, model)."""
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = estimator.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = regression_metrics(y_test, y_pred)
    metrics["fit_time_s"] = fit_time
    metrics["predict_time_ms"] = predict_time * 1000.0
    return metrics, estimator


def print_metrics(metrics: dict, digits: int = 4) -> None:
    """In gọn một dòng chỉ số (dùng trong notebook cho dễ đọc)."""
    keys = [k for k in metrics if k.startswith(("R2", "RMSE", "MAE", "MAPE"))]
    print(" | ".join(f"{k}={metrics[k]:.{digits}f}" for k in keys))


def choose_best_model(results: pd.DataFrame, metric: str = "R2_mean") -> str:
    """Chọn mô hình theo R2 trung bình cao nhất, hoà thì ưu tiên RMSE thấp hơn."""
    return (
        results.sort_values([metric, "RMSE_mean"], ascending=[False, True])
        .iloc[0]["model"]
    )


def cross_validate_multioutput(estimator, X, y, cv, target_names: list[str] = TARGETS) -> pd.DataFrame:
    """Chạy K-Fold thủ công cho bài toán đa mục tiêu, trả về chỉ số của TỪNG fold.

    Vì sao không dùng cross_val_score? Vì nó chỉ trả về MỘT điểm trung bình cho mỗi
    fold (trung bình trên 2 mục tiêu), trong khi báo cáo cần R2/RMSE/MAE **riêng**
    cho Y1 và Y2, kèm thời gian huấn luyện của từng fold.

    Lưu ý chống rò rỉ: `estimator` là cả Pipeline, nên mỗi fold chỉ fit
    imputer/scaler/encoder trên phần train của fold đó.
    """
    from sklearn.base import clone

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        fit_time = time.perf_counter() - t0

        pred = est.predict(X.iloc[va_idx])
        row = regression_metrics(y.iloc[va_idx], pred, target_names)
        row["fold"] = fold
        row["fit_time_s"] = fit_time
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cv(fold_df: pd.DataFrame) -> dict:
    """Tổng hợp kết quả CV: mean và std của các chỉ số chính.

    Quy ước tên: 'CV_R2_mean' là trung bình qua các fold của R2 trung bình 2 mục tiêu;
    'CV_R2_mean_std' là độ lệch chuẩn giữa các fold.
    """
    out = {}
    for col in ["R2_mean", "RMSE_mean", "MAE_mean", "MAPE_mean",
                "R2_Y1", "R2_Y2", "RMSE_Y1", "RMSE_Y2",
                "MAE_Y1", "MAE_Y2", "MAPE_Y1", "MAPE_Y2", "fit_time_s"]:
        out[f"CV_{col}"] = float(fold_df[col].mean())
        out[f"CV_{col}_std"] = float(fold_df[col].std())
    return out
