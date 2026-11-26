"""Leak-free feature engineering for an h-day-ahead forecast.

Every feature available for target date t is computable using data up to t - h only:
lags are >= h, rolling statistics are taken on the h-shifted series, and calendar
features are known in advance.
"""
import numpy as np
import pandas as pd

LNY = {2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
       2025: "2025-01-29", 2026: "2026-02-17", 2027: "2027-02-06", 2028: "2028-01-26"}


def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["day_of_year"] = idx.dayofyear
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["golden_week"] = ((idx.month == 10) & (idx.day <= 7)).astype(int)
    f["labour_day"] = ((idx.month == 5) & (idx.day <= 5)).astype(int)
    f["summer"] = idx.month.isin([7, 8]).astype(int)
    f["xmas_ny"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))).astype(int)
    lny_flag = np.zeros(len(idx), dtype=int)
    for d in LNY.values():
        d = pd.Timestamp(d)
        lny_flag |= ((idx >= d - pd.Timedelta(days=1)) & (idx <= d + pd.Timedelta(days=6))).astype(int)
    f["lny_week"] = lny_flag
    return f


def make_features(y: pd.Series, horizon: int = 7) -> pd.DataFrame:
    h = horizon
    X = _calendar(y.index)
    X[f"lag_{h}"] = y.shift(h)
    X[f"lag_{h+7}"] = y.shift(h + 7)
    X[f"lag_{h+21}"] = y.shift(h + 21)
    shifted = y.shift(h)
    X["roll_mean_7"] = shifted.rolling(7).mean()
    X["roll_mean_28"] = shifted.rolling(28).mean()
    X["roll_std_28"] = shifted.rolling(28).std()
    X["trend_ratio"] = X["roll_mean_7"] / X["roll_mean_28"]
    return X

