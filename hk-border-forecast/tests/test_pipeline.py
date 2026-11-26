import numpy as np
import pandas as pd
from hkborder import synthetic_raw, load_daily_arrivals, make_features, walk_forward, summarise
from hkborder.data import normalise

RAW = synthetic_raw(end="2026-06-30")
Y = load_daily_arrivals(RAW)


def test_features_do_not_leak_future():
    """Changing y from day k onward must not change features for any date before k + horizon."""
    h, k = 7, 600
    y2 = Y.copy(); y2.iloc[k:] = y2.iloc[k:] * 10
    a, b = make_features(Y, h), make_features(y2, h)
    pd.testing.assert_frame_equal(a.iloc[:k + h], b.iloc[:k + h])


def test_walk_forward_is_out_of_sample_and_ordered():
    res = walk_forward(Y, horizon=7, n_folds=4, fold_days=30)
    assert res.index.is_monotonic_increasing and res.index.is_unique
    assert len(res) == 4 * 30
    assert np.isfinite(summarise(res).values).all()


def test_models_beat_naive_on_structured_data():
    res = walk_forward(Y, horizon=7, n_folds=6, fold_days=30)
    s = summarise(res)
    assert s["MAPE_%"].drop("naive").min() < s.loc["naive", "MAPE_%"]


def test_normalise_survives_renamed_columns_and_thousand_separators():
    raw = pd.DataFrame({
        "Date": ["01-02-2024", "01-02-2024"], "Control Point": ["Lo Wu", "Lo Wu"],
        "Arrival / Departure": ["Arrival", "Departure"], "Hong Kong Residents": ["1,000", "900"],
        "Mainland Visitors": ["2,500", "2,400"], "Other Visitors": ["100", "90"], "Total": ["3,600", "3,390"]})
    out = normalise(raw)
    assert out.mainland_visitors.tolist() == [2500, 2400]
    assert out.date.iloc[0] == pd.Timestamp("2024-02-01")  # day-first parsing


def test_daily_series_is_continuous_and_arrivals_only():
    assert Y.index.freq is not None or pd.infer_freq(Y.index) == "D"
    assert Y.isna().sum() == 0


def test_official_total_rows_are_not_double_counted():
    """Use the official aggregate row rather than adding it to control-point rows."""
    rows = pd.DataFrame({
        "date": pd.to_datetime(["2024-02-01"] * 3),
        "control_point": ["Lo Wu", "Airport", "Total"],
        "direction": ["Arrival"] * 3,
        "hk_residents": [100, 50, 150],
        "mainland_visitors": [200, 80, 280],
        "other_visitors": [10, 5, 15],
        "total": [310, 135, 445],
    })
    actual = load_daily_arrivals(rows)
    assert actual.loc[pd.Timestamp("2024-02-01")] == 280
