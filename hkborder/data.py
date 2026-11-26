"""Loading and cleaning the Immigration Department daily passenger traffic data.

Source: data.gov.hk "Statistics on Daily Passenger Traffic" (Immigration Department),
daily inbound/outbound trips at every control point, split into Hong Kong residents,
Mainland visitors and other visitors.
"""
import numpy as np
import pandas as pd

DATA_URL = ("https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/"
            "statistics_on_daily_passenger_traffic.csv")

# Normal cross-border travel fully resumed after the COVID closures (6 Feb 2023).
REOPENING = pd.Timestamp("2023-02-06")


def _find(cols, *keywords):
    """Return the first column whose lower-cased name contains all keywords."""
    for c in cols:
        if all(k in c.lower() for k in keywords):
            return c
    raise KeyError(f"No column containing {keywords}. Columns found: {list(cols)}")


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the raw CSV onto a fixed schema:
    date, control_point, direction, hk_residents, mainland_visitors, other_visitors, total.
    Column matching is keyword-based so small header changes do not break the loader."""
    cols = list(raw.columns)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[_find(cols, "date")], dayfirst=True, errors="coerce"),
        "control_point": raw[_find(cols, "control")].astype(str).str.strip(),
        "direction": raw[_find(cols, "arrival")].astype(str).str.strip().str.lower(),
        "hk_residents": raw[_find(cols, "resident")],
        "mainland_visitors": raw[_find(cols, "mainland")],
        "other_visitors": raw[_find(cols, "other")],
        "total": raw[_find(cols, "total")],
    })
    for c in ["hk_residents", "mainland_visitors", "other_visitors", "total"]:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
    return out.dropna(subset=["date"])


def load_raw(source: str = DATA_URL) -> pd.DataFrame:
    return normalise(pd.read_csv(source, encoding="utf-8-sig"))


def synthetic_raw(seed: int = 0, end=None) -> pd.DataFrame:
    """Realistic fake data in the same schema, for offline use and tests:
    near-zero traffic until the Feb 2023 reopening, then a ramp-up, weekly and holiday patterns."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range("2021-01-01", end)
    points = {"Lo Wu": 0.30, "Lok Ma Chau Spur Line": 0.22, "Shenzhen Bay": 0.18,
              "Airport": 0.12, "Hung Hom": 0.08, "HZMB Hong Kong Port": 0.10}
    rows = []
    t = (dates - REOPENING).days.values
    ramp = np.where(t < 0, 0.02, 1 - 0.55 * np.exp(-np.clip(t, 0, None) / 250))
    dow = dates.dayofweek.values
    weekly = np.array([0.95, 0.90, 0.92, 0.95, 1.10, 1.15, 1.00])[dow]
    peak = np.zeros(len(dates))
    for i, d in enumerate(dates):
        if d.month == 10 and d.day <= 7: peak[i] = 0.5
        if d.month == 5 and d.day <= 5: peak[i] = 0.25
        if d.month in (7, 8): peak[i] = max(peak[i], 0.12)
    for year, lny in {2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22",
                      2024: "2024-02-10", 2025: "2025-01-29", 2026: "2026-02-17"}.items():
        lny = pd.Timestamp(lny)
        m = (dates >= lny - pd.Timedelta(days=1)) & (dates <= lny + pd.Timedelta(days=6))
        peak[m] = np.maximum(peak[m], 0.6)
    base_visitors = 150_000 * ramp * weekly * (1 + peak)
    for cp, share in points.items():
        for direction, lag_shift in [("arrival", 0), ("departure", 1)]:
            noise = rng.lognormal(0, 0.06, len(dates))
            mainland = base_visitors * share * noise
            if lag_shift:
                mainland = np.roll(mainland, 1)
            rows.append(pd.DataFrame({
                "date": dates, "control_point": cp, "direction": direction,
                "hk_residents": (mainland * 1.8 * rng.lognormal(0, 0.05, len(dates))).round(),
                "mainland_visitors": mainland.round(),
                "other_visitors": (mainland * 0.12 * rng.lognormal(0, 0.1, len(dates))).round(),
            }))
    df = pd.concat(rows, ignore_index=True)
    df["total"] = df[["hk_residents", "mainland_visitors", "other_visitors"]].sum(axis=1)
    return df


def load_daily_arrivals(raw: pd.DataFrame, column: str = "mainland_visitors",
                        since: pd.Timestamp = REOPENING) -> pd.Series:
    """Daily total *arrivals* of one visitor group across all control points.
    Starts at the reopening date because the COVID closure era is a structural break."""
    arr = raw[raw["direction"].str.contains("arriv", case=False, na=False)].copy()

    # The Immigration Department file includes a pre-aggregated "Total" row
    # alongside individual control points. Prefer that row when present;
    # otherwise sum individual control points. Never sum both, which would
    # double-count the daily traffic.
    total_rows = arr[arr["control_point"].astype(str).str.strip().str.casefold().eq("total")]
    if not total_rows.empty:
        s = total_rows.groupby("date")[column].sum().sort_index()
    else:
        detail = arr[~arr["control_point"].astype(str).str.strip().str.casefold().eq("total")]
        s = detail.groupby("date")[column].sum().sort_index()
    s = s[s.index >= since]
    s = s.asfreq("D")
    s = s.interpolate(limit=3)  # patch rare missing days only; do not invent long gaps
    return s.rename(column)
