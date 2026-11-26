"""Run the full analysis and save figures + metrics.

    python main.py                  # download live data from the Immigration Department
    python main.py --csv file.csv   # use a file you downloaded manually
    python main.py --offline        # synthetic data, no internet needed
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor

from hkborder import load_raw, load_daily_arrivals, synthetic_raw, make_features, walk_forward, summarise
from hkborder.data import normalise


def get_raw(args):
    if args.offline:
        return synthetic_raw(), "SYNTHETIC data (offline mode)"
    try:
        if args.csv:
            return normalise(pd.read_csv(args.csv, encoding="utf-8-sig")), f"local file {args.csv}"
        return load_raw(), "data.gov.hk / Immigration Department"
    except Exception as e:
        sys.exit(f"Could not load the data: {e}\n"
                 "Download the CSV from data.gov.hk and use --csv, or try --offline.")


def plot_overview(y, label):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index, y / 1e3, lw=.6, alpha=.6, label="daily")
    ax.plot(y.index, y.rolling(28).mean() / 1e3, lw=2, label="28-day average")
    ax.set(ylabel="arrivals (thousands)", title=f"Daily Mainland visitor arrivals: {label}")
    ax.legend(); fig.tight_layout(); fig.savefig("figures/overview.png", dpi=150); plt.close(fig)


def plot_seasonality(y):
    d = pd.DataFrame({"y": y})
    d["rel"] = d.y / d.y.rolling(28, center=True, min_periods=14).mean()  # remove the trend
    d["dow"], d["month"] = d.index.dayofweek, d.index.month
    piv = d.pivot_table(index="dow", columns="month", values="rel", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title("Arrivals relative to 28-day trend, by weekday and month")
    fig.colorbar(im); fig.tight_layout(); fig.savefig("figures/seasonality.png", dpi=150); plt.close(fig)


def plot_backtest(res, summary):
    last = res.iloc[-120:]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(last.index, last.actual / 1e3, "k", lw=2, label="actual")
    for m, st in [("naive", ":"), ("ridge", "--"), ("gbm", "-")]:
        ax[0].plot(last.index, last[m] / 1e3, st, lw=1.2, label=m)
    ax[0].set(ylabel="arrivals (thousands)", title="7-day-ahead forecasts vs actual (last 120 days of backtest)")
    ax[0].legend()
    summary["MAPE_%"].plot.bar(ax=ax[1], color=["grey", "C0", "C1"], rot=0)
    ax[1].set(ylabel="MAPE (%)", title="Walk-forward error (lower is better)")
    fig.tight_layout(); fig.savefig("figures/backtest.png", dpi=150); plt.close(fig)


def plot_importance(y, horizon=7):
    X = make_features(y, horizon)
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    train, test = data.iloc[:-90 - horizon], data.iloc[-90:]
    feats = [c for c in data.columns if c != "y"]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                      l2_regularization=1.0, random_state=0).fit(train[feats], train.y)
    pi = permutation_importance(m, test[feats], test.y, n_repeats=15, random_state=0,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.tail(10).plot.barh(ax=ax)
    ax.set(xlabel="increase in MAE when shuffled", title="What drives the GBM forecast? (permutation importance)")
    fig.tight_layout(); fig.savefig("figures/importance.png", dpi=150); plt.close(fig)
    return imp.sort_values(ascending=False).head(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    raw, label = get_raw(args)
    y = load_daily_arrivals(raw, "mainland_visitors")
    print(f"Data: {label}\nSeries: {y.index.min().date()} to {y.index.max().date()} ({len(y)} days), "
          f"mean {y.mean():,.0f} Mainland arrivals/day")

    res = walk_forward(y, horizon=args.horizon)
    summary = summarise(res)
    print("\nWalk-forward backtest (12 folds x 30 days, expanding window):")
    print(summary.to_string())
    best = summary["MAPE_%"].drop("naive").idxmin()
    lift = (1 - summary.loc[best, "MAPE_%"] / summary.loc["naive", "MAPE_%"]) * 100
    print(f"\nBest model: {best}, {lift:.0f}% lower MAPE than the seasonal-naive baseline.")

    feats = make_features(y, args.horizon).reindex(res.index)
    holiday = (feats.golden_week | feats.lny_week | feats.labour_day).astype(bool)
    err = (res[best] - res.actual).abs() / res.actual * 100
    print(f"{best} MAPE on holiday weeks: {err[holiday].mean():.1f}% vs normal days: {err[~holiday].mean():.1f}%")

    res.to_csv("figures/backtest_predictions.csv"); summary.to_csv("figures/backtest_metrics.csv")
    plot_overview(y, label); plot_seasonality(y); plot_backtest(res, summary)
    print("\nTop drivers (GBM permutation importance):\n" + plot_importance(y, args.horizon).to_string())
    print("\nSaved figures to ./figures")


if __name__ == "__main__":
    main()
