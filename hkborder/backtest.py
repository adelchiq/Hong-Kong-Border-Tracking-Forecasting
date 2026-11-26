"""Walk-forward (expanding window) backtest with a naive baseline."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import make_features


def _models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
        "gbm": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                             l2_regularization=1.0, random_state=0),
    }


def walk_forward(y: pd.Series, horizon: int = 7, n_folds: int = 12, fold_days: int = 30,
                 min_train: int = 180) -> pd.DataFrame:
    """Predict each test block using models trained only on data before it.

    Returns a frame with columns: actual, naive (same weekday last week), ridge, gbm, fold.
    The naive baseline is the same weekday, at least `horizon` days back, so it only uses data available at forecast time.
    """
    X = make_features(y, horizon)
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    n = len(data)
    first_test = n - n_folds * fold_days
    if first_test < min_train:
        raise ValueError(f"Not enough history: {n} usable days for {n_folds} folds of {fold_days} days.")
    out = []
    for k in range(n_folds):
        start = first_test + k * fold_days
        # The first test day is forecast from origin (start - horizon), so only targets
        # known by then may be used for training: no peeking at the days just before the block.
        train, test = data.iloc[:start - horizon + 1], data.iloc[start:start + fold_days]
        feats = [c for c in data.columns if c != "y"]
        res = pd.DataFrame({"actual": test["y"], "fold": k})
        naive_lag = 7 * int(np.ceil(horizon / 7))  # same weekday, at least `horizon` days back
        res["naive"] = y.shift(naive_lag).reindex(test.index)
        for name, model in _models().items():
            model.fit(train[feats], train["y"])
            res[name] = model.predict(test[feats])
        out.append(res)
    return pd.concat(out)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for m in ["naive", "ridge", "gbm"]:
        err = res[m] - res["actual"]
        rows[m] = {"MAE": err.abs().mean(),
                   "MAPE_%": (err.abs() / res["actual"]).mean() * 100,
                   "RMSE": float(np.sqrt((err ** 2).mean()))}
    return pd.DataFrame(rows).T.round(2)
