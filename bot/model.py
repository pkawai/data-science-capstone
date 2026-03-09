# model.py — XGBoost training, persistence, and prediction

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

import config

N_CLASSES = 3
LABEL_MAP = {0: "Sell", 1: "Hold", 2: "Buy"}


def train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Train XGBoost with class-imbalance correction via sample weights."""
    counts = y.value_counts().sort_index()
    print(f"[model] Class distribution:\n{counts.to_string()}")

    max_count  = counts.max()
    weight_map = {cls: max_count / cnt for cls, cnt in counts.items()}
    sample_weight = y.map(weight_map).values

    params = {k: v for k, v in config.XGBOOST_PARAMS.items()
              if k != "use_label_encoder"}

    model = XGBClassifier(**params, num_class=N_CLASSES, objective="multi:softprob")
    model.fit(X, y, sample_weight=sample_weight,
              eval_set=[(X, y)], verbose=False)

    print(f"[model] Training done. Estimators: {model.n_estimators}")
    return model


def save(model: XGBClassifier, symbol: str) -> None:
    path = config.model_path(symbol)
    joblib.dump(model, path)
    print(f"[model] Saved → {path}")


def load(symbol: str) -> XGBClassifier:
    path = config.model_path(symbol)
    model = joblib.load(path)
    print(f"[model] Loaded ← {path}")
    return model


def predict_proba(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Returns shape (n_samples, 3) — [P(Sell), P(Hold), P(Buy)]"""
    return model.predict_proba(X)


def predict_signal(model: XGBClassifier,
                   X: pd.DataFrame,
                   threshold: float = config.CONFIDENCE_THRESHOLD
                   ) -> tuple[int, float]:
    """
    Predict signal for the LAST row of X.
    Returns (signal, confidence). Forces Hold if confidence < threshold.
    """
    proba      = predict_proba(model, X.iloc[[-1]])
    signal     = int(np.argmax(proba[0]))
    confidence = float(proba[0][signal])

    if confidence < threshold:
        return 1, confidence
    return signal, confidence


def feature_importance_plot(model: XGBClassifier,
                             feature_names: list[str],
                             symbol: str = "",
                             top_n: int = 20,
                             save_path: str | None = None) -> None:
    scores = pd.Series(
        model.get_booster().get_score(importance_type="gain"),
        name="Importance",
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    scores.plot.barh(ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Top-{top_n} Feature Importances — {symbol}")
    ax.set_xlabel("Gain")
    plt.tight_layout()

    path = save_path or f"feature_importance_{symbol}.png"
    fig.savefig(path, dpi=150)
    print(f"[model] Feature importance chart saved → {path}")
    plt.show()
