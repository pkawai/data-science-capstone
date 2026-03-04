# model.py — XGBoost training, persistence, and prediction

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import config


# Label mapping: Sell=0, Hold=1, Buy=2
LABEL_MAP   = {0: "Sell", 1: "Hold", 2: "Buy"}
N_CLASSES   = 3


def train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost classifier with class-imbalance correction.

    Parameters
    ----------
    X : feature DataFrame
    y : integer labels (0=Sell, 1=Hold, 2=Buy)

    Returns
    -------
    Fitted XGBClassifier
    """
    counts = y.value_counts().sort_index()
    print(f"[model] Class distribution:\n{counts.to_string()}")

    # Use max_class_count / each_class_count as sample_weight
    max_count = counts.max()
    weight_map = {cls: max_count / cnt for cls, cnt in counts.items()}
    sample_weight = y.map(weight_map).values

    params = {k: v for k, v in config.XGBOOST_PARAMS.items()
              if k != "use_label_encoder"}   # deprecated param removed in XGB 2.x

    model = XGBClassifier(**params, num_class=N_CLASSES, objective="multi:softprob")
    model.fit(X, y, sample_weight=sample_weight,
              eval_set=[(X, y)], verbose=False)

    print(f"[model] Training done. Estimators: {model.n_estimators}")
    return model


def save(model: XGBClassifier, path: str = config.MODEL_PATH) -> None:
    joblib.dump(model, path)
    print(f"[model] Saved → {path}")


def load(path: str = config.MODEL_PATH) -> XGBClassifier:
    model = joblib.load(path)
    print(f"[model] Loaded ← {path}")
    return model


def predict_proba(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Returns shape (n_samples, 3) probability array.
    Columns: [P(Sell), P(Hold), P(Buy)]
    """
    return model.predict_proba(X)


def predict_signal(model: XGBClassifier,
                   X: pd.DataFrame,
                   threshold: float = config.CONFIDENCE_THRESHOLD
                   ) -> tuple[int, float]:
    """
    Predict the most likely signal for the LAST row of X.

    Returns
    -------
    (signal, confidence)
    signal     : 0=Sell, 1=Hold, 2=Buy
    confidence : max class probability
    If max probability < threshold, returns (1, confidence) — Hold.
    """
    proba = predict_proba(model, X.iloc[[-1]])
    signal    = int(np.argmax(proba[0]))
    confidence = float(proba[0][signal])

    if confidence < threshold:
        return 1, confidence   # force Hold
    return signal, confidence


def feature_importance_plot(model: XGBClassifier,
                             feature_names: list[str],
                             top_n: int = 20,
                             save_path: str | None = "feature_importance.png"
                             ) -> None:
    """Bar chart of the top-N features by gain importance."""
    scores = pd.Series(
        model.get_booster().get_fscore(importance_type="gain"),
        name="Importance",
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    scores.plot.barh(ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Top-{top_n} Feature Importances (Gain)")
    ax.set_xlabel("Gain")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[model] Feature importance chart saved → {save_path}")
    plt.show()
