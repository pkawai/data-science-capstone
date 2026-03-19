# model.py — Ensemble training, persistence, and prediction with meta-labeling
#
# ModelBundle (dict) stored in each .pkl:
#   primary_models      : list of [XGBClassifier, LGBMClassifier, RandomForest]
#   meta_model          : binary XGBClassifier or None
#   feature_cols        : list[str] — feature order used at training time
#   confidence_threshold: float — optimised per-pair confidence cut-off
#   meta_threshold      : float — meta-model minimum win-probability

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

import config

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

N_CLASSES = 3
LABEL_MAP = {0: "Sell", 1: "Hold", 2: "Buy"}


# ── Sample weights ──────────────────────────────────────────────────────────────

def _sample_weights(y: pd.Series) -> np.ndarray:
    counts    = y.value_counts().sort_index()
    max_count = counts.max()
    weight_map = {cls: max_count / cnt for cls, cnt in counts.items()}
    return y.map(weight_map).values


# ── Individual model trainers ───────────────────────────────────────────────────

def _train_xgb(X, y, params: dict = None) -> XGBClassifier:
    sw = _sample_weights(y)
    p  = {k: v for k, v in (params or config.XGBOOST_PARAMS).items()
          if k != "use_label_encoder"}
    model = XGBClassifier(**p, num_class=N_CLASSES, objective="multi:softprob")
    model.fit(X, y, sample_weight=sw, eval_set=[(X, y)], verbose=False)
    return model


def _train_lgbm(X, y) -> "LGBMClassifier | None":
    if not LGBM_AVAILABLE:
        return None
    sw = _sample_weights(y)
    model = LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        num_class=N_CLASSES, objective="multiclass",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y, sample_weight=sw)
    return model


def _train_rf(X, y) -> "RandomForestClassifier | None":
    if not RF_AVAILABLE:
        return None
    sw = _sample_weights(y)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42, n_jobs=-1,
    )
    model.fit(X, y, sample_weight=sw)
    return model


# ── Public training API ─────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series, xgb_params: dict = None) -> list:
    """
    Train an ensemble of [XGBoost, LightGBM, RandomForest].
    Returns a list of fitted estimators (list acts as the 'model' in backtest).
    """
    counts = y.value_counts().sort_index()
    print(f"[model] Class distribution:\n{counts.to_string()}")

    models = []
    xgb = _train_xgb(X, y, params=xgb_params)
    models.append(xgb)
    print("[model] XGBoost trained.")

    lgbm = _train_lgbm(X, y)
    if lgbm:
        models.append(lgbm)
        print("[model] LightGBM trained.")
    else:
        print("[model] LightGBM not available — skipping.")

    rf = _train_rf(X, y)
    if rf:
        models.append(rf)
        print("[model] RandomForest trained.")

    print(f"[model] Ensemble: {len(models)} models.")
    return models


def train_meta(X_meta: np.ndarray, y_meta: np.ndarray) -> XGBClassifier:
    """Train a binary meta-labeling model (will this signal win?)."""
    from collections import Counter
    counts = Counter(y_meta.tolist())
    scale  = counts.get(0, 1) / max(counts.get(1, 1), 1)

    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_meta, y_meta)
    print(f"[model] Meta-model trained on {len(y_meta)} samples "
          f"(wins={counts.get(1,0)}, losses={counts.get(0,0)}).")
    return model


# ── Persistence ─────────────────────────────────────────────────────────────────

def save(bundle: dict, symbol: str) -> None:
    path = config.model_path(symbol)
    joblib.dump(bundle, path)
    print(f"[model] Bundle saved → {path}")


def load(symbol: str) -> dict:
    path   = config.model_path(symbol)
    bundle = joblib.load(path)
    print(f"[model] Bundle loaded ← {path}")

    # Backward compatibility: old format was a bare XGBClassifier
    if not isinstance(bundle, dict):
        print("[model] Old single-model format detected — wrapping in bundle.")
        bundle = {
            "primary_models":       [bundle],
            "meta_model":           None,
            "feature_cols":         None,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "meta_threshold":       config.META_CONFIDENCE_THRESHOLD,
        }
    return bundle


# ── Inference ───────────────────────────────────────────────────────────────────

def predict_proba(model_or_bundle, X: pd.DataFrame) -> np.ndarray:
    """
    Average class probabilities across all ensemble members.
    Accepts: dict bundle | list of models | single model (backward compat).
    Returns shape (n_samples, 3).
    """
    if isinstance(model_or_bundle, dict):
        models = model_or_bundle["primary_models"]
    elif isinstance(model_or_bundle, list):
        models = model_or_bundle
    else:
        return model_or_bundle.predict_proba(X)

    probas = [m.predict_proba(X) for m in models]
    return np.mean(probas, axis=0)


def predict_signal(bundle_or_model,
                   X: pd.DataFrame,
                   threshold: float = None) -> tuple[int, float]:
    """
    Predict signal for the LAST row of X.
    Pipeline: ensemble → confidence filter → meta-model filter → signal.
    Returns (signal, confidence). Forces Hold (1) if either filter fails.
    """
    # ── Backward compat: bare single model ──────────────────────────────────
    if not isinstance(bundle_or_model, dict):
        proba      = bundle_or_model.predict_proba(X.iloc[[-1]])
        signal     = int(np.argmax(proba[0]))
        confidence = float(proba[0][signal])
        thr        = threshold or config.CONFIDENCE_THRESHOLD
        return (1, confidence) if confidence < thr else (signal, confidence)

    bundle    = bundle_or_model
    feat_cols = bundle.get("feature_cols")

    # Align feature columns to training order
    X_last = X.iloc[[-1]]
    if feat_cols is not None:
        available = [f for f in feat_cols if f in X.columns]
        X_last    = X_last[available]

    # ── Step 1: ensemble probabilities ──────────────────────────────────────
    proba      = predict_proba(bundle["primary_models"], X_last)  # (1, 3)
    signal     = int(np.argmax(proba[0]))
    confidence = float(proba[0][signal])

    # ── Step 2: confidence threshold ────────────────────────────────────────
    thr = threshold or bundle.get("confidence_threshold", config.CONFIDENCE_THRESHOLD)
    if confidence < thr or signal == 1:
        return 1, confidence

    # ── Step 3: meta-model filter ────────────────────────────────────────────
    meta_model = bundle.get("meta_model")
    if meta_model is not None:
        X_meta      = np.concatenate([X_last.values, proba], axis=1)
        meta_proba  = meta_model.predict_proba(X_meta)[0]
        meta_win    = float(meta_proba[1])
        meta_thr    = bundle.get("meta_threshold", config.META_CONFIDENCE_THRESHOLD)
        if meta_win < meta_thr:
            return 1, confidence   # meta says this will lose

    return signal, confidence


# ── Diagnostics ─────────────────────────────────────────────────────────────────

def feature_importance_plot(bundle_or_model,
                             feature_names: list[str],
                             symbol: str = "",
                             top_n: int = 20,
                             save_path: str | None = None) -> None:
    # Extract XGBoost model from whatever format is passed
    if isinstance(bundle_or_model, dict):
        xgb_model = bundle_or_model["primary_models"][0]
    elif isinstance(bundle_or_model, list):
        xgb_model = bundle_or_model[0]
    else:
        xgb_model = bundle_or_model

    scores = pd.Series(
        xgb_model.get_booster().get_score(importance_type="gain"),
        name="Importance",
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    scores.plot.barh(ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title(f"Top-{top_n} Feature Importances (XGBoost) — {symbol}")
    ax.set_xlabel("Gain")
    plt.tight_layout()

    path = save_path or f"feature_importance_{symbol}.png"
    fig.savefig(path, dpi=150)
    print(f"[model] Feature importance chart saved → {path}")
    plt.show()
