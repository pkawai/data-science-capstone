# test_features_and_model.py — feature-list hygiene + train/serve parity with
# the EXISTING model bundles (the "bot works like nothing happened" guarantee).
#
# 1. Raw price-LEVEL columns (BB_upper/BB_lower/MA_20/MA_50) must not be
#    offered as training features: tree models split on absolute price levels,
#    which act as a date proxy — memorised in-sample, garbage out-of-sample.
#    Normalised versions (BB_pct, BB_width, MA_cross, Price_MA_dist) stay.
# 2. build_features must keep COMPUTING those columns: the bundles already
#    trained (tracked .pkl files) list them in feature_cols and must keep
#    finding them at serve time.
# 3. predict_signal must fail LOUDLY if a stored feature column is missing
#    from the frame — silently subsetting hid exactly this class of bug.

import os

import numpy as np
import pandas as pd
import pytest

import config
from features import build_features, get_feature_columns
from model import load, predict_signal

RAW_LEVEL_COLS = {"BB_upper", "BB_lower", "MA_20", "MA_50"}
BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def feature_df():
    """~500 H1 bars of a synthetic random walk → full feature frame."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2026-01-05", periods=n, freq="h", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    spread = np.abs(rng.normal(0, 0.0004, n))
    df = pd.DataFrame({
        "Open":   close + rng.normal(0, 0.0002, n),
        "High":   close + spread,
        "Low":    close - spread,
        "Close":  close,
        "Volume": rng.integers(100, 1000, n).astype(float),
    }, index=idx)
    out = build_features(df)
    assert len(out) > 50, "synthetic frame collapsed in build_features"
    return out


def test_raw_price_levels_are_not_training_features(feature_df):
    cols = set(get_feature_columns(feature_df))
    leaked = cols & RAW_LEVEL_COLS
    assert not leaked, (
        f"Raw price-level columns offered as features: {sorted(leaked)} — "
        "absolute levels act as a date proxy for tree models."
    )
    # The normalised counterparts must still be there.
    assert {"BB_pct", "BB_width", "MA_cross", "Price_MA_dist"} <= cols


def test_raw_price_levels_are_still_computed(feature_df):
    # Old bundles trained WITH these columns must keep finding them.
    assert RAW_LEVEL_COLS <= set(feature_df.columns)


def test_predict_signal_fails_loudly_on_missing_feature_columns():
    bundle = {
        "primary_models": [object()],  # never reached — check happens first
        "meta_model": None,
        "feature_cols": ["RSI", "DOES_NOT_EXIST"],
        "confidence_threshold": 0.6,
        "meta_threshold": 0.5,
    }
    X = pd.DataFrame({"RSI": [50.0]})
    with pytest.raises(ValueError, match="DOES_NOT_EXIST"):
        predict_signal(bundle, X)


@pytest.mark.parametrize("symbol", config.PAIRS)
def test_existing_tracked_bundles_still_predict(feature_df, symbol):
    """End-to-end serve-path parity: the bundles currently tracked in git
    (the ones the live bot loads) must predict from a full build_features
    frame — i.e. every stored feature column still exists and aligns."""
    path = os.path.join(BOT_DIR, config.model_path(symbol))
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    cwd = os.getcwd()
    os.chdir(BOT_DIR)  # model_path() is relative
    try:
        bundle = load(symbol)
    finally:
        os.chdir(cwd)

    stored = bundle.get("feature_cols") or []
    missing = [c for c in stored if c not in feature_df.columns]
    assert missing == [], (
        f"build_features no longer produces columns the {symbol} bundle was "
        f"trained on: {missing}"
    )

    signal, confidence = predict_signal(bundle, feature_df)
    assert signal in (0, 1, 2)
    assert 0.0 <= confidence <= 1.0
