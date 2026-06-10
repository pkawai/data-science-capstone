# test_config_overrides.py — local_overrides.json must not re-enable the
# retired directional-override band-aid (see config.py "DISABLED" section).
#
# Background: calibrate_floor.py used to write
#   {"DIRECTIONAL_OVERRIDE": true, "DIRECTIONAL_FLOOR": ...}
# and config.py loads local_overrides.json LAST, so a stale file on the live
# machine silently re-enabled the band-aid even after it was disabled in
# config.py. The loader must ignore those retired keys (loudly) while still
# applying legitimate machine-specific overrides.

import importlib
import json
import os

import config as config_module

OVERRIDES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(config_module.__file__)),
    "local_overrides.json",
)


def _reload_with_overrides(overrides: dict):
    """Write a local_overrides.json, reload config, return the module."""
    with open(OVERRIDES_PATH, "w") as f:
        json.dump(overrides, f)
    return importlib.reload(config_module)


def _cleanup():
    if os.path.exists(OVERRIDES_PATH):
        os.remove(OVERRIDES_PATH)
    importlib.reload(config_module)


def test_stale_overrides_cannot_reenable_directional_override():
    assert not os.path.exists(OVERRIDES_PATH), (
        "Pre-existing local_overrides.json found — remove it before testing."
    )
    try:
        cfg = _reload_with_overrides(
            {"DIRECTIONAL_OVERRIDE": True, "DIRECTIONAL_FLOOR": 0.25}
        )
        assert cfg.DIRECTIONAL_OVERRIDE is False, (
            "A stale local_overrides.json re-enabled the retired directional "
            "override — the bot would bypass the confidence gate and meta-model."
        )
        assert cfg.DIRECTIONAL_FLOOR == 0.50, (
            "DIRECTIONAL_FLOOR from a stale overrides file must be ignored."
        )
    finally:
        _cleanup()


def test_legitimate_overrides_still_apply():
    assert not os.path.exists(OVERRIDES_PATH)
    try:
        cfg = _reload_with_overrides({"ADX_THRESHOLD": 22})
        assert cfg.ADX_THRESHOLD == 22, (
            "Non-retired keys in local_overrides.json must still apply."
        )
    finally:
        _cleanup()
