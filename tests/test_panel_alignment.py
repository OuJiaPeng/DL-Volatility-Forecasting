"""One alignment path: panel rows are well-formed and targets match hand computation."""
import numpy as np
import pandas as pd


def test_panel_shape_and_index(panel):
    assert panel.index.is_monotonic_increasing
    assert panel.index.is_unique
    assert panel.index.name == "t0"
    for h in (1, 5, 10, 21):
        assert f"tgt_rv_{h}" in panel.columns
    for c in ("feat_rv_d", "feat_rv_w", "feat_rv_m", "feat_atm_iv"):
        assert c in panel.columns


def test_target_matches_manual(panel, rv_var):
    """tgt_rv_1 at t0 == sqrt of the realized variance of the very next session."""
    t0 = panel.index[5]
    nxt = rv_var.loc[rv_var.index > t0].iloc[0]
    assert np.isclose(panel.loc[t0, "tgt_rv_1"], np.sqrt(nxt))


def test_target_5d_matches_manual(panel, rv_var):
    t0 = panel.index[5]
    seg = rv_var.loc[rv_var.index > t0].iloc[:5]
    assert np.isclose(panel.loc[t0, "tgt_rv_5"], np.sqrt(seg.mean()))


def test_feature_is_not_target(panel, rv_var):
    """Trailing feat_rv_d != forward tgt_rv_1 (the legacy bug conflated them)."""
    t0 = panel.index[5]
    assert not np.isclose(panel.loc[t0, "feat_rv_d"], panel.loc[t0, "tgt_rv_1"])
