"""Splits are time-ordered, disjoint, and purged so target windows don't cross boundaries."""
from volforecast.splits import make_splits


def test_splits_disjoint_and_ordered(panel, cfg):
    s = make_splits(panel, cfg)
    assert len(s.train) and len(s.val) and len(s.test)
    assert set(s.train).isdisjoint(s.val)
    assert set(s.train).isdisjoint(s.test)
    assert set(s.val).isdisjoint(s.test)
    assert s.train.max() < s.val.min() < s.test.min()


def test_purge_prevents_target_overlap(panel, cfg):
    s = make_splits(panel, cfg)
    max_h = max(cfg.horizons)
    pos = {t: i for i, t in enumerate(panel.index)}
    # gap between last train origin and first val origin must exceed the horizon
    assert pos[s.val.min()] - pos[s.train.max()] > max_h


def test_norm_stats_fit_on_train_only(panel, cfg):
    s = make_splits(panel, cfg)
    assert list(s.mean.index) == s.feat_cols
    assert (s.std > 0).all()
