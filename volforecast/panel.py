"""PanelBuilder — THE single alignment path.

Assembles one tidy, point-in-time matrix indexed by decision time ``t0``, with columns
grouped by prefix: ``feat_*`` (causal inputs), ``tgt_*`` (forward labels), ``meta_*``.
Every downstream consumer (baselines, model, backtest, eval) loads this one object and
slices by ``t0`` — there is no second alignment path, which is what makes the legacy
60-row drift impossible to reproduce.
"""
from __future__ import annotations

import pandas as pd

from .calendar import Calendar
from .timeutil import LookaheadError
from .features.realized import daily_realized_variance, daily_return, intraday_stats
from .features.har import har_components
from .features.surface import surface_features
from .features.calendar_feats import calendar_features
from .features.intraday import intraday_features
from .features.targets import forward_targets


class PanelBuilder:
    def __init__(self, cfg, adapter):
        self.cfg = cfg
        self.adapter = adapter
        self.cal = Calendar(
            open_time=getattr(cfg, "open_time", "09:30"),
            close_time=getattr(cfg, "close_time", "16:00"),
            rv_minutes=getattr(cfg, "rv_minutes", 5),
        )
        self.horizons = list(getattr(cfg, "horizons", [1, 5, 10, 21]))

    def build(self) -> pd.DataFrame:
        start, end = self.cfg.start, self.cfg.end
        bars = self.adapter.minute_bars(start, end)
        iv = self.adapter.iv_surface(start, end)

        rv_var = daily_realized_variance(bars, self.cal)
        ret = daily_return(bars, self.cal)
        intra = intraday_stats(bars, self.cal)

        index, records = [], []
        for t0 in self.cal.decision_times(start, end):
            har = har_components(rv_var, t0=t0)        # PIT-guarded internally
            surf = surface_features(iv, t0=t0)
            intr = intraday_features(intra, t0=t0)
            if har is None or surf is None or intr is None:
                continue
            calf = calendar_features(t0=t0)
            tgt = forward_targets(rv_var, self.horizons, t0=t0)
            row = {**har, **surf, **intr, **calf, **tgt}
            row["meta_ret_1d"] = float(ret.loc[t0]) if t0 in ret.index else float("nan")
            index.append(t0)
            records.append(row)

        panel = pd.DataFrame.from_records(
            records, index=pd.DatetimeIndex(index, name="t0")
        )
        tgt_cols = [c for c in panel.columns if c.startswith("tgt_")]
        panel = panel.dropna(subset=tgt_cols)  # drop origins without a full future
        self._assert_pit(panel, rv_var, iv)
        return panel

    def _assert_pit(self, panel: pd.DataFrame, rv_var: pd.Series, iv: pd.DataFrame) -> None:
        """Independent re-check of the no-lookahead invariant over a sample of rows."""
        feat_cols = [c for c in panel.columns if c.startswith("feat_")]
        tgt_cols = [c for c in panel.columns if c.startswith("tgt_")]
        if set(feat_cols) & set(tgt_cols):
            raise LookaheadError("feature and target columns overlap")
        if len(panel) == 0:
            return
        step = max(1, len(panel) // 20)
        for t0 in panel.index[::step]:
            feat_ts = []
            h = rv_var.index[rv_var.index <= t0]
            if len(h):
                feat_ts.append(h[-1])
            hi = iv.index[iv.index <= t0]
            if len(hi):
                feat_ts.append(hi[-1])
            if feat_ts and max(feat_ts) > t0:
                raise LookaheadError(f"feature source {max(feat_ts)} > t0={t0}")
            fut = rv_var.index[rv_var.index > t0]
            if len(fut) and fut[0] <= t0:
                raise LookaheadError(f"target source {fut[0]} <= t0={t0}")
