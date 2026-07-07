"""MassiveAdapter (Polygon.io rebrand) — structured stub for the real data path.

Wiring guide (v1 leaves the network calls behind a credential check so the package
installs and tests without an API key):

  * Minute bars  -> Aggregates v2:
      GET {base}/v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}
      (ticker e.g. "I:SPX" for the index, or "SPY"/"ES" as a proxy)
  * IV surface   -> Options snapshot / chain (Massive ships computed IV + Greeks):
      GET {base}/v3/snapshot/options/{underlying}
      Summarize ATM IV, 25-delta skew, and term slope per session close.

``base`` defaults to https://api.polygon.io (still active post-rebrand). Set
``cfg.api_key`` (or MASSIVE_API_KEY / POLYGON_API_KEY env var). Until implemented,
methods raise NotImplementedError with a clear pointer.
"""
from __future__ import annotations

import os
import pandas as pd

from .adapter import MarketDataAdapter


class MassiveAdapter(MarketDataAdapter):
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = getattr(cfg, "base_url", "https://api.polygon.io")
        self.ticker = getattr(cfg, "ticker", "I:SPX")
        self.api_key = (
            getattr(cfg, "api_key", None)
            or os.getenv("MASSIVE_API_KEY")
            or os.getenv("POLYGON_API_KEY")
        )

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "MassiveAdapter needs an API key. Set cfg.api_key or MASSIVE_API_KEY / "
                "POLYGON_API_KEY. (v1 runs end-to-end on the SyntheticAdapter without one.)"
            )

    def minute_bars(self, start, end) -> pd.DataFrame:  # pragma: no cover - network path
        self._require_key()
        raise NotImplementedError(
            "Wire GET /v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}; "
            "normalize results to ts-indexed OHLCV (ts = bar end)."
        )

    def iv_surface(self, start, end) -> pd.DataFrame:  # pragma: no cover - network path
        self._require_key()
        raise NotImplementedError(
            "Wire GET /v3/snapshot/options/{underlying}; summarize ATM IV / skew / "
            "term slope per session close into ts-indexed columns."
        )
