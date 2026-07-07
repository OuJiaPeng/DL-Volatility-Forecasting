"""DatabentoAdapter — real SPX data path (decided over Massive, July 2026).

Why Databento: Massive has no *historical* IV/Greeks (snapshots only), so the IV
surface must be solved from raw quotes either way — and Databento's narrow-slice
usage pricing, 2013+ 1-min consolidated NBBO (``cbbo-1m``), and ``SPX.OPT`` parent
symbology make it the cheaper, deeper, more point-in-time-honest source.

Data plan:
  * ``minute_bars``  — ES futures (GLBX.MDP3, continuous front contract), converted
    to America/New_York and tz-stripped to match the tz-naive Calendar. ES is a
    traded instrument, avoiding the stale-component RV bias of the cash index.
  * ``iv_surface``   — per session: SPX.OPT chain definitions + close-window
    cbbo-1m quotes -> volsolve (parity forward, Black-76 inversion) -> tenor-matched
    ATM IV / 25-delta skew / term slope, in DAILY vol units; VIX column from Cboe's
    free daily CSV.

Every remote pull is cached to parquet under ``data_cache/`` first — API spend is
one-time, and reruns are free/offline. Requires the ``databento`` package
(``pip install -e ".[databento]"``) and DATABENTO_API_KEY (or cfg.api_key).

NOTE: written against Databento's documented API (timeseries.get_range, parent
symbology, DBN->DataFrame column conventions) but not yet exercised against a live
key. First live run should start with ``metadata.get_cost`` on a single day.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd

from .adapter import MarketDataAdapter
from .schema import validate_bars, validate_iv
from .cache import save_parquet, load_parquet
from . import volsolve
from ..calendar import Calendar

VIX_CSV_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
ANN = np.sqrt(252.0)


def _key_from_dotenv(name: str = "DATABENTO_API_KEY", path: str = ".env") -> str | None:
    """Minimal .env fallback (gitignored file) so no key ever lands in config/repo."""
    if not os.path.exists(path):
        return None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


class DatabentoAdapter(MarketDataAdapter):
    def __init__(self, cfg):
        self.cfg = cfg
        self.api_key = (
            getattr(cfg, "api_key", None)
            or os.getenv("DATABENTO_API_KEY")
            or _key_from_dotenv()
        )
        db_cfg = getattr(cfg, "databento", None)
        # underlying: defaults to ES futures (SPX config); single names override to
        # their equity venue (e.g. XNAS.ITCH / NVDA / raw_symbol)
        self.u_dataset = getattr(db_cfg, "underlying_dataset", "GLBX.MDP3") if db_cfg else "GLBX.MDP3"
        self.u_stype = getattr(db_cfg, "underlying_stype", "continuous") if db_cfg else "continuous"
        self.es_symbol = (getattr(db_cfg, "underlying_symbol", None)
                          or getattr(db_cfg, "es_symbol", "ES.c.0")) if db_cfg else "ES.c.0"
        # corporate actions: [{date, ratio}] — prices BEFORE date divided by ratio so
        # cross-day returns/gaps are continuous (intraday RV and same-day chain solves
        # are unaffected either way)
        self.splits = list(getattr(db_cfg, "splits", []) or []) if db_cfg else []
        self.close_window_min = int(getattr(db_cfg, "close_window_minutes", 15)) if db_cfg else 15
        # SPX weeklies/0DTE trade under the separate SPXW root — both parents are needed
        # for short tenors (confirmed live 2026-07: SPXW.OPT resolves, ~2.4x SPX's contracts).
        default_roots = ["SPX.OPT", "SPXW.OPT"]
        self.option_roots = list(getattr(db_cfg, "option_roots", default_roots)) if db_cfg else default_roots
        # cache-name prefix per underlying so multi-asset caches never collide
        # ("SPX.OPT" -> "spx", keeping all existing SPX cache files valid)
        self.root_tag = self.option_roots[0].split(".")[0].lower()
        self.cache_dir = getattr(cfg, "cache_dir", "data_cache")
        self.horizons = list(getattr(cfg, "horizons", [1, 5, 10, 21]))
        self.cal = Calendar(
            open_time=getattr(cfg, "open_time", "09:30"),
            close_time=getattr(cfg, "close_time", "16:00"),
        )
        self._client_obj = None

    # --- plumbing ------------------------------------------------------------------
    def _client(self):
        if self._client_obj is None:
            if not self.api_key:
                raise RuntimeError(
                    "DatabentoAdapter needs an API key: set DATABENTO_API_KEY or cfg.api_key."
                )
            try:
                import databento as db
            except ImportError as e:  # pragma: no cover - optional dep
                raise ImportError(
                    "pip install -e '.[databento]' to use the Databento adapter"
                ) from e
            self._client_obj = db.Historical(self.api_key)
        return self._client_obj

    def _cached(self, name: str, fetch):
        path = os.path.join(self.cache_dir, f"{name}.parquet")
        if os.path.exists(path):
            return load_parquet(path)
        df = fetch()
        save_parquet(df, path)
        return df

    @staticmethod
    def _to_ny_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(idx)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return idx.tz_convert("America/New_York").tz_localize(None)

    # --- underlying: ES futures minute bars ----------------------------------------
    def minute_bars(self, start, end) -> pd.DataFrame:
        s, e = pd.Timestamp(start), pd.Timestamp(end)

        def fetch():
            data = self._client().timeseries.get_range(
                dataset=self.u_dataset,
                schema="ohlcv-1m",
                symbols=[self.es_symbol],
                stype_in=self.u_stype,
                start=s.strftime("%Y-%m-%d"),
                end=(e + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            df = data.to_df()
            df.index = self._to_ny_naive(df.index)
            df.index.name = "ts"
            # bar timestamps are bar-OPEN in DBN; our convention is bar-END
            df.index = df.index + pd.Timedelta(minutes=1)
            out = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
            out = out[~out.index.duplicated(keep="last")]
            for sp in self.splits:  # split-adjust pre-split prices
                d, r = pd.Timestamp(sp["date"]), float(sp["ratio"])
                mask = out.index < d
                out.loc[mask, ["open", "high", "low", "close"]] /= r
                out.loc[mask, "volume"] *= r
            return out

        name = f"es_ohlcv1m_{self.es_symbol.replace('.', '-')}_{s.date()}_{e.date()}"
        bars = self._cached(name, fetch)
        # restrict to RTH so realized-vol sees the same session the options close on;
        # inclusive='right' drops the bar LABELLED 09:30 (it covers the pre-open
        # minute 09:29-09:30 under our bar-end convention) and keeps 09:31..16:00
        rth = bars.between_time(self.cal.open_time, self.cal.close_time, inclusive="right")
        # a date-only end means "through that whole session", matching iv_surface's
        # inclusive trading_days(s, e) — otherwise the last day's bars silently drop
        e_eff = e.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1) if e == e.normalize() else e
        return validate_bars(rth.loc[(rth.index >= s) & (rth.index <= e_eff)])

    # --- IV surface: SPX chain -> volsolve ------------------------------------------
    def iv_surface(self, start, end) -> pd.DataFrame:
        s, e = pd.Timestamp(start), pd.Timestamp(end)

        def build():
            vix = self._vix_series()
            rows, index = [], []
            for day in self.cal.trading_days(s, e):
                chain = self._session_chain(day)
                if chain is None or chain.empty:
                    continue
                summary = volsolve.surface_summary(chain, day, self.horizons)
                if summary is None:
                    continue
                row = {k: v / ANN for k, v in summary.items()}  # annualized -> daily vol
                row["vix"] = float(vix.get(pd.Timestamp(day).normalize(), np.nan))
                rows.append(row)
                index.append(self.cal.session_close(day))
            # vix stays in annualized % (matches the synthetic adapter's convention)
            return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="ts"))

        # cache the SOLVED surface (re-solving ~500 sessions of chains costs ~15 min
        # of CPU each panel rebuild; the underlying per-day pulls are cached anyway)
        name = f"{self.root_tag}_surface_{s.date()}_{e.date()}_w{self.close_window_min}"
        return validate_iv(self._cached(name, build))

    def _session_chain(self, day) -> pd.DataFrame | None:
        """Definitions + close-window quotes for one session, cached, joined."""
        day = pd.Timestamp(day)
        tag = day.strftime("%Y%m%d")

        def fetch_defs():
            data = self._client().timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="definition",
                symbols=self.option_roots,
                stype_in="parent",
                start=day.strftime("%Y-%m-%d"),
                end=(day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            df = data.to_df()
            keep = df[["instrument_id", "raw_symbol", "expiration", "strike_price",
                       "instrument_class"]].copy()
            return keep.drop_duplicates(subset=["instrument_id"], keep="last")

        def fetch_quotes():
            close_utc = (
                self.cal.session_close(day)
                .tz_localize("America/New_York")
                .tz_convert("UTC")
            )
            start_utc = close_utc - pd.Timedelta(minutes=self.close_window_min)
            # cbbo interval records are stamped at interval END (ts_recv) and
            # get_range's end is EXCLUSIVE — extend one interval past the close so
            # the 16:00-stamped snapshot (the actual closing NBBO) is included.
            data = self._client().timeseries.get_range(
                dataset="OPRA.PILLAR",
                schema="cbbo-1m",
                symbols=self.option_roots,
                stype_in="parent",
                start=start_utc.isoformat(),
                end=(close_utc + pd.Timedelta(minutes=1)).isoformat(),
            )
            df = data.to_df().reset_index()
            # last COHERENT snapshot per instrument: drop rows missing either side
            # first, then take whole rows — GroupBy.last() would otherwise stitch a
            # stale bid to a fresh ask across snapshots. Order by ts_recv (the
            # snapshot stamp; ts_event can be NaT on snapshot rows — verified live).
            tcol = "ts_recv" if "ts_recv" in df.columns else "ts_event"
            df = df.dropna(subset=["bid_px_00", "ask_px_00"])
            df = df.sort_values(tcol).groupby("instrument_id").tail(1)
            return df[["instrument_id", "bid_px_00", "ask_px_00"]].reset_index(drop=True)

        try:
            defs = self._cached(f"{self.root_tag}_defs_{tag}", fetch_defs)
            quotes = self._cached(f"{self.root_tag}_cbbo_{tag}", fetch_quotes)
        except Exception as e:
            print(f"[chain-error] {day.date()}: {type(e).__name__}: {e}", flush=True)
            return None

        m = quotes.merge(defs, on="instrument_id", how="inner")
        m = m[m["instrument_class"].isin(["C", "P"])]
        chain = pd.DataFrame({
            "expiry": pd.to_datetime(m["expiration"]).dt.tz_localize(None).dt.normalize(),
            "strike": m["strike_price"].astype(float),
            "right": m["instrument_class"],
            "bid": m["bid_px_00"].astype(float),
            "ask": m["ask_px_00"].astype(float),
        })
        return chain[(chain["bid"] > 0) & np.isfinite(chain["ask"])]

    # --- VIX from Cboe's free CSV ---------------------------------------------------
    def _vix_series(self) -> pd.Series:
        def fetch():
            import requests

            r = requests.get(VIX_CSV_URL, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip().upper() for c in df.columns]
            df["DATE"] = pd.to_datetime(df["DATE"])
            return df.set_index("DATE")[["CLOSE"]].astype(float)

        vix = self._cached("vix_history", fetch)
        return vix["CLOSE"]
