"""Trading calendar: sessions, the 5-minute realized-vol grid, and the t0 grid.

v1 is timezone-naive and uses business days minus an optional holiday list. Real
half-days / DST / exchange holidays are a known TODO for the Massive data path
(swap in ``pandas_market_calendars`` there); the grid math below already adapts to
whatever session bounds it is given.
"""
from __future__ import annotations

from datetime import time
import pandas as pd


class Calendar:
    def __init__(
        self,
        open_time: str = "09:30",
        close_time: str = "16:00",
        holidays=None,
        bar_minutes: int = 1,
        rv_minutes: int = 5,
    ):
        self.open_time = _parse_time(open_time)
        self.close_time = _parse_time(close_time)
        self.holidays = pd.DatetimeIndex(pd.to_datetime(holidays)) if holidays else pd.DatetimeIndex([])
        self.bar_minutes = bar_minutes
        self.rv_minutes = rv_minutes

    def trading_days(self, start, end) -> pd.DatetimeIndex:
        days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
        if len(self.holidays):
            days = days.difference(self.holidays.normalize())
        return days

    def session_open(self, day) -> pd.Timestamp:
        d = pd.Timestamp(day).normalize()
        return pd.Timestamp.combine(d.date(), self.open_time)

    def session_close(self, day) -> pd.Timestamp:
        d = pd.Timestamp(day).normalize()
        return pd.Timestamp.combine(d.date(), self.close_time)

    def _grid(self, day, step_minutes: int) -> pd.DatetimeIndex:
        """Bar-END timestamps within RTH: first bar ends at open+step, last at close."""
        o, c = self.session_open(day), self.session_close(day)
        return pd.date_range(o + pd.Timedelta(minutes=step_minutes), c, freq=f"{step_minutes}min")

    def minute_grid(self, day) -> pd.DatetimeIndex:
        return self._grid(day, self.bar_minutes)

    def rv_grid(self, day) -> pd.DatetimeIndex:
        return self._grid(day, self.rv_minutes)

    def decision_times(self, start, end) -> pd.DatetimeIndex:
        """The t0 grid: one decision per trading day, at session close."""
        return pd.DatetimeIndex([self.session_close(d) for d in self.trading_days(start, end)])


def _parse_time(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))
