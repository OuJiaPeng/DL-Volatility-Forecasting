"""volforecast — leakage-safe SPX/0DTE realized-volatility forecasting.

The cardinal rule of this package: every panel row carries one decision timestamp
``t0``; features use only data with ``ts <= t0`` and targets use only data with
``ts > t0``. There is exactly ONE alignment path (``volforecast.panel.PanelBuilder``).
See ``volforecast.timeutil`` for the point-in-time guards that enforce it.
"""
__version__ = "0.1.0"
