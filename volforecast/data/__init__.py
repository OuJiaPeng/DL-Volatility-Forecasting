"""Vendor-agnostic market-data adapters."""
from .adapter import MarketDataAdapter, get_adapter

__all__ = ["MarketDataAdapter", "get_adapter"]
