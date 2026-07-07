"""CLI: build the panel, run baselines, print/save the comparison table."""
from __future__ import annotations

import argparse
import os

from .config import get_config
from .data import get_adapter
from .panel import PanelBuilder
from .splits import make_splits
from .baselines import build_baselines
from .eval import compare
from .data.cache import save_parquet, load_parquet

PANEL_PATH = "artifacts/panel.parquet"  # overridden per-config via cfg.name
METRICS_PATH = "artifacts/metrics.csv"


def _paths(cfg):
    name = getattr(cfg, "name", "panel")
    return f"artifacts/{name}.parquet", f"artifacts/{name}_cube.npy"


def cmd_panel(cfg) -> None:
    import numpy as np

    from .calendar import Calendar
    from .datasets import build_intraday_cube

    adapter = get_adapter(cfg)
    builder = PanelBuilder(cfg, adapter)
    panel = builder.build()
    panel_path, cube_path = _paths(cfg)
    save_parquet(panel, panel_path)
    print(f"panel: {panel.shape[0]} rows x {panel.shape[1]} cols -> {panel_path}")

    # companion raw-intraday cube (E3 input arm), aligned row-for-row with the panel
    bars = adapter.minute_bars(cfg.start, cfg.end)
    cube = build_intraday_cube(bars, builder.cal, panel.index)
    np.save(cube_path, cube)
    print(f"intraday cube: {cube.shape} -> {cube_path}")


def cmd_eval(cfg, with_model: bool = False) -> None:
    panel = load_parquet(_paths(cfg)[0])
    split = make_splits(panel, cfg)
    forecasters = build_baselines(cfg.horizons)
    if with_model:
        try:
            from .models.hybrid import HybridResidualForecaster
        except ImportError as e:
            raise SystemExit(f"--model needs torch (pip install -e '.[model]'): {e}")
        hybrid = HybridResidualForecaster(cfg.horizons, cfg.model).bind_split(split)
        forecasters.append(hybrid)
    table = compare(panel, split, forecasters, cfg.horizons)
    os.makedirs("artifacts", exist_ok=True)
    table.to_csv(METRICS_PATH)
    print(f"\ntest origins: {len(split.test)}  (train {len(split.train)}, val {len(split.val)})")
    print(table.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nsaved -> {METRICS_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(prog="vf", description="volforecast pipeline")
    p.add_argument("command", choices=["panel", "baselines", "eval", "all"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", action="store_true", help="include the v2 hybrid model (torch)")
    args = p.parse_args()
    cfg = get_config(args.config)
    if args.command in ("panel", "all"):
        cmd_panel(cfg)
    if args.command in ("baselines", "eval", "all"):
        cmd_eval(cfg, with_model=args.model)


if __name__ == "__main__":
    main()
