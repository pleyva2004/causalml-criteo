"""Command-line entry point: ``uv run causalml <stage> [--mode dev|full]``.

Each stage is a ``run(mode) -> dict`` function that writes its own metrics/tables/figures and
returns a small dict of headline numbers. Stages are imported lazily so a broken stage cannot
block the others.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

from src.utils import get_logger, set_run_mode, set_seed, timer

log = get_logger("causalml")

# stage name -> (module, function). Order is the execution order of ``all``.
STAGES: dict[str, tuple[str, str]] = {
    "data": ("src.data.load", "build_processed"),
    "eda": ("src.data.preprocess", "run_eda"),
    "stats": ("src.causal.ate", "run_stats"),
    "predict": ("src.models.evaluation", "run_predictive"),
    "causal": ("src.causal.ate", "run_causal_ate"),
    "cate": ("src.causal.learners", "run_cate"),
    "targeting": ("src.optimization.targeting", "run_targeting"),
    "robustness": ("src.robustness", "run_robustness"),
    "scaling": ("src.scaling", "run_scaling"),
    "report": ("src.report", "run_report"),
}


def run_stage(stage: str, mode: str) -> dict:
    module, fn = STAGES[stage]
    func = getattr(importlib.import_module(module), fn)
    set_seed()
    with timer(f"stage {stage} ({mode})", log):
        return func() if stage == "data" else func(mode=mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalml", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--mode", choices=["dev", "full"], default="dev",
                        help="dev: ~5%% stratified subsample, outputs to results/dev/; full: all 14M rows")
    args = parser.parse_args(argv)
    set_run_mode(args.mode)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in stages:
        out = run_stage(stage, args.mode)
        if out:
            log.info("%s headline: %s", stage, json.dumps(out, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
