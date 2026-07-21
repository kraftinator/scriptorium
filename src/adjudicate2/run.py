#!/usr/bin/env python3
"""CLI entry point for the adjudicate2 experimentation tree.

    python src/adjudicate2/run.py --corpus corpora/us_census_1850 \
        --reel populationschedu0604unix --frame 23 --strategy v4

Strategies are auto-discovered from strategies/. Add a new one by dropping
strategies/vNEW_yourthing.py (see strategies/__init__.py for the contract) —
it shows up in --strategy choices automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make `import adjudicate2.*` work when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adjudicate2.orchestrator import run_page
from adjudicate2.strategies import STRATEGIES


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Adjudicate one page with a chosen strategy (adjudicate2 tree).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="registered strategies:\n" + "\n".join(
            f"  {name:12s} {info['module']:20s} {info['doc']}"
            for name, info in sorted(STRATEGIES.items())
        ),
    )
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--reel", required=True)
    ap.add_argument("--frame", required=True, type=int)
    ap.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    ap.add_argument("--lines", nargs="*", type=int, default=None,
                    help="only adjudicate these line numbers (writes .partial.json)")
    args = ap.parse_args()

    info = STRATEGIES[args.strategy]
    run_page(
        corpus=args.corpus.resolve(),
        reel=args.reel,
        frame=args.frame,
        strategy_name=args.strategy,
        crop_fn=info["crop_fn"],
        adjudicate_fn=info["adjudicate_fn"],
        lines=args.lines,
    )


if __name__ == "__main__":
    main()
