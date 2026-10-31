"""CLI: run the panel grid and the checker cells against local Ollama.

  python3 run_panels.py --pilot 8      # 8 cases, smoke test -> results/pilot/
  python3 run_panels.py --full         # all cases -> results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from panels import runner

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="number of cases for a smoke test")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--grid-only", action="store_true")
    ap.add_argument("--checkers-only", action="store_true")
    args = ap.parse_args()

    cases = runner.load_battery()
    if args.pilot:
        out_dir = RESULTS / "pilot"
        out_dir.mkdir(parents=True, exist_ok=True)
        runner.write_manifest(out_dir / "model_manifest.json")
        runner.run_grid(cases, out_dir / "grid.jsonl", limit=args.pilot)
        runner.run_checkers(cases, out_dir / "grid.jsonl", out_dir / "checkers.jsonl",
                            limit=args.pilot)
        print(f"pilot complete -> {out_dir}")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    runner.write_manifest(RESULTS / "model_manifest.json")
    print(f"{len(cases)} cases; grid records = {len(cases)*12}; "
          f"checker records = {len(cases)*4}")
    if not args.checkers_only:
        runner.run_grid(cases, RESULTS / "grid.jsonl")
    if not args.grid_only:
        runner.run_checkers(cases, RESULTS / "grid.jsonl", RESULTS / "checkers.jsonl")
    print("done")


if __name__ == "__main__":
    main()
