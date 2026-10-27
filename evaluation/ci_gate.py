"""CI quality gate (Feature 8.2).

Runs the Ragas-style benchmark through the offline-deterministic pipeline and
fails (exit 1) if the average composite score falls below a floor, so quality
regressions break the build.

Usage:
    python -m evaluation.ci_gate                 # default floor
    python -m evaluation.ci_gate --min 60        # custom floor

Runs fully offline: with no LLM backend configured the pipeline uses the
deterministic offline fallback, so this is reproducible in CI.
"""
from __future__ import annotations

import argparse
import sys

from evaluation.benchmark import run_benchmark

DEFAULT_MIN_COMPOSITE = 50.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=DEFAULT_MIN_COMPOSITE,
                    help="Minimum acceptable avg_composite_score (0-100).")
    args = ap.parse_args()

    res = run_benchmark()
    score = float(res.get("avg_composite_score", 0.0))
    samples = res.get("total_samples", 0)

    print(f"[ci_gate] samples={samples} avg_composite_score={score} (floor={args.min})")
    print(f"[ci_gate] faithfulness={res.get('avg_faithfulness')} "
          f"answer_relevance={res.get('avg_answer_relevance')} "
          f"context_recall={res.get('avg_context_recall')}")

    if score < args.min:
        print(f"[ci_gate] FAIL: avg_composite_score {score} < floor {args.min}")
        return 1
    print("[ci_gate] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
