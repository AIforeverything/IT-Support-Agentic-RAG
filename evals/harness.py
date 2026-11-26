# evals/harness.py
"""
Shared harness for the eval suite.

The eval files each run their own DeepEval evaluation and return results; this
module holds the two pieces they all share:

  load_goldens(path)          -- read a golden JSON file
  summarize_by_metric(result) -- turn a DeepEval EvaluationResult into a
                                 PER-METRIC summary (pass rate + score stats)

Why per-metric and not pooled: a single eval file often runs several metrics at
once (correctness + completeness + style; the RAG triad; faithfulness +
relevancy). Pooling them into one number would let a regression in ONE metric
hide behind the others. summarize_by_metric keeps each metric separate, which is
exactly what the regression suite compares against the baseline.

The extractor is defensive across DeepEval versions: test results may live on
`.test_results` (or be a bare list), and per-test metrics on `.metrics_data`
(newer) or `.metrics` (older). It keys pass/fail off each metric's own `success`
flag, so it stays correct whether higher or lower score is "good".
"""
import json
from datetime import datetime
from pathlib import Path


def load_golden(path):
    with open(path) as f:
        return json.load(f)


def summarize_by_metric(result):
    """DeepEval EvaluationResult -> {metric_name: {n, pass_rate, avg/min/max score}}."""
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    # metric_name -> {"scores": [...], "passed": int, "total": int}
    buckets = {}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(
            tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(
                name, {"scores": [], "passed": 0, "total": 0})
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title, summary, output_dir="evals/results"):
    """Print and save the per-metric evaluation summary as JSON."""

    print("\n" + "=" * 60)
    print(f"{title} (per-metric summary)")
    print("=" * 60)

    for name, s in summary.items():
        # nan != nan
        avg = (
            f"{s['avg_score']:.2f}"
            if s["avg_score"] == s["avg_score"]
            else "nan"
        )

        print(
            f"  {name:<26} "
            f"pass_rate={s['pass_rate']:5.0f}%  "
            f"avg={avg}  n={s['n']}"
        )

    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare JSON data
    results = {
        "title": title,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": summary
    }

    # Save JSON file
    filename = (
        title.lower().replace(" ", "_").replace(":", "")
        + "_results.json"
    )

    file_path = output_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False,
            allow_nan=False
        )

    print(f"\nResults saved to: {file_path}")
