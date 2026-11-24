import csv
import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.rag.vectorstore import rerank_documents
from evals.gemini_judge import GeminiJudge

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

settings = get_settings()

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

GOLDEN_DATASET = (
    BASE_DIR / "golden_dataset" / "deepevals_novacore_retrieval_golden_dataset.json"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "retrieval_evaluation_results.csv"
SUMMARY_PATH = RESULTS_DIR / "retrieval_evaluation_summary.csv"

THRESHOLD = 0.7
DELAY_SECONDS = 5

# --------------------------------------------------
# 2. Initialize Gemini judge
# --------------------------------------------------

judge = GeminiJudge(model=settings.GEMINI_MODEL)

# --------------------------------------------------
# 3. Initialize evaluation metrics
# --------------------------------------------------

recall_metric = ContextualRecallMetric(
    threshold=THRESHOLD,
    model=judge,
    include_reason=True,
)

precision_metric = ContextualPrecisionMetric(
    threshold=THRESHOLD,
    model=judge,
    include_reason=True,
)

# --------------------------------------------------
# 4. Load golden dataset
# --------------------------------------------------

with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
    goldens = json.load(f)

print("Total golden queries:", len(goldens))

# --------------------------------------------------
# 5. Prepare CSV
# --------------------------------------------------

FIELDNAMES = [
    "id",
    "query",
    "contextual_recall",
    "recall_reason",
    "contextual_precision",
    "precision_reason",
]

if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

# Load previously completed results for resume support
completed = {}

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        completed[row["id"]] = row

# --------------------------------------------------
# 6. Run retrieval evaluation sequentially
# --------------------------------------------------

for index, golden in enumerate(goldens, start=1):

    test_id = golden["id"]
    query = golden["query"]

    previous = completed.get(test_id, {})

    # Skip only if both metrics were successfully evaluated
    if (
        previous.get("contextual_recall", "") != ""
        and previous.get("contextual_precision", "") != ""
    ):
        print(f"\nSkipping {test_id}: already evaluated")
        continue

    print(f"\nEvaluating {index}/{len(goldens)}")
    print("ID:", test_id)
    print("Query:", query)

    row = {
        "id": test_id,
        "query": query,
        "contextual_recall": previous.get("contextual_recall", ""),
        "recall_reason": previous.get("recall_reason", ""),
        "contextual_precision": previous.get("contextual_precision", ""),
        "precision_reason": previous.get("precision_reason", ""),
    }

    # Retrieve context
    try:
        retrieved = rerank_documents(
            query=query,
            top_n=settings.TOP_N_RANKED,
        )

        retrieval_context = [doc.page_content for doc in retrieved]

    except Exception as e:
        print("Retrieval failed:", e)
        continue

    # Create DeepEval test case
    test_case = LLMTestCase(
        input=query,
        expected_output=golden["ideal_answer"],
        retrieval_context=retrieval_context,
        actual_output="Retrieval evaluation only.",
    )

    # Contextual Recall
    if row["contextual_recall"] == "":
        try:
            recall_metric.measure(test_case)

            row["contextual_recall"] = recall_metric.score
            row["recall_reason"] = recall_metric.reason or ""

            print(
                "Contextual Recall:",
                f"{recall_metric.score:.4f}",
            )

        except Exception as e:
            print("Recall evaluation failed:", e)

        time.sleep(DELAY_SECONDS)

    # Contextual Precision
    if row["contextual_precision"] == "":
        try:
            precision_metric.measure(test_case)

            row["contextual_precision"] = precision_metric.score
            row["precision_reason"] = precision_metric.reason or ""

            print(
                "Contextual Precision:",
                f"{precision_metric.score:.4f}",
            )

        except Exception as e:
            print("Precision evaluation failed:", e)

        time.sleep(DELAY_SECONDS)

    # Save results immediately.
    existing_rows = []

    if CSV_PATH.exists():
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            existing_rows = [r for r in csv.DictReader(
                f) if r["id"] != test_id]

    existing_rows.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows)

    completed[test_id] = row

    print("Saved:", CSV_PATH)

    time.sleep(DELAY_SECONDS)

# --------------------------------------------------
# 7. Calculate average scores from saved CSV
# --------------------------------------------------

recall_scores = []
precision_scores = []

with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):

        if row["contextual_recall"] != "":
            recall_scores.append(float(row["contextual_recall"]))

        if row["contextual_precision"] != "":
            precision_scores.append(float(row["contextual_precision"]))

avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None

avg_precision = (
    sum(precision_scores) / len(precision_scores) if precision_scores else None
)

# --------------------------------------------------
# 8. Save summary CSV
# --------------------------------------------------

summary = [
    {
        "metric": "Contextual Recall",
        "average_score": avg_recall,
        "successful_evaluations": len(recall_scores),
        "total_queries": len(goldens),
    },
    {
        "metric": "Contextual Precision",
        "average_score": avg_precision,
        "successful_evaluations": len(precision_scores),
        "total_queries": len(goldens),
    },
]

with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "metric",
            "average_score",
            "successful_evaluations",
            "total_queries",
        ],
    )
    writer.writeheader()
    writer.writerows(summary)

# --------------------------------------------------
# 9. Print final report
# --------------------------------------------------

print("\n========== RETRIEVAL EVALUATION SUMMARY ==========")

print("Total queries:", len(goldens))
print("Recall evaluations:", len(recall_scores))
print("Precision evaluations:", len(precision_scores))

print(
    "Average Contextual Recall:",
    f"{avg_recall:.4f}" if avg_recall is not None else "N/A",
)

print(
    "Average Contextual Precision:",
    f"{avg_precision:.4f}" if avg_precision is not None else "N/A",
)

print("\nDetailed results:", CSV_PATH)
print("Summary results:", SUMMARY_PATH)

print("==================================================")
