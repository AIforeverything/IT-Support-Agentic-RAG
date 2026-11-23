# # import json
# # from pathlib import Path

# # from app.core.config import get_settings
# # from app.rag.vectorstore import rerank_documents
# # from app.services.llm import llm

# # from deepeval import evaluate
# # from deepeval.test_case import LLMTestCase
# # from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

# # settings = get_settings()
# # BASE_DIR = Path(__file__).resolve().parent
# # GOLDEN_DATASET = (
# #     BASE_DIR / "golden_dataset" / "deepevals_novacore_retrieval_golden_dataset.json"
# # )
# # THRESHOLD = 0.7
# # model = settings.GEMINI_MODEL

# # # 1. load the golden dataset
# # with open(GOLDEN_DATASET, "r") as f:
# #     goldens = json.load(f)

# # # 2. Run the retriever to extract the retrieved context
# # test_cases = []

# # for g in goldens:
# #     retrieved = rerank_documents(query=g["query"], top_n=settings.TOP_N_RANKED)
# #     retrieval_context = [doc.page_content for doc in retrieved]

# #     test_cases.append(
# #         LLMTestCase(
# #             input=g["query"],
# #             expected_output=g["ideal_answer"],
# #             retrieval_context=retrieval_context,
# #             actual_output="generator will be evaluated later.",
# #         )
# #     )

# # print("test_cases: ", len(test_cases))
# # # 3. Metrics: recall(Did we miss?); precision (ranked well?)
# # metrics = [
# #     ContextualRecallMetric(threshold=THRESHOLD, model=model, include_reason=True),
# #     ContextualPrecisionMetric(threshold=THRESHOLD, model=model, include_reason=True),
# # ]

# # # 4. Evaluate -- every metric on every case, batched+parallel, with a printed report
# # evaluate(
# #     test_cases=test_cases,
# #     metrics=metrics,
# #     hyperparameters={
# #         "retriever": settings.RERANKING_MODEL_NAME,
# #         "embedding_model": settings.EMBEDDING_MODEL_NAME,
# #         "chunk_size": settings.CHUNK_SIZE,
# #         "chunk_overlap": settings.CHUNK_OVERLAP,
# #         "top_n": settings.TOP_N_RANKED,
# #         "judge_model": "groq",
# #         "golden_dataset": str(GOLDEN_DATASET),
# #     },
# # )


# import json
# import csv
# from pathlib import Path

# from app.core.config import get_settings
# from app.rag.vectorstore import rerank_documents
# from evals.gemini_judge import GeminiJudge

# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     ContextualRecallMetric,
#     ContextualPrecisionMetric,
# )

# settings = get_settings()

# BASE_DIR = Path(__file__).resolve().parent
# GOLDEN_DATASET = (
#     BASE_DIR / "golden_dataset" / "deepevals_novacore_retrieval_golden_dataset.json"
# )

# THRESHOLD = 0.7

# # 1. Initialize Gemini as the evaluation judge
# judge = GeminiJudge(model=settings.GEMINI_MODEL)

# # 2. Load the golden dataset
# with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
#     goldens = json.load(f)

# # 3. Retrieve context for every golden query
# test_cases = []

# for g in goldens:
#     retrieved = rerank_documents(
#         query=g["query"],
#         top_n=settings.TOP_N_RANKED,
#     )

#     retrieval_context = [doc.page_content for doc in retrieved]

#     test_cases.append(
#         LLMTestCase(
#             input=g["query"],
#             expected_output=g["ideal_answer"],
#             retrieval_context=retrieval_context,
#             actual_output="Generator evaluation will be performed later.",
#         )
#     )

# print("Total test cases:", len(test_cases))

# # 4. Configure retrieval evaluation metrics
# metrics = [
#     ContextualRecallMetric(
#         threshold=THRESHOLD,
#         model=judge,
#         include_reason=True,
#     ),
#     ContextualPrecisionMetric(
#         threshold=THRESHOLD,
#         model=judge,
#         include_reason=True,
#     ),
# ]

# # 5. Run evaluation parallel
# # evaluate(
# #     test_cases=test_cases,
# #     metrics=metrics,
# #     hyperparameters={
# #         "retriever": settings.RERANKING_MODEL_NAME,
# #         "embedding_model": settings.EMBEDDING_MODEL_NAME,
# #         "chunk_size": settings.CHUNK_SIZE,
# #         "chunk_overlap": settings.CHUNK_OVERLAP,
# #         "top_n": settings.TOP_N_RANKED,
# #         "judge_model": judge.get_model_name(),
# #         "golden_dataset": str(GOLDEN_DATASET),
# #     },
# # )

# import time

# # Run each test case sequentially
# for index, test_case in enumerate(test_cases, start=1):

#     print(f"\nEvaluating query {index}/{len(test_cases)}")

#     for metric in metrics:
#         try:
#             metric.measure(test_case)

#             print(f"{metric.__class__.__name__}: " f"{metric.score:.3f}")

#             if metric.reason:
#                 print("Reason:", metric.reason)

#         except Exception as e:
#             print(f"Error in {metric.__class__.__name__}: {e}")

#         # Pause between metric API calls
#         time.sleep(3)

#     # Pause before next query
#     time.sleep(3)


# # CSV output location
# RESULTS_DIR = BASE_DIR / "results"
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CSV_PATH = RESULTS_DIR / "retrieval_evaluation_results.csv"

# # Create CSV with headers if it doesn't exist
# if not CSV_PATH.exists():
#     with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "id",
#                 "query",
#                 "contextual_recall",
#                 "recall_reason",
#                 "contextual_precision",
#                 "precision_reason",
#             ],
#         )
#         writer.writeheader()
# recall_metric = ContextualRecallMetric(
#     threshold=THRESHOLD,
#     model=judge,
#     include_reason=True,
# )

# precision_metric = ContextualPrecisionMetric(
#     threshold=THRESHOLD,
#     model=judge,
#     include_reason=True,
# )

# recall_scores = []
# precision_scores = []

# # Sequential evaluation
# for index, (golden, test_case) in enumerate(zip(goldens, test_cases), start=1):
#     print(f"\nEvaluating {index}/{len(test_cases)}")
#     print("Query:", golden["query"])

#     row = {
#         "id": golden["id"],
#         "query": golden["query"],
#         "contextual_recall": "",
#         "recall_reason": "",
#         "contextual_precision": "",
#         "precision_reason": "",
#     }

#     # Contextual Recall
#     try:
#         recall_metric.measure(test_case)

#         recall_score = recall_metric.score
#         row["contextual_recall"] = recall_score
#         row["recall_reason"] = recall_metric.reason or ""
#         recall_scores.append(recall_score)

#         print("Recall:", recall_score)

#     except Exception as e:
#         print("Recall evaluation failed:", e)

#     time.sleep(5)

#     # Contextual Precision
#     try:
#         precision_metric.measure(test_case)

#         precision_score = precision_metric.score
#         row["contextual_precision"] = precision_score
#         row["precision_reason"] = precision_metric.reason or ""
#         precision_scores.append(precision_score)

#         print("Precision:", precision_score)

#     except Exception as e:
#         print("Precision evaluation failed:", e)

#     # Save after every query
#     with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=row.keys())
#         writer.writerow(row)

#     print("Saved:", CSV_PATH)

#     time.sleep(5)

# # Calculate average scores
# avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

# avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0

# print("\n========== RETRIEVAL EVALUATION SUMMARY ==========")
# print(f"Successful Recall evaluations: {len(recall_scores)}")
# print(f"Successful Precision evaluations: {len(precision_scores)}")
# print(f"Average Contextual Recall: {avg_recall:.4f}")
# print(f"Average Contextual Precision: {avg_precision:.4f}")
# print("Results CSV:", CSV_PATH)
# print("==================================================")


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
DELAY_SECONDS = 10

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
    # Replace any previous incomplete row for this ID.
    existing_rows = []

    if CSV_PATH.exists():
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            existing_rows = [r for r in csv.DictReader(f) if r["id"] != test_id]

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
