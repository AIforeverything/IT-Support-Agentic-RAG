
from langsmith import evaluate

from app.rag.workflow import ask

DATASET_NAME = "novacore_golden_dataset"


def target(inputs: dict) -> dict:
    """Execute the existing LangGraph agent."""

    result = ask(inputs["question"])

    kb_docs = result.get("kb_docs") or []

    return {
        "answer": result.get("answer", ""),
        "source_used": result.get("source_used", ""),
        "citations": result.get("citations") or [],
        "kb_contexts": [
            doc.page_content for doc in kb_docs
        ],
        "web_results": result.get("web_results", ""),
        "trace": result.get("trace") or [],
        "kb_grade": result.get("kb_grade", ""),
        "web_grade": result.get("web_grade", ""),
    }


def answer_not_empty(run, example) -> dict:
    answer = (run.outputs or {}).get("answer", "")

    passed = (
        isinstance(answer, str)
        and bool(answer.strip())
    )

    return {
        "key": "answer_not_empty",
        "score": 1 if passed else 0,
        "comment": (
            "Answer generated"
            if passed else "Empty answer"
        ),
    }


def citation_check(run, example) -> dict:
    metadata = example.metadata or {}
    outputs = run.outputs or {}

    if not metadata.get("answerable_from_handbook", False):
        return {
            "key": "citation_check",
            "score": 1,
            "comment": "Not applicable",
        }

    citations = outputs.get("citations") or []
    passed = len(citations) > 0

    return {
        "key": "citation_check",
        "score": 1 if passed else 0,
        "comment": (
            "Citations present"
            if passed else "Expected handbook citations"
        ),
    }


def source_check(run, example) -> dict:
    metadata = example.metadata or {}
    outputs = run.outputs or {}

    if not metadata.get("answerable_from_handbook", False):
        return {
            "key": "source_check",
            "score": 1,
            "comment": "Not applicable",
        }

    source = outputs.get("source_used", "")
    passed = source == "private_kb"

    return {
        "key": "source_check",
        "score": 1 if passed else 0,
        "comment": f"Actual source: {source}",
    }


if __name__ == "__main__":

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[
            answer_not_empty,
            citation_check,
            source_check,
        ],
        experiment_prefix="novacore-baseline",
        description="Baseline evaluation of NovaCore Agent",
        max_concurrency=1,
    )

    print(results)
