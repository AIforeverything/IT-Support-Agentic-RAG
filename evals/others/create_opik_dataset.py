import json
# import os
from pathlib import Path
from app.core.config import Settings
import opik
# from dotenv import load_dotenv


# Load .env from the project root.
# load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = (
    BASE_DIR
    / "evals"
    / "golden_dataset"
    / "novacore_golden_dataset.json"
)

settings=Settings()
PROJECT_NAME = settings.OPIK_PROJECT_NAME
DESCRIPTION = "NovaCore IT Support Agent golden evaluation dataset"
OPIK_API_KEY=settings.OPIK_API_KEY
OPIK_WORKSPACE=settings.OPIK_WORKSPACE

def load_golden_examples():
    """Load and validate the project's actual golden-dataset structure."""

    if not JSON_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset JSON was not found: {JSON_PATH}"
        )

    with JSON_PATH.open("r", encoding="utf-8") as file:
        dataset_json = json.load(file)

    examples = dataset_json.get("examples")
    if not isinstance(examples, list):
        raise ValueError(
            "Invalid golden dataset: expected an 'examples' list."
        )

    expected_total = dataset_json.get("total_examples")
    if expected_total is not None and expected_total != len(examples):
        raise ValueError(
            f"Dataset declares {expected_total} examples, "
            f"but contains {len(examples)}."
        )

    items = []
    seen_ids = set()

    for index, item in enumerate(examples, start=1):
        try:
            test_id = item["id"]
            question = item["inputs"]["question"]
            reference_answer = item["outputs"]["reference_answer"]
            metadata = item.get("metadata", {})
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Invalid example at position {index}: {exc}"
            ) from exc

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Example {test_id!r} has an empty or invalid question."
            )

        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(
                f"Example {test_id!r} has an empty or invalid reference answer."
            )

        if test_id in seen_ids:
            raise ValueError(f"Duplicate test id in JSON: {test_id}")
        seen_ids.add(test_id)

        # Keep fields flat so the evaluation task can access
        # dataset_item["question"], dataset_item["reference_answer"], etc.
        # source_pages is preserved as supplied (a string in this JSON).
        items.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "test_id": test_id,
                "category": metadata.get("category", ""),
                "answerable_from_handbook": metadata.get(
                    "answerable_from_handbook", False
                ),
                "expected_behavior": metadata.get(
                    "expected_behavior", ""
                ),
                "source_pages": metadata.get("source_pages", ""),
            }
        )

    return dataset_json, items


def main():
    # Opik reads OPIK_API_KEY and OPIK_WORKSPACE from the environment
    # after load_dotenv() has loaded the project's .env file.
    if not OPIK_API_KEY:
        raise EnvironmentError(
            "OPIK_API_KEY is missing. Add it to your project-root .env file."
        )

    if not OPIK_WORKSPACE:
        raise EnvironmentError(
            "OPIK_WORKSPACE is missing. Add it to your project-root .env file."
        )

    dataset_json, items = load_golden_examples()

    # Use the dataset name declared inside the uploaded JSON:
    # novacore-it-support-golden-dataset.
    dataset_name = dataset_json.get(
        "dataset_name",
        "novacore_golden_dataset",
    )

    print("API key loaded:", bool(OPIK_API_KEY))
    print("Workspace loaded:", bool(OPIK_WORKSPACE))    

    client = opik.Opik(
        api_key= OPIK_API_KEY,
        workspace=OPIK_WORKSPACE,
        project_name=PROJECT_NAME,
    )

    dataset = client.get_or_create_dataset(
        name=dataset_name,
        description=DESCRIPTION,
        project_name=PROJECT_NAME,
    )

    # Opik's Python SDK deduplicates identical items by default.
    # Re-running with unchanged data should not create duplicates.
    dataset.insert(
        items,
        deduplication=True,
        num_threads=1,
    )

    print("\n========== OPIK DATASET IMPORT ==========")
    print(f"Dataset name       : {dataset.name}")
    print(f"Project            : {PROJECT_NAME}")
    print(f"JSON source        : {JSON_PATH}")
    print(f"Examples in JSON   : {len(items)}")
    print("Deduplication      : enabled")
    print("========================================")
    print("Dataset import completed.")
    print(
        "Open Opik -> Evaluation -> Datasets and select "
        f"'{dataset.name}' to verify the examples."
    )


if __name__ == "__main__":
    main()
