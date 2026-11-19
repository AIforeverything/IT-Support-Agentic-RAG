import json
from pathlib import Path

from langsmith import Client

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "evals" / "golden_dataset" / "novacore_golden_dataset.json"

DATASET_NAME = "novacore_golden_dataset"


def main():
    client = Client()

    # 1. Load golden dataset
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        dataset_data = json.load(file)

    examples = dataset_data["examples"]

    print(f"Golden dataset examples found: {len(examples)}")

    # 2. Check whether dataset already exists
    existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))

    if existing_datasets:
        dataset = existing_datasets[0]
        print(f"Using existing dataset: {dataset.name}")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=("NovaCore IT Support Agent golden evaluation dataset"),
        )
        print(f"Created dataset: {dataset.name}")

    # 3. Get existing examples to prevent duplicates
    existing_examples = list(client.list_examples(dataset_id=dataset.id))

    existing_test_ids = {
        example.metadata.get("test_id")
        for example in existing_examples
        if example.metadata
    }

    print(f"Examples already uploaded: {len(existing_examples)}")

    # 4. Prepare only missing examples
    langsmith_examples = []

    for item in examples:
        test_id = item.get("id")

        if test_id in existing_test_ids:
            continue

        metadata = item.get("metadata", {})

        langsmith_examples.append(
            {
                "inputs": {"question": item["inputs"]["question"]},
                "outputs": {"reference_answer": item["outputs"]["reference_answer"]},
                "metadata": {
                    **metadata,
                    "test_id": test_id,
                    "source_pages": metadata.get(
                        "source_pages", item.get("source_pages", [])
                    ),
                },
            }
        )

    # 5. Upload missing examples
    if langsmith_examples:
        client.create_examples(
            dataset_id=dataset.id,
            examples=langsmith_examples,
        )

        print(f"Uploaded {len(langsmith_examples)} new examples.")
    else:
        print("No new examples to upload.")

    # 6. Verify final dataset count
    final_examples = list(client.list_examples(dataset_id=dataset.id))

    print("\n========== DATASET SUMMARY ==========")
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset ID: {dataset.id}")
    print(f"Expected examples: {len(examples)}")
    print(f"Uploaded examples: {len(final_examples)}")
    print("=====================================")


if __name__ == "__main__":
    main()
