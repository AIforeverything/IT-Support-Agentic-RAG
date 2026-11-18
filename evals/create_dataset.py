import json
from pathlib import Path

from langsmith import Client

BASE_DIR = Path(__file__).resolve().parent

JSON_PATH = BASE_DIR/"evals"/"golden_dataset"/"novacore_golden_dataset.json"

DATASET_NAME = "novacore_golden_dataset"


def main():
    client = Client()

    with open(JSON_PATH, 'r', encoding='utf-8') as file:
        dataset_data = json.load(file)

    examples = dataset_data["examples"]

    # reuse the dataset if already exists
    existing = client.list_datasets(dataset_name=DATASET_NAME)

    if existing:
        print(f"Dataset already exists: {DATASET_NAME}")
        print("No examples were uploaded again.")
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME, description="NovaCore IT Support Agent golden evaluation dataset")

    langsmith_examples = []

    for item in examples:
        langsmith_examples.append({
            "inputs": {
                "question": item["inputs"]["question"]
            },
            "outputs": {
                "reference_answer": item["outputs"]["reference_answer"]
            },
            "metadata": {
                **item.get("metadata", {}),
                "test_id": item.get("id")
                "category": "company",
                "answerable_from_handbook": true,
                "expected_behavior": "answer_from_handbook",
                "source_pages": "1"
            }
        })
