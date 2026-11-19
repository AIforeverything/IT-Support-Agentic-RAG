
from dotenv import load_dotenv
from langsmith import Client
import os

load_dotenv(override=True)

api_key = os.getenv("LANGSMITH_API_KEY")

print("API key loaded:", bool(api_key))
print("API key prefix:", api_key[:8] if api_key else None)
print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))

try:
    client = Client(api_key=api_key)

    datasets = list(client.list_datasets())

    print("\nLangSmith authentication successful!")
    print("Available datasets:")

    for dataset in datasets:
        print("-", dataset.name, dataset.id)

except Exception as e:
    print("\nLangSmith connection failed:")
    print(type(e).__name__, str(e))
