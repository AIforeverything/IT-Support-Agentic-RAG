from app.core.config import Settings
import opik

settings= Settings()

client= opik.Opik(
    project_name=settings.OPIK_PROJECT_NAME,
    workspace=settings.OPIK_WORKSPACE,
    api_key=settings.OPIK_API_KEY
)

print("Opik client is initiated successfully!")

test_datasets= client.get_datasets(max_results=2)

print("Available Datasets: ")
for dataset in test_datasets:
    print(dataset.name)