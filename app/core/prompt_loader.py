import yaml
from pathlib import Path
from functools import lru_cache

PROMPT_FILE = Path(__file__).resolve().parent.parent / \
    "prompts" / "prompts_against_promptInjection.yaml"


@lru_cache
def load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_prompt(prompt_name: str, version: str):
    data = load_prompts()
    prompt_data = data["prompts"][prompt_name].get(version)
    if prompt_data is None:
        raise ValueError(f"Prompt {prompt_name} is not found.")
    return prompt_data["template"]


if __name__ == "__main__":
    print(get_prompt("decision_router_prompt", "version_1.1.0"))
