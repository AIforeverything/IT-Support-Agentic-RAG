import yaml
from pathlib import Path
from app.rag.state import AgentState
from functools import lru_cache

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "prompts.yaml"

@lru_cache
def load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
   


def get_prompt(prompt_name: str):
    data = load_prompts()
    prompt_data = data["prompts"].get(prompt_name)
    if prompt_data is None:
        raise ValueError(f"Prompt {prompt_name} is not found.")
    return prompt_data["template"]


def get_prompt_version(prompt_name: str):
    data = load_prompts()
    return data["prompts"][prompt_name]["version"]


# def loading_prompt_formatting_using_langchain(prompt_name: str, state: AgentState):
#     # step-1: loading prompt template
#     prompt_template = get_prompt(prompt_name)
#     # step-2: creating langchain prompt
#     langchain_prompt = PromptTemplate.from_template(template=prompt_template)
#     # step-3: insert user question
#     langchain_prompt_user_query = langchain_prompt.format(query=state["query"])
#     return langchain_prompt_user_query


if __name__ == "__main__":
    load_prompts()
    get_prompt_version("")
