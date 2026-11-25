"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""
import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.services import llm
from evals.gemini_judge import GeminiJudge

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import evaluate

# loading the prompt from prompts.yaml
from app.core.prompt_loader import get_prompt
prompt_name = "faithfullness_prompt"
prompt_yaml = get_prompt(prompt_name)

# loading the golden dataset
BASE_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET = (BASE_DIR / "golden_dataset" /
                  "novacore_faithfulness_answer_relevancy_golden_dataset_v2.json")
with open(GOLDEN_DATASET, 'r', encoding='utf-8') as f:
    goldens = json.load(f)

llm = llm()
settings = get_settings()

# faithfulness-first prompt: ground every claim in the context, abstain if unsure
prompt = ChatPromptTemplate.from_template(
    prompt_yaml
)

chain = prompt | llm | StrOutputParser()


def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


def generate_stream(query: str, context: list[str]):
    context_text = "\n\n".join(context)
    for chunk in chain.stream({query: str, context: list[str]}):
        if chunk:
            yield chunk


# quick manual test: python src/generator.py
if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]

    # non-streaming
    print(generate("what is online eval?", ctx))

    # streaming (prints tokens as they arrive)
    print("\n--- streaming ---")
    for piece in generate_stream("what is online eval?", ctx):
        print(piece, end="", flush=True)
    print()
