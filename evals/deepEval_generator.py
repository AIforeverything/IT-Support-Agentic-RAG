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
import time
from pathlib import Path
from app.core.config import get_settings
from evals.harness import load_golden, summarize_by_metric, print_summary
from evals.llm_generator import generate
from evals.llm_judge import groq_judge, gpt_judge, gemini_judge

from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval import evaluate
THRESHOLD = 0.7
DELAY_SECONDS = 1
settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR/"evals"/"golden_dataset" / \
    "novacore_faithfulness_answer_relevancy_golden_dataset_v2.json"


def run():
    # Step-1: Loading faithfulness golden dataset
    goldens = load_golden(FILE_PATH)
    # print(dataset)
    answers = []
    # Step-2: RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    i = 1
    for g in goldens:
        query = g["query"]
        context = g["ideal_context"]  # known good context
        # we get the generated output
        answer = generate(query=query, context=context)
        time.sleep(DELAY_SECONDS)
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,      # generated answer we are judging
                retrieval_context=context,  # faithfulness checks the answer aginst this context
            )
        )
        print(f"answer-{i}: ", answer[:50])
        i = i+1
        answers.append(answer)

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=gpt_judge,
            include_reason=True  # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=gpt_judge,
            include_reason=True  # prints WHY each score --- shows which claims were unsupported
        )
    ]

    # Evaluate : runs the metrics for every testcase and prints report
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result=result)


if __name__ == "__main__":
    print_summary("generator_results", run())
