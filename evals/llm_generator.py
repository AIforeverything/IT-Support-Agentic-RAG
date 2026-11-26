from app.core.prompt_loader import get_prompt
import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.llm import llm
from app.core.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

settings = get_settings()

# loading the prompt from prompts.yaml

prompt_name = "faithfullness_prompt"
prompt_yaml = get_prompt(prompt_name)
prompt = ChatPromptTemplate.from_template(prompt_yaml)

llm_gemini = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=settings.TEMPERATURE,
    google_api_key=settings.GEMINI_API_KEY,
)

llm_groq = ChatGroq(
    model=settings.GROQ_MODEL,
    temperature=settings.TEMPERATURE,
    api_key=settings.GROQ_API_KEY
)
chain = prompt | llm_groq | StrOutputParser()


def generate(query: str, context: list[str]) -> str:
    """Generate a grounded answer from the query and context chunks."""
    context_text = "\n\n".join(context)
    return chain.invoke({"question": query, "context": context_text})


if __name__ == "__main__":
    ctx = [
        "Online eval means evaluating your system on live production traffic "
        "after deployment. It works without an answer key, unlike offline eval."
    ]

    # non-streaming
    print(generate("what is online eval?", ctx))
