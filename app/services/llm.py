from app.core.logging import configure_logging
from app.core.config import get_settings
from langchain_groq import ChatGroq

# logger = configure_logging()
settings = get_settings()

_llm = None


# initializing groq LLM
def llm():
    global _llm
    if _llm is None:
        if settings.GROQ_API_KEY is None:
            raise RuntimeError("Groq API Key is missing.")
        _llm = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.TEMPERATURE,
        )
    return _llm


# import litellm

# response = litellm.completion(
#     model="gemini/gemini-2.5-flash",
#     messages=[
#         {"role": "user", "content": "Evaluate this retrieved context."}
#     ],
#     api_key="YOUR_GEMINI_API_KEY"
# )

# print(response.choices[0].message.content)

if __name__ == "__main__":
    llm()
