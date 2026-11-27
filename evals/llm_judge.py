from deepeval.models import OpenAIModel
from deepeval.models import GeminiModel
from deepeval.models import GrokModel

from app.core.config import get_settings
settings = get_settings()

gpt_judge = OpenAIModel(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    temperature=settings.TEMPERATURE
)

gemini_judge = GeminiModel(
    model=settings.GEMINI_MODEL,
    api_key=settings.ADMIN_API_KEY,
    temperature=settings.TEMPERATURE
)

groq_judge = GrokModel(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=settings.TEMPERATURE
)
