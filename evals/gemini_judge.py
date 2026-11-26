import asyncio
import json
from app.core.config import get_settings

from google import genai
from deepeval.models import DeepEvalBaseLLM


class GeminiJudge(DeepEvalBaseLLM):

    def __init__(self, model: str):
        self.model_name = model
        self.client = genai.Client()

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                settings=get_settings(),
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )

            if response.parsed is not None:
                if isinstance(response.parsed, schema):
                    return response.parsed
                return schema.model_validate(response.parsed)

            # Fallback if the SDK returns JSON text but no parsed object
            return schema.model_validate_json(response.text)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)
