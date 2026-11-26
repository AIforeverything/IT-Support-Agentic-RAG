import asyncio
from copy import deepcopy

from groq import Groq
from deepeval.models import DeepEvalBaseLLM


def make_groq_compatible_schema(schema: dict) -> dict:
    """
    Convert a Pydantic JSON schema to Groq strict JSON Schema format.

    Groq strict mode requires:
    - additionalProperties=False on every object.
    - Every property to be listed in required.
    - Properties that were optional in the original schema to accept null.
    """
    schema = deepcopy(schema)

    def make_nullable(property_schema: dict) -> dict:
        """Allow null for a property that was optional in the original schema."""
        property_schema = deepcopy(property_schema)

        if "anyOf" in property_schema:
            options = property_schema["anyOf"]
            if not any(
                isinstance(option, dict) and option.get("type") == "null"
                for option in options
            ):
                options.append({"type": "null"})

        elif "oneOf" in property_schema:
            options = property_schema.pop("oneOf")
            if not any(
                isinstance(option, dict) and option.get("type") == "null"
                for option in options
            ):
                options.append({"type": "null"})
            property_schema = {"anyOf": options}

        elif "allOf" in property_schema:
            property_schema = {
                "anyOf": [
                    property_schema,
                    {"type": "null"},
                ]
            }

        elif "$ref" in property_schema:
            property_schema = {
                "anyOf": [
                    property_schema,
                    {"type": "null"},
                ]
            }

        elif isinstance(property_schema.get("type"), str):
            property_schema["type"] = [
                property_schema["type"],
                "null",
            ]

        elif isinstance(property_schema.get("type"), list):
            if "null" not in property_schema["type"]:
                property_schema["type"].append("null")

        else:
            property_schema = {
                "anyOf": [
                    property_schema,
                    {"type": "null"},
                ]
            }

        return property_schema

    def process(node):
        """Recursively process nested objects and schema definitions."""
        if isinstance(node, list):
            for item in node:
                process(item)
            return

        if not isinstance(node, dict):
            return

        properties = node.get("properties")

        if isinstance(properties, dict):
            original_required = set(node.get("required", []))

            # Strict mode requires every property to be required.
            for key, property_schema in list(properties.items()):
                if key not in original_required:
                    property_schema = make_nullable(property_schema)

                properties[key] = property_schema

            node["required"] = list(properties.keys())
            node["additionalProperties"] = False

        # Traverse all nested schema structures, including $defs,
        # definitions, items, anyOf, oneOf, and allOf.
        for value in node.values():
            process(value)

    process(schema)
    return schema


class GroqJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq()

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt: str, schema=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert LLM evaluation judge. "
                    "Evaluate objectively and follow the requested output format."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if schema is not None:
            groq_schema = make_groq_compatible_schema(
                schema.model_json_schema()
            )

            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "deepeval_response",
                    "strict": True,
                    "schema": groq_schema,
                },
            }

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content

        if not content:
            raise ValueError("Groq returned an empty response.")

        if schema is not None:
            return schema.model_validate_json(content)

        return content

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)
