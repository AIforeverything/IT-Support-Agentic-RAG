from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class DecisionRoute(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use 'kb' when we need to answer using Agentic RAG; Use 'direct' for simple questions where 'kb' is not required."
    )


class GradingAnswer(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="grade the answer as 'good' if the evidence is enough to answer the query, otherwise return 'weak'."
    )


class AgentState(TypedDict):
    query: str
    current_query: str

    kb_docs: List[Document]
    web_results: str

    kb_grade: str
    web_grade: str

    answer: str
    source_used: str
    retry_count: int
    trace: str
    citations: List[str]
