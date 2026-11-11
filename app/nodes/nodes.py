# import json
from typing import Literal
from app.services.llm import llm
from app.tools.tools import web_search_tool
from app.rag.state import AgentState, DecisionRoute, GradingAnswer
from app.rag.vectorstore import rerank_documents
from app.core.prompt_loader import (
    # loading_prompt_formatting_using_langchain,
    get_prompt,
    get_prompt_version,
)

from app.core.config import get_settings

settings = get_settings()


# adding trace
def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# 1. Node 1
def route_question(state: AgentState):
    router = llm().with_structured_output(DecisionRoute, method="json_schema")

    # loading the prompt from prompts.yaml
    prompt_name = "decision_router_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["query"])
    decision = router.invoke(formatted_prompt)
    return {
        "source_used": decision.route,
        "trace": add_trace(state=state, message=f"Router -> {decision.route.upper()}"),
    }


# function for conditional edges
def route_after_routing(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"


# Node-2 : Retrive from the private KB
def retrieve_kb(state: AgentState) -> dict:
    docs = rerank_documents(query=state["current_query"], top_n=settings.TOP_N_RANKED)
    return {
        "kb_docs": docs,
        "trace": add_trace(
            state=state, message=f"Private KB retrieval: {len(docs)} chunks"
        ),
    }


# Node 3: Grade Private KB Evidence


def grade_kb_evidence(state: AgentState):
    kb_grader_llm = llm().with_structured_output(
        schema=GradingAnswer, method="json_schema"
    )

    # query = state["query"]
    context = "\n\n".join(
        f"source: {doc.metadata.get('source')} \n {doc.page_content}"
        for doc in state["kb_docs"]
    )
    # loading the prompt from prompts.yaml
    prompt_name = "kb_grader_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["current_query"],context=context)
    grade = kb_grader_llm.invoke(formatted_prompt)

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state=state, message=f"KB evidence -> {grade.grade.upper()}"
        ),
    }


def decide_after_kb_grade(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"


## Node 4 — Tavily Web Search Fallback
def search_web(state: AgentState):
    query = state["current_query"]
    lines, citations = [], []
    result = web_search_tool().invoke(query)

    # Tavily can return dict/list structures depending on package version.
    # Convert it into readable text for grading and generation.
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search Answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = (
                item.get("title", ""),
                item.get("url", ""),
                item.get("content", ""),
            )
            lines.append(f"Title:{title}\nURL:{url}\nContent:{content}")
            citations.append({"Title": title or url, "URL": url, "type": "web"})

    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state=state, message=f"Web Fallback -> Tavily Web Search"),
    }


## Node 5 — Grade Web Evidence
def grade_web_evidence(state: AgentState):
    # query = state["query"]
    web_results = state["web_results"]
    web_grader_llm = llm().with_structured_output(
        schema=GradingAnswer, method="json_schema"
    )
    # loading the prompt from prompts.yaml
    prompt_name = "web_grader_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["query"],web_results=web_results)
    grade = web_grader_llm.invoke(formatted_prompt)
    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state=state, message=f"Web evidence grade: {grade.grade.upper()}"
        ),
    }


def decide_after_web_grade(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.MAX_RETRIES:
        return "rewrite_query"
    return "answer_insufficient"


# Node 6 — Rewrite the Query
def rewrite_query(state: AgentState):
    # query = state["query"]
    retry_count = state["retry_count"] + 1
    prompt_name = "rewrite_query_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["query"])
    rewritten = llm().invoke(formatted_prompt)
    return {
        "current_query": rewritten.content,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state=state, message=f"Query Rewrite -> {rewritten}"),
    }


## Node 7 — Generate from Private KB
def generate_from_kb(state: AgentState):
    # query = state["query"]
    context = "\n\n".join(
        f"[kb_source: {doc.metadata.get('source')}]\n {doc.page_content}"
        for doc in state["kb_docs"]
    )
    # loading the prompt from prompts.yaml
    prompt_name = "generate_from_kb_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["query"],context=context)
    answer = llm().invoke(formatted_prompt)

    citations = []
    seen = set()
    for d in state["kb_docs"]:
        src = d.metadata.get("source", "Private KB")
        if src not in seen:
            seen.add(src)
            citations.append(
                {"title": src.split("/")[-1], "url": "", "type": "private_kb"}
            )
    return {
        "answer": answer.content,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state=state, message=f"Answer generation -> Private KB"),
    }


# Node-8: Generate from web search
def generate_from_web(state: AgentState):
    # query = state["query"]
    web_results = state["web_results"]
    # loading the prompt from prompts.yaml
    prompt_name = "generate_from_web_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["query"],web_results=web_results)
    answer = llm().invoke(formatted_prompt)
    return {
        "answer": answer.content,
        "source_used": "web_search",
        "trace": add_trace(state=state, message=f"Answer generation -> Web Search"),
    }


# Node-9: Direct answer
def direct_answer(state: AgentState):
    # query = state["query"]
    # loading the prompt from prompts.yaml
    prompt_name = "direct_answer_prompt"
    prompt=get_prompt(prompt_name) 
    formatted_prompt= prompt.format(query= state["query"])
    answer = llm().invoke(formatted_prompt)
    return {
        "answer": answer.content,
        "source_used": "direct",
        "trace": add_trace(state=state, message=f"Direct Response -> No retrieval"),
    }


# Node-10: Insufficient evident answer
def answer_insufficient(state: AgentState):
    answer = (
        "Sorry.I could not find enough evidence to answer from private knowledge base or web search."
        "Please specify more specific documents or rephrase the question "
    )
    return {
        "answer": answer,
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }
