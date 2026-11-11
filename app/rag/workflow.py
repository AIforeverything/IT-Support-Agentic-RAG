from app.core.logging import configure_logging
from app.core.config import get_settings
from app.rag.state import AgentState
from langgraph.graph import StateGraph, START, END
from app.nodes.nodes import (
    route_question,
    route_after_routing,
    retrieve_kb,
    grade_kb_evidence,
    decide_after_kb_grade,
    search_web,
    grade_web_evidence,
    decide_after_web_grade,
    rewrite_query,
    generate_from_kb,
    generate_from_web,
    direct_answer,
    answer_insufficient,
)
from typing import List, Literal, TypedDict

logger = configure_logging()
settings = get_settings()


def build_graph():
    graph = StateGraph(state_schema=AgentState)

    # making nodes for the graph
    for node_name, function in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb_evidence": grade_kb_evidence,
        "search_web": search_web,
        "grade_web_evidence": grade_web_evidence,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "answer_insufficient": answer_insufficient,
    }.items():
        graph.add_node(node_name, function)

    # adding edges between nodes
    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        route_after_routing,
        {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
    )

    graph.add_edge("retrieve_kb", "grade_kb_evidence")
    graph.add_conditional_edges(
        "grade_kb_evidence",
        decide_after_kb_grade,
        {"generate_from_kb": "generate_from_kb", "search_web": "search_web"},
    )

    graph.add_edge("search_web", "grade_web_evidence")
    graph.add_conditional_edges(
        "grade_web_evidence",
        decide_after_web_grade,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "answer_insufficient": "answer_insufficient",
        },
    )

    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial_state: AgentState = {
        "query": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial_state)
