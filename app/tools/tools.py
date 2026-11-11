from app.core.logging import configure_logging
from app.core.config import get_settings
from langchain_tavily import TavilySearch

logger = configure_logging()
settings = get_settings()
_web_search = None


# initiate Tavily web search tool
def web_search_tool():
    global _web_search
    if _web_search is None:
        if settings.TAVILY_API_KEY is None:
            raise RuntimeError("TAVILY_API_KEY is missing.")
        _web_search = TavilySearch(
            tavily_api_key=settings.TAVILY_API_KEY,
            max_results=3,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


if __name__ == "__main__":
    web_search_tool()
