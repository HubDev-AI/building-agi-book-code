# tools/web_search.py

import aiohttp

from ..tool_manager import Tool

async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web and return results."""
    # Using a search API (e.g., SearXNG self-hosted,
    # or Brave Search API)
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:8080/search",  # SearXNG instance
            params={"q": query, "format": "json"},
        ) as resp:
            data = await resp.json()
            results = data.get("results", [])[:num_results]
            return "\n\n".join(
                f"**{r['title']}**\n{r['url']}\n"
                f"{r.get('content', '')}"
                for r in results
            )

web_search_tool = Tool(
    name="web_search",
    description="Search the web for current information. "
                "Use when you need up-to-date facts or "
                "information not in your training data.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Search query"},
            "num_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
    handler=web_search,
    category="information",
    risk_level="low",
)
