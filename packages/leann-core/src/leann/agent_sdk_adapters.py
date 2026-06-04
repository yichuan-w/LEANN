"""Optional adapters from LEANN research tools to external agent SDK tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .research_tools import ResearchTool


def _missing_sdk_error(package_name: str) -> RuntimeError:
    return RuntimeError(
        f"{package_name} is required for this optional agent SDK adapter. "
        "Install the relevant agent SDK extra before using this integration."
    )


def create_openai_function_tools(tools: Iterable[ResearchTool]) -> list[Any]:
    """Wrap LEANN research tools as OpenAI Agents SDK function tools."""
    try:
        from agents import function_tool
    except ImportError as exc:
        raise _missing_sdk_error("openai-agents") from exc

    sdk_tools = []
    for research_tool in tools:

        def invoke(query: str, top_k: int = 5, _tool=research_tool) -> str:
            """Run a LEANN research tool."""
            return _tool.run(query, top_k=top_k).observation

        invoke.__name__ = research_tool.name
        invoke.__doc__ = research_tool.description
        sdk_tools.append(function_tool(invoke))

    return sdk_tools


def create_claude_sdk_tools(tools: Iterable[ResearchTool]) -> list[Any]:
    """Wrap LEANN research tools as Claude Agent SDK MCP tools."""
    try:
        from claude_agent_sdk import tool
    except ImportError as exc:
        raise _missing_sdk_error("claude-agent-sdk") from exc

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "default": 5},
        },
        "required": ["query"],
    }

    sdk_tools = []
    for research_tool in tools:

        async def invoke(args: dict[str, Any], _tool=research_tool) -> dict[str, Any]:
            query = str(args["query"])
            top_k = int(args.get("top_k", 5))
            result = _tool.run(query, top_k=top_k)
            return {"content": [{"type": "text", "text": result.observation}]}

        invoke.__name__ = research_tool.name
        invoke.__doc__ = research_tool.description
        sdk_tools.append(
            tool(
                research_tool.name,
                research_tool.description,
                input_schema,
            )(invoke)
        )

    return sdk_tools
