import asyncio
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest
from leann.agent_sdk_adapters import (
    create_claude_sdk_tools,
    create_openai_function_tools,
)
from leann.research_tools import ResearchToolResult


class FakeAgentsModule(types.ModuleType):
    function_tool: Callable[[Callable[..., str]], Callable[..., str]]


class FakeClaudeModule(types.ModuleType):
    tool: Callable[[str, str, dict[str, Any]], Callable[[Callable[..., Any]], Callable[..., Any]]]


class FakeResearchTool:
    name = "leann_search"
    description = "Search LEANN."
    source = "local"

    def run(self, query: str, top_k: int = 5) -> ResearchToolResult:
        return ResearchToolResult(
            observation=f"{query}:{top_k}",
            results_count=1,
            source="local",
        )


def test_openai_adapter_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "agents", None)

    with pytest.raises(RuntimeError, match="openai-agents"):
        create_openai_function_tools([FakeResearchTool()])


def test_openai_adapter_wraps_research_tool(monkeypatch):
    fake_agents = FakeAgentsModule("agents")

    def function_tool(func):
        return func

    fake_agents.function_tool = function_tool
    monkeypatch.setitem(sys.modules, "agents", fake_agents)

    [sdk_tool] = create_openai_function_tools([FakeResearchTool()])

    assert sdk_tool.__name__ == "leann_search"
    assert sdk_tool("hello", top_k=3) == "hello:3"


def test_claude_adapter_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

    with pytest.raises(RuntimeError, match="claude-agent-sdk"):
        create_claude_sdk_tools([FakeResearchTool()])


def test_claude_adapter_wraps_research_tool(monkeypatch):
    fake_claude = FakeClaudeModule("claude_agent_sdk")

    def tool(name, description, input_schema):
        def decorate(func):
            func.sdk_name = name
            func.sdk_description = description
            func.sdk_input_schema = input_schema
            return func

        return decorate

    fake_claude.tool = tool
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_claude)

    [sdk_tool] = create_claude_sdk_tools([FakeResearchTool()])
    result = asyncio.run(sdk_tool({"query": "hello", "top_k": 3}))

    assert sdk_tool.sdk_name == "leann_search"
    assert result == {"content": [{"type": "text", "text": "hello:3"}]}
