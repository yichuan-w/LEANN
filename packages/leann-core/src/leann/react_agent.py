"""
Simple ReAct agent for multiturn retrieval with LEANN.

This implements a basic ReAct (Reasoning + Acting) agent pattern:
- Thought: LLM reasons about what to do next
- Action: Performs a search action (local or web)
- Observation: Gets results from search
- Repeat until final answer

Reference: Inspired by mini-swe-agent pattern, kept simple for multiturn retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .api import LeannSearcher, SearchResult
from .chat import LLMInterface, get_llm
from .web_search import WebSearcher

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct agent for multiturn retrieval with local and web search.

    Supports three tools:
    - leann_search: search the local knowledge base
    - web_search: search the public internet via Serper API (when configured)
    - visit_page: fetch full page content via Jina Reader (when configured)

    The agent dynamically adapts its prompt and behavior based on which
    tools are available (i.e., whether API keys are configured).
    """

    def __init__(
        self,
        searcher: LeannSearcher,
        llm: LLMInterface | None = None,
        llm_config: dict[str, Any] | None = None,
        max_iterations: int = 5,
        serper_api_key: str | None = None,
        jina_api_key: str | None = None,
    ):
        self.searcher = searcher
        if llm is None:
            self.llm = get_llm(llm_config)
        else:
            self.llm = llm
        self.max_iterations = max_iterations
        self.search_history: list[dict[str, Any]] = []
        self.web_searcher = WebSearcher(api_key=serper_api_key, jina_api_key=jina_api_key)
        self.web_search_available = bool(self.web_searcher.api_key)

    def _format_search_results(self, results: list[SearchResult]) -> str:
        """Format search results as a string for the LLM."""
        if not results:
            return "No results found."
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"[Result {i}] (Score: {result.score:.3f})\n{result.text[:500]}...")
            if result.metadata.get("source"):
                formatted[-1] += f"\nSource: {result.metadata['source']}"
        return "\n\n".join(formatted)

    def _create_react_prompt(
        self, question: str, iteration: int, previous_observations: list[str]
    ) -> str:
        """Create the ReAct prompt, dynamically adapted to available tools."""
        if self.web_search_available:
            tools_block = (
                "You have access to these tools:\n"
                '1. leann_search("query"): Search the local private knowledge base (code, docs, history).\n'
                '2. web_search("query"): Search the public internet for up-to-date information.\n'
                '3. visit_page("url"): Read the full content of a specific URL.\n'
                "\nStrategies:\n"
                "- Use `leann_search` for internal project details, code implementation, or private history.\n"
                "- Use `web_search` for public documentation, latest news, or general concepts.\n"
                "- Use `visit_page` if you found a relevant link but need the full details.\n"
                "- You can combine both!"
            )
            action_examples = (
                'Action: leann_search("your query")\n\nOR\n\n'
                "Thought: [your reasoning]\n"
                'Action: web_search("your query")\n\nOR\n\n'
                "Thought: [your reasoning]\n"
                "Action: Final Answer: [your answer]"
            )
        else:
            tools_block = (
                "You have access to this tool:\n"
                '1. leann_search("query"): Search the local private knowledge base (code, docs, history).\n'
                "\nNote: Web search is not available (no API key configured). "
                "Answer using only the local knowledge base."
            )
            action_examples = (
                'Action: leann_search("your query")\n\nOR\n\n'
                "Thought: [your reasoning]\n"
                "Action: Final Answer: [your answer]"
            )

        prompt = (
            "You are a helpful assistant that answers questions by searching through "
            "a knowledge base"
        )
        if self.web_search_available:
            prompt += " AND the internet"
        prompt += f".\n\nQuestion: {question}\n\n{tools_block}\n\nPrevious observations:\n"

        if previous_observations:
            for i, obs in enumerate(previous_observations, 1):
                prompt += f"\nObservation {i}:\n{obs}\n"
        else:
            prompt += "None yet.\n"

        prompt += (
            f"\nCurrent iteration: {iteration}/{self.max_iterations}\n\n"
            "Think step by step.\n"
            "Format your response EXACTLY like this:\n\n"
            f"Thought: [your reasoning]\n{action_examples}\n\n"
            'IMPORTANT: You MUST start a new line with "Action:" to trigger a tool.\n'
        )

        return prompt

    def _parse_llm_response(self, response: str) -> tuple[str, str | None]:
        """
        Parse LLM response to extract thought and action.

        Returns:
            (thought, action) where action is a prefixed string like
            "leann_search:query", "web_search:query", "visit_page:url",
            or None if the agent wants to give a final answer.
        """
        thought = ""
        action = None

        if "Thought:" in response:
            thought_part = response.split("Thought:")[1]
            if "Action:" in thought_part:
                thought = thought_part.split("Action:")[0].strip()
            elif "Final Answer:" in thought_part:
                thought = thought_part.split("Final Answer:")[0].strip()
            else:
                thought = thought_part.strip()
        else:
            if "Action:" in response or "Final Answer:" in response:
                thought = response.split("Action:")[0].split("Final Answer:")[0].strip()
            else:
                thought = response.strip()

        if "Final Answer:" in response:
            action = None
        elif "Action:" in response:
            action_part = response.split("Action:")[1].strip()

            match = re.search(
                r'(web_search|leann_search|visit_page|search)\(["\']([^"\']+)["\']\)',
                action_part,
            )
            if match:
                tool_name = match.group(1)
                if tool_name == "search":
                    tool_name = "leann_search"
                action = f"{tool_name}:{match.group(2)}"
        elif "search(" in response.lower():
            match = re.search(r'search\(["\']([^"\']+)["\']\)', response, re.IGNORECASE)
            if match:
                action = f"leann_search:{match.group(1)}"

        return thought, action

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform a local search and return results."""
        logger.info(f"Searching: {query}")
        results = self.searcher.search(query, top_k=top_k)
        return results

    def run(self, question: str, top_k: int = 5) -> str:
        """
        Run the ReAct agent to answer a question.

        The agent routes between local (leann_search) and web (web_search,
        visit_page) tools based on LLM reasoning. When web tools are
        unavailable, the agent gracefully falls back to local-only search.
        """
        logger.info(f"Starting ReAct agent for question: {question}")
        self.search_history = []
        previous_observations: list[str] = []
        all_context: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            prompt = self._create_react_prompt(question, iteration, previous_observations)

            logger.info("Getting LLM reasoning...")
            response = self.llm.ask(prompt)

            thought, action = self._parse_llm_response(response)
            logger.info(f"Thought: {thought}")

            if action is None:
                if "Final Answer:" in response:
                    final_answer = response.split("Final Answer:")[1].strip()
                else:
                    final_answer = response.strip()
                    if "Action:" in final_answer:
                        final_answer = final_answer.split("Action:")[0].strip()

                logger.info(f"Final answer: {final_answer}")
                return final_answer

            logger.info(f"Action: {action}")

            results_count = 0

            if action.startswith("web_search:"):
                query_str = action.split(":", 1)[1]

                if not self.web_search_available:
                    observation = (
                        "Web search is not available (no SERPER_API_KEY configured). "
                        "Use leann_search to search the local knowledge base instead."
                    )
                    results_count = 0
                else:
                    web_results = self.web_searcher.search(query_str, top_k=top_k)

                    is_error = len(web_results) == 1 and web_results[0].get("title") == "Error"
                    if is_error:
                        observation = (
                            f"Web search failed: {web_results[0].get('snippet', 'Unknown error')}. "
                            "Try leann_search for local results instead."
                        )
                        results_count = 0
                    elif not web_results:
                        observation = "No web results found."
                        results_count = 0
                    else:
                        formatted = []
                        for i, res in enumerate(web_results, 1):
                            formatted.append(
                                f"[Web Result {i}]\nTitle: {res['title']}\n"
                                f"Link: {res['link']}\nSnippet: {res['snippet']}"
                            )
                        observation = "\n\n".join(formatted)
                        results_count = len(web_results)

            elif action.startswith("visit_page:"):
                url = action.split(":", 1)[1]
                try:
                    content = self.web_searcher.get_page_content(url)
                except Exception as e:
                    content = f"Error fetching page: {e!s}"
                results_count = 1 if not content.startswith("Error") else 0
                observation = f"Content of {url}:\n{content[:15000]}"

            else:
                query_str = action.split(":", 1)[1] if ":" in action else action
                results = self.search(query_str, top_k=top_k)
                results_count = len(results)
                observation = self._format_search_results(results)

            previous_observations.append(observation)
            all_context.append(f"Action: {action}\n{observation}")

            if action.startswith("web_search:") or action.startswith("visit_page:"):
                source = "web"
            else:
                source = "local"

            self.search_history.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "results_count": results_count,
                    "source": source,
                }
            )

            if results_count == 0 and iteration >= 2:
                logger.warning("No results found, asking LLM for final answer...")
                final_prompt = f"""Based on the previous searches, provide your best answer to the question.

Question: {question}

Previous searches and results:
{chr(10).join(all_context)}

Since no new results were found, provide your final answer based on what you know.
"""
                final_answer = self.llm.ask(final_prompt)
                return final_answer.strip()

        logger.warning(f"Reached max iterations ({self.max_iterations}), getting final answer...")
        final_prompt = f"""Based on all the searches performed, provide your final answer to the question.

Question: {question}

All search results:
{chr(10).join(all_context)}

Provide your final answer now.
"""
        final_answer = self.llm.ask(final_prompt)
        return final_answer.strip()


def create_react_agent(
    index_path: str,
    llm_config: dict[str, Any] | None = None,
    max_iterations: int = 5,
    serper_api_key: str | None = None,
    jina_api_key: str | None = None,
    **searcher_kwargs,
) -> ReActAgent:
    """Convenience function to create a ReActAgent."""
    searcher = LeannSearcher(index_path, **searcher_kwargs)
    return ReActAgent(
        searcher=searcher,
        llm_config=llm_config,
        max_iterations=max_iterations,
        serper_api_key=serper_api_key,
        jina_api_key=jina_api_key,
    )
