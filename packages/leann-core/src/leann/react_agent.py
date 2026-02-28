"""
Simple ReAct agent for multiturn retrieval with LEANN.

This implements a basic ReAct (Reasoning + Acting) agent pattern:
- Thought: LLM reasons about what to do next
- Action: Performs a search action
- Observation: Gets results from search
- Repeat until final answer

Reference: Inspired by mini-swe-agent pattern, kept simple for multiturn retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

from .api import LeannSearcher, SearchResult
from .chat import LLMInterface, get_llm
from .web_search import WebSearcher

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    Simple ReAct agent for multiturn retrieval.

    The agent follows this pattern:
    1. Thought: LLM reasons about what information is needed
    2. Action: Performs a search query
    3. Observation: Gets search results
    4. Repeat until LLM decides it has enough information to answer
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
        """
        Initialize the ReAct agent.

        Args:
            searcher: LeannSearcher instance for performing searches
            llm: LLM interface (if None, will create from llm_config)
            llm_config: Configuration for creating LLM if llm is None
            max_iterations: Maximum number of search iterations (default: 5)
        """
        self.searcher = searcher
        if llm is None:
            self.llm = get_llm(llm_config)
        else:
            self.llm = llm
        self.max_iterations = max_iterations
        self.search_history: list[dict[str, Any]] = []
        self.web_searcher = WebSearcher(api_key=serper_api_key, jina_api_key=jina_api_key)

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
        """Create the ReAct prompt for the LLM."""
        prompt = f"""You are a helpful assistant that answers questions by searching through a knowledge base AND the internet.

Question: {question}

You have access to two tools:
1. leann_search("query"): Search the local private knowledge base (code, docs, history).
2. web_search("query"): Search the public internet for up-to-date information.
3. visit_page("url"): Read the full content of a specific URL.

Strategies:
- Use `leann_search` for internal project details, code implementation, or private history.
- Use `web_search` for public documentation, latest news, or general concepts.
- Use `visit_page` if you found a relevant link but need the full details.
- You can Combine both!

Previous observations:
"""
        if previous_observations:
            for i, obs in enumerate(previous_observations, 1):
                prompt += f"\nObservation {i}:\n{obs}\n"
        else:
            prompt += "None yet.\n"

        prompt += f"""
Current iteration: {iteration}/{self.max_iterations}

Think step by step.
Format your response EXACTLY like this:

Thought: [your reasoning]
Action: web_search("your query")

OR

Thought: [your reasoning]
Action: leann_search("your query")

OR

Thought: [your reasoning]
Action: Final Answer: [your answer]

IMPORTANT: You MUST start a new line with "Action:" to trigger a tool.
"""

        return prompt

    def _parse_llm_response(self, response: str) -> tuple[str, str | None]:
        """
        Parse LLM response to extract thought and action.

        Returns:
            (thought, action) where action is either a search query string or None if final answer
        """
        thought = ""
        action = None

        # Extract thought
        if "Thought:" in response:
            thought_part = response.split("Thought:")[1]
            if "Action:" in thought_part:
                thought = thought_part.split("Action:")[0].strip()
            elif "Final Answer:" in thought_part:
                thought = thought_part.split("Final Answer:")[0].strip()
            else:
                thought = thought_part.strip()
        else:
            # If no explicit Thought, use everything before Action/Final Answer
            if "Action:" in response or "Final Answer:" in response:
                thought = response.split("Action:")[0].split("Final Answer:")[0].strip()
            else:
                thought = response.strip()

        # Extract action
        if "Final Answer:" in response:
            # Agent wants to provide final answer
            action = None
        elif "Action:" in response:
            action_part = response.split("Action:")[1].strip()

            # Use regex to extract action - handles all cases correctly
            import re

            # Try to match web_search, leann_search, search, or visit_page
            match = re.search(
                r'(web_search|leann_search|visit_page|search)\(["\']([^"\']+)["\']\)',
                action_part,
            )
            if match:
                action = f"{match.group(1)}:{match.group(2)}"

        return thought, action

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Perform a search and return results.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        logger.info(f"🔍 Searching: {query}")
        results = self.searcher.search(query, top_k=top_k)
        return results

    def run(self, question: str, top_k: int = 5) -> str:
        """
        Run the ReAct agent to answer a question.

        Args:
            question: The question to answer
            top_k: Number of search results per iteration

        Returns:
            Final answer string
        """
        logger.info(f"🤖 Starting ReAct agent for question: {question}")
        self.search_history = []
        previous_observations: list[str] = []
        all_context: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            # Create prompt for this iteration
            prompt = self._create_react_prompt(question, iteration, previous_observations)

            # Get LLM response
            logger.info("💭 Getting LLM reasoning...")
            response = self.llm.ask(prompt)

            # Parse response
            thought, action = self._parse_llm_response(response)
            logger.info(f"Thought: {thought}")

            if action is None:
                # LLM wants to provide final answer
                if "Final Answer:" in response:
                    final_answer = response.split("Final Answer:")[1].strip()
                else:
                    # Extract answer from response
                    final_answer = response.strip()
                    # Remove any action markers if present
                    if "Action:" in final_answer:
                        final_answer = final_answer.split("Action:")[0].strip()

                logger.info(f"✅ Final answer: {final_answer}")
                return final_answer

            # Perform search action
            logger.info(f"🔍 Action: {action}")

            results_count = 0

            if action.startswith("web_search:"):
                query = action.split(":", 1)[1]
                # Perform web search directly to get objects and count
                web_results = self.web_searcher.search(query, top_k=top_k)
                results_count = len(web_results)

                if not web_results:
                    observation = "No web results found."
                else:
                    formatted = []
                    for i, res in enumerate(web_results, 1):
                        formatted.append(
                            f"[Web Result {i}]\nTitle: {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}"
                        )
                    observation = "\n\n".join(formatted)

            elif action.startswith("visit_page:"):
                url = action.split(":", 1)[1]
                content = self.web_searcher.get_page_content(url)
                results_count = 1
                # Truncate content to avoid token limit overflow (adjust limit as needed)
                observation = f"Content of {url}:\n{content[:15000]}"

            else:
                # Default to local search
                query = action.split(":", 1)[1] if ":" in action else action
                results = self.search(query, top_k=top_k)
                results_count = len(results)
                observation = self._format_search_results(results)

            previous_observations.append(observation)
            all_context.append(f"Action: {action}\n{observation}")

            # Store in history
            self.search_history.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "results_count": results_count,
                }
            )

            # If no results, might want to stop early
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

        # Max iterations reached, get final answer
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
    """
    Convenience function to create a ReActAgent.

    Args:
        index_path: Path to LEANN index
        llm_config: LLM configuration dict
        max_iterations: Maximum search iterations
        **searcher_kwargs: Additional kwargs for LeannSearcher

    Returns:
        ReActAgent instance
    """
    searcher = LeannSearcher(index_path, **searcher_kwargs)
    return ReActAgent(
        searcher=searcher,
        llm_config=llm_config,
        max_iterations=max_iterations,
        serper_api_key=serper_api_key,
        jina_api_key=jina_api_key,
    )
