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
        prompt = f"""You are a helpful assistant that answers questions by searching through a knowledge base.

Question: {question}

You can search the knowledge base by using the action: search("query")

Previous observations:
"""
        if previous_observations:
            for i, obs in enumerate(previous_observations, 1):
                prompt += f"\nObservation {i}:\n{obs}\n"
        else:
            prompt += "None yet.\n"

        prompt += f"""
Current iteration: {iteration}/{self.max_iterations}

Think step by step:
1. If you need more information, use search("your search query")
2. If you have enough information, provide your final answer

Format your response as:
Thought: [your reasoning]
Action: search("query") OR Final Answer: [your answer]
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
            # Try to extract search query
            if 'search("' in action_part:
                start = action_part.find('search("') + 7
                end = action_part.find('")', start)
                if end != -1:
                    action = action_part[start:end]
                else:
                    # Try with single quote
                    end = action_part.find('")', start)
                    if end != -1:
                        action = action_part[start:end]
            elif "search(" in action_part:
                # Handle without quotes
                start = action_part.find("search(") + 7
                end = action_part.find(")", start)
                if end != -1:
                    action = action_part[start:end].strip('"').strip("'")
        elif "search(" in response.lower():
            # Try to extract search query even if format is slightly different
            import re

            match = re.search(r'search\(["\']([^"\']+)["\']\)', response, re.IGNORECASE)
            if match:
                action = match.group(1)

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
            logger.info(f'🔍 Action: search("{action}")')
            results = self.search(action, top_k=top_k)

            # Format observation
            observation = self._format_search_results(results)
            previous_observations.append(observation)
            all_context.append(f"Search: {action}\n{observation}")

            # Store in history
            self.search_history.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "results_count": len(results),
                }
            )

            # If no results, might want to stop early
            if not results and iteration >= 2:
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
    return ReActAgent(searcher=searcher, llm_config=llm_config, max_iterations=max_iterations)
