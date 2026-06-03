"""Markdown formatting helpers for benchmark artifacts."""

from __future__ import annotations


def command_markdown_lines(command: str | None) -> list[str]:
    """Return Markdown lines for a benchmark script command."""
    if not command:
        return ["- Command: unavailable"]
    return ["- Command:", "", "```bash", command, "```"]
