import argparse
import re
import sys
import time
from pathlib import Path
from statistics import mean

from leann.chat import get_llm


def parse_prompts_from_file(file_path: str) -> list[str]:
    """
    Parse a prompt dump file into individual prompt strings.

    Splits by lines that look like: "PROMPT #<n>:".
    Keeps the content from each marker up to the next marker (or EOF).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    matches = list(re.finditer(r"^PROMPT\s+#\d+:\s*$", text, flags=re.MULTILINE))
    if not matches:
        # Fallback: try a more permissive pattern
        matches = list(
            re.finditer(r"^=+\nPROMPT\s+#\d+:\n=+\s*$", text, flags=re.MULTILINE)
        )

    prompts: list[str] = []
    if not matches:
        # No explicit markers; treat the whole file as a single prompt
        return [text]

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        # Reattach the marker line content above the block for full context
        header_line_start = text.rfind("\n", 0, m.start()) + 1
        header = text[header_line_start : m.end()].strip()
        prompts.append(f"{header}\n{block}".strip())

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Iterate prompts in a dump file, time generations, print outputs, and report last-10 average time."
        )
    )
    parser.add_argument(
        "--path",
        default="benchmarks/data/prompts_g5/prompt_dump_nq_hnsw.txt",
        help="Path to the prompt dump file",
    )
    parser.add_argument(
        "--type",
        default="ollama",
        choices=["hf", "openai", "ollama", "gemini", "simulated"],
        help="LLM backend type",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="Model identifier (depends on backend)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max new tokens to generate per prompt",
    )
    args = parser.parse_args()

    llm_config = {"type": args.type, "model": args.model}
    chat = get_llm(llm_config)

    prompts = parse_prompts_from_file(args.path)
    print(f"Found {len(prompts)} prompts in {args.path}")

    times: list[float] = []
    for idx, prompt in enumerate(prompts, start=1):
        print("\n" + "=" * 80)
        print(f"PROMPT {idx}/{len(prompts)}")
        print("-" * 80)
        start = time.perf_counter()
        try:
            output = chat.ask(prompt, max_tokens=args.max_tokens)
        except Exception as e:
            output = f"<error: {e}>"
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Time: {elapsed:.3f}s")
        print("-" * 80)
        print(output)
        print("=" * 80)

    if times:
        window = times[-10:] if len(times) >= 10 else times
        avg_last_10 = mean(window)
        print(
            f"\nAverage time over last {len(window)} prompts: {avg_last_10:.3f}s"
        )
    else:
        print("No prompts processed.")


if __name__ == "__main__":
    main()





