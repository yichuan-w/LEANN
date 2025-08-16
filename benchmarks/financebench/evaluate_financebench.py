#!/usr/bin/env python3
"""
FinanceBench Evaluation Script
Uses intelligent evaluation similar to VectifyAI/Mafin approach
"""

import argparse
import json
import os
import re
import time
from typing import Optional

import openai
from leann import LeannChat, LeannSearcher


class FinanceBenchEvaluator:
    def __init__(self, index_path: str, openai_api_key: Optional[str] = None):
        self.index_path = index_path
        self.openai_client = openai.OpenAI(api_key=openai_api_key) if openai_api_key else None

        self.searcher = LeannSearcher(index_path)
        self.chat = LeannChat(index_path) if openai_api_key else None

    def load_dataset(self, dataset_path: str = "data/financebench_merged.jsonl"):
        """Load FinanceBench dataset"""
        data = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"📊 Loaded {len(data)} FinanceBench examples")
        return data

    def evaluate_retrieval_intelligent(self, data: list[dict], top_k: int = 10) -> dict:
        """
        Intelligent retrieval evaluation
        Uses semantic similarity instead of strict word overlap
        """
        print(f"🔍 Evaluating retrieval performance (top_k={top_k})...")

        metrics = {
            "total_questions": 0,
            "questions_with_relevant_retrieved": 0,
            "exact_matches": 0,
            "semantic_matches": 0,
            "number_matches": 0,
            "search_times": [],
            "detailed_results": [],
        }

        for item in data:
            question = item["question"]
            evidence_texts = [ev["evidence_text"] for ev in item.get("evidence", [])]
            expected_answer = item["answer"]

            if not evidence_texts:
                continue

            metrics["total_questions"] += 1

            # Search for relevant documents
            start_time = time.time()
            results = self.searcher.search(question, top_k=top_k, complexity=64)
            search_time = time.time() - start_time
            metrics["search_times"].append(search_time)

            # Evaluate retrieved results
            found_relevant = False
            match_types = []

            for evidence_text in evidence_texts:
                for i, result in enumerate(results):
                    retrieved_text = result.text

                    # Method 1: Exact substring match
                    if self._has_exact_overlap(evidence_text, retrieved_text):
                        found_relevant = True
                        match_types.append(f"exact_match_rank_{i + 1}")
                        metrics["exact_matches"] += 1
                        break

                    # Method 2: Key numbers match
                    elif self._has_number_match(evidence_text, retrieved_text, expected_answer):
                        found_relevant = True
                        match_types.append(f"number_match_rank_{i + 1}")
                        metrics["number_matches"] += 1
                        break

                    # Method 3: Semantic similarity (word overlap with lower threshold)
                    elif self._has_semantic_similarity(
                        evidence_text, retrieved_text, threshold=0.2
                    ):
                        found_relevant = True
                        match_types.append(f"semantic_match_rank_{i + 1}")
                        metrics["semantic_matches"] += 1
                        break

            if found_relevant:
                metrics["questions_with_relevant_retrieved"] += 1

            # Store detailed result
            metrics["detailed_results"].append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "found_relevant": found_relevant,
                    "match_types": match_types,
                    "search_time": search_time,
                    "top_results": [
                        {"text": r.text[:200] + "...", "score": r.score, "metadata": r.metadata}
                        for r in results[:3]
                    ],
                }
            )

        # Calculate metrics
        if metrics["total_questions"] > 0:
            metrics["question_coverage"] = (
                metrics["questions_with_relevant_retrieved"] / metrics["total_questions"]
            )
            metrics["avg_search_time"] = sum(metrics["search_times"]) / len(metrics["search_times"])

            # Match type breakdown
            metrics["exact_match_rate"] = metrics["exact_matches"] / metrics["total_questions"]
            metrics["number_match_rate"] = metrics["number_matches"] / metrics["total_questions"]
            metrics["semantic_match_rate"] = (
                metrics["semantic_matches"] / metrics["total_questions"]
            )

        return metrics

    def evaluate_qa_intelligent(self, data: list[dict], max_samples: Optional[int] = None) -> dict:
        """
        Intelligent QA evaluation using LLM-based answer comparison
        Similar to VectifyAI/Mafin approach
        """
        if not self.chat or not self.openai_client:
            print("⚠️  Skipping QA evaluation (no OpenAI API key provided)")
            return {"accuracy": 0.0, "total_questions": 0}

        print("🤖 Evaluating QA performance...")

        if max_samples:
            data = data[:max_samples]
            print(f"📝 Using first {max_samples} samples for QA evaluation")

        results = []
        correct_answers = 0

        for i, item in enumerate(data):
            question = item["question"]
            expected_answer = item["answer"]

            print(f"Question {i + 1}/{len(data)}: {question[:80]}...")

            try:
                # Get answer from LEANN
                start_time = time.time()
                generated_answer = self.chat.ask(question)
                qa_time = time.time() - start_time

                # Intelligent evaluation using LLM
                is_correct = self._evaluate_answer_with_llm(
                    question, expected_answer, generated_answer
                )

                if is_correct:
                    correct_answers += 1

                results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "is_correct": is_correct,
                        "qa_time": qa_time,
                    }
                )

                print(f"  ✅ {'Correct' if is_correct else '❌ Incorrect'}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": f"ERROR: {e}",
                        "is_correct": False,
                        "qa_time": 0.0,
                    }
                )

        metrics = {
            "total_questions": len(data),
            "correct_answers": correct_answers,
            "accuracy": correct_answers / len(data) if data else 0.0,
            "avg_qa_time": sum(r["qa_time"] for r in results) / len(results) if results else 0.0,
            "detailed_results": results,
        }

        return metrics

    def _has_exact_overlap(self, evidence_text: str, retrieved_text: str) -> bool:
        """Check for exact substring overlap"""
        # Check if evidence is contained in retrieved text or vice versa
        return (
            evidence_text.lower() in retrieved_text.lower()
            or retrieved_text.lower() in evidence_text.lower()
        )

    def _has_number_match(
        self, evidence_text: str, retrieved_text: str, expected_answer: str
    ) -> bool:
        """Check if key numbers from evidence/answer appear in retrieved text"""
        # Extract numbers from evidence and expected answer
        evidence_numbers = set(re.findall(r"\$?[\d,]+\.?\d*", evidence_text))
        answer_numbers = set(re.findall(r"\$?[\d,]+\.?\d*", expected_answer))
        retrieved_numbers = set(re.findall(r"\$?[\d,]+\.?\d*", retrieved_text))

        # Check if any key numbers match
        key_numbers = evidence_numbers.union(answer_numbers)
        return bool(key_numbers.intersection(retrieved_numbers))

    def _has_semantic_similarity(
        self, evidence_text: str, retrieved_text: str, threshold: float = 0.2
    ) -> bool:
        """Check semantic similarity using word overlap"""
        words1 = set(evidence_text.lower().split())
        words2 = set(retrieved_text.lower().split())

        if len(words1) == 0:
            return False

        overlap = len(words1.intersection(words2))
        similarity = overlap / len(words1)

        return similarity >= threshold

    def _evaluate_answer_with_llm(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> bool:
        """
        Use LLM to evaluate answer equivalence
        Based on VectifyAI/Mafin approach
        """
        prompt = f"""You are an expert evaluator for AI-generated responses to financial questions. Your task is to determine whether the AI-generated answer correctly answers the question based on the golden answer provided by a human expert.

Evaluation Criteria:
- Numerical Accuracy: Rounding differences should be ignored if they don't meaningfully change the conclusion. Numbers like 1.2 and 1.23 are considered similar.
- Fractions, percentages, and numerics could be considered similar. For example: "11 of 14" ≈ "79%" ≈ "0.79".
- If the golden answer or any of its equivalence can be inferred from the AI answer, then the AI answer is correct.
- The AI answer is correct if it conveys the same meaning, conclusion, or rationale as the golden answer.
- If the AI answer is a superset of the golden answer, it is also considered correct.
- Subjective judgments are correct as long as they are reasonable and justifiable.

Question: {question}
AI-Generated Answer: {generated_answer}
Golden Answer: {expected_answer}

Your output should be ONLY a boolean value: `True` or `False`, nothing else."""

        # retry in exponential backoff
        for i in range(3):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                result = response.choices[0].message.content.strip().lower()
                return "true" in result
            except Exception as e:
                print(f"LLM evaluation error: {e}")
                time.sleep(2**i)

        return False

    def run_evaluation(
        self,
        dataset_path: str = "data/financebench_merged.jsonl",
        top_k: int = 10,
        qa_samples: Optional[int] = None,
    ) -> dict:
        """Run complete FinanceBench evaluation"""
        print("🏦 FinanceBench Evaluation with LEANN")
        print("=" * 50)
        print(f"📁 Index: {self.index_path}")
        print(f"🔍 Top-k: {top_k}")
        if qa_samples:
            print(f"🤖 QA samples: {qa_samples}")
        print()

        # Load dataset
        data = self.load_dataset(dataset_path)

        # Run retrieval evaluation
        retrieval_metrics = self.evaluate_retrieval_intelligent(data, top_k=top_k)

        # Run QA evaluation
        qa_metrics = self.evaluate_qa_intelligent(data, max_samples=qa_samples)

        # Print results
        self._print_results(retrieval_metrics, qa_metrics)

        return {
            "retrieval": retrieval_metrics,
            "qa": qa_metrics,
        }

    def _print_results(self, retrieval_metrics: dict, qa_metrics: dict):
        """Print evaluation results"""
        print("\n🎯 EVALUATION RESULTS")
        print("=" * 50)

        print("\n📊 Retrieval Metrics:")
        print(f"  Question Coverage: {retrieval_metrics.get('question_coverage', 0):.1%}")
        print(f"  Exact Match Rate: {retrieval_metrics.get('exact_match_rate', 0):.1%}")
        print(f"  Number Match Rate: {retrieval_metrics.get('number_match_rate', 0):.1%}")
        print(f"  Semantic Match Rate: {retrieval_metrics.get('semantic_match_rate', 0):.1%}")
        print(f"  Avg Search Time: {retrieval_metrics.get('avg_search_time', 0):.3f}s")

        if qa_metrics.get("total_questions", 0) > 0:
            print("\n🤖 QA Metrics:")
            print(f"  Accuracy: {qa_metrics.get('accuracy', 0):.1%}")
            print(f"  Questions Evaluated: {qa_metrics.get('total_questions', 0)}")
            print(f"  Avg QA Time: {qa_metrics.get('avg_qa_time', 0):.3f}s")

        # Show some example results
        print("\n📝 Example Results:")
        for i, result in enumerate(retrieval_metrics.get("detailed_results", [])[:3]):
            print(f"\n  Example {i + 1}:")
            print(f"    Q: {result['question'][:80]}...")
            print(f"    Found relevant: {'✅' if result['found_relevant'] else '❌'}")
            if result["match_types"]:
                print(f"    Match types: {', '.join(result['match_types'])}")

    def cleanup(self):
        """Cleanup resources"""
        if self.searcher:
            self.searcher.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinanceBench with LEANN")
    parser.add_argument("--index", required=True, help="Path to LEANN index")
    parser.add_argument("--dataset", default="data/financebench_merged.jsonl", help="Dataset path")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--qa-samples", type=int, default=None, help="Limit QA evaluation samples")
    parser.add_argument("--openai-api-key", help="OpenAI API key for QA evaluation")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Get OpenAI API key
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and args.qa_samples != 0:
        print("⚠️  No OpenAI API key provided. QA evaluation will be skipped.")
        print("   Set OPENAI_API_KEY environment variable or use --openai-api-key")

    try:
        # Run evaluation
        evaluator = FinanceBenchEvaluator(args.index, api_key)
        results = evaluator.run_evaluation(
            dataset_path=args.dataset, top_k=args.top_k, qa_samples=args.qa_samples
        )

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")

        evaluator.cleanup()

        print("\n✅ Evaluation completed!")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
