"""
Enron Emails Benchmark Setup Script
Prepares passages from emails.csv, builds LEANN index, and FAISS Flat baseline
"""

import argparse
import csv
import json
import os
import re
from collections.abc import Iterable
from email import message_from_string
from email.policy import default
from pathlib import Path
from typing import Optional

from leann import LeannBuilder


class EnronSetup:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.passages_preview = self.data_dir / "enron_passages_preview.jsonl"
        self.index_path = self.data_dir / "enron_index_hnsw.leann"
        self.queries_file = self.data_dir / "evaluation_queries.jsonl"
        self.downloads_dir = self.data_dir / "downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Dataset acquisition
    # ----------------------------
    def ensure_emails_csv(self, emails_csv: Optional[str]) -> str:
        """Return a path to emails.csv, downloading from Kaggle if needed."""
        if emails_csv:
            p = Path(emails_csv)
            if not p.exists():
                raise FileNotFoundError(f"emails.csv not found: {emails_csv}")
            return str(p)

        print(
            "📥 Trying to download Enron emails.csv from Kaggle (wcukierski/enron-email-dataset)..."
        )
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                "wcukierski/enron-email-dataset", path=str(self.downloads_dir), unzip=True
            )
            candidate = self.downloads_dir / "emails.csv"
            if candidate.exists():
                print(f"✅ Downloaded emails.csv: {candidate}")
                return str(candidate)
            else:
                raise FileNotFoundError(
                    f"emails.csv was not found in {self.downloads_dir} after Kaggle download"
                )
        except Exception as e:
            print(
                "❌ Could not download via Kaggle automatically. Provide --emails-csv or configure Kaggle API."
            )
            print(
                "   Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place emails.csv locally and pass --emails-csv."
            )
            raise e

    # ----------------------------
    # Data preparation
    # ----------------------------
    @staticmethod
    def _extract_message_id(raw_email: str) -> str:
        msg = message_from_string(raw_email, policy=default)
        val = msg.get("Message-ID", "")
        if val.startswith("<") and val.endswith(">"):
            val = val[1:-1]
        return val or ""

    @staticmethod
    def _split_header_body(raw_email: str) -> tuple[str, str]:
        parts = raw_email.split("\n\n", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        # Heuristic fallback
        first_lines = raw_email.splitlines()
        if first_lines and ":" in first_lines[0]:
            return raw_email.strip(), ""
        return "", raw_email.strip()

    @staticmethod
    def _split_fixed_words(text: str, chunk_words: int, keep_last: bool) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if chunk_words <= 0:
            return [text]
        words = text.split()
        if not words:
            return []
        limit = len(words)
        if not keep_last:
            limit = (len(words) // chunk_words) * chunk_words
        if limit == 0:
            return []
        chunks = [" ".join(words[i : i + chunk_words]) for i in range(0, limit, chunk_words)]
        return [c for c in (s.strip() for s in chunks) if c]

    def _iter_passages_from_csv(
        self,
        emails_csv: Path,
        chunk_words: int = 256,
        keep_last_header: bool = True,
        keep_last_body: bool = True,
        max_emails: int | None = None,
    ) -> Iterable[dict]:
        with open(emails_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for i, row in enumerate(reader):
                if max_emails is not None and count >= max_emails:
                    break

                raw_message = row.get("message", "")
                email_file_id = row.get("file", "")

                if not raw_message.strip():
                    continue

                message_id = self._extract_message_id(raw_message)
                if not message_id:
                    # Fallback ID based on CSV position and file path
                    safe_file = re.sub(r"[^A-Za-z0-9_.-]", "_", email_file_id)
                    message_id = f"enron_{i}_{safe_file}"

                header, body = self._split_header_body(raw_message)

                # Header chunks
                for chunk in self._split_fixed_words(header, chunk_words, keep_last_header):
                    yield {
                        "text": chunk,
                        "metadata": {
                            "message_id": message_id,
                            "is_header": True,
                            "email_file_id": email_file_id,
                        },
                    }

                # Body chunks
                for chunk in self._split_fixed_words(body, chunk_words, keep_last_body):
                    yield {
                        "text": chunk,
                        "metadata": {
                            "message_id": message_id,
                            "is_header": False,
                            "email_file_id": email_file_id,
                        },
                    }

                count += 1

    # ----------------------------
    # Build LEANN index and FAISS baseline
    # ----------------------------
    def build_leann_index(
        self,
        emails_csv: Optional[str],
        backend: str = "hnsw",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_words: int = 256,
        max_emails: int | None = None,
    ) -> str:
        emails_csv_path = self.ensure_emails_csv(emails_csv)
        print(f"🏗️ Building LEANN index from {emails_csv_path}...")

        builder = LeannBuilder(
            backend_name=backend,
            embedding_model=embedding_model,
            embedding_mode="sentence-transformers",
            graph_degree=32,
            complexity=64,
            is_recompute=True,
            is_compact=True,
            num_threads=4,
        )

        # Stream passages and add to builder
        preview_written = 0
        with open(self.passages_preview, "w", encoding="utf-8") as preview_out:
            for p in self._iter_passages_from_csv(
                Path(emails_csv_path), chunk_words=chunk_words, max_emails=max_emails
            ):
                builder.add_text(p["text"], metadata=p["metadata"])
                if preview_written < 200:
                    preview_out.write(json.dumps({"text": p["text"][:200], **p["metadata"]}) + "\n")
                    preview_written += 1

        print(f"🔨 Building index at {self.index_path}...")
        builder.build_index(str(self.index_path))
        print("✅ LEANN index built!")
        return str(self.index_path)

    def build_faiss_flat_baseline(self, index_path: str, output_dir: str = "baseline") -> str:
        print("🔨 Building FAISS Flat baseline from LEANN passages...")

        import pickle

        import numpy as np
        from leann.api import compute_embeddings
        from leann_backend_hnsw import faiss

        os.makedirs(output_dir, exist_ok=True)
        baseline_path = os.path.join(output_dir, "faiss_flat.index")
        metadata_path = os.path.join(output_dir, "metadata.pkl")

        if os.path.exists(baseline_path) and os.path.exists(metadata_path):
            print(f"✅ Baseline already exists at {baseline_path}")
            return baseline_path

        # Read meta for passage source and embedding model
        meta_path = f"{index_path}.meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        embedding_model = meta["embedding_model"]
        passage_source = meta["passage_sources"][0]
        passage_file = passage_source["path"]

        if not os.path.isabs(passage_file):
            index_dir = os.path.dirname(index_path)
            passage_file = os.path.join(index_dir, os.path.basename(passage_file))

        # Load passages from builder output so IDs match LEANN
        passages: list[str] = []
        passage_ids: list[str] = []
        with open(passage_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                passages.append(data["text"])
                passage_ids.append(data["id"])  # builder-assigned ID

        print(f"📄 Loaded {len(passages)} passages for baseline")
        print(f"🤖 Embedding model: {embedding_model}")

        embeddings = compute_embeddings(
            passages,
            embedding_model,
            mode="sentence-transformers",
            use_server=False,
        )

        # Build FAISS IndexFlatIP
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        emb_f32 = embeddings.astype(np.float32)
        index.add(emb_f32.shape[0], faiss.swig_ptr(emb_f32))

        faiss.write_index(index, baseline_path)
        with open(metadata_path, "wb") as pf:
            pickle.dump(passage_ids, pf)

        print(f"✅ FAISS baseline saved: {baseline_path}")
        print(f"✅ Metadata saved: {metadata_path}")
        print(f"📊 Total vectors: {index.ntotal}")
        return baseline_path

    # ----------------------------
    # Queries (optional): prepare evaluation queries file
    # ----------------------------
    def prepare_queries(self, min_realism: float = 0.85) -> Path:
        print(
            "📝 Preparing evaluation queries from HuggingFace dataset corbt/enron_emails_sample_questions ..."
        )
        try:
            from datasets import load_dataset

            ds = load_dataset("corbt/enron_emails_sample_questions", split="train")
        except Exception as e:
            print(f"⚠️  Failed to load dataset: {e}")
            return self.queries_file

        kept = 0
        with open(self.queries_file, "w", encoding="utf-8") as out:
            for i, item in enumerate(ds):
                how_realistic = float(item.get("how_realistic", 0.0))
                if how_realistic < min_realism:
                    continue
                qid = str(item.get("id", f"enron_q_{i}"))
                query = item.get("question", "")
                if not query:
                    continue
                record = {
                    "id": qid,
                    "query": query,
                    # For reference only, not used in recall metric below
                    "gt_message_ids": item.get("message_ids", []),
                }
                out.write(json.dumps(record) + "\n")
                kept += 1
        print(f"✅ Wrote {kept} queries to {self.queries_file}")
        return self.queries_file


def main():
    parser = argparse.ArgumentParser(description="Setup Enron Emails Benchmark")
    parser.add_argument(
        "--emails-csv",
        help="Path to emails.csv (Enron dataset). If omitted, attempt Kaggle download.",
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--backend", choices=["hnsw", "diskann"], default="hnsw")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model for LEANN",
    )
    parser.add_argument("--chunk-words", type=int, default=256, help="Fixed word chunk size")
    parser.add_argument("--max-emails", type=int, help="Limit number of emails to process")
    parser.add_argument("--skip-queries", action="store_true", help="Skip creating queries file")
    parser.add_argument("--skip-build", action="store_true", help="Skip building LEANN index")

    args = parser.parse_args()

    setup = EnronSetup(args.data_dir)

    # Build index
    if not args.skip_build:
        index_path = setup.build_leann_index(
            emails_csv=args.emails_csv,
            backend=args.backend,
            embedding_model=args.embedding_model,
            chunk_words=args.chunk_words,
            max_emails=args.max_emails,
        )

        # Build FAISS baseline from the same passages & embeddings
        setup.build_faiss_flat_baseline(index_path)
    else:
        print("⏭️  Skipping LEANN index build and baseline")

    # Queries file (optional)
    if not args.skip_queries:
        setup.prepare_queries()
    else:
        print("⏭️  Skipping query preparation")

    print("\n🎉 Enron Emails setup completed!")
    print(f"📁 Data directory: {setup.data_dir.absolute()}")
    print("Next steps:")
    print(
        "1) Evaluate recall: python evaluate_enron_emails.py --index data/enron_index_hnsw.leann --stage 2"
    )


if __name__ == "__main__":
    main()
