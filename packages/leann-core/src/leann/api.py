"""
This file contains the core API for the LEANN project, now definitively updated
with the correct, original embedding logic from the user's reference code.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import torch

from .registry import BACKEND_REGISTRY
from .interface import LeannBackendFactoryInterface

# --- The Correct, Verified Embedding Logic from old_code.py ---

def compute_embeddings(chunks: List[str], model_name: str) -> np.ndarray:
    """Computes embeddings using sentence-transformers for consistent results."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            f"sentence-transformers not available. Install with: pip install sentence-transformers"
        ) from e

    # Load model using sentence-transformers
    model = SentenceTransformer(model_name)

    model = model.half()
    print(f"INFO: Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}'...")
    # use acclerater GPU or MAC GPU

    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")

    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=64)

    return embeddings

# --- Core API Classes (Restored and Unchanged) ---

@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class PassageManager:
    def __init__(self, passage_sources: List[Dict[str, Any]]):
        self.offset_maps = {}
        self.passage_files = {}
        self.global_offset_map = {}  # Combined map for fast lookup
        
        for source in passage_sources:
            if source["type"] == "jsonl":
                passage_file = source["path"]
                index_file = source["index_path"]
                if not Path(index_file).exists():
                    raise FileNotFoundError(f"Passage index file not found: {index_file}")
                with open(index_file, 'rb') as f:
                    offset_map = pickle.load(f)
                    self.offset_maps[passage_file] = offset_map
                    self.passage_files[passage_file] = passage_file
                    
                    # Build global map for O(1) lookup
                    for passage_id, offset in offset_map.items():
                        self.global_offset_map[passage_id] = (passage_file, offset)

    def get_passage(self, passage_id: str) -> Dict[str, Any]:
        if passage_id in self.global_offset_map:
            passage_file, offset = self.global_offset_map[passage_id]
            with open(passage_file, 'r', encoding='utf-8') as f:
                f.seek(offset)
                return json.loads(f.readline())
        raise KeyError(f"Passage ID not found: {passage_id}")

class LeannBuilder:
    def __init__(self, backend_name: str, embedding_model: str = "facebook/contriever-msmarco", dimensions: Optional[int] = None, **backend_kwargs):
        self.backend_name = backend_name
        backend_factory: LeannBackendFactoryInterface | None = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found or not registered.")
        self.backend_factory = backend_factory
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.backend_kwargs = backend_kwargs
        self.chunks: List[Dict[str, Any]] = []

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        if metadata is None: metadata = {}
        passage_id = metadata.get('id', str(uuid.uuid4()))
        chunk_data = {"id": passage_id, "text": text, "metadata": metadata}
        self.chunks.append(chunk_data)

    def build_index(self, index_path: str):
        if not self.chunks: raise ValueError("No chunks added.")
        if self.dimensions is None: self.dimensions = len(compute_embeddings(["dummy"], self.embedding_model)[0])
        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        index_dir.mkdir(parents=True, exist_ok=True)
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        offset_map = {}
        with open(passages_file, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                offset = f.tell()
                json.dump({"id": chunk["id"], "text": chunk["text"], "metadata": chunk["metadata"]}, f, ensure_ascii=False)
                f.write('\n')
                offset_map[chunk["id"]] = offset
        with open(offset_file, 'wb') as f: pickle.dump(offset_map, f)
        texts_to_embed = [c["text"] for c in self.chunks]
        embeddings = compute_embeddings(texts_to_embed, self.embedding_model)
        string_ids = [chunk["id"] for chunk in self.chunks]
        current_backend_kwargs = {**self.backend_kwargs, 'dimensions': self.dimensions}
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        builder_instance.build(embeddings, string_ids, index_path, **current_backend_kwargs)
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        meta_data = {
            "version": "1.0", "backend_name": self.backend_name, "embedding_model": self.embedding_model,
            "dimensions": self.dimensions, "backend_kwargs": self.backend_kwargs,
            "passage_sources": [{"type": "jsonl", "path": str(passages_file), "index_path": str(offset_file)}]
        }
        
        # Add storage status flags for HNSW backend
        if self.backend_name == "hnsw":
            is_compact = self.backend_kwargs.get("is_compact", True)
            is_recompute = self.backend_kwargs.get("is_recompute", True)
            meta_data["is_compact"] = is_compact
            meta_data["is_pruned"] = is_compact and is_recompute  # Pruned only if compact and recompute
        with open(leann_meta_path, 'w', encoding='utf-8') as f: json.dump(meta_data, f, indent=2)

class LeannSearcher:
    def __init__(self, index_path: str, **backend_kwargs):
        meta_path_str = f"{index_path}.meta.json"
        if not Path(meta_path_str).exists(): raise FileNotFoundError(f"Leann metadata file not found at {meta_path_str}")
        with open(meta_path_str, 'r', encoding='utf-8') as f: self.meta_data = json.load(f)
        backend_name = self.meta_data['backend_name']
        self.embedding_model = self.meta_data['embedding_model']
        self.passage_manager = PassageManager(self.meta_data.get('passage_sources', []))
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None: raise ValueError(f"Backend '{backend_name}' not found.")
        final_kwargs = {**self.meta_data.get('backend_kwargs', {}), **backend_kwargs}
        self.backend_impl = backend_factory.searcher(index_path, **final_kwargs)

    def search(self, query: str, top_k: int = 5, **search_kwargs) -> List[SearchResult]:
        print(f"🔍 DEBUG LeannSearcher.search() called:")
        print(f"  Query: '{query}'")
        print(f"  Top_k: {top_k}")
        print(f"  Search kwargs: {search_kwargs}")
        
        query_embedding = compute_embeddings([query], self.embedding_model)
        print(f"  Generated embedding shape: {query_embedding.shape}")
        print(f"🔍 DEBUG Query embedding first 10 values: {query_embedding[0][:10]}")
        print(f"🔍 DEBUG Query embedding norm: {np.linalg.norm(query_embedding[0])}")
        
        results = self.backend_impl.search(query_embedding, top_k, **search_kwargs)
        print(f"  Backend returned: labels={len(results.get('labels', [[]])[0])} results")
        
        enriched_results = []
        if 'labels' in results and 'distances' in results:
            print(f"  Processing {len(results['labels'][0])} passage IDs:")
            for i, (string_id, dist) in enumerate(zip(results['labels'][0], results['distances'][0])):
                try:
                    passage_data = self.passage_manager.get_passage(string_id)
                    enriched_results.append(SearchResult(
                        id=string_id, score=dist, text=passage_data['text'], metadata=passage_data.get('metadata', {})
                    ))
                    print(f"    {i+1}. passage_id='{string_id}' -> SUCCESS: {passage_data['text'][:60]}...")
                except KeyError: 
                    print(f"    {i+1}. passage_id='{string_id}' -> ERROR: Passage not found in PassageManager!")
        
        print(f"  Final enriched results: {len(enriched_results)} passages")
        return enriched_results

from .chat import get_llm

class LeannChat:
    def __init__(self, index_path: str, llm_config: Optional[Dict[str, Any]] = None, **kwargs):
        self.searcher = LeannSearcher(index_path, **kwargs)
        self.llm = get_llm(llm_config)

    def ask(self, question: str, top_k=5, **kwargs):
        results = self.searcher.search(question, top_k=top_k, **kwargs)
        context = "\n\n".join([r.text for r in results])
        prompt = (
            "Here is some retrieved context that might help answer your question:\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Please provide the best answer you can based on this context and your knowledge."
        )
        return self.llm.ask(prompt, **kwargs.get("llm_kwargs", {}))

    def start_interactive(self):
        print("\nLeann Chat started (type 'quit' to exit)")
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                if not user_input:
                    continue
                response = self.ask(user_input)
                print(f"Leann: {response}")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break