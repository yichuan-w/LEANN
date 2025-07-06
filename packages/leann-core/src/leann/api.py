from .registry import BACKEND_REGISTRY
from .interface import LeannBackendFactoryInterface
from typing import List, Dict, Any, Optional
import numpy as np
import os
import json
from pathlib import Path
import openai
from dataclasses import dataclass, field
import uuid
import pickle

# --- Helper Functions for Embeddings ---

def _get_openai_client():
    """Initializes and returns an OpenAI client, ensuring the API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set, which is required for OpenAI models.")
    return openai.OpenAI(api_key=api_key)

def _is_openai_model(model_name: str) -> bool:
    """Checks if the model is likely an OpenAI embedding model."""
    # This is a simple check, can be improved with a more robust list.
    return "ada" in model_name or "babbage" in model_name or model_name.startswith("text-embedding-")

def _compute_embeddings(chunks: List[str], model_name: str) -> np.ndarray:
    """Computes embeddings for a list of text chunks using either SentenceTransformers or OpenAI."""
    if _is_openai_model(model_name):
        print(f"INFO: Computing embeddings for {len(chunks)} chunks using OpenAI model '{model_name}'...")
        client = _get_openai_client()
        response = client.embeddings.create(model=model_name, input=chunks)
        embeddings = [item.embedding for item in response.data]
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        print(f"INFO: Computing embeddings for {len(chunks)} chunks using SentenceTransformer model '{model_name}'...")
        embeddings = model.encode(chunks, show_progress_bar=True)
    
    return np.asarray(embeddings, dtype=np.float32)

def _get_embedding_dimensions(model_name: str) -> int:
    """Gets the embedding dimensions for a given model."""
    print(f"INFO: Calculating dimensions for model '{model_name}'...")
    if _is_openai_model(model_name):
        client = _get_openai_client()
        response = client.embeddings.create(model=model_name, input=["dummy text"])
        return len(response.data[0].embedding)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Model '{model_name}' does not have a valid embedding dimension.")
        return dimension


@dataclass
class SearchResult:
    """Represents a single search result."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PassageManager:
    """Manages passage data and lazy loading from JSONL files."""
    
    def __init__(self, passage_sources: List[Dict[str, Any]]):
        self.offset_maps = {}
        self.passage_files = {}
        
        for source in passage_sources:
            if source["type"] == "jsonl":
                passage_file = source["path"]
                index_file = source["index_path"]
                
                if not os.path.exists(index_file):
                    raise FileNotFoundError(f"Passage index file not found: {index_file}")
                
                with open(index_file, 'rb') as f:
                    offset_map = pickle.load(f)
                
                self.offset_maps[passage_file] = offset_map
                self.passage_files[passage_file] = passage_file
    
    def get_passage(self, passage_id: str) -> Dict[str, Any]:
        """Lazy load a passage by ID."""
        for passage_file, offset_map in self.offset_maps.items():
            if passage_id in offset_map:
                offset = offset_map[passage_id]
                with open(passage_file, 'r', encoding='utf-8') as f:
                    f.seek(offset)
                    line = f.readline()
                    return json.loads(line)
        
        raise KeyError(f"Passage ID not found: {passage_id}")

# --- Core Classes ---

class LeannBuilder:
    """
    The builder is responsible for building the index, it will compute the embeddings and then build the index.
    It will also save the metadata of the index.
    """
    def __init__(self, backend_name: str, embedding_model: str = "sentence-transformers/all-mpnet-base-v2", dimensions: Optional[int] = None, **backend_kwargs):
        self.backend_name = backend_name
        backend_factory: LeannBackendFactoryInterface | None = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found or not registered.")
        self.backend_factory = backend_factory

        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.backend_kwargs = backend_kwargs
        self.chunks: List[Dict[str, Any]] = []
        print(f"INFO: LeannBuilder initialized with '{backend_name}' backend.")

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        if metadata is None:
            metadata = {}
        
        # Check if ID is provided in metadata
        passage_id = metadata.get('id')
        if passage_id is None:
            passage_id = str(uuid.uuid4())
        else:
            # Validate uniqueness
            existing_ids = {chunk['id'] for chunk in self.chunks}
            if passage_id in existing_ids:
                raise ValueError(f"Duplicate passage ID: {passage_id}")
        
        # Store the definitive ID with the chunk
        chunk_data = {
            "id": passage_id,
            "text": text,
            "metadata": metadata
        }
        self.chunks.append(chunk_data)

    def build_index(self, index_path: str):
        if not self.chunks:
            raise ValueError("No chunks added. Use add_text() first.")

        if self.dimensions is None:
            self.dimensions = _get_embedding_dimensions(self.embedding_model)
            print(f"INFO: Auto-detected dimensions for '{self.embedding_model}': {self.dimensions}")

        path = Path(index_path)
        index_dir = path.parent
        index_name = path.name
        
        # Ensure the directory exists
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the passages.jsonl file and offset index
        passages_file = index_dir / f"{index_name}.passages.jsonl"
        offset_file = index_dir / f"{index_name}.passages.idx"
        
        offset_map = {}
        
        with open(passages_file, 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                offset = f.tell()
                passage_data = {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
                json.dump(passage_data, f, ensure_ascii=False)
                f.write('\n')
                offset_map[chunk["id"]] = offset
        
        # Save the offset map
        with open(offset_file, 'wb') as f:
            pickle.dump(offset_map, f)
        
        # Compute embeddings
        texts_to_embed = [c["text"] for c in self.chunks]
        embeddings = _compute_embeddings(texts_to_embed, self.embedding_model)
        
        # Extract string IDs for the backend
        string_ids = [chunk["id"] for chunk in self.chunks]
        
        # Build the vector index
        current_backend_kwargs = self.backend_kwargs.copy()
        current_backend_kwargs['dimensions'] = self.dimensions
        builder_instance = self.backend_factory.builder(**current_backend_kwargs)
        
        builder_instance.build(embeddings, string_ids, index_path, **current_backend_kwargs)

        # Create the lightweight meta.json file
        leann_meta_path = index_dir / f"{index_name}.meta.json"
        
        meta_data = {
            "version": "1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "backend_kwargs": self.backend_kwargs,
            "passage_sources": [
                {
                    "type": "jsonl",
                    "path": str(passages_file),
                    "index_path": str(offset_file)
                }
            ]
        }
        with open(leann_meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)
        print(f"INFO: Leann metadata saved to {leann_meta_path}")


class LeannSearcher:
    """
    The searcher is responsible for loading the index and performing the search.
    It will also load the metadata of the index.
    """
    def __init__(self, index_path: str, **backend_kwargs):
        leann_meta_path = Path(index_path).parent / f"{Path(index_path).name}.meta.json"
        if not leann_meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {leann_meta_path}. Was the index built with LeannBuilder?")

        with open(leann_meta_path, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)

        backend_name = self.meta_data['backend_name']
        self.embedding_model = self.meta_data['embedding_model']
        
        # Initialize the passage manager
        passage_sources = self.meta_data.get('passage_sources', [])
        self.passage_manager = PassageManager(passage_sources)
        
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' (from index file) not found or not registered.")

        final_kwargs = backend_kwargs.copy()
        final_kwargs['meta'] = self.meta_data

        self.backend_impl = backend_factory.searcher(index_path, **final_kwargs)
        print(f"INFO: LeannSearcher initialized with '{backend_name}' backend using index '{index_path}'.")
    
    def search(self, query: str, top_k: int = 5, **search_kwargs):
        query_embedding = _compute_embeddings([query], self.embedding_model)
        
        search_kwargs['embedding_model'] = self.embedding_model
        results = self.backend_impl.search(query_embedding, top_k, **search_kwargs)
        
        enriched_results = []
        for string_id, dist in zip(results['labels'][0], results['distances'][0]):
            try:
                passage_data = self.passage_manager.get_passage(string_id)
                enriched_results.append(SearchResult(
                    id=string_id,
                    score=dist,
                    text=passage_data['text'],
                    metadata=passage_data.get('metadata', {})
                ))
            except KeyError:
                print(f"WARNING: Passage ID '{string_id}' not found in passage files")
        return enriched_results


class LeannChat:
    """
    The chat is responsible for the conversation with the LLM.
    It will use the searcher to get the results and then use the LLM to generate the response.
    """
    def __init__(self, index_path: str, backend_name: Optional[str] = None, llm_model: str = "gpt-4o", **kwargs):
        if backend_name is None:
            leann_meta_path = Path(index_path).parent / f"{Path(index_path).name}.meta.json"
            if not leann_meta_path.exists():
                raise FileNotFoundError(f"Leann metadata file not found at {leann_meta_path}.")
            with open(leann_meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            backend_name = meta_data['backend_name']
        
        self.searcher = LeannSearcher(index_path, **kwargs)
        self.llm_model = llm_model
        
    def ask(self, question: str, top_k=5, **kwargs):
        """
        Additional keyword arguments (kwargs) for advanced search customization. Example usage:
            chat.ask(
                "What is ANN?",
                top_k=10,
                complexity=64,
                beam_width=8,
                USE_DEFERRED_FETCH=True,
                skip_search_reorder=True,
                recompute_beighbor_embeddings=True,
                dedup_node_dis=True,
                prune_ratio=0.1,
                batch_recompute=True,
                global_pruning=True
            )
        
        Supported kwargs:
            - complexity (int): Search complexity parameter (default: 32)
            - beam_width (int): Beam width for search (default: 4)
            - USE_DEFERRED_FETCH (bool): Enable deferred fetch mode (default: False)
            - skip_search_reorder (bool): Skip search reorder step (default: False)
            - recompute_beighbor_embeddings (bool): Enable ZMQ embedding server for neighbor recomputation (default: False)
            - dedup_node_dis (bool): Deduplicate nodes by distance (default: False)
            - prune_ratio (float): Pruning ratio for search (default: 0.0)
            - batch_recompute (bool): Enable batch recomputation (default: False)
            - global_pruning (bool): Enable global pruning (default: False)
        """

        results = self.searcher.search(question, top_k=top_k, **kwargs)
        context = "\n\n".join([r.text for r in results])

        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        print(f"DEBUG: Calling LLM with prompt: {prompt}...")
        try:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"ERROR: Failed to call OpenAI API: {e}")
            return f"Error: Could not get a response from the LLM. {e}"
    
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
