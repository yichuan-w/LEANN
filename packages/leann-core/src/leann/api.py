from .registry import BACKEND_REGISTRY
from .interface import LeannBackendFactoryInterface
from typing import List, Dict, Any, Optional
import numpy as np
import os
import json
from pathlib import Path
import openai # Import openai library

# 一个辅助函数，用于临时计算 embedding
def _compute_embeddings(chunks: List[str], model_name: str) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        print(f"INFO: Computing embeddings for {len(chunks)} chunks using '{model_name}'...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        return np.asarray(embeddings, dtype=np.float32)
    except ImportError:
        print("WARNING: sentence-transformers not installed. Falling back to random embeddings.")
        # 如果没有安装，则生成随机向量用于测试
        # TODO: 应该从一个固定的地方获取维度信息
        return np.random.rand(len(chunks), 768).astype(np.float32)


class LeannBuilder:
    """
    负责构建 Leann 索引的上层 API。
    它协调 embedding 计算和后端索引构建。
    """
    def __init__(self, backend_name: str, embedding_model: str = "sentence-transformers/all-mpnet-base-v2", **backend_kwargs):
        self.backend_name = backend_name
        self.backend_factory = BACKEND_REGISTRY.get(backend_name)
        if self.backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' not found or not registered.")
        
        self.embedding_model = embedding_model
        self.backend_kwargs = backend_kwargs
        self.chunks: List[Dict[str, Any]] = []
        print(f"INFO: LeannBuilder initialized with '{backend_name}' backend.")

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        # 简单的分块逻辑
        self.chunks.append({"text": text, "metadata": metadata or {}})

    def build_index(self, index_path: str):
        if not self.chunks:
            raise ValueError("No chunks added. Use add_text() first.")

        # 1. 计算 embedding (这是 leann-core 的职责)
        texts_to_embed = [c["text"] for c in self.chunks]
        embeddings = _compute_embeddings(texts_to_embed, self.embedding_model)

        # 2. 创建 builder 实例并构建索引
        builder_instance = self.backend_factory.builder(**self.backend_kwargs)
        builder_instance.build(embeddings, index_path, **self.backend_kwargs)

        # 3. 保存 leann 特有的元数据（不包含向量）
        index_dir = Path(index_path).parent
        leann_meta_path = index_dir / f"{Path(index_path).name}.meta.json"
        
        meta_data = {
            "version": "0.1.0",
            "backend_name": self.backend_name,
            "embedding_model": self.embedding_model,
            "num_chunks": len(self.chunks),
            "chunks": self.chunks,
        }
        with open(leann_meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2)
        print(f"INFO: Leann metadata saved to {leann_meta_path}")


class LeannSearcher:
    """
    负责加载索引并执行检索的上层 API。
    """
    def __init__(self, index_path: str, **backend_kwargs):
        leann_meta_path = Path(index_path).parent / f"{Path(index_path).name}.meta.json"
        if not leann_meta_path.exists():
            raise FileNotFoundError(f"Leann metadata file not found at {leann_meta_path}. Was the index built with LeannBuilder?")

        with open(leann_meta_path, 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)

        backend_name = self.meta_data['backend_name']
        self.embedding_model = self.meta_data['embedding_model']
        
        backend_factory = BACKEND_REGISTRY.get(backend_name)
        if backend_factory is None:
            raise ValueError(f"Backend '{backend_name}' (from index file) not found or not registered.")

        # 创建 searcher 实例
        self.backend_impl = backend_factory.searcher(index_path, **backend_kwargs)
        print(f"INFO: LeannSearcher initialized with '{backend_name}' backend using index '{index_path}'.")
    
    def search(self, query: str, top_k: int = 5, **search_kwargs):
        query_embedding = _compute_embeddings([query], self.embedding_model)
        
        # 委托给后端的 search 方法
        results = self.backend_impl.search(query_embedding, top_k, **search_kwargs)
        
        # 丰富返回结果，加入原始文本和元数据
        enriched_results = []
        for label, dist in zip(results['labels'][0], results['distances'][0]):
            if label < len(self.meta_data['chunks']):
                chunk_info = self.meta_data['chunks'][label]
                enriched_results.append({
                    "id": label,
                    "score": dist,
                    "text": chunk_info['text'],
                    "metadata": chunk_info['metadata']
                })
        return enriched_results


class LeannChat:
    """
    封装了 Searcher 和 LLM 的对话式 RAG 接口。
    """
    def __init__(self, index_path: str, backend_name: Optional[str] = None, llm_model: str = "gpt-4o", **kwargs):
        # 如果用户没有指定后端，尝试从索引元数据中读取
        if backend_name is None:
            leann_meta_path = Path(index_path).parent / f"{Path(index_path).name}.meta.json"
            if not leann_meta_path.exists():
                raise FileNotFoundError(f"Leann metadata file not found at {leann_meta_path}.")
            with open(leann_meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            backend_name = meta_data['backend_name']
        
        self.searcher = LeannSearcher(index_path, **kwargs)
        self.llm_model = llm_model
        self.openai_client = None # Lazy load

    def _get_openai_client(self):
        if self.openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self.openai_client = openai.OpenAI(api_key=api_key)
        return self.openai_client
        
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
        context = "\n\n".join([r['text'] for r in results])

        # 2. 构建 Prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # 3. 调用 LLM
        print(f"DEBUG: Calling LLM with prompt: {prompt}...")
        try:
            client = self._get_openai_client()
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
