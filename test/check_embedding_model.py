from llama_index.core import VectorStoreIndex, Document
from llama_index.core.embeddings import resolve_embed_model

# Check the default embedding model
embed_model = resolve_embed_model("default")
print(f"Default embedding model: {embed_model}")

# Create a simple test
doc = Document(text="This is a test document")
index = VectorStoreIndex.from_documents([doc])

# Get the embedding model from the index
index_embed_model = index.embed_model
print(f"Index embedding model: {index_embed_model}")

# Check if it's OpenAI or local
if hasattr(index_embed_model, 'model_name'):
    print(f"Model name: {index_embed_model.model_name}")
else:
    print(f"Embedding model type: {type(index_embed_model)}") 