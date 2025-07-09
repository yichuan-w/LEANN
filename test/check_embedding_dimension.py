from llama_index.core import VectorStoreIndex, Document
from llama_index.core.embeddings import resolve_embed_model

# Check the default embedding model
embed_model = resolve_embed_model("default")
print(f"Default embedding model: {embed_model}")

# Create a simple test document
doc = Document(text="This is a test document")

# Get embedding dimension
try:
    # Test embedding
    test_embedding = embed_model.get_text_embedding("test")
    print(f"Embedding dimension: {len(test_embedding)}")
    print(f"Embedding type: {type(test_embedding)}")
except Exception as e:
    print(f"Error getting embedding: {e}")

# Alternative way to check dimension
if hasattr(embed_model, 'embed_dim'):
    print(f"Model embed_dim attribute: {embed_model.embed_dim}")
elif hasattr(embed_model, 'dimension'):
    print(f"Model dimension attribute: {embed_model.dimension}") 