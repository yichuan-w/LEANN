import os

from leann.api import LeannBuilder, LeannChat

# Define the path for our new MLX-based index
INDEX_PATH = "./mlx_diskann_index/leann"

if os.path.exists(INDEX_PATH + ".meta.json"):
    print(f"Index already exists at {INDEX_PATH}. Skipping build.")
else:
    print("Initializing LeannBuilder with MLX support...")
    # 1. Configure LeannBuilder to use MLX
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        embedding_mode="mlx",
    )

    # 2. Add documents
    print("Adding documents...")
    docs = [
        "MLX is an array framework for machine learning on Apple silicon.",
        "It was designed by Apple's machine learning research team.",
        "The mlx-community organization provides pre-trained models in MLX format.",
        "It supports operations on multi-dimensional arrays.",
        "Leann can now use MLX for its embedding models.",
    ]
    for doc in docs:
        builder.add_text(doc)

    # 3. Build the index
    print(f"Building the MLX-based index at: {INDEX_PATH}")
    builder.build_index(INDEX_PATH)
    print("\nSuccessfully built the index with MLX embeddings!")
    print(f"Check the metadata file: {INDEX_PATH}.meta.json")


chat = LeannChat(index_path=INDEX_PATH)
# add query
query = "MLX is an array framework for machine learning on Apple silicon."
print(f"Query: {query}")
response = chat.ask(query, top_k=3, recompute_beighbor_embeddings=True, complexity=3, beam_width=1)
print(f"Response: {response}")
