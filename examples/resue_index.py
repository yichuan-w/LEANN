import asyncio
from leann.api import LeannChat
from pathlib import Path

INDEX_DIR = Path("./test_pdf_index")
INDEX_PATH = str(INDEX_DIR / "pdf_documents.leann")

async def main():
    print(f"\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=INDEX_PATH)
    query = "What is the main idea of RL and give me 5 exapmle of classic RL algorithms?"
    query = "Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?"

    response = chat.ask(query,top_k=20,recompute_beighbor_embeddings=True,complexity=64,beam_width=1)
    print(f"\n[PHASE 2] Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())