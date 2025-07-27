import asyncio
from pathlib import Path

from leann.api import LeannChat

INDEX_DIR = Path("./test_pdf_index_huawei")
INDEX_PATH = str(INDEX_DIR / "pdf_documents.leann")


async def main():
    print("\n[PHASE 2] Starting Leann chat session...")
    chat = LeannChat(index_path=INDEX_PATH)
    query = "What is the main idea of RL and give me 5 exapmle of classic RL algorithms?"
    query = "Based on the paper, what are the main techniques LEANN explores to reduce the storage overhead and DLPM explore to achieve Fairness and Efiiciency trade-off?"
    # query = "什么是盘古大模型以及盘古开发过程中遇到了什么阴暗面,任务令一般在什么城市颁发"
    response = chat.ask(
        query, top_k=20, recompute_beighbor_embeddings=True, complexity=32, beam_width=1
    )
    print(f"\n[PHASE 2] Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
