"""
ChatGPT RAG example. Indexes ChatGPT export data (chat.html / .zip files).
"""
from apps.chat_export_rag import ChatExportRAG
from apps.chatgpt_data.chatgpt_reader import ChatGPTReader


class ChatGPTRAG(ChatExportRAG):
    def __init__(self):
        super().__init__(
            name="ChatGPT",
            description="Process and query ChatGPT conversation exports with LEANN",
            default_index_name="chatgpt_conversations_index",
            reader_factory=lambda concat: ChatGPTReader(concatenate_conversations=concat),
            export_keyword="chatgpt",
            file_extensions=[".zip", ".html"],
            default_export_dir="./chatgpt_export",
            example_queries=[
                "What did I ask about Python programming?",
                "Show me conversations about machine learning",
                "Find discussions about travel planning",
                "What advice did ChatGPT give me about career development?",
                "Search for conversations about cooking recipes",
            ],
            export_setup_instructions=[
                "1. Sign in to ChatGPT",
                "2. Click on your profile icon → Settings → Data Controls",
                "3. Click 'Export' under Export Data",
                "4. Download the zip file from the email link",
                "5. Extract or place the file/directory at the specified path",
            ],
        )


if __name__ == "__main__":
    import asyncio

    print("\n🤖 ChatGPT RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    for q in ChatGPTRAG().example_queries:
        print(f"- '{q}'")
    print("\nTo get started:")
    print("1. Export your ChatGPT data from Settings → Data Controls → Export")
    print("2. Place the downloaded zip file or extracted HTML in ./chatgpt_export/")
    print("3. Run this script to build your personal ChatGPT knowledge base!")
    print("\nOr run without --query for interactive mode\n")

    rag = ChatGPTRAG()
    asyncio.run(rag.run())
