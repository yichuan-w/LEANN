"""
Claude RAG example. Indexes Claude conversation export data (.json / .zip files).
"""
from apps.chat_export_rag import ChatExportRAG
from apps.claude_data.claude_reader import ClaudeReader


class ClaudeRAG(ChatExportRAG):
    def __init__(self):
        super().__init__(
            name="Claude",
            description="Process and query Claude conversation exports with LEANN",
            default_index_name="claude_conversations_index",
            reader_factory=lambda concat: ClaudeReader(concatenate_conversations=concat),
            export_keyword="claude",
            file_extensions=[".zip", ".json"],
            default_export_dir="./claude_export",
            example_queries=[
                "What did I ask Claude about Python programming?",
                "Show me conversations about machine learning",
                "Find discussions about code optimization",
                "What advice did Claude give me about software design?",
                "Search for conversations about debugging techniques",
            ],
            export_setup_instructions=[
                "1. Open Claude in your browser",
                "2. Look for export/download options in settings or conversation menu",
                "3. Download the conversation data (usually in JSON format)",
                "4. Place the file/directory at the specified path",
                "",
                "Note: Claude export methods may vary. Check Claude's help documentation for current instructions.",
            ],
        )


if __name__ == "__main__":
    import asyncio

    print("\n🤖 Claude RAG Example")
    print("=" * 50)
    print("\nExample queries you can try:")
    for q in ClaudeRAG().example_queries:
        print(f"- '{q}'")
    print("\nTo get started:")
    print("1. Export your Claude conversation data")
    print("2. Place the JSON/ZIP file in ./claude_export/")
    print("3. Run this script to build your personal Claude knowledge base!")
    print("\nOr run without --query for interactive mode\n")

    rag = ClaudeRAG()
    asyncio.run(rag.run())
