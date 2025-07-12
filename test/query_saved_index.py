import os
from llama_index.core import VectorStoreIndex, StorageContext

def load_index(save_dir: str = "mail_index"):
    """
    Load the saved index from disk.
    
    Args:
        save_dir: Directory where the index is saved
    
    Returns:
        Loaded index or None if loading fails
    """
    try:
        # Load storage context
        storage_context = StorageContext.from_defaults(persist_dir=save_dir)
        
        # Load index
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context
        )
        
        print(f"Index loaded from {save_dir}")
        return index
    
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def query_index(index, query: str):
    """
    Query the loaded index.
    
    Args:
        index: The loaded index
        query: The query string
    """
    if index is None:
        print("No index available for querying.")
        return
    
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(f"\nQuery: {query}")
    print(f"Response: {response}")

def main():
    save_dir = "mail_index"
    
    # Check if index exists
    if not os.path.exists(save_dir) or not os.path.exists(os.path.join(save_dir, "vector_store.json")):
        print(f"Index not found in {save_dir}")
        print("Please run mail_reader_save_load.py first to create the index.")
        return
    
    # Load the index
    index = load_index(save_dir)
    
    if not index:
        print("Failed to load index.")
        return
    
    print("\n" + "="*60)
    print("Email Query Interface")
    print("="*60)
    print("Type 'quit' to exit")
    print("Type 'help' for example queries")
    print("="*60)
    
    # Interactive query loop
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nExample queries:")
                print("- Hows Berkeley Graduate Student Instructor")
                print("- What emails mention GSR appointments?")
                print("- Find emails about deadlines")
                print("- Search for emails from specific sender")
                print("- Find emails about meetings")
                continue
            elif not query:
                continue
            
            query_index(index, query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main() 