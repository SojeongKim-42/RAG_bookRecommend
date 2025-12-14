"""
Main entry point for RAG Book Recommendation System.
"""

import argparse
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_agent import RAGAgent
from utils import print_statistics


def setup_system() -> tuple:
    """
    Set up the RAG system components.

    Returns:
        Tuple of (vectorstore_manager, document_processor, chunks, original_docs)
    """
    # Setup environment
    Config.setup_environment()

    # Initialize components
    doc_processor = DocumentProcessor()
    vectorstore_manager = VectorStoreManager()

    # Process documents or load existing vector store
    if vectorstore_manager.exists():
        print("Loading existing vector store and documents...")
        vectorstore_manager.load_vectorstore()
        chunks, original_docs = doc_processor.process()
    else:
        print("Creating new vector store...")
        chunks, original_docs = doc_processor.process()
        vectorstore_manager.create_vectorstore(chunks, save=True)

    # Print statistics
    print_statistics(original_docs, chunks)

    return vectorstore_manager, doc_processor, chunks, original_docs


def run_single_query(query: str, vectorstore_manager: VectorStoreManager) -> None:
    """
    Run a single query and display results.

    Args:
        query: User query
        vectorstore_manager: Initialized VectorStoreManager
    """
    print(f"\n=== Processing Query ===")
    print(f"Query: {query}\n")

    # Create and run agent
    agent = RAGAgent(vectorstore_manager)
    response = agent.query(query, verbose=True)

    # Display response
    response_text = agent.get_response_text(response)
    print("\n=== Agent Response ===")
    print(response_text)
    print("\n" + "=" * 80)


def run_interactive_mode(vectorstore_manager: VectorStoreManager) -> None:
    """
    Run interactive mode for continuous queries.

    Args:
        vectorstore_manager: Initialized VectorStoreManager
    """
    agent = RAGAgent(vectorstore_manager)
    agent.interactive_mode()


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Book Recommendation System")
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (if not provided, runs in interactive mode)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild of vector store"
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Only show statistics and exit"
    )
    parser.add_argument(
        "--test_vector_store",
        action="store_true",
        help="Test vector store functionality",
    )

    args = parser.parse_args()

    try:
        # Handle rebuild flag
        if args.rebuild:
            import shutil

            if Config.VECTOR_STORE_DIR.exists():
                print(f"Removing existing vector store at {Config.VECTOR_STORE_DIR}")
                shutil.rmtree(Config.VECTOR_STORE_DIR)

        # Setup system
        vectorstore_manager, doc_processor, chunks, original_docs = setup_system()

        # Exit if only showing stats
        if args.stats_only:
            return

        # Run appropriate mode
        if args.query:
            run_single_query(args.query, vectorstore_manager)
        elif args.test_vector_store:
            vectorstore_manager.test_vector_store()
        else:
            run_interactive_mode(vectorstore_manager)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()
