"""
Utility functions for RAG book recommendation system.
"""

from typing import List, Dict, Set
from langchain_core.documents import Document


class BookDeduplicator:
    """Handles deduplication of book recommendations."""

    @staticmethod
    def deduplicate_by_row(
        documents: List[Document],
        max_books: int = 5
    ) -> List[Dict]:
        """
        Deduplicate documents by row number to get unique books.

        Args:
            documents: List of retrieved documents
            max_books: Maximum number of unique books to return

        Returns:
            List of unique book dictionaries
        """
        seen_rows: Set[int] = set()
        unique_books: List[Dict] = []

        for doc in documents:
            row = doc.metadata.get("row")

            if row is not None and row not in seen_rows:
                seen_rows.add(row)

                book_info = {
                    "row": row,
                    "구분": doc.metadata.get("구분", "N/A"),
                    "상품명": doc.metadata.get("상품명", "N/A"),
                    "matched_content": doc.page_content,
                    "full_metadata": doc.metadata
                }

                unique_books.append(book_info)

                if len(unique_books) >= max_books:
                    break

        return unique_books

    @staticmethod
    def format_book_info(book: Dict, index: int = None) -> str:
        """
        Format book information for display.

        Args:
            book: Book information dictionary
            index: Optional index number for display

        Returns:
            Formatted string
        """
        prefix = f"{index}. " if index is not None else ""

        return (
            f"{prefix}[{book['구분']}] {book['상품명']}\n"
            f"   Row: {book['row']}\n"
            f"   매칭 내용: {book['matched_content'][:100]}...\n"
        )


class ResultFormatter:
    """Formats various types of results for display."""

    @staticmethod
    def format_document_list(
        documents: List[Document],
        max_content_length: int = 100
    ) -> str:
        """
        Format a list of documents for display.

        Args:
            documents: List of documents
            max_content_length: Maximum length of content to display

        Returns:
            Formatted string
        """
        if not documents:
            return "No documents found."

        output = [f"Found {len(documents)} documents:\n"]

        for i, doc in enumerate(documents, 1):
            content_preview = doc.page_content[:max_content_length]
            if len(doc.page_content) > max_content_length:
                content_preview += "..."

            output.append(f"{i}. {content_preview}")
            output.append(f"   Metadata: {doc.metadata}\n")

        return "\n".join(output)

    @staticmethod
    def format_search_results(
        documents: List[Document],
        query: str,
        show_full_content: bool = False
    ) -> str:
        """
        Format search results with query context.

        Args:
            documents: Retrieved documents
            query: Original search query
            show_full_content: Whether to show full content

        Returns:
            Formatted string
        """
        deduplicator = BookDeduplicator()
        unique_books = deduplicator.deduplicate_by_row(documents)

        output = [
            f"=== Search Results for: '{query}' ===",
            f"Found {len(unique_books)} unique books:\n"
        ]

        for i, book in enumerate(unique_books, 1):
            formatted = deduplicator.format_book_info(book, index=i)
            output.append(formatted)

            if show_full_content:
                output.append(f"   Full content: {book['matched_content']}\n")

        return "\n".join(output)


def validate_documents(documents: List[Document]) -> bool:
    """
    Validate that documents have required metadata.

    Args:
        documents: List of documents to validate

    Returns:
        True if valid, False otherwise
    """
    if not documents:
        return False

    required_fields = ["row"]

    for doc in documents:
        for field in required_fields:
            if field not in doc.metadata:
                print(f"Warning: Document missing required field: {field}")
                return False

    return True


def print_statistics(documents: List[Document], chunks: List[Document]) -> None:
    """
    Print statistics about documents and chunks.

    Args:
        documents: Original documents
        chunks: Document chunks
    """
    print("\n=== Document Statistics ===")
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Average chunks per document: {len(chunks) / len(documents):.2f}")

    # Count unique categories
    categories = set()
    for doc in documents:
        content = doc.page_content
        for line in content.split("\n"):
            if line.startswith("구분:"):
                categories.add(line.replace("구분:", "").strip())

    print(f"Unique categories: {len(categories)}")
    print(f"Categories: {', '.join(sorted(categories))}")
