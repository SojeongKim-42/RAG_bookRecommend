"""
Document processing module for loading and chunking documents.
"""

import pandas as pd
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config


class DocumentProcessor:
    """Handles document loading and processing from CSV files."""

    def __init__(
        self,
        file_path: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        content_columns: List[str] = None,
    ):
        """
        Initialize DocumentProcessor.

        Args:
            file_path: Path to the CSV file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            content_columns: Columns to use as content
        """
        self.file_path = file_path or str(Config.DATA_FILE)
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.content_columns = content_columns or Config.CONTENT_COLUMNS

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def load_csv(self) -> List[Document]:
        """
        Load documents from CSV file.

        Returns:
            List of Document objects

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        print(f"Loading data from CSV: {self.file_path}")

        loader = CSVLoader(
            file_path=self.file_path,
            encoding=Config.CSV_ENCODING,
            csv_args={"delimiter": Config.CSV_DELIMITER},
            content_columns=self.content_columns,
        )

        try:
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")
            return documents
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of chunked documents
        """
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def enrich_metadata(
        self, chunks: List[Document], original_documents: List[Document]
    ) -> List[Document]:
        """
        Enrich chunk metadata with original document information.

        Args:
            chunks: List of document chunks
            original_documents: Original documents before chunking

        Returns:
            Chunks with enriched metadata
        """
        print("Enriching chunk metadata...")

        # Load CSV to get all metadata directly
        try:
            df = pd.read_csv(self.file_path)
            rank_column = Config.RANK_COLUMN
            has_csv = True
        except Exception as e:
            print(f"Warning: Could not load CSV for metadata: {e}")
            has_csv = False
            df = None

        for i, chunk in enumerate(chunks):
            row_num = chunk.metadata.get("row")
            if row_num is None:
                continue

            # Get metadata directly from CSV
            if has_csv and df is not None and row_num < len(df):
                try:
                    csv_row = df.iloc[row_num]

                    # Always add these fields from CSV if they exist
                    if "구분" in df.columns:
                        chunk.metadata["구분"] = str(csv_row["구분"])

                    if "상품명" in df.columns:
                        chunk.metadata["상품명"] = str(csv_row["상품명"])

                    # Add rank if column exists
                    if rank_column in df.columns:
                        chunk.metadata[rank_column] = csv_row[rank_column]

                except Exception as e:
                    print(f"Warning: Could not extract metadata for row {row_num}: {e}")

                    # Fallback: try to extract from page_content
                    if row_num < len(original_documents):
                        original_doc = original_documents[row_num]
                        content_lines = original_doc.page_content.split("\n")
                        for line in content_lines:
                            if (
                                line.startswith("구분:")
                                and "구분" not in chunk.metadata
                            ):
                                chunk.metadata["구분"] = line.replace(
                                    "구분:", ""
                                ).strip()
                            elif (
                                line.startswith("상품명:")
                                and "상품명" not in chunk.metadata
                            ):
                                chunk.metadata["상품명"] = line.replace(
                                    "상품명:", ""
                                ).strip()
            else:
                # Fallback: extract from page_content if CSV not available
                if row_num < len(original_documents):
                    original_doc = original_documents[row_num]
                    content_lines = original_doc.page_content.split("\n")
                    for line in content_lines:
                        if line.startswith("구분:"):
                            chunk.metadata["구분"] = line.replace("구분:", "").strip()
                        elif line.startswith("상품명:"):
                            chunk.metadata["상품명"] = line.replace(
                                "상품명:", ""
                            ).strip()

            # Add chunk index
            chunk.metadata["chunk_id"] = i

        return chunks

    def process(self) -> Tuple[List[Document], List[Document]]:
        """
        Complete document processing pipeline.

        Returns:
            Tuple of (chunks, original_documents)
        """
        # Load documents
        documents = self.load_csv()

        # Split into chunks
        chunks = self.split_documents(documents)

        # Enrich metadata
        enriched_chunks = self.enrich_metadata(chunks, documents)

        return enriched_chunks, documents
