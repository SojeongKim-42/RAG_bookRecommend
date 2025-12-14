"""
Vector store module for managing document embeddings and similarity search.
"""

import os
from typing import List, Optional
from pathlib import Path

import torch
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config import Config


class VectorStoreManager:
    """Manages FAISS vector store operations including creation, saving, and loading."""

    def __init__(self, model_name: str = None, store_path: str = None):
        """
        Initialize VectorStoreManager.

        Args:
            model_name: Name of the HuggingFace embedding model
            store_path: Path to save/load the vector store
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.store_path = store_path or str(Config.VECTOR_STORE_DIR)
        self.embeddings = None
        self.vectorstore = None

    def _get_device_config(self) -> dict:
        """
        Determine the best available device for embeddings.

        Returns:
            Dictionary with device configuration
        """
        if torch.cuda.is_available():
            print("Using CUDA device")
            return {"device": "cuda"}
        elif torch.backends.mps.is_available():
            print("Using MPS device")
            return {"device": "mps"}
        else:
            print("Using CPU device")
            return {"device": "cpu"}

    def initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initialize HuggingFace embeddings model.

        Returns:
            Initialized embeddings model
        """
        if self.embeddings is None:
            print(f"Initializing embeddings model: {self.model_name}")
            model_kwargs = self._get_device_config()

            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name, model_kwargs=model_kwargs
            )

        return self.embeddings

    def create_vectorstore(self, documents: List[Document], save: bool = True) -> FAISS:
        """
        Create a new FAISS vector store from documents.

        Args:
            documents: List of documents to index
            save: Whether to save the vector store to disk

        Returns:
            FAISS vector store
        """
        print("Creating FAISS vector store...")

        if self.embeddings is None:
            self.initialize_embeddings()

        self.vectorstore = FAISS.from_documents(
            documents=documents, embedding=self.embeddings
        )

        if save:
            self.save_vectorstore()

        print(f"Vector store created with {len(documents)} documents")
        return self.vectorstore

    def save_vectorstore(self, path: str = None) -> None:
        """
        Save vector store to disk.

        Args:
            path: Optional custom path to save the vector store
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save. Create or load one first.")

        save_path = path or self.store_path
        print(f"Saving vector store to: {save_path}")

        self.vectorstore.save_local(save_path)
        print("Vector store saved successfully")

    def load_vectorstore(self, path: str = None) -> FAISS:
        """
        Load vector store from disk.

        Args:
            path: Optional custom path to load the vector store from

        Returns:
            Loaded FAISS vector store

        Raises:
            FileNotFoundError: If the vector store doesn't exist
        """
        load_path = path or self.store_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at: {load_path}")

        print(f"Loading existing vector store from: {load_path}")

        if self.embeddings is None:
            self.initialize_embeddings()

        self.vectorstore = FAISS.load_local(
            load_path, self.embeddings, allow_dangerous_deserialization=True
        )

        print("Vector store loaded successfully")
        return self.vectorstore

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of documents to return

        Returns:
            List of similar documents

        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load one first.")

        k = k or Config.DEFAULT_K
        print(f"Searching for {k} similar documents...")

        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def exists(self, path: str = None) -> bool:
        """
        Check if a vector store exists at the specified path.

        Args:
            path: Optional custom path to check

        Returns:
            True if vector store exists, False otherwise
        """
        check_path = path or self.store_path
        return os.path.exists(check_path)

    def get_or_create_vectorstore(self, documents: List[Document] = None) -> FAISS:
        """
        Load existing vector store or create a new one if it doesn't exist.

        Args:
            documents: Documents to use for creating a new vector store

        Returns:
            FAISS vector store

        Raises:
            ValueError: If vector store doesn't exist and no documents provided
        """
        if self.exists():
            return self.load_vectorstore()
        else:
            if documents is None:
                raise ValueError(
                    "Vector store doesn't exist and no documents provided for creation"
                )
            return self.create_vectorstore(documents, save=True)

    def test_vector_store(self) -> None:
        """
        Test vector store functionality with a sample query.
        """
        sample_query = input("Enter a sample query to test the vector store: ")
        print(f"Testing vector store with query: '{sample_query}'")

        results = self.similarity_search(sample_query, k=Config.DEFAULT_K)
        if results:
            print("Test successful. Retrieved document:")
            print(results[0].page_content)
        else:
            print("Test failed. No documents retrieved.")
