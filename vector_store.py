"""
Vector store module for managing document embeddings and similarity search.
Uses FAISS with advanced retrieval features:
- MMR (Maximal Marginal Relevance) for diversity
- Metadata filtering
- Bestseller rank-based reranking
- Adaptive Top-k strategy
"""

import os
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import torch
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from config import Config


class VectorStoreManager:
    """
    Manages FAISS vector store operations with advanced retrieval features.

    Features:
    - MMR search for diversity
    - Metadata filtering
    - Bestseller rank-based reranking
    - Adaptive Top-k strategy
    """

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
        Save FAISS vector store to disk.

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
        Load FAISS vector store from disk.

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

    def similarity_search(
        self, query: str, k: int = None, filter: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Search for similar documents with optional metadata filtering.

        Args:
            query: Search query
            k: Number of documents to return
            filter: Optional metadata filter (e.g., {"구분": "소설"})

        Returns:
            List of similar documents

        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load one first.")

        k = k or Config.DEFAULT_K
        print(f"Searching for {k} similar documents...")

        if filter:
            print(f"Applying metadata filter: {filter}")
            results = self.vectorstore.similarity_search(query, k=k, filter=filter)
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        return results

    def similarity_search_with_score(
        self, query: str, k: int = None, filter: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.

        Args:
            query: Search query
            k: Number of documents to return
            filter: Optional metadata filter

        Returns:
            List of (document, score) tuples

        Raises:
            ValueError: If vector store is not initialized
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load one first.")

        k = k or Config.DEFAULT_K

        if filter:
            results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter
            )
        else:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

        return results

    def mmr_search(
        self,
        query: str,
        k: int = None,
        fetch_k: int = None,
        lambda_mult: float = None,
        filter: Dict[str, Any] = None,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search for diversity.

        MMR optimizes for both relevance and diversity by:
        1. Fetching more candidates (fetch_k) than needed
        2. Iteratively selecting documents that are relevant but diverse

        Args:
            query: Search query
            k: Number of documents to return
            fetch_k: Number of documents to fetch before MMR filtering
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Optional metadata filter

        Returns:
            List of diverse similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Create or load one first.")

        k = k or Config.DEFAULT_K
        fetch_k = fetch_k or Config.MMR_FETCH_K
        lambda_mult = lambda_mult if lambda_mult is not None else Config.MMR_LAMBDA

        print(f"MMR search: k={k}, fetch_k={fetch_k}, lambda={lambda_mult}")

        try:
            if filter:
                results = self.vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
                )
            else:
                results = self.vectorstore.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
            return results
        except Exception as e:
            print(f"MMR search failed, falling back to similarity search: {str(e)}")
            return self.similarity_search(query, k=k, filter=filter)

    def rerank_by_bestseller(
        self,
        results: List[Tuple[Document, float]],
        alpha: float = None,
        beta: float = None,
        rank_column: str = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank search results using bestseller rank.

        Final score = alpha * similarity_score + beta * (1 / rank)

        Args:
            results: List of (document, similarity_score) tuples
            alpha: Weight for similarity score (default from config)
            beta: Weight for rank score (default from config)
            rank_column: Metadata column name for bestseller rank

        Returns:
            Reranked list of (document, new_score) tuples
        """
        alpha = alpha if alpha is not None else Config.RANK_ALPHA
        beta = beta if beta is not None else Config.RANK_BETA
        rank_column = rank_column or Config.RANK_COLUMN

        print(f"Reranking with bestseller: alpha={alpha}, beta={beta}")

        reranked = []
        for doc, sim_score in results:
            # Extract rank from metadata
            rank = doc.metadata.get(rank_column)

            # Calculate rank score
            if rank is not None:
                try:
                    rank_value = float(rank)
                    # Normalize: higher rank (lower number) = higher score
                    rank_score = 1.0 / (rank_value + 1)  # +1 to avoid division by zero
                except (ValueError, TypeError):
                    rank_score = 0.0
            else:
                rank_score = 0.0

            # Calculate final score
            final_score = alpha * sim_score + beta * rank_score
            reranked.append((doc, final_score))

        # Sort by final score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked

    def adaptive_top_k(
        self, query: str, min_k: int = None, max_k: int = None, threshold: float = None
    ) -> int:
        """
        Determine optimal k based on query characteristics and similarity distribution.

        Strategy:
        1. Query length: longer queries may need more context
        2. Query ambiguity: keywords like "추천", "비슷한" increase k
        3. Similarity distribution: if scores drop sharply, use fewer results

        Args:
            query: Search query
            min_k: Minimum k value
            max_k: Maximum k value
            threshold: Similarity threshold for filtering

        Returns:
            Optimal k value
        """
        min_k = min_k or Config.MIN_K
        max_k = max_k or Config.MAX_ADAPTIVE_K
        threshold = threshold if threshold is not None else Config.SIMILARITY_THRESHOLD

        print(f"Calculating adaptive k for query: '{query}'")

        # Factor 1: Query length
        query_length = len(query)
        length_factor = min(1.0, query_length / 100)  # Normalize to [0, 1]

        # Factor 2: Query ambiguity (keywords that suggest need for more results)
        ambiguity_keywords = [
            "추천",
            "비슷한",
            "같은",
            "유사",
            "여러",
            "다양한",
            "종류",
        ]
        ambiguity_score = sum(1 for kw in ambiguity_keywords if kw in query)
        ambiguity_factor = min(1.0, ambiguity_score * 0.3)

        # Factor 3: Similarity score distribution
        # Fetch max_k results and analyze score distribution
        try:
            results_with_scores = self.similarity_search_with_score(query, k=max_k)

            if not results_with_scores:
                return min_k

            scores = [score for _, score in results_with_scores]

            # Count how many scores are above threshold
            above_threshold = sum(1 for score in scores if score >= threshold)
            distribution_factor = min(1.0, above_threshold / max_k)

        except Exception as e:
            print(f"Error in adaptive k calculation: {str(e)}")
            distribution_factor = 0.5

        # Combine factors
        combined_factor = (length_factor + ambiguity_factor + distribution_factor) / 3

        # Calculate k
        adaptive_k = int(min_k + (max_k - min_k) * combined_factor)
        adaptive_k = max(min_k, min(max_k, adaptive_k))

        print(
            f"Adaptive k determined: {adaptive_k} (factors: length={length_factor:.2f}, ambiguity={ambiguity_factor:.2f}, distribution={distribution_factor:.2f})"
        )

        return adaptive_k

    def advanced_search(
        self,
        query: str,
        use_mmr: bool = None,
        use_reranking: bool = None,
        use_adaptive_k: bool = None,
        k: int = None,
        filter: Dict[str, Any] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Advanced search combining multiple strategies.

        Combines:
        - Adaptive Top-k
        - MMR for diversity
        - Metadata filtering
        - Bestseller reranking

        Args:
            query: Search query
            use_mmr: Whether to use MMR (default from config)
            use_reranking: Whether to use reranking (default from config)
            use_adaptive_k: Whether to use adaptive k (default from config)
            k: Number of documents (used if adaptive_k is disabled)
            filter: Optional metadata filter
            **kwargs: Additional parameters for MMR, reranking, etc.

        Returns:
            List of documents (best matches considering all factors)
        """
        use_mmr = use_mmr if use_mmr is not None else Config.USE_MMR
        use_reranking = (
            use_reranking if use_reranking is not None else Config.USE_RERANKING
        )
        use_adaptive_k = (
            use_adaptive_k if use_adaptive_k is not None else Config.USE_ADAPTIVE_K
        )

        print("\n=== Advanced Search ===")

        # Step 1: Determine k
        if use_adaptive_k:
            k = self.adaptive_top_k(
                query,
                **{
                    key: val
                    for key, val in kwargs.items()
                    if key in ["min_k", "max_k", "threshold"]
                },
            )
        else:
            k = k or Config.DEFAULT_K

        # Step 2: Retrieve documents
        if use_mmr:
            # MMR search for diversity
            results = self.mmr_search(
                query,
                k=k,
                filter=filter,
                **{
                    key: val
                    for key, val in kwargs.items()
                    if key in ["fetch_k", "lambda_mult"]
                },
            )
            # Need scores for reranking, so fetch again with scores
            if use_reranking:
                results_with_scores = self.similarity_search_with_score(
                    query, k=k, filter=filter
                )
            else:
                return results
        else:
            # Standard similarity search with scores
            results_with_scores = self.similarity_search_with_score(
                query, k=k, filter=filter
            )

        # Step 3: Rerank if enabled
        if use_reranking:
            reranked = self.rerank_by_bestseller(
                results_with_scores,
                **{
                    key: val
                    for key, val in kwargs.items()
                    if key in ["alpha", "beta", "rank_column"]
                },
            )
            results = [doc for doc, score in reranked]
        else:
            results = [doc for doc, score in results_with_scores]

        print(f"Advanced search completed: returned {len(results)} documents\n")

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
        Load existing FAISS vector store or create a new one if it doesn't exist.

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

    def test_vector_store(self, disable_advanced: bool = False) -> None:
        """
        Test vector store functionality with a sample query.
        """
        sample_query = input("Enter a sample query to test the vector store: ")
        print(f"Testing vector store with query: '{sample_query}'")

        if disable_advanced:
            results = self.similarity_search(sample_query, k=Config.DEFAULT_K)
        else:
            results = self.advanced_search(sample_query)

        if results:
            print("Test successful. Retrieved document:")
            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc.page_content}...\n")  # Print first 200 chars

        else:
            print("Test failed. No documents retrieved.")
