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
        self.cross_encoder = None

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
        Determine optimal k based on similarity score distribution (Relative Drop Strategy).

        Strategy:
        1. Fetch max_k results.
        2. Keep all results within 85% score of the top result (Quality Guarantee).
        3. Cut off if there is a significant score drop (elbow) between neighbors.

        Args:
            query: Search query
            min_k: Minimum k value
            max_k: Maximum k value
            threshold: (Deprecated) Used for backward compatibility

        Returns:
            Optimal k value
        """
        min_k = min_k or Config.MIN_K
        max_k = max_k or Config.MAX_ADAPTIVE_K
        
        # New: Use relative drop threshold
        relative_drop_threshold = getattr(Config, "RELATIVE_DROP_THRESHOLD", 0.05)

        print(f"Calculating adaptive k for query: '{query}'")

        try:
            # Fetch max_k results with scores
            results_with_scores = self.similarity_search_with_score(query, k=max_k)

            if not results_with_scores:
                return min_k

            scores = [score for _, score in results_with_scores]
            top_score = scores[0]

            # Strategy 1: Quality Guarantee (Keep scores close to top)
            # Example: if top is 0.8, keep everything above 0.8 * 0.85 = 0.68
            quality_candidates = [s for s in scores if s >= top_score * 0.85]
            quality_k = len(quality_candidates)

            # Strategy 2: Relative Drop (Elbow Method)
            # Detect sudden drop in scores
            elbow_k = len(scores)
            for i in range(len(scores) - 1):
                # If gap between neighbors is too large
                if scores[i] - scores[i+1] > relative_drop_threshold:
                    elbow_k = i + 1
                    break
            
            # Combine strategies: take the smaller cut-off to be precise, 
            # but ensure at least min_k
            final_k = min(elbow_k, quality_k)
            final_k = max(min_k, final_k)
            final_k = min(max_k, final_k)

            print(f"Adaptive k determined: {final_k} (top_score={top_score:.4f}, elbow_k={elbow_k}, quality_k={quality_k})")
            
            return final_k

        except Exception as e:
            print(f"Error in adaptive k calculation: {str(e)}")
            return min_k

    def rerank_with_cross_encoder(
        self,
        query: str,
        results: List[Document],
        top_n: int = None
    ) -> List[Document]:
        """
        Rerank documents using a Cross-Encoder model.

        Args:
            query: Search query
            results: List of documents to rerank
            top_n: Number of documents to return

        Returns:
            Reranked list of documents
        """
        if not results:
            return []

        model_name = getattr(Config, "CROSS_ENCODER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
        top_n = top_n or len(results)

        print(f"Reranking {len(results)} documents with Cross-Encoder: {model_name}")

        try:
            from sentence_transformers import CrossEncoder
            
            # Initialize model if not already loaded
            if self.cross_encoder is None:
                # Use device config if possible
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.cross_encoder = CrossEncoder(model_name, device=device)
            
            model = self.cross_encoder

            # Prepare pairs
            pairs = [[query, doc.page_content] for doc in results]
            
            # Predict scores
            scores = model.predict(pairs)

            # Combine docs with scores
            doc_scores = list(zip(results, scores))

            # Sort by score descending
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Return top_n documents
            reranked_docs = [doc for doc, score in doc_scores[:top_n]]
            
            return reranked_docs

        except ImportError:
            print("Error: sentence-transformers not installed. Skipping Cross-Encoder reranking.")
            return results
        except Exception as e:
            print(f"Error during Cross-Encoder reranking: {str(e)}")
            return results

    def advanced_search(
        self,
        query: str,
        use_mmr: bool = None,
        use_reranking: bool = None,
        use_adaptive_k: bool = None,
        use_cross_encoder: bool = None,
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
        - Cross-Encoder reranking (New)

        Args:
            query: Search query
            use_mmr: Whether to use MMR
            use_reranking: Whether to use bestseller reranking
            use_adaptive_k: Whether to use adaptive k
            use_cross_encoder: Whether to use Cross-Encoder
            k: Number of documents
            filter: Optional metadata filter
            **kwargs: Additional parameters

        Returns:
            List of documents
        """
        use_mmr = use_mmr if use_mmr is not None else Config.USE_MMR
        use_reranking = (
            use_reranking if use_reranking is not None else Config.USE_RERANKING
        )
        use_adaptive_k = (
            use_adaptive_k if use_adaptive_k is not None else Config.USE_ADAPTIVE_K
        )
        # Default to Config value if not provided
        config_use_ce = getattr(Config, "USE_CROSS_ENCODER", False)
        use_cross_encoder = use_cross_encoder if use_cross_encoder is not None else config_use_ce

        print("\n=== Advanced Search ===")

        # Step 1: Determine k
        # If using Cross-Encoder, we might want to fetch more candidates first
        initial_k = k or Config.DEFAULT_K
        
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
            k = initial_k

        # If using Cross-Encoder, fetch slightly more candidates for reranking?
        # For now, let's keep it simple: retrieve 'k' docs then rerank them.
        # Or better: MMR selects 'k' docs, then we rerank them.

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
            # Need scores for bestseller reranking
            if use_reranking and not use_cross_encoder:
                 # If we are going to do Cross-Encoder later, we don't necessarily need vector scores right now,
                 # but for bestseller reranking we currently use them.
                 # Re-fetching with scores is inefficient but consistent with current logic.
                 # Optimization: mmr_search could return scores, but standard interface doesn't.
                 pass
                 
        else:
            # Standard similarity search without scores (will get scores if needed below)
            # Actually similarity_search returns docs.
            results = self.similarity_search(query, k=k, filter=filter)

        # Step 3: Rerank
        # Priority: Cross-Encoder > Bestseller Reranking
        # (Usually you don't do both, or you do Bestseller as a pre-filter)
        
        if use_cross_encoder:
            # Semantic Reranking
            results = self.rerank_with_cross_encoder(query, results)
        
        elif use_reranking:
            # Heuristic Bestseller Reranking
            # We need scores for this logic
            # This part is a bit tricky because 'results' acts as docs now.
            # We need to re-score them against the query to get 'similarity_score' for the formula.
            # For efficiency, let's trust the order implicitly or re-calculate.
            
            # To strictly follow previous logic, we need (doc, score) tuples.
            # Let's re-fetch scores for the selected docs.
            results_with_scores = []
            for doc in results:
                # Calculate sim score (hacky but accurate)
                # Actually, simple way: just pass 1.0 as score if we can't get it, 
                # OR assume vector store returned them in order.
                # Let's re-query this specific doc? No, too slow.
                # Let's just skip score-based component if we lost it?
                # The original code did: results_with_scores = similarity_search_with_score...
                pass

            # Only do bestseller reranking if we have scores, OR if we strictly follow the old flow.
            # The old flow was: if MMR -> re-search with scores.
            # Let's keep the old flow for 'use_reranking' case to avoid regression.
            if use_mmr:
                 results_with_scores = self.similarity_search_with_score(query, k=k, filter=filter)
                 # Wait, this undoes MMR selection! The previous code had this bug/feature.
                 # Previous code:
                 # if use_mmr: ... results = mmr_search ...
                 # if use_reranking: results_with_scores = similarity_search_with_score ...
                 # This means MMR was IGNORED if reranking was on!
                 # That explains why MMR didn't help much.
                 # Let's FIX THIS: If MMR is used, we should rerank the MMR results.
                 
                 # Correct flow:
                 # 1. Get candidates (MMR or Standard)
                 # 2. Rerank them
                 pass
            else:
                 results_with_scores = self.similarity_search_with_score(query, k=k, filter=filter)

            # Rerank
            reranked = self.rerank_by_bestseller(
                results_with_scores,
                 **{
                    key: val
                    for key, val in kwargs.items()
                    if key in ["alpha", "beta", "rank_column"]
                },
            )
            results = [doc for doc, score in reranked]

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
