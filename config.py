"""
Configuration module for RAG Book Recommendation System.
Handles environment variables, model settings, and system configurations.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import streamlit as st


try:
    # Use streamlit secret key
    print("First Try loading API keys from Streamlit secrets")
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    LANGSMITH_API_KEY = st.secrets["LANGSMITH_API_KEY"]

except (KeyError, FileNotFoundError):
    # If streamlit secrets are not configured, load from .env file
    print("FAILED. Try loading API keys from .env file")
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")


class Config:
    """Configuration class for managing system settings."""

    # Project paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = BASE_DIR / "faiss_index"

    # Data files
    DATA_FILE = DATA_DIR / "combined_preprocessed.csv"

    # API Keys (loaded from environment variables or Streamlit secrets)
    GOOGLE_API_KEY: Optional[str] = GOOGLE_API_KEY
    LANGSMITH_API_KEY: Optional[str] = LANGSMITH_API_KEY

    # Model settings
    EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
    # Try gemini-1.5-flash if gemini-2.0-flash-lite quota is exceeded
    CHAT_MODEL_NAME = "openai:gpt-5-mini"
    
    SUMMARIZATION_MODEL_NAME = "google_genai:gemini-2.0-flash-lite"

    # Chain model settings (for orchestrator chains - use lightweight model)
    CHAIN_MODEL_NAME = "google_genai:gemini-2.0-flash-lite"

    # Available chat models
    AVAILABLE_MODELS = {
        "gpt-5-nano": "openai:gpt-5-nano",
        "gpt-5-mini": "openai:gpt-5-mini",
        "gemini-2.5-flash": "google_genai:gemini-2.5-flash",
        "gemini-2.0-flash": "google_genai:gemini-2.0-flash",
        "gemini-2.0-flash-lite": "google_genai:gemini-2.0-flash-lite",
        "gemini-1.5-flash": "google_genai:gemini-1.5-flash",
    }
    
    MODEL_PROVIDERS = {
        "Google": "google_genai",
        "OpenAI": "openai",
        "Anthropic": "anthropic",
        "Hugging Face": "huggingface",
    }

    # Text processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100

    # Retrieval settings
    DEFAULT_K = 5  # Number of documents to retrieve
    MAX_K = 7  # Maximum number of documents for deduplication

    # MMR settings
    USE_MMR = True  # Use Maximal Marginal Relevance for diversity
    MMR_FETCH_K = 20  # Number of documents to fetch before MMR filtering
    MMR_LAMBDA = 0.8  # 0=max diversity, 1=max relevance

    # Reranking settings
    USE_RERANKING = True  # Use bestseller rank based reranking
    RANK_ALPHA = 0.8  # Weight for similarity score (0-1)
    RANK_BETA = 0.2  # Weight for rank score (0-1)
    RANK_COLUMN = "순번/순위"  # Column name for bestseller rank

    # Adaptive Top-k settings
    USE_ADAPTIVE_K = True  # Use adaptive top-k strategy
    MIN_K = 3  # Minimum number of documents (Increased from 2)
    MAX_ADAPTIVE_K = 10  # Maximum number of documents for adaptive retrieval
    SIMILARITY_THRESHOLD = 0.5  # Threshold for adaptive k (Lowered from 0.7)
    RELATIVE_DROP_THRESHOLD = 0.05  # Threshold for elbow method

    # Cross-Encoder settings
    USE_CROSS_ENCODER = False  # Default to False (enable in experiments)
    CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # Lightweight yet powerful reranker

    # CSV loader settings
    CSV_ENCODING = "utf-8"
    CSV_DELIMITER = ","
    CONTENT_COLUMNS = ["순번/순위", "구분", "상품명", "책소개", "목차"]

    @classmethod
    def setup_environment(cls) -> None:
        """Set up environment variables for API access."""
        if cls.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = cls.GOOGLE_API_KEY
        else:
            print("Warning: GOOGLE_API_KEY not found in environment variables")

        if cls.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_API_KEY"] = cls.LANGSMITH_API_KEY
        else:
            print("Warning: LANGSMITH_API_KEY not found in environment variables")

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configurations are present."""
        if not cls.DATA_FILE.exists():
            raise FileNotFoundError(f"Data file not found: {cls.DATA_FILE}")

        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required but not set")

        return True
