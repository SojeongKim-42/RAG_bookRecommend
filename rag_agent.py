"""
RAG Agent module for handling book recommendation queries with advanced retrieval.
Supports MMR, metadata filtering, reranking, and adaptive top-k strategies.
"""

from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

from config import Config
from vector_store import VectorStoreManager


class RAGAgent:
    """
    RAG-based agent for book recommendations with advanced retrieval features.

    Features:
    - MMR search for diversity
    - Metadata filtering
    - Bestseller rank-based reranking
    - Adaptive top-k strategy
    """

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        model_name: str = None,
        k: int = None,
        use_advanced_search: bool = True,
    ):
        """
        Initialize RAG Agent.

        Args:
            vectorstore_manager: VectorStoreManager instance
            model_name: Name of the chat model
            k: Number of documents to retrieve for context
            use_advanced_search: Whether to use advanced search features
        """
        self.vectorstore_manager = vectorstore_manager
        self.model_name = model_name or Config.CHAT_MODEL_NAME
        self.k = k or Config.DEFAULT_K
        self.use_advanced_search = use_advanced_search
        self.agent = None
        self.model = None

    def _initialize_model(self):
        """Initialize the chat model with fallback options."""
        if self.model is None:
            print(f"Initializing chat model: {self.model_name}")
            try:
                self.model = init_chat_model(self.model_name)
            except Exception as e:
                print(f"Failed to initialize {self.model_name}: {str(e)}")
                # Try fallback model
                fallback_model = "google_genai:gemini-1.5-flash"
                print(f"Trying fallback model: {fallback_model}")
                try:
                    self.model = init_chat_model(fallback_model)
                    self.model_name = fallback_model
                except Exception as fallback_error:
                    print(f"Fallback model also failed: {str(fallback_error)}")
                    raise
        return self.model

    def _create_prompt_middleware(self):
        """
        Create dynamic prompt middleware that injects retrieved context.
        Uses advanced search if enabled.

        Returns:
            Dynamic prompt function
        """
        vectorstore_manager = self.vectorstore_manager
        k = self.k
        use_advanced_search = self.use_advanced_search

        @dynamic_prompt
        def prompt_with_context(request: ModelRequest) -> str:
            """Inject retrieved context into the prompt."""
            try:
                # Get the last user message
                last_query = request.state["messages"][-1].text

                # Retrieve relevant documents using advanced search or standard search
                if use_advanced_search:
                    print(
                        "Using advanced search with MMR, adaptive k, and reranking..."
                    )
                    retrieved_docs = vectorstore_manager.advanced_search(
                        last_query,
                        use_mmr=Config.USE_MMR,
                        use_reranking=Config.USE_RERANKING,
                        use_adaptive_k=Config.USE_ADAPTIVE_K,
                        k=k,
                    )
                else:
                    print(f"Using standard similarity search...")
                    retrieved_docs = vectorstore_manager.similarity_search(
                        last_query, k=k
                    )

                # Combine document contents
                docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # Create system message with context
                system_message = (
                    "You are a helpful book recommendation assistant. "
                    "Use the following context from our book database to provide accurate recommendations:\n\n"
                    f"{docs_content}\n\n"
                    "Based on this context, provide relevant book recommendations. "
                    "If you recommend a book, mention its category (구분) and title (상품명). "
                    "Consider the bestseller rankings and diverse options in your recommendations."
                )

                return system_message

            except Exception as e:
                print(f"Error in prompt middleware: {str(e)}")
                return "You are a helpful book recommendation assistant."

        return prompt_with_context

    def create_agent(self, tools: List = None, verbose: bool = False):
        """
        Create the agent with middleware.

        Args:
            tools: List of tools for the agent
            verbose: Whether to enable verbose mode

        Returns:
            Created agent
        """
        if self.agent is None:
            model = self._initialize_model()
            prompt_middleware = self._create_prompt_middleware()

            print("Creating RAG agent...")
            self.agent = create_agent(
                model=model,
                tools=tools or [],
                middleware=[prompt_middleware],
            )

        return self.agent

    def query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Query the agent with a question.

        Args:
            question: User's question
            verbose: Whether to print verbose output

        Returns:
            Agent response
        """
        if self.agent is None:
            self.create_agent(verbose=verbose)

        if verbose:
            print(f"\nProcessing query: {question}\n")

        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            return response

        except Exception as e:
            print(f"Error during query: {str(e)}")
            raise

    def get_response_text(self, response: Dict[str, Any]) -> str:
        """
        Extract response text from agent response.

        Args:
            response: Agent response dictionary

        Returns:
            Response text
        """
        try:
            return response["messages"][-1].content
        except (KeyError, IndexError) as e:
            print(f"Error extracting response text: {str(e)}")
            return "Error: Could not extract response"

    def interactive_mode(self):
        """Run agent in interactive mode for continuous queries."""
        print("\n=== RAG Book Recommendation Agent ===")
        print("Ask me about book recommendations! (Type 'quit' to exit)\n")

        self.create_agent(verbose=False)

        while True:
            try:
                query = input("You: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                response = self.query(query)
                response_text = self.get_response_text(response)

                print(f"\nAgent: {response_text}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                continue
