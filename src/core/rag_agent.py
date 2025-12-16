"""
RAG Agent module for handling book recommendation queries with advanced retrieval.
Supports MMR, metadata filtering, reranking, and adaptive top-k strategies.
"""

from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import dynamic_prompt, ModelRequest, SummarizationMiddleware
from langchain.agents import create_agent


from src.config import Config
from src.data.vector_store import VectorStoreManager


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
        self.chat_history = []  # Store conversation history
        self.summarization_model = init_chat_model(Config.SUMMARIZATION_MODEL_NAME)

    def _initialize_model(self):
        """Initialize the chat model with fallback options."""
        if self.model is None:
            print(f"Initializing chat model: {self.model_name}")
            try:
                self.model = init_chat_model(self.model_name)
            except Exception as e:
                print(f"Failed to initialize {self.model_name}: {str(e)}")
                # Try fallback models in order
                fallback_models = [
                    "google_genai:gemini-1.5-flash",
                    "huggingface:openai/gpt-oss-20b"
                ]
                for fallback_model in fallback_models:
                    print(f"Trying fallback model: {fallback_model}")
                    try:
                        self.model = init_chat_model(fallback_model)
                        self.model_name = fallback_model
                        print(f"Successfully initialized {fallback_model}")
                        break
                    except Exception as fallback_error:
                        print(f"Fallback model {fallback_model} also failed: {str(fallback_error)}")
                        continue
                else:
                    raise Exception("All models failed to initialize")
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
                    "Based on this context, provide relevant book recommendations."
                    "If you recommend a book, mention its category and title. "
                    "Consider the bestseller rankings and diverse options in your recommendations."
                )

                return system_message

            except Exception as e:
                print(f"Error in prompt middleware: {str(e)}")
                return "You are a helpful book recommendation assistant."

        return prompt_with_context

    def create_agent(self, tools: List = None, verbose: bool = False, force_recreate: bool = False):
        """
        Create the agent with middleware.

        Args:
            tools: List of tools for the agent
            verbose: Whether to enable verbose mode
            force_recreate: Force recreation of the agent even if it exists

        Returns:
            Created agent
        """
        if self.agent is None or force_recreate:
            # Reset model if forcing recreation
            if force_recreate:
                self.model = None
                self.agent = None

            model = self._initialize_model()

            print("Creating RAG agent...")
            self.agent = create_agent(
                model=model,
                tools=tools or [],
                middleware=[
                    self._create_prompt_middleware(),
                    SummarizationMiddleware(
                        model=self.summarization_model,
                        trigger=[
                            ("messages", 5),
                            ("tokens", 2000),
                        ]
                    )
                    ],
            )

        return self.agent

    def query(self, question: str, verbose: bool = False, use_history: bool = False) -> Dict[str, Any]:
        """
        Query the agent with a question.

        Args:
            question: User's question
            verbose: Whether to print verbose output
            use_history: Whether to use conversation history

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

    def interactive_mode(self, select_model: bool = False):
        """
        Run agent in interactive mode for continuous queries with conversation history.

        Args:
            select_model: If True, prompt user to select a model before starting
        """
        print("\n=== RAG Book Recommendation Agent ===")

        # Model selection
        if select_model:
            print("\nAvailable models:")
            models = list(Config.AVAILABLE_MODELS.items())
            for i, (name, model_id) in enumerate(models, 1):
                print(f"{i}. {name} ({model_id})")
            print(f"{len(models) + 1}. Use default ({self.model_name})")

            try:
                choice = input(f"\nSelect model (1-{len(models) + 1}): ").strip()
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected_name, selected_model = models[choice_num - 1]
                    self.model_name = selected_model
                    print(f"Selected model: {selected_name} ({selected_model})")
            except (ValueError, IndexError):
                print(f"Invalid choice. Using default model: {self.model_name}")

        print("\nAsk me about book recommendations! (Type 'quit' to exit)")
        print("Your conversation history will be remembered.\n")

        self.create_agent(verbose=False)
        self.chat_history = []  # Reset chat history

        while True:
            try:
                query = input("You: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not query:
                    continue

                response = self.query(query, use_history=True)
                response_text = self.get_response_text(response)

                print(f"\nAgent: {response_text}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                continue
