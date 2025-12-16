"""
Orchestrator Agent with Stateful Flow for Ambiguity-Aware Book Recommendation.

This module implements a single agent that acts as a decision-making orchestrator,
delegating actual intelligence work to specialized chains.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.core.chains import (
    AmbiguityDetector,
    QueryRewriter,
    RetrieveQualityEvaluator,
    ClarificationQuestionGenerator,
    FinalRecommender
)
from src.data.vector_store import VectorStoreManager
from src.config import Config


class AgentState(Enum):
    """Agent state machine states."""
    INIT = "init"
    AMBIGUITY_CHECK = "ambiguity_check"
    QUERY_REWRITE = "query_rewrite"
    RETRIEVE = "retrieve"
    QUALITY_CHECK = "quality_check"
    CLARIFICATION = "clarification"
    FINAL_RECOMMENDATION = "final_recommendation"
    DONE = "done"


@dataclass
class ConversationState:
    """State container for the recommendation flow."""

    # User input
    user_query: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)

    # Ambiguity analysis
    is_ambiguous: bool = False
    ambiguity_type: Optional[str] = None
    ambiguity_confidence: float = 0.0

    # Query rewriting
    rewritten_query: Optional[str] = None

    # Retrieval
    retrieved_books: List[Dict[str, Any]] = field(default_factory=list)

    # Quality check
    retrieval_sufficient: bool = False
    missing_info: List[str] = field(default_factory=list)

    # Clarification
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    clarification_history: List[Dict[str, str]] = field(default_factory=list)

    # Current state
    current_state: AgentState = AgentState.INIT

    # Final output
    final_response: Optional[str] = None


class AmbiguityAwareOrchestrator:
    """
    Orchestrator agent that manages the stateful flow for book recommendations.

    This agent acts as a decision-maker (control logic) and delegates
    intelligence tasks to specialized chains.

    Flow:
    1. Ambiguity Detection
    2. Query Rewriting (if ambiguous)
    3. Retrieve
    4. Quality Check
    5. Clarification (if insufficient)
    6. Final Recommendation
    """

    def __init__(
        self,
        vectorstore_manager: VectorStoreManager,
        model_name: str = None,
        chain_model_name: str = None,
        k: int = None,
        verbose: bool = False,
        # Ablation flags
        use_rewriter: bool = True,
        use_quality_check: bool = True,
        # Session-specific config
        retrieval_config: Dict[str, Any] = None
    ):
        """
        Initialize orchestrator.

        Args:
            vectorstore_manager: VectorStoreManager instance
            model_name: LLM model name for final recommendation
            chain_model_name: LLM model name for chain operations (default: lightweight model)
            k: Number of documents to retrieve
            verbose: Enable verbose logging
            use_rewriter: Whether to use query rewriter (ablation)
            use_quality_check: Whether to use quality evaluation (ablation)
            retrieval_config: Dictionary containing retrieval parameters (override global Config)
        """
        self.vectorstore_manager = vectorstore_manager
        self.model_name = model_name or Config.CHAT_MODEL_NAME
        self.chain_model_name = chain_model_name or Config.CHAIN_MODEL_NAME
        self.k = k or Config.DEFAULT_K
        self.verbose = verbose
        self.use_rewriter = use_rewriter
        self.use_quality_check = use_quality_check
        self.retrieval_config = retrieval_config or {}

        # Initialize chains with lightweight model
        self.ambiguity_detector = AmbiguityDetector(model_name=self.chain_model_name)
        self.query_rewriter = QueryRewriter(model_name=self.chain_model_name)
        self.quality_evaluator = RetrieveQualityEvaluator(model_name=self.chain_model_name)
        self.clarification_generator = ClarificationQuestionGenerator(model_name=self.chain_model_name)
        # Final recommender uses main model for better quality
        self.recommender = FinalRecommender(model_name=self.model_name)

        # State
        self.state: Optional[ConversationState] = None
        self.include_links: bool = True  # Default to include links

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Orchestrator] {message}")

    def process_query(self, user_query: str, chat_history: List[Dict[str, str]] = None, include_links: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the complete flow.

        Args:
            user_query: User's query
            chat_history: List of conversation history [{"role": "user", "content": "..."}, ...]
            include_links: Whether to include purchase links in final recommendation

        Returns:
            Result dictionary with response and state
        """
        # Initialize state
        self.state = ConversationState(
            user_query=user_query,
            chat_history=chat_history or []
        )
        self.state.current_state = AgentState.AMBIGUITY_CHECK
        self.include_links = include_links  # Store for later use

        # Run state machine
        while self.state.current_state != AgentState.DONE:
            self._execute_current_state()

        return {
            "response": self.state.final_response,
            "needs_clarification": self.state.clarification_needed,
            "clarification_question": self.state.clarification_question,
            "state": self.state
        }

    def process_clarification_response(self, user_answer: str) -> Dict[str, Any]:
        """
        Process user's answer to a clarification question.

        Args:
            user_answer: User's answer to clarification question

        Returns:
            Result dictionary with response and state
        """
        if self.state is None:
            raise ValueError("No active conversation state. Call process_query first.")

        if not self.state.clarification_needed:
            raise ValueError("No clarification was requested.")

        # Record clarification
        self.state.clarification_history.append({
            "question": self.state.clarification_question,
            "answer": user_answer
        })

        # Reset clarification state
        self.state.clarification_needed = False
        self.state.clarification_question = None

        # Update query with clarification
        clarification_context = " ".join([
            f"{item['answer']}" for item in self.state.clarification_history
        ])
        enhanced_query = f"{self.state.user_query} {clarification_context}"

        self._log(f"Enhanced query with clarification: {enhanced_query}")

        # Re-retrieve with enhanced query
        self.state.current_state = AgentState.RETRIEVE
        self.state.rewritten_query = enhanced_query

        # Continue state machine
        while self.state.current_state != AgentState.DONE:
            self._execute_current_state()

        return {
            "response": self.state.final_response,
            "needs_clarification": self.state.clarification_needed,
            "clarification_question": self.state.clarification_question,
            "state": self.state
        }

    def _execute_current_state(self):
        """Execute the current state's logic."""
        state = self.state.current_state

        if state == AgentState.AMBIGUITY_CHECK:
            self._step_ambiguity_check()
        elif state == AgentState.QUERY_REWRITE:
            self._step_query_rewrite()
        elif state == AgentState.RETRIEVE:
            self._step_retrieve()
        elif state == AgentState.QUALITY_CHECK:
            self._step_quality_check()
        elif state == AgentState.CLARIFICATION:
            self._step_clarification()
        elif state == AgentState.FINAL_RECOMMENDATION:
            self._step_final_recommendation(include_links=self.include_links)
        else:
            raise ValueError(f"Unknown state: {state}")

    def _step_ambiguity_check(self):
        """Step 1: Check if query is ambiguous."""
        self._log("Step 1: Checking ambiguity...")

        result = self.ambiguity_detector.detect(
            self.state.user_query,
            self.state.chat_history
        )

        self.state.is_ambiguous = result.get("is_ambiguous", False)
        self.state.ambiguity_type = result.get("ambiguity_type")
        self.state.ambiguity_confidence = result.get("confidence", 0.0)

        self._log(f"Ambiguous: {self.state.is_ambiguous}, Type: {self.state.ambiguity_type}, Confidence: {self.state.ambiguity_confidence}")

        # Decision: rewrite or retrieve directly
        # Ablation check: Only enter rewrite state if use_rewriter is True
        # OR if there is chat history, we might want to "merge" context even if not strictly ambiguous
        # so let's allow rewriting if we have history to ensure context continuity.
        has_history = len(self.state.chat_history) > 0
        
        if self.use_rewriter and (self.state.is_ambiguous or has_history):
            self.state.current_state = AgentState.QUERY_REWRITE
        else:
            self.state.current_state = AgentState.RETRIEVE

    def _step_query_rewrite(self):
        """Step 2: Rewrite query for better search."""
        self._log("Step 2: Rewriting query...")

        rewritten = self.query_rewriter.rewrite(
            self.state.user_query,
            self.state.chat_history,
            self.state.ambiguity_type
        )

        self.state.rewritten_query = rewritten
        self._log(f"Rewritten query: {rewritten}")

        # Next: retrieve
        self.state.current_state = AgentState.RETRIEVE

    def _step_retrieve(self):
        """Step 3: Retrieve documents."""
        self._log("Step 3: Retrieving documents...")

        # Use rewritten query if available, otherwise original
        search_query = self.state.rewritten_query or self.state.user_query

        # Retrieve using advanced search
        # Retrieve using advanced search
        # Use session config if available, otherwise fallback to global Config
        use_mmr = self.retrieval_config.get("use_mmr", Config.USE_MMR)
        use_reranking = self.retrieval_config.get("use_reranking", Config.USE_RERANKING)
        use_adaptive_k = self.retrieval_config.get("use_adaptive_k", Config.USE_ADAPTIVE_K)
        mmr_lambda = self.retrieval_config.get("mmr_lambda", Config.MMR_LAMBDA)

        retrieved_docs = self.vectorstore_manager.advanced_search(
            search_query,
            use_mmr=use_mmr,
            use_reranking=use_reranking,
            use_adaptive_k=use_adaptive_k,
            mmr_lambda=mmr_lambda,
            k=self.k
        )

        # Convert documents to dict format
        self.state.retrieved_books = [
            doc.metadata for doc in retrieved_docs
        ]

        self._log(f"Retrieved {len(self.state.retrieved_books)} books")

        # Next: quality check or final recommendation
        if self.use_quality_check:
            self.state.current_state = AgentState.QUALITY_CHECK
        else:
            # Skip QC/Clarification loop
            self.state.retrieval_sufficient = True # Assume sufficient
            self.state.current_state = AgentState.FINAL_RECOMMENDATION

    def _step_quality_check(self):
        """Step 4: Check if retrieval is sufficient."""
        self._log("Step 4: Checking retrieval quality...")

        evaluation = self.quality_evaluator.evaluate(
            self.state.user_query,
            self.state.retrieved_books,
            ambiguity_type=self.state.ambiguity_type
        )

        self.state.retrieval_sufficient = evaluation.get("sufficient", True)
        self.state.missing_info = evaluation.get("missing_info", [])

        self._log(f"Sufficient: {self.state.retrieval_sufficient}, Missing: {self.state.missing_info}")

        # Decision: clarify or recommend
        if not self.state.retrieval_sufficient and self.state.missing_info:
            # Only ask for clarification once
            if len(self.state.clarification_history) == 0:
                self.state.current_state = AgentState.CLARIFICATION
            else:
                # Already asked once, proceed anyway
                self.state.current_state = AgentState.FINAL_RECOMMENDATION
        else:
            self.state.current_state = AgentState.FINAL_RECOMMENDATION

    def _step_clarification(self):
        """Step 5: Generate clarification question."""
        self._log("Step 5: Generating clarification question...")

        question = self.clarification_generator.generate(
            self.state.user_query,
            self.state.missing_info
        )

        self.state.clarification_question = question
        self.state.clarification_needed = True

        self._log(f"Clarification: {question}")

        # Pause here - need user input
        self.state.current_state = AgentState.DONE
        self.state.final_response = None  # No final response yet

    def _step_final_recommendation(self, include_links: bool = True):
        """Step 6: Generate final recommendation."""
        self._log("Step 6: Generating final recommendation...")

        # Create user state summary
        user_state_summary = f"원래 요청: {self.state.user_query}"
        if self.state.is_ambiguous:
            user_state_summary += f"\n모호성 타입: {self.state.ambiguity_type}"
        if self.state.rewritten_query:
            user_state_summary += f"\n해석된 의도: {self.state.rewritten_query}"

        recommendation = self.recommender.recommend(
            user_query=self.state.user_query,
            retrieved_books=self.state.retrieved_books,
            user_state_summary=user_state_summary,
            clarification_history=self.state.clarification_history,
            include_links=include_links
        )

        self.state.final_response = recommendation
        self._log("Recommendation generated")

        # Done
        self.state.current_state = AgentState.DONE

    def get_state(self) -> Optional[ConversationState]:
        """Get current conversation state."""
        return self.state

    def reset_state(self):
        """Reset conversation state."""
        self.state = None
        self._log("State reset")
