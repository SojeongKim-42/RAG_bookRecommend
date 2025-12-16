"""
Streamlit RAG Book Recommendation Chatbot
AI 기반 도서 추천 챗봇 웹 애플리케이션
"""

import streamlit as st
from src.config import Config
from src.data.document_processor import DocumentProcessor
from src.data.vector_store import VectorStoreManager
from src.core.rag_agent import RAGAgent
from src.core.orchestrator import AmbiguityAwareOrchestrator


# Page configuration
st.set_page_config(
    page_title="AI 도서 추천 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",  # 사이드바를 기본적으로 숨김
)


@st.cache_resource
def initialize_rag_system():
    """
    Initialize RAG system components with caching.
    This ensures the system is loaded only once and reused across sessions.
    """
    try:
        with st.spinner("RAG 시스템을 초기화하는 중..."):
            # Setup environment
            Config.setup_environment()

            # Initialize components
            doc_processor = DocumentProcessor()
            vectorstore_manager = VectorStoreManager()

            # Load or create vector store
            if vectorstore_manager.exists():
                vectorstore_manager.load_vectorstore()
                chunks, original_docs = doc_processor.process()
            else:
                chunks, original_docs = doc_processor.process()
                vectorstore_manager.create_vectorstore(chunks, save=True)

            return vectorstore_manager, True, None

    except Exception as e:
        return None, False, str(e)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "awaiting_clarification" not in st.session_state:
        st.session_state.awaiting_clarification = False
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False
    if "config_changed" not in st.session_state:
        st.session_state.config_changed = False
    if "use_orchestrator" not in st.session_state:
        st.session_state.use_orchestrator = True  # Default to new orchestrator
    if "include_purchase_links" not in st.session_state:
        st.session_state.include_purchase_links = True  # Default to include links

    # Initialize config values in session state
    if "use_mmr" not in st.session_state:
        st.session_state.use_mmr = Config.USE_MMR
    if "use_reranking" not in st.session_state:
        st.session_state.use_reranking = Config.USE_RERANKING
    if "use_adaptive_k" not in st.session_state:
        st.session_state.use_adaptive_k = Config.USE_ADAPTIVE_K
    if "default_k" not in st.session_state:
        st.session_state.default_k = Config.DEFAULT_K
    if "mmr_lambda" not in st.session_state:
        st.session_state.mmr_lambda = Config.MMR_LAMBDA
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = Config.CHAT_MODEL_NAME


def display_chat_messages():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_query(user_query: str, agent: RAGAgent = None, orchestrator: AmbiguityAwareOrchestrator = None):
    """
    Process user query and generate response.

    Args:
        user_query: User's question
        agent: RAGAgent instance (legacy mode)
        orchestrator: AmbiguityAwareOrchestrator instance (new mode)

    Returns:
        Tuple of (response_text, error, result_dict)
    """
    try:
        if orchestrator is not None:
            # Use new orchestrator
            if st.session_state.awaiting_clarification:
                # This is a clarification response
                result = orchestrator.process_clarification_response(user_query)
            else:
                # This is a new query
                # Pass history excluding the current message (which was just appended)
                chat_history = st.session_state.messages[:-1]
                result = orchestrator.process_query(
                    user_query,
                    chat_history=chat_history,
                    include_links=st.session_state.include_purchase_links
                )

            # Check if needs clarification
            if result["needs_clarification"]:
                st.session_state.awaiting_clarification = True
                response_text = result["clarification_question"]
            else:
                st.session_state.awaiting_clarification = False
                response_text = result["response"]

            return response_text, None, result

        elif agent is not None:
            # Use legacy agent
            response = agent.query(user_query, verbose=False, use_history=True)
            response_text = agent.get_response_text(response)
            return response_text, None, None

        else:
            return None, "No agent or orchestrator available", None

    except Exception as e:
        return None, str(e), None


def apply_config_changes():
    """Apply config changes to Config class and recreate agent/orchestrator."""
    # Config globals are NOT modified anymore to ensure session isolation.
    # Instead, we pass the local config to the instances.

    # Recreate based on mode
    if st.session_state.system_ready:
        if st.session_state.use_orchestrator:
            # Create config dict for current session
            retrieval_config = {
                "use_mmr": st.session_state.use_mmr,
                "use_reranking": st.session_state.use_reranking,
                "use_adaptive_k": st.session_state.use_adaptive_k,
                "mmr_lambda": st.session_state.mmr_lambda
            }
            
            # Create new orchestrator instance with session config
            new_orchestrator = AmbiguityAwareOrchestrator(
                st.session_state.vectorstore_manager,
                model_name=st.session_state.selected_model,
                k=st.session_state.default_k,
                verbose=False,
                retrieval_config=retrieval_config
            )
            st.session_state.orchestrator = new_orchestrator
        else:
            # Legacy Agent support - warning
            # RAGAgent might still rely on global config in some places, 
            # but we are moving towards Orchestrator. 
            # For now, we recreate it but it might still read global defaults unless refactored.
            
            # Save current chat history
            old_chat_history = []
            if st.session_state.agent:
                old_chat_history = st.session_state.agent.chat_history.copy()

            # Create new agent instance
            new_agent = RAGAgent(
                st.session_state.vectorstore_manager,
                model_name=st.session_state.selected_model,
                k=st.session_state.default_k,
                use_advanced_search=any(
                    [st.session_state.use_mmr, st.session_state.use_reranking, st.session_state.use_adaptive_k]
                ),
            )
            # Force recreation
            new_agent.create_agent(verbose=False, force_recreate=True)
            new_agent.chat_history = old_chat_history
            st.session_state.agent = new_agent

    st.session_state.config_changed = False


def sidebar_settings():
    """Display sidebar with system settings and information."""
    with st.sidebar:
        st.title("⚙️ 설정")

        if st.session_state.system_ready:
            st.success("✅ 시스템 준비 완료")

            # 기본 설정 (항상 표시)
            with st.expander("🧠 Agent 모드", expanded=True):
                use_orchestrator = st.checkbox(
                    "Ambiguity-Aware Orchestrator 사용",
                    value=st.session_state.use_orchestrator,
                    help="모호한 질문을 자동으로 감지하고 처리하는 새로운 Agent"
                )
                if use_orchestrator != st.session_state.use_orchestrator:
                    st.session_state.use_orchestrator = use_orchestrator
                    st.session_state.config_changed = True
                    st.session_state.awaiting_clarification = False

                if st.session_state.use_orchestrator:
                    st.caption("🆕 모호한 질문 자동 감지 및 명확화")
                else:
                    st.caption("📚 표준 RAG Agent")

            # 출력 옵션
            with st.expander("📋 출력 옵션", expanded=False):
                include_links = st.checkbox(
                    "구매 링크 포함",
                    value=st.session_state.include_purchase_links,
                    help="추천 결과에 Google 쇼핑, YES24, 알라딘 구매 링크 추가"
                )
                if include_links != st.session_state.include_purchase_links:
                    st.session_state.include_purchase_links = include_links

            # 모델 설정
            with st.expander("🤖 모델 설정", expanded=False):
                model_options = list(Config.AVAILABLE_MODELS.keys())
                model_values = list(Config.AVAILABLE_MODELS.values())

                # Find current model index
                try:
                    current_index = model_values.index(st.session_state.selected_model)
                except ValueError:
                    current_index = 0

                selected_model_name = st.selectbox(
                    "채팅 모델",
                    options=model_options,
                    index=current_index,
                    help="사용할 LLM 모델을 선택하세요",
                    label_visibility="collapsed"
                )

                new_model = Config.AVAILABLE_MODELS[selected_model_name]
                if new_model != st.session_state.selected_model:
                    st.session_state.selected_model = new_model
                    st.session_state.config_changed = True

                st.caption(f"현재: {selected_model_name}")

            # 검색 설정
            with st.expander("🔍 검색 설정", expanded=False):
                new_k = st.slider(
                    "검색 문서 수 (K)",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.default_k,
                    help="검색 시 반환할 문서 개수",
                )
                if new_k != st.session_state.default_k:
                    st.session_state.default_k = new_k
                    st.session_state.config_changed = True

                st.markdown("**고급 검색 기능**")

                # MMR setting
                use_mmr = st.checkbox(
                    "MMR (다양성 검색)",
                    value=st.session_state.use_mmr,
                    help="검색 결과의 다양성을 높입니다",
                )
                if use_mmr != st.session_state.use_mmr:
                    st.session_state.use_mmr = use_mmr
                    st.session_state.config_changed = True

                # MMR Lambda setting (only if MMR is enabled)
                if use_mmr:
                    mmr_lambda = st.slider(
                        "MMR Lambda",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.mmr_lambda,
                        step=0.1,
                        help="0=다양성 우선, 1=관련성 우선",
                    )
                    if mmr_lambda != st.session_state.mmr_lambda:
                        st.session_state.mmr_lambda = mmr_lambda
                        st.session_state.config_changed = True

                # Reranking setting
                use_reranking = st.checkbox(
                    "Reranking (베스트셀러 고려)",
                    value=st.session_state.use_reranking,
                    help="베스트셀러 순위를 고려하여 재정렬합니다",
                )
                if use_reranking != st.session_state.use_reranking:
                    st.session_state.use_reranking = use_reranking
                    st.session_state.config_changed = True

                # Adaptive K setting
                use_adaptive_k = st.checkbox(
                    "Adaptive K (자동 조절)",
                    value=st.session_state.use_adaptive_k,
                    help="유사도에 따라 검색 결과 개수를 자동 조절합니다",
                )
                if use_adaptive_k != st.session_state.use_adaptive_k:
                    st.session_state.use_adaptive_k = use_adaptive_k
                    st.session_state.config_changed = True

                # Apply button
                if st.session_state.config_changed:
                    st.divider()
                    if st.button(
                        "✅ 설정 적용", use_container_width=True, type="primary"
                    ):
                        apply_config_changes()
                        st.success("설정이 적용되었습니다!")
                        st.rerun()
                else:
                    st.divider()
                    st.caption("**현재 설정:**")
                    st.caption(f"• MMR: {'✅' if st.session_state.use_mmr else '❌'} • Reranking: {'✅' if st.session_state.use_reranking else '❌'} • Adaptive K: {'✅' if st.session_state.use_adaptive_k else '❌'}")

            # 시스템 관리
            with st.expander("🔧 시스템 관리", expanded=False):
                if st.button("🔄 시스템 재시작", help="Orchestrator를 강제로 재생성합니다", use_container_width=True):
                    st.cache_resource.clear()
                    # Create config dict for current session
                    retrieval_config = {
                        "use_mmr": st.session_state.use_mmr,
                        "use_reranking": st.session_state.use_reranking,
                        "use_adaptive_k": st.session_state.use_adaptive_k,
                        "mmr_lambda": st.session_state.mmr_lambda
                    }

                    st.session_state.orchestrator = AmbiguityAwareOrchestrator(
                        st.session_state.vectorstore_manager,
                        model_name=st.session_state.selected_model,
                        k=st.session_state.default_k,
                        verbose=False,
                        retrieval_config=retrieval_config
                    )
                    st.success("✅ 시스템이 재시작되었습니다!")
                    st.rerun()

            # 사용 팁
            with st.expander("💡 사용 팁", expanded=False):
                st.markdown(
                    """
                    - 원하는 장르나 주제를 구체적으로 말씀해주세요
                    - "추천해줘"라고 요청하면 다양한 도서를 추천받을 수 있습니다
                    - 특정 카테고리(소설, 자기계발 등)를 언급해보세요
                    """
                )

        else:
            st.warning("⏳ 시스템 초기화 중...")

        # 대화 기록 삭제 버튼 (항상 하단에 표시)
        st.divider()
        if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
            st.session_state.messages = []
            # Reset agent's chat history as well
            if st.session_state.agent:
                st.session_state.agent.chat_history = []
            # Reset orchestrator state
            if st.session_state.orchestrator:
                st.session_state.orchestrator.reset_state()
            st.session_state.awaiting_clarification = False
            st.rerun()


def main():
    """Main application function."""
    # Page title
    st.title("📚 AI 도서 추천 챗봇")
    st.markdown("**RAG 기반 개인화 도서 추천 시스템**")
    st.markdown("---")

    # Initialize session state
    initialize_session_state()

    # Initialize RAG system
    if not st.session_state.system_ready:
        vectorstore_manager, success, error = initialize_rag_system()

        if success:
            st.session_state.vectorstore_manager = vectorstore_manager

            # Initialize both agent and orchestrator
            st.session_state.agent = RAGAgent(
                vectorstore_manager,
                model_name=st.session_state.selected_model
            )
            # Create config dict for current session
            retrieval_config = {
                "use_mmr": st.session_state.use_mmr,
                "use_reranking": st.session_state.use_reranking,
                "use_adaptive_k": st.session_state.use_adaptive_k,
                "mmr_lambda": st.session_state.mmr_lambda
            }

            st.session_state.orchestrator = AmbiguityAwareOrchestrator(
                vectorstore_manager,
                model_name=st.session_state.selected_model,
                k=st.session_state.default_k,
                verbose=False,
                retrieval_config=retrieval_config
            )

            st.session_state.system_ready = True
            st.success("✅ 시스템이 준비되었습니다! 질문을 입력해주세요.")
        else:
            st.error(f"❌ 시스템 초기화 실패: {error}")
            st.stop()

    # Display sidebar
    sidebar_settings()

    # Display chat messages
    display_chat_messages()

    # Add welcome message if no messages
    if len(st.session_state.messages) == 0:
        welcome_message = """
        안녕하세요! 저는 AI 도서 추천 챗봇입니다. 📚
        
        어떤 책을 찾고 계신가요? 다음과 같이 질문해보세요:
        - "SF 소설 추천해줘"
        - "자기계발 책 중에서 좋은 거 있어?"
        - "여행 관련 책 추천해줘"
        - "베스트셀러 중에서 추천해줘"
        """
        with st.chat_message("assistant"):
            st.markdown(welcome_message)

    # Chat input
    if prompt := st.chat_input("무엇을 도와드릴까요?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("생각하는 중..."):
                # Choose which agent to use
                if st.session_state.use_orchestrator:
                    response_text, error, result = process_user_query(
                        prompt,
                        orchestrator=st.session_state.orchestrator
                    )
                else:
                    response_text, error, result = process_user_query(
                        prompt,
                        agent=st.session_state.agent
                    )

                if error:
                    st.error(f"오류가 발생했습니다: {error}")
                    response_text = (
                        "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
                    )

                st.markdown(response_text)

                # Show debug info if in orchestrator mode
                if st.session_state.use_orchestrator and result:
                    with st.expander("🔍 처리 과정 정보 (디버그)", expanded=False):
                        state = result.get("state")
                        if state:
                            st.write(f"**모호성 감지**: {state.is_ambiguous}")
                            if state.is_ambiguous:
                                st.write(f"**모호성 유형**: {state.ambiguity_type}")
                                st.write(f"**신뢰도**: {state.ambiguity_confidence:.2f}")
                            if state.rewritten_query:
                                st.write(f"**재작성된 쿼리**: {state.rewritten_query}")
                            st.write(f"**검색된 책 수**: {len(state.retrieved_books)}")
                            if state.clarification_history:
                                st.write(f"**명확화 이력**: {len(state.clarification_history)}회")

        # Add assistant message to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


if __name__ == "__main__":
    main()
