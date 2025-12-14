"""
Streamlit RAG Book Recommendation Chatbot
AI 기반 도서 추천 챗봇 웹 애플리케이션
"""

import sys
from pathlib import Path

# Add parent directory to path to import project modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
from config import Config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from rag_agent import RAGAgent


# Page configuration
st.set_page_config(
    page_title="AI 도서 추천 챗봇",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
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
    if "system_ready" not in st.session_state:
        st.session_state.system_ready = False


def display_chat_messages():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_query(user_query: str, agent: RAGAgent):
    """
    Process user query and generate response.

    Args:
        user_query: User's question
        agent: RAGAgent instance

    Returns:
        Response text from the agent
    """
    try:
        response = agent.query(user_query, verbose=False)
        response_text = agent.get_response_text(response)
        return response_text, None
    except Exception as e:
        return None, str(e)


def sidebar_settings():
    """Display sidebar with system settings and information."""
    with st.sidebar:
        st.title("⚙️ 설정")

        st.markdown("---")
        st.subheader("📊 시스템 정보")

        if st.session_state.system_ready:
            st.success("✅ 시스템 준비 완료")
            st.info(f"🤖 모델: {Config.CHAT_MODEL_NAME}")
            st.info(f"🔍 검색 문서 수: {Config.DEFAULT_K}")

            with st.expander("고급 설정"):
                st.write(f"**MMR 사용**: {'✅' if Config.USE_MMR else '❌'}")
                st.write(f"**Reranking**: {'✅' if Config.USE_RERANKING else '❌'}")
                st.write(f"**Adaptive K**: {'✅' if Config.USE_ADAPTIVE_K else '❌'}")
        else:
            st.warning("⏳ 시스템 초기화 중...")

        st.markdown("---")
        st.subheader("💡 사용 팁")
        st.markdown(
            """
        - 원하는 장르나 주제를 구체적으로 말씀해주세요
        - "추천해줘"라고 요청하면 다양한 도서를 추천받을 수 있습니다
        - 특정 카테고리(소설, 자기계발 등)를 언급해보세요
        """
        )

        st.markdown("---")
        if st.button("🗑️ 대화 기록 삭제", use_container_width=True):
            st.session_state.messages = []
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
            st.session_state.agent = RAGAgent(vectorstore_manager)
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
                response_text, error = process_user_query(
                    prompt, st.session_state.agent
                )

                if error:
                    st.error(f"오류가 발생했습니다: {error}")
                    response_text = (
                        "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."
                    )

                st.markdown(response_text)

        # Add assistant message to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )


if __name__ == "__main__":
    main()
