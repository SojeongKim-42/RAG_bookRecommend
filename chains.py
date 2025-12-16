"""
Chain modules for ambiguity-aware book recommendation.
Each chain is a specialized LLM call for a specific decision-making step.
"""

from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from config import Config


def generate_google_shopping_link(book_title: str, author: str = None) -> str:
    """
    Generate Google Shopping search link for a book.

    Args:
        book_title: Book title
        author: Author name (optional)

    Returns:
        Google Shopping search URL
    """
    # Create search query
    search_query = book_title
    if author:
        search_query = f"{book_title} {author}"

    # Add "책 구매" to search query for better results
    search_query = f"{search_query} 책 구매"

    # URL encode
    encoded_query = quote_plus(search_query)

    # Google Shopping search URL
    return f"https://www.google.com/search?q={encoded_query}&tbm=shop"


def generate_search_links(book_title: str, author: str = None) -> Dict[str, str]:
    """
    Generate multiple shopping links for a book.

    Args:
        book_title: Book title
        author: Author name (optional)

    Returns:
        Dictionary with platform names and URLs
    """
    search_query = book_title
    if author:
        search_query = f"{book_title} {author}"

    encoded_query = quote_plus(search_query)

    return {
        "google_shopping": f"https://www.google.com/search?q={encoded_query}+책+구매&tbm=shop",
        "google_search": f"https://www.google.com/search?q={encoded_query}+책",
        "yes24": f"https://www.yes24.com/Product/Search?query={encoded_query}",
        "aladin": f"https://www.aladin.co.kr/search/wsearchresult.aspx?SearchTarget=All&SearchWord={encoded_query}",
    }


class AmbiguityDetector:
    """Detects whether a query is ambiguous and classifies the type."""

    AMBIGUITY_TYPES = [
        "emotional_only",        # ex: "요즘 너무 공허해"
        "situational",           # ex: "군대 가기 전에 읽을 책"
        "vague_topic",           # ex: "인생에 도움되는 책"
        "multi_intent",          # ex: "재밌고 의미 있는 소설"
        "not_ambiguous"          # 명확한 요청
    ]

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.CHAIN_MODEL_NAME
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = init_chat_model(self.model_name)
        return self.model

    def detect(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Detect ambiguity in user query, considering conversation context.

        Returns:
            {
                "is_ambiguous": bool,
                "ambiguity_type": str,
                "confidence": float (0-1),
                "reason": str
            }
        """
        history_text = ""
        if chat_history:
            # Format last few turns for context
            relevant_history = chat_history[-6:]  # Last 3 turns
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in relevant_history])

        system_prompt = f"""당신은 도서 추천 요청의 모호성을 판별하는 분류기입니다.

                        **대화 맥락**:
                        {history_text if history_text else "없음"}

                        **판단 기준**:
                        1. 사용자가 **이전 질문에 대해 구체적인 답변**을 했다면 `not_ambiguous`로 분류하세요.
                           (예: AI "어떤 장르가 좋으세요?" -> 사용자 "소설이요" => **명확함**)
                        2. 문맥을 고려했을 때 여전히 정보가 부족하면 해당되는 모호성 타입을 선택하세요.

                        사용자 질문을 다음 카테고리 중 하나로 분류하세요:

                        1. emotional_only: 감정만 표현하고 구체적인 선호도가 없음
                        예: "요즘 너무 공허해", "기분이 안 좋아"

                        2. situational: 상황만 설명하고 장르/스타일 언급 없음
                        예: "군대 가기 전에 읽을 책", "출퇴근할 때 읽을만한 거"

                        3. vague_topic: 주제가 너무 광범위하거나 추상적
                        예: "인생에 도움되는 책", "성장할 수 있는 책"

                        4. multi_intent: 여러 요구사항이 섞여있고 우선순위 불명확
                        예: "재밌고 의미 있고 짧은 소설", "감동적이면서 유익한 책"

                        5. not_ambiguous: 장르, 주제, 스타일이 명확함 (또는 문맥상 명확해짐)
                        예: "SF 소설 추천해줘", "소설" (이전 질문이 장르였을 때)

                        JSON 형식으로 답하세요:
                        {{
                        "is_ambiguous": true/false,
                        "ambiguity_type": "카테고리",
                        "confidence": 0.0~1.0,
                        "reason": "판단 이유"
                        }}"""

        model = self._get_model()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"사용자 질문: {user_query}")
        ]

        response = model.invoke(messages)

        # Parse response (assuming structured output)
        import json
        try:
            # Extract JSON from response
            content = response.content
            # Find JSON in markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result
        except (json.JSONDecodeError, IndexError) as e:
            # Fallback: assume ambiguous
            return {
                "is_ambiguous": True,
                "ambiguity_type": "vague_topic",
                "confidence": 0.5,
                "reason": f"파싱 실패: {str(e)}"
            }


class QueryRewriter:
    """Rewrites ambiguous queries into search-optimized versions."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.CHAIN_MODEL_NAME
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = init_chat_model(self.model_name)
        return self.model

    def rewrite(self, user_query: str, chat_history: List[Dict[str, str]], ambiguity_type: str) -> str:
        """
        Rewrite query for better vector search using conversation history.

        Args:
            user_query: Original user query
            chat_history: Conversation history
            ambiguity_type: Type of ambiguity detected

        Returns:
            Rewritten query optimized for vector search
        """
        history_text = ""
        if chat_history:
            relevant_history = chat_history[-6:]
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in relevant_history])

        system_prompt = f"""당신은 검색 쿼리 최적화 전문가입니다.

                        사용자의 요청을 **벡터 검색에 최적화된 문장**으로 변환하세요.
                        이전 대화 맥락을 고려하여, **누적된 사용자의 요구사항을 모두 포함**해야 합니다.

                        **대화 맥락**:
                        {history_text if history_text else "없음"}

                        **현재 모호성 유형**: {ambiguity_type}

                        **작업 지침**:
                        1. 이전 대화에서 사용자가 언급한 선호도(장르, 분위기, 분량 등)를 모두 기억하세요.
                        2. 현재 요청과 이전 요구사항을 합쳐서 **하나의 구체적인 검색 문장**을 만드세요.
                           (예: 이전="소설 추천해줘", 현재="밝은 분위기" -> "밝고 희망찬 분위기의 장편 소설, 해피엔딩")
                        3. 불필요한 인사말이나 서술어는 제외하고 키워드 위주로 작성하세요.
                        4. **중요**: 사용자가 장르(소설, 에세이 등)를 명시하지 않았다면, 임의로 장르를 단정 짓지 말고 **분위기와 주제 위주**로 작성하세요. "책"이나 "도서" 같은 포괄적 표현을 사용하세요.

                        **변환 전략**:
                        1. emotional_only → 감정에 맞는 분위기/키워드 추가 (장르 고정 X)
                        2. situational → 상황에 맞는 독서 스타일/분량 추가
                        3. vague_topic → 구체적인 하위 주제/관점 추가
                        4. multi_intent → 우선순위 명시 + 구체적 조건

                        **출력 형식**:
                        - 형용사 + 상황 + 장르 + 제약조건
                        - 한국어로 자연스럽게

                        **오직 rewritten query만 출력하세요. 설명이나 추천은 금지.**"""

        model = self._get_model()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"사용자 요청: {user_query}")
        ]

        response = model.invoke(messages)
        return response.content.strip()


class RetrieveQualityEvaluator:
    """Evaluates whether retrieved documents are sufficient."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.CHAIN_MODEL_NAME
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = init_chat_model(self.model_name)
        return self.model

    def evaluate(self, user_query: str, retrieved_books: List[Dict[str, Any]], ambiguity_type: str = "not_ambiguous") -> Dict[str, Any]:
        """
        Evaluate quality of retrieved documents.

        Args:
            user_query: User query
            retrieved_books: List of retrieved books
            ambiguity_type: Type of ambiguity detected

        Returns:
            {
                "sufficient": bool,
                "reason": str,
                "missing_info": List[str]
            }
        """
        # Format books for evaluation
        books_summary = "\n".join([
            f"{i+1}. [{book.get('구분', 'N/A')}] {book.get('상품명', 'N/A')}"
            for i, book in enumerate(retrieved_books)
        ])

        # Conditional Logic based on Ambiguity Type
        if ambiguity_type == "not_ambiguous":
            # Permissive logic (Enhanced F1 optimized)
            criteria = """
                        1. **관련성**: 검색된 책들 중 사용자의 의도와 조금이라도 연관된 책이 있는가?
                        2. **최소 충족**: 완벽하지 않더라도, 사용자에게 추천해 줄 만한 후보가 1권이라도 있는가?
                        3. **다양성 인정**: 장르나 테마가 다양하더라도, 긍정적으로 평가하라.
                        """
            strictness_instruction = "정말로 엉뚱한 책만 있거나, 추천할 만한 책이 단 한 권도 없을 때만 `sufficient: false`로 판단하세요."
        else:
            # Strict logic for vague/situational/multi_intent/emotional_only (Triggers Clarification)
            criteria = """
                        1. **구체적 적합성**: 검색 결과가 사용자의 복잡하거나 모호한 요구(상황, 다중 의도)를 **명확히 해소**해주는가?
                        2. **정보 부족 여부**: 사용자의 의도를 만족시키기에 **정보가 부족**하여, 추가 질문(장르, 분위기 등)을 하는 것이 더 나은가?
                        3. **다양성 주의**: 결과가 너무 중구난방이어서 사용자에게 혼란을 줄 것 같다면 `sufficient: false`로 판단하라.
                        """
            strictness_instruction = "추천하기에 조금이라도 애매하거나, 추가 정보를 묻는 것이 사용자에게 **더 나은 추천**을 줄 수 있다면 과감하게 `sufficient: false`로 판단하세요."

        system_prompt = f"""당신은 도서 추천 검색 결과의 품질을 평가하는 전문가입니다.

                        검색 결과가 사용자 요청에 **충분히 응답 가능한지** 판단하세요.
                        
                        **현재 모호성 타입**: {ambiguity_type}

                        **평가 기준**:
                        {criteria}

                        **주의**: 
                        - {strictness_instruction}

                        JSON 형식으로 답하세요:
                        {{
                        "sufficient": true/false,
                        "reason": "판단 근거",
                        "missing_info": ["추가로 필요한 정보1"] (필수적인 경우에만 작성)
                        }}

                        missing_info는 `sufficient: false`일 때 **반드시** 작성하세요."""

        user_message = f"""사용자 요청: {user_query}

                        검색된 책들:
                        {books_summary}

                        이 결과가 충분한가요?"""

        model = self._get_model()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = model.invoke(messages)

        # Parse JSON response
        import json
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return result
        except (json.JSONDecodeError, IndexError):
            # Fallback: assume sufficient
            return {
                "sufficient": True,
                "reason": "파싱 실패, 기본값 사용",
                "missing_info": []
            }


class ClarificationQuestionGenerator:
    """Generates minimal, choice-based clarification questions."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.CHAIN_MODEL_NAME
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = init_chat_model(self.model_name)
        return self.model

    def generate(self, user_query: str, missing_info: List[str]) -> str:
        """
        Generate a single clarification question with choices.

        Args:
            user_query: Original user query
            missing_info: List of missing information pieces

        Returns:
            Clarification question with 2-3 choices
        """
        system_prompt = """당신은 명확한 질문을 만드는 전문가입니다.

                        **원칙**:
                        1. 질문은 반드시 **1개**만
                        2. **선택지 형태**로 제공 (2-3개 옵션)
                        3. 사용자가 번호나 키워드로 쉽게 답할 수 있게

                        예시:
                        "지금 상황에 더 맞는 쪽은 어느 쪽인가요?
                        1) 가볍게 읽히는 위로 위주
                        2) 생각할 거리를 주는 내용"

                        또는:
                        "어떤 형식을 선호하시나요?
                        1) 소설 (이야기 중심)
                        2) 에세이 (산문 형식)
                        3) 실용서 (정보 제공)"

                        **오직 질문과 선택지만 출력하세요.**"""

        missing_str = ", ".join(missing_info)
        user_message = f"""사용자 요청: {user_query}

                        부족한 정보: {missing_str}

                        이 중 가장 중요한 것 1개에 대해 선택지 질문을 만드세요."""

        model = self._get_model()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = model.invoke(messages)
        return response.content.strip()


class FinalRecommender:
    """Generates final book recommendation with context."""

    def __init__(self, model_name: str = None):
        # Use main chat model for final recommendation (needs better quality)
        self.model_name = model_name or Config.CHAT_MODEL_NAME
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = init_chat_model(self.model_name)
        return self.model

    def recommend(
        self,
        user_query: str,
        retrieved_books: List[Dict[str, Any]],
        user_state_summary: str = "",
        clarification_history: List[Dict[str, str]] = None,
        include_links: bool = True
    ) -> str:
        """
        Generate final recommendation.

        Args:
            user_query: Original user query
            retrieved_books: List of retrieved book documents
            user_state_summary: Summary of user's inferred state
            clarification_history: List of Q&A during clarification
            include_links: Whether to include purchase links (default: True)

        Returns:
            Recommendation text with purchase links
        """
        # Format books context
        books_context = "\n\n".join([
            f"[{book.get('구분', 'N/A')}] {book.get('상품명', 'N/A')}\n"
            f"소개: {book.get('책소개', 'N/A')[:200]}..."
            for book in retrieved_books
        ])

        # Format clarification history
        clarification_text = ""
        if clarification_history:
            clarification_text = "\n\n추가 확인 내용:\n" + "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}"
                for item in clarification_history
            ])

        system_prompt = f"""당신은 전문 도서 추천 큐레이터입니다.

                            **추천 원칙**:
                            1. 이 추천은 **불완전한 사용자 정보**를 기반으로 함을 인정
                            2. 사용자 응답에 따라 추천을 **갱신할 수 있음**을 명시
                            3. 각 책의 특징과 추천 이유를 구체적으로 설명
                            4. 베스트셀러 순위가 있다면 참고하되, 맹목적으로 따르지 않음

                            **사용자 현재 상태 추론**:
                            {user_state_summary if user_state_summary else "명시적 정보 없음"}

                            **추천 스타일**:
                            - 2-3권 추천
                            - 각 책마다: [카테고리] **제목** - 추천 이유 (1-2문장)
                            - **중요**: 책 제목은 반드시 **볼드체**로 표시하세요 (예: **책 제목**)
                            - 따뜻하고 친근한 어조
                            - 마지막에 "다른 스타일을 원하시면 말씀해주세요" 추가"""

        user_message = f"""사용자 요청: {user_query}
                        {clarification_text}

                        추천 가능한 책들:
                        {books_context}

                        이 중에서 사용자에게 가장 적합한 책을 추천해주세요.
                        **책 제목은 반드시 볼드체로 표시하세요.**"""

        model = self._get_model()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = model.invoke(messages)
        recommendation_text = response.content.strip()

        # Add purchase links if requested
        if include_links:
            recommendation_text = self._add_purchase_links(recommendation_text, retrieved_books)

        return recommendation_text

    def _add_purchase_links(self, recommendation_text: str, retrieved_books: List[Dict[str, Any]]) -> str:
        """
        Add purchase links to recommendation text based on books mentioned in the final recommendation.

        Args:
            recommendation_text: Original recommendation text
            retrieved_books: List of retrieved book documents

        Returns:
            Recommendation text with purchase links appended
        """
        import re

        # Extract book titles that are bolded in the recommendation text (using **title** format)
        # Pattern: **책 제목** format
        mentioned_titles = re.findall(r'\*\*([^*]+)\*\*', recommendation_text)

        if not mentioned_titles:
            return recommendation_text

        # Create a mapping of book titles to book data for quick lookup
        book_map = {book.get('상품명', ''): book for book in retrieved_books if book.get('상품명')}

        # Generate links section
        links_section = "\n\n---\n\n### 🛒 구매 링크\n\n"

        added_books = set()  # Track books we've already added to avoid duplicates

        for title in mentioned_titles:
            # Skip if this title was already added
            if title in added_books:
                continue

            # Find matching book in retrieved books (exact match or fuzzy match)
            matched_book = None

            # First try exact match
            if title in book_map:
                matched_book = book_map[title]
            else:
                # Try fuzzy match: check if the mentioned title is a substring of any retrieved book
                for book_title, book_data in book_map.items():
                    if title in book_title or book_title in title:
                        matched_book = book_data
                        break

            if matched_book:
                author = matched_book.get('저자/아티스트', None)
            else:
                # If no match found in retrieved books, still generate links based on the mentioned title
                author = None

            # Generate links
            links = generate_search_links(title, author)

            links_section += f"**{title}**\n"
            links_section += f"- [📚 YES24 검색]({links['yes24']})\n"
            links_section += f"- [📖 알라딘 검색]({links['aladin']})\n\n"

            added_books.add(title)

        return recommendation_text + links_section
