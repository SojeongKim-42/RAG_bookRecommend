"""
Evaluation dataset for RAG book recommendation system.
Contains test queries with ground truth for evaluating retrieval and recommendation quality.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Type of query based on specificity."""
    SPECIFIC = "specific"  # 명확한 장르/주제 지정
    EMOTIONAL = "emotional"  # 감정 기반
    SITUATIONAL = "situational"  # 상황 기반
    VAGUE = "vague"  # 모호한 표현
    MULTI_INTENT = "multi_intent"  # 복합 의도


class GenreCategory(Enum):
    """Main genre categories based on preprocessed data."""
    NOVEL = "소설/시/희곡"
    GENRE_NOVEL = "장르소설"
    ESSAY = "에세이"
    SELF_HELP = "자기계발"
    HUMANITIES = "인문학"
    SOCIAL = "사회과학"
    HISTORY = "역사"
    COMIC = "만화"
    TEXTBOOK = "대학교재/전문서적"
    CHILDREN = "어린이"
    TODDLER = "유아"
    TEEN = "청소년"
    TRAVEL = "여행"
    RELIGION = "종교/역학"
    ART = "예술/대중문화"
    COOKING = "요리/살림"
    PARENTING = "좋은부모"


@dataclass
class TestQuery:
    """Test query with ground truth."""

    # Query information
    query_id: str
    query: str
    query_type: QueryType

    # Ground truth - Expected genres
    expected_genres: List[GenreCategory]

    # Ground truth - Relevant book titles (if any)
    relevant_books: Optional[List[str]] = None

    # Additional context for evaluation
    expected_themes: Optional[List[str]] = None  # 예: ["위로", "성장", "사랑"]
    expected_mood: Optional[str] = None  # 예: "따뜻한", "긴장감 있는"

    # Notes for evaluation
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "query_type": self.query_type.value,
            "expected_genres": [g.value for g in self.expected_genres],
            "relevant_books": self.relevant_books,
            "expected_themes": self.expected_themes,
            "expected_mood": self.expected_mood,
            "notes": self.notes
        }


class EvaluationDataset:
    """Dataset of test queries for evaluation."""

    def __init__(self):
        self.queries: List[TestQuery] = []
        self._initialize_dataset()

    def _initialize_dataset(self):
        """Initialize the evaluation dataset with diverse test queries."""

        # Category 1: Specific genre/topic queries (명확한 요청)
        self.queries.extend([
            TestQuery(
                query_id="S001",
                query="SF 소설 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.NOVEL, GenreCategory.GENRE_NOVEL],
                expected_themes=["SF", "공상과학"],
                notes="명확한 장르 지정"
            ),
            TestQuery(
                query_id="S002",
                query="마케팅 관련 실용서 필요해",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.SOCIAL, GenreCategory.TEXTBOOK],
                expected_themes=["마케팅"],
                notes="비즈니스 분야 명확 (전문서적 포함)"
            ),
            TestQuery(
                query_id="S003",
                query="한국 현대 소설 중에서 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.NOVEL],
                expected_themes=["한국문학", "현대소설"],
                notes="장르와 국적 지정"
            ),
            TestQuery(
                query_id="S004",
                query="역사 관련 인문서 추천",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.HUMANITIES, GenreCategory.HISTORY],
                expected_themes=["역사"],
                notes="인문 분야 세부 주제 지정"
            ),
            TestQuery(
                query_id="S005",
                query="에세이 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.ESSAY],
                notes="장르만 명확"
            ),
        ])

        # Category 2: Emotional queries (감정 기반)
        self.queries.extend([
            TestQuery(
                query_id="E001",
                query="요즘 너무 우울해",
                query_type=QueryType.EMOTIONAL,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.SELF_HELP, GenreCategory.NOVEL],
                expected_themes=["위로", "힐링", "공감"],
                expected_mood="따뜻한",
                notes="우울감 -> 위로 필요"
            ),
            TestQuery(
                query_id="E002",
                query="기분이 안 좋아서 뭔가 읽고 싶어",
                query_type=QueryType.EMOTIONAL,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.NOVEL],
                expected_themes=["위로", "힐링"],
                notes="감정 표현만 있음"
            ),
            TestQuery(
                query_id="E003",
                query="무기력한데 동기부여 받고 싶어",
                query_type=QueryType.EMOTIONAL,
                expected_genres=[GenreCategory.SELF_HELP, GenreCategory.ESSAY],
                expected_themes=["동기부여", "성장", "자기계발"],
                expected_mood="에너지 넘치는",
                notes="무기력 -> 동기부여"
            ),
            TestQuery(
                query_id="E004",
                query="외로움을 달래줄 책",
                query_type=QueryType.EMOTIONAL,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.NOVEL],
                expected_themes=["위로", "공감", "관계"],
                notes="외로움 해소"
            ),
            TestQuery(
                query_id="E005",
                query="스트레스 받아서 현실도피하고 싶어",
                query_type=QueryType.EMOTIONAL,
                expected_genres=[GenreCategory.NOVEL, GenreCategory.GENRE_NOVEL, GenreCategory.COMIC],
                expected_themes=["판타지", "로맨스", "가벼움"],
                expected_mood="가벼운",
                notes="현실도피 욕구"
            ),
        ])

        # Category 3: Situational queries (상황 기반)
        self.queries.extend([
            TestQuery(
                query_id="SI001",
                query="군대 가기 전에 읽을 책",
                query_type=QueryType.SITUATIONAL,
                expected_genres=[GenreCategory.SELF_HELP, GenreCategory.HUMANITIES, GenreCategory.NOVEL],
                expected_themes=["성찰", "인생", "가치관"],
                notes="입대 전 상황"
            ),
            TestQuery(
                query_id="SI002",
                query="출퇴근할 때 읽을만한 거",
                query_type=QueryType.SITUATIONAL,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.NOVEL],
                expected_themes=["가벼움", "짧은 호흡"],
                notes="통근 시간 -> 짧은 분량"
            ),
            TestQuery(
                query_id="SI003",
                query="취업 준비하는데 도움될 책",
                query_type=QueryType.SITUATIONAL,
                expected_genres=[GenreCategory.SELF_HELP, GenreCategory.SOCIAL, GenreCategory.ESSAY],
                expected_themes=["취업", "자기계발", "경력"],
                notes="취업 준비 상황 (취업 에세이 포함)"
            ),
            TestQuery(
                query_id="SI004",
                query="잠들기 전에 읽기 좋은 책",
                query_type=QueryType.SITUATIONAL,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.NOVEL],
                expected_themes=["잔잔함", "평화"],
                expected_mood="차분한",
                notes="취침 전 -> 자극적이지 않은"
            ),
            TestQuery(
                query_id="SI005",
                query="대학 신입생인데 읽으면 좋을 책",
                query_type=QueryType.SITUATIONAL,
                expected_genres=[GenreCategory.SELF_HELP, GenreCategory.HUMANITIES, GenreCategory.ESSAY],
                expected_themes=["성장", "인생", "교양"],
                notes="대학 입학 시점"
            ),
        ])

        # Category 4: Vague queries (모호한 표현)
        self.queries.extend([
            TestQuery(
                query_id="V001",
                query="인생에 도움되는 책",
                query_type=QueryType.VAGUE,
                expected_genres=[GenreCategory.SELF_HELP, GenreCategory.HUMANITIES, GenreCategory.ESSAY],
                expected_themes=["성장", "지혜", "교훈"],
                notes="매우 광범위한 요청"
            ),
            TestQuery(
                query_id="V002",
                query="성장할 수 있는 책",
                query_type=QueryType.VAGUE,
                expected_genres=[GenreCategory.SELF_HELP, GenreCategory.HUMANITIES],
                expected_themes=["성장", "자기계발"],
                notes="추상적 목표"
            ),
            TestQuery(
                query_id="V003",
                query="재밌는 책 추천",
                query_type=QueryType.VAGUE,
                expected_genres=[GenreCategory.NOVEL, GenreCategory.GENRE_NOVEL, GenreCategory.ESSAY, GenreCategory.COMIC],
                notes="'재밌다'의 기준 불명확"
            ),
            TestQuery(
                query_id="V004",
                query="교양 쌓을 수 있는 책",
                query_type=QueryType.VAGUE,
                expected_genres=[GenreCategory.HUMANITIES, GenreCategory.SOCIAL, GenreCategory.ART, GenreCategory.HISTORY],
                expected_themes=["교양", "지식"],
                notes="교양의 범위 광범위 (전문서적 제외)"
            ),
            TestQuery(
                query_id="V005",
                query="의미있는 책 찾아",
                query_type=QueryType.VAGUE,
                expected_genres=[GenreCategory.HUMANITIES, GenreCategory.ESSAY, GenreCategory.NOVEL],
                notes="'의미'의 기준 주관적"
            ),
        ])

        # Category 5: Multi-intent queries (복합 의도)
        self.queries.extend([
            TestQuery(
                query_id="M001",
                query="재밌고 의미 있는 소설",
                query_type=QueryType.MULTI_INTENT,
                expected_genres=[GenreCategory.NOVEL, GenreCategory.GENRE_NOVEL],
                expected_themes=["재미", "의미", "교훈"],
                notes="재미와 의미 둘 다 요구"
            ),
            TestQuery(
                query_id="M002",
                query="감동적이면서 유익한 책",
                query_type=QueryType.MULTI_INTENT,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.HUMANITIES, GenreCategory.SELF_HELP],
                expected_themes=["감동", "유익", "교훈"],
                notes="감성과 실용성 동시 요구"
            ),
            TestQuery(
                query_id="M003",
                query="가볍게 읽히면서 생각할 거리를 주는 책",
                query_type=QueryType.MULTI_INTENT,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.NOVEL],
                expected_themes=["가벼움", "사색", "철학"],
                notes="가벼움과 깊이 동시 요구"
            ),
            TestQuery(
                query_id="M004",
                query="짧으면서도 울림이 큰 에세이",
                query_type=QueryType.MULTI_INTENT,
                expected_genres=[GenreCategory.ESSAY],
                expected_themes=["간결함", "감동", "울림"],
                notes="분량과 감동 모두 중요"
            ),
            TestQuery(
                query_id="M005",
                query="실용적이면서 흥미로운 경제 책",
                query_type=QueryType.MULTI_INTENT,
                expected_genres=[GenreCategory.SOCIAL],
                expected_themes=["실용성", "흥미", "경제"],
                notes="실용성과 재미 병행"
            ),
        ])

        # Category 6: Additional diverse queries
        self.queries.extend([
            TestQuery(
                query_id="A001",
                query="여행 가서 읽을 책",
                query_type=QueryType.SITUATIONAL,
                expected_genres=[GenreCategory.NOVEL, GenreCategory.ESSAY, GenreCategory.TRAVEL],
                expected_themes=["여행", "휴식", "가벼움"],
                notes="여행지 독서"
            ),
            TestQuery(
                query_id="A002",
                query="20대 여성이 공감할 만한 에세이",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.ESSAY],
                expected_themes=["공감", "20대", "여성"],
                notes="타겟 독자 명확"
            ),
            TestQuery(
                query_id="A003",
                query="인간관계에 대해 생각하게 하는 책",
                query_type=QueryType.VAGUE,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.HUMANITIES, GenreCategory.NOVEL, GenreCategory.SELF_HELP],
                expected_themes=["관계", "인간", "소통"],
                notes="주제는 명확하나 장르 미지정"
            ),
            TestQuery(
                query_id="A004",
                query="추리 소설 중에서 반전이 좋은 거",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.NOVEL, GenreCategory.GENRE_NOVEL],
                expected_themes=["추리", "반전", "미스터리"],
                notes="장르와 특징 모두 지정"
            ),
            TestQuery(
                query_id="A005",
                query="아침에 읽으면 하루가 달라질 것 같은 책",
                query_type=QueryType.EMOTIONAL,
                expected_genres=[GenreCategory.ESSAY, GenreCategory.SELF_HELP],
                expected_themes=["동기부여", "긍정", "아침"],
                expected_mood="에너지 넘치는",
                notes="시간대와 효과 명시"
            ),
        ])

        # Category 7: Missing genre coverage (누락된 장르 커버리지)
        self.queries.extend([
            # 좋은부모 (PARENTING)
            TestQuery(
                query_id="P001",
                query="아이 독서 습관 키우는 방법 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.PARENTING],
                expected_themes=["독서교육", "자녀교육", "습관"],
                notes="육아/교육 분야"
            ),
            TestQuery(
                query_id="P002",
                query="초등학생 자녀 교육서 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.PARENTING],
                expected_themes=["초등교육", "자녀교육"],
                notes="학령기 자녀 교육"
            ),
            # 요리/살림 (COOKING)
            TestQuery(
                query_id="C001",
                query="집에서 간단하게 만들 수 있는 요리책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.COOKING],
                expected_themes=["요리", "레시피", "간편식"],
                notes="요리 입문"
            ),
            TestQuery(
                query_id="C002",
                query="살림 초보를 위한 가정 관리 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.COOKING],
                expected_themes=["살림", "가정관리", "정리"],
                notes="살림/가사 분야"
            ),
            # 종교/역학 (RELIGION)
            TestQuery(
                query_id="R001",
                query="불교 입문서 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.RELIGION],
                expected_themes=["불교", "명상", "수행"],
                notes="종교 입문서"
            ),
            TestQuery(
                query_id="R002",
                query="명상이나 마음 수련 관련 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.RELIGION, GenreCategory.SELF_HELP],
                expected_themes=["명상", "마음챙김", "수련"],
                notes="명상/영성 분야"
            ),
            # 유아 (TODDLER)
            TestQuery(
                query_id="T001",
                query="3살 아이에게 읽어줄 그림책 추천",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TODDLER],
                expected_themes=["그림책", "유아", "읽어주기"],
                notes="유아 그림책"
            ),
            TestQuery(
                query_id="T002",
                query="유아 발달에 좋은 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TODDLER, GenreCategory.PARENTING],
                expected_themes=["유아발달", "조기교육"],
                notes="유아 발달/교육"
            ),
            # 어린이 (CHILDREN)
            TestQuery(
                query_id="CH001",
                query="초등학생이 읽기 좋은 동화책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.CHILDREN],
                expected_themes=["동화", "초등", "어린이문학"],
                notes="초등학생 대상"
            ),
            TestQuery(
                query_id="CH002",
                query="어린이 과학책 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.CHILDREN],
                expected_themes=["과학", "어린이", "학습"],
                notes="어린이 학습서"
            ),
            # 청소년 (TEEN)
            TestQuery(
                query_id="TE001",
                query="중학생 추천 도서 알려줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TEEN],
                expected_themes=["청소년", "중학생", "성장"],
                notes="중학생 권장도서"
            ),
            TestQuery(
                query_id="TE002",
                query="고등학생이 읽으면 좋은 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TEEN, GenreCategory.HUMANITIES],
                expected_themes=["고등학생", "입시", "교양"],
                notes="고등학생 권장도서"
            ),
        ])

        # Category 8: Underrepresented genre reinforcement (부족한 장르 보강)
        self.queries.extend([
            # 만화 (COMIC) - 기존 1회
            TestQuery(
                query_id="CO001",
                query="재미있는 만화책 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.COMIC],
                expected_themes=["만화", "웹툰", "그래픽"],
                notes="만화 일반"
            ),
            TestQuery(
                query_id="CO002",
                query="일본 만화 명작 추천",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.COMIC],
                expected_themes=["일본만화", "명작"],
                notes="일본 만화"
            ),
            # 여행 (TRAVEL) - 기존 1회
            TestQuery(
                query_id="TR001",
                query="유럽 여행 가이드북 추천",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TRAVEL],
                expected_themes=["유럽", "여행", "가이드"],
                notes="해외여행 가이드"
            ),
            TestQuery(
                query_id="TR002",
                query="국내 여행지 소개하는 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TRAVEL],
                expected_themes=["국내여행", "여행지"],
                notes="국내여행"
            ),
            # 예술/대중문화 (ART) - 기존 1회
            TestQuery(
                query_id="AR001",
                query="미술 입문자를 위한 책",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.ART],
                expected_themes=["미술", "예술", "입문"],
                notes="미술 입문"
            ),
            TestQuery(
                query_id="AR002",
                query="영화 관련 책 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.ART],
                expected_themes=["영화", "시네마", "대중문화"],
                notes="영화/대중문화"
            ),
            # 대학교재/전문서적 (TEXTBOOK) - 기존 1회
            TestQuery(
                query_id="TX001",
                query="프로그래밍 입문서 추천",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TEXTBOOK],
                expected_themes=["프로그래밍", "코딩", "개발"],
                notes="IT/개발 전문서"
            ),
            TestQuery(
                query_id="TX002",
                query="통계학 기초 책 추천해줘",
                query_type=QueryType.SPECIFIC,
                expected_genres=[GenreCategory.TEXTBOOK],
                expected_themes=["통계", "수학", "데이터"],
                notes="통계/수학 전문서"
            ),
        ])

    def get_all_queries(self) -> List[TestQuery]:
        """Get all test queries."""
        return self.queries

    def get_queries_by_type(self, query_type: QueryType) -> List[TestQuery]:
        """Get queries filtered by type."""
        return [q for q in self.queries if q.query_type == query_type]

    def get_query_by_id(self, query_id: str) -> Optional[TestQuery]:
        """Get a specific query by ID."""
        for query in self.queries:
            if query.query_id == query_id:
                return query
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_queries": len(self.queries),
            "by_type": {}
        }

        for query_type in QueryType:
            count = len(self.get_queries_by_type(query_type))
            stats["by_type"][query_type.value] = count

        return stats

    def get_genre_coverage(self) -> Dict[str, int]:
        """
        Get genre coverage statistics.

        Returns:
            Dictionary mapping genre name to count of appearances in expected_genres
        """
        coverage = {genre.value: 0 for genre in GenreCategory}

        for query in self.queries:
            for genre in query.expected_genres:
                coverage[genre.value] += 1

        return coverage

    def get_uncovered_genres(self) -> List[str]:
        """
        Get list of genres with zero coverage.

        Returns:
            List of genre names that are never expected in any query
        """
        coverage = self.get_genre_coverage()
        return [genre for genre, count in coverage.items() if count == 0]

    def get_underrepresented_genres(self, threshold: int = 2) -> List[str]:
        """
        Get list of genres with coverage below threshold.

        Args:
            threshold: Minimum expected coverage count

        Returns:
            List of genre names below threshold
        """
        coverage = self.get_genre_coverage()
        return [genre for genre, count in coverage.items() if count < threshold]

    def print_genre_coverage_report(self):
        """Print a formatted genre coverage report."""
        coverage = self.get_genre_coverage()

        print("\n=== Genre Coverage Report ===")
        print(f"{'Genre':<25} {'Count':>6} {'Status':<15}")
        print("-" * 50)

        # Sort by count descending
        sorted_coverage = sorted(coverage.items(), key=lambda x: x[1], reverse=True)

        for genre, count in sorted_coverage:
            if count == 0:
                status = "🔴 MISSING"
            elif count < 2:
                status = "🟡 LOW"
            else:
                status = "✅ OK"
            print(f"{genre:<25} {count:>6} {status:<15}")

        print("-" * 50)
        print(f"Total genres: {len(coverage)}")
        print(f"Missing: {len(self.get_uncovered_genres())}")
        print(f"Underrepresented (<2): {len(self.get_underrepresented_genres())}")

    def to_json(self) -> List[Dict[str, Any]]:
        """Convert entire dataset to JSON-serializable format."""
        return [q.to_dict() for q in self.queries]


if __name__ == "__main__":
    # Test the dataset
    dataset = EvaluationDataset()
    stats = dataset.get_statistics()

    print("=== Evaluation Dataset Statistics ===")
    print(f"Total queries: {stats['total_queries']}")
    print("\nBy query type:")
    for qtype, count in stats['by_type'].items():
        print(f"  {qtype}: {count}")

    # Print genre coverage report
    dataset.print_genre_coverage_report()

    print("\n=== Sample Queries ===")
    for query_type in QueryType:
        samples = dataset.get_queries_by_type(query_type)[:2]
        if samples:
            print(f"\n{query_type.value}:")
            for sample in samples:
                print(f"  - {sample.query_id}: {sample.query}")
