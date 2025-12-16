"""
Evaluation metrics for RAG book recommendation system.
Focuses on genre appropriateness, retrieval quality, and recommendation quality.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from evaluation.dataset import TestQuery, GenreCategory, QueryType


@dataclass
class RetrievalResult:
    """Result from retrieval system."""
    query_id: str
    query: str
    retrieved_books: List[Dict[str, Any]]  # List of book metadata
    retrieval_scores: Optional[List[float]] = None  # Similarity scores if available


@dataclass
class GenreMetrics:
    """Metrics for genre appropriateness."""
    # Genre matching
    genre_precision: float  # 검색된 책 중 올바른 장르 비율
    genre_recall: float  # 기대 장르를 모두 커버했는가
    genre_f1: float  # F1 score

    # Diversity
    genre_diversity: float  # 장르 다양성 (0-1)
    unique_genres: int  # 유니크 장르 수

    # Distribution
    genre_distribution: Dict[str, float]  # 장르별 비율

    # Details
    expected_genres: List[str]
    retrieved_genres: List[str]
    matched_genres: List[str]
    missing_genres: List[str]


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    # Ranking metrics
    precision_at_k: Dict[int, float]  # P@1, P@2, P@3, etc.
    recall_at_k: Dict[int, float]  # R@1, R@2, R@3, etc.
    ndcg_at_k: Dict[int, float]  # NDCG@1, NDCG@2, etc.
    mrr: float  # Mean Reciprocal Rank

    # Coverage
    coverage: float  # 기대하는 책들을 얼마나 검색했는가


@dataclass
class SemanticMetrics:
    """Metrics for semantic similarity."""
    avg_similarity: float  # 평균 코사인 유사도
    max_similarity: float  # 최대 코사인 유사도 (최고의 검색 결과)
    min_similarity: float  # 최소 코사인 유사도
    similarity_at_k: List[float]  # 각 순위별 유사도


@dataclass
class ThemeMetrics:
    """Metrics for theme matching."""
    theme_recall: float  # 기대 테마가 책 소개에 등장하는 비율
    theme_precision: float  # 책들이 테마와 매칭되는 비율
    matched_themes: List[str]  # 매칭된 테마들
    missing_themes: List[str]  # 누락된 테마들
    theme_coverage_per_book: List[int]  # 각 책별 매칭된 테마 수


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""
    query_id: str
    query: str
    query_type: QueryType

    # Metrics
    genre_metrics: GenreMetrics
    retrieval_metrics: Optional[RetrievalMetrics] = None
    semantic_metrics: Optional[SemanticMetrics] = None
    theme_metrics: Optional[ThemeMetrics] = None

    # Retrieved books
    retrieved_books: List[Dict[str, Any]] = None

    # Notes
    notes: str = ""


class GenreEvaluator:
    """Evaluates genre appropriateness of retrieved books."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def _extract_genre(self, book_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract genre from book metadata.

        Args:
            book_metadata: Book metadata dictionary

        Returns:
            Genre string or None
        """
        # Try to get genre from '구분' field
        genre = book_metadata.get("구분")
        if genre:
            return str(genre).strip()

        # Try from other fields
        category = book_metadata.get("대표분류(대분류명)")
        if category:
            return str(category).strip()

        return None

    def _normalize_genre(self, genre: str) -> str:
        """
        Normalize genre string to standard category.

        Args:
            genre: Raw genre string

        Returns:
            Normalized genre string
        """
        if not genre:
            return "기타"

        genre = genre.lower().strip()

        # Mapping rules - aligned with updated GenreCategory
        # IMPORTANT: Order matters! Specific terms must come before broad/substring terms.
        # e.g., "인문학" before "소설" (as "문학" is in "소설" variations)
        # "장르소설" before "소설"
        genre_mappings = {
            "장르소설": ["장르소설", "추리", "미스터리", "스릴러", "판타지", "SF", "무협", "로맨스", "공포"],
            "인문학": ["인문", "철학", "심리", "윤리"],
            "소설/시/희곡": ["소설", "시", "희곡", "문학", "고전"],
            "사회과학": ["사회", "정치", "행정", "법", "경제", "경영", "비즈니스", "마케팅"],
            "에세이": ["에세이", "수필", "산문"],
            "자기계발": ["자기계발", "성공", "처세", "리더십"],
            "역사": ["역사", "한국사", "세계사", "문화사"],
            "만화": ["만화", "웹툰", "그래픽노블", "코믹"],
            "대학교재/전문서적": ["대학교재", "전문서적", "공학", "자연과학", "컴퓨터", "IT", "기술"],
            "어린이": ["어린이", "아동", "초등"],
            "유아": ["유아", "그림책", "0~"],
            "청소년": ["청소년", "1318"],
            "여행": ["여행", "기행", "답사"],
            "종교/역학": ["종교", "기독교", "불교", "천주교", "역학", "사주"],
            "예술/대중문화": ["예술", "대중문화", "미술", "음악", "영화", "디자인"],
            "요리/살림": ["요리", "살림", "가정", "건강", "취미", "레저"],
            "좋은부모": ["부모", "육아", "임신", "출산", "자녀"],
        }

        for standard, variations in genre_mappings.items():
            if any(var in genre for var in variations):
                return standard

        return genre

    def calculate_genre_precision(
        self,
        expected_genres: List[str],
        retrieved_genres: List[str]
    ) -> float:
        """
        Calculate genre precision.

        Precision = (검색된 책 중 올바른 장르 개수) / (전체 검색된 책 개수)

        Args:
            expected_genres: List of expected genre strings
            retrieved_genres: List of retrieved genre strings

        Returns:
            Precision score (0-1)
        """
        if not retrieved_genres:
            return 0.0

        # Normalize genres
        expected_normalized = [self._normalize_genre(g) for g in expected_genres]
        retrieved_normalized = [self._normalize_genre(g) for g in retrieved_genres]

        # Count matches
        matches = sum(1 for g in retrieved_normalized if g in expected_normalized)

        return matches / len(retrieved_normalized)

    def calculate_genre_recall(
        self,
        expected_genres: List[str],
        retrieved_genres: List[str]
    ) -> float:
        """
        Calculate genre recall.

        Recall = (커버된 기대 장르 수) / (전체 기대 장르 수)

        Args:
            expected_genres: List of expected genre strings
            retrieved_genres: List of retrieved genre strings

        Returns:
            Recall score (0-1)
        """
        if not expected_genres:
            # If no specific genres expected, we can't calculate recall meaningfully.
            # Assuming 1.0 might be misleading, but current logic assumes "any match is good".
            return 1.0

        # Normalize genres
        expected_normalized = set([self._normalize_genre(g) for g in expected_genres])
        retrieved_normalized = set([self._normalize_genre(g) for g in retrieved_genres])

        # Count covered genres
        covered = len(expected_normalized & retrieved_normalized)

        return covered / len(expected_normalized)

    def calculate_genre_f1(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score.

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_genre_diversity(self, retrieved_genres: List[str]) -> Tuple[float, int]:
        """
        Calculate genre diversity.

        Diversity = (유니크 장르 수) / (전체 검색 결과 수)

        Args:
            retrieved_genres: List of retrieved genre strings

        Returns:
            Tuple of (diversity_score, unique_count)
        """
        if not retrieved_genres:
            return 0.0, 0

        # Normalize genres
        normalized = [self._normalize_genre(g) for g in retrieved_genres]

        # Calculate unique genres
        unique_genres = len(set(normalized))

        # Diversity score
        diversity = unique_genres / len(normalized)

        return diversity, unique_genres

    def calculate_genre_distribution(self, retrieved_genres: List[str]) -> Dict[str, float]:
        """
        Calculate genre distribution.

        Args:
            retrieved_genres: List of retrieved genre strings

        Returns:
            Dictionary mapping genre to proportion
        """
        if not retrieved_genres:
            return {}

        # Normalize genres
        normalized = [self._normalize_genre(g) for g in retrieved_genres]

        # Count genres
        counter = Counter(normalized)

        # Calculate proportions
        total = len(normalized)
        distribution = {genre: count / total for genre, count in counter.items()}

        return distribution

    def evaluate(
        self,
        test_query: TestQuery,
        retrieved_books: List[Dict[str, Any]]
    ) -> GenreMetrics:
        """
        Evaluate genre appropriateness.

        Args:
            test_query: Test query with ground truth
            retrieved_books: List of retrieved book metadata

        Returns:
            GenreMetrics object
        """
        # Extract genres
        expected_genres = [g.value for g in test_query.expected_genres]
        retrieved_genres = [
            self._extract_genre(book) for book in retrieved_books
        ]
        retrieved_genres = [g for g in retrieved_genres if g]  # Filter None

        # Calculate metrics
        precision = self.calculate_genre_precision(expected_genres, retrieved_genres)
        recall = self.calculate_genre_recall(expected_genres, retrieved_genres)
        f1 = self.calculate_genre_f1(precision, recall)

        diversity, unique_count = self.calculate_genre_diversity(retrieved_genres)
        distribution = self.calculate_genre_distribution(retrieved_genres)

        # Determine matched and missing genres
        expected_normalized = set([self._normalize_genre(g) for g in expected_genres])
        retrieved_normalized = set([self._normalize_genre(g) for g in retrieved_genres])

        matched = list(expected_normalized & retrieved_normalized)
        missing = list(expected_normalized - retrieved_normalized)

        return GenreMetrics(
            genre_precision=precision,
            genre_recall=recall,
            genre_f1=f1,
            genre_diversity=diversity,
            unique_genres=unique_count,
            genre_distribution=distribution,
            expected_genres=expected_genres,
            retrieved_genres=retrieved_genres,
            matched_genres=matched,
            missing_genres=missing
        )


class RetrievalEvaluator:
    """Evaluates retrieval quality metrics."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def calculate_precision_at_k(
        self,
        retrieved_books: List[Dict[str, Any]],
        relevant_books: List[str],
        k_values: List[int] = [1, 2, 3, 5]
    ) -> Dict[int, Optional[float]]:
        """
        Calculate Precision@K.

        P@K = (관련 문서 수 in top-K) / K

        Args:
            retrieved_books: List of retrieved book metadata (in rank order)
            relevant_books: List of relevant book titles
            k_values: List of K values to evaluate

        Returns:
            Dictionary mapping K to precision score (or None if no ground truth)
        """
        if not relevant_books:
            return {k: None for k in k_values}  # Corrected: Return None if no ground truth

        results = {}

        for k in k_values:
            top_k = retrieved_books[:k]
            top_k_titles = [book.get("상품명", "") for book in top_k]

            # Count relevant in top-k
            relevant_count = sum(
                1 for title in top_k_titles
                if any(rel in title for rel in relevant_books)
            )

            results[k] = relevant_count / k if k > 0 else 0.0

        return results

    def calculate_recall_at_k(
        self,
        retrieved_books: List[Dict[str, Any]],
        relevant_books: List[str],
        k_values: List[int] = [1, 2, 3, 5]
    ) -> Dict[int, Optional[float]]:
        """
        Calculate Recall@K.

        R@K = (관련 문서 수 in top-K) / (전체 관련 문서 수)

        Args:
            retrieved_books: List of retrieved book metadata
            relevant_books: List of relevant book titles
            k_values: List of K values to evaluate

        Returns:
            Dictionary mapping K to recall score (or None if no ground truth)
        """
        if not relevant_books:
            return {k: None for k in k_values} # Corrected: Return None if no ground truth

        results = {}

        for k in k_values:
            top_k = retrieved_books[:k]
            top_k_titles = [book.get("상품명", "") for book in top_k]

            # Count relevant in top-k
            relevant_count = sum(
                1 for title in top_k_titles
                if any(rel in title for rel in relevant_books)
            )

            results[k] = relevant_count / len(relevant_books)

        return results

    def calculate_mrr(
        self,
        retrieved_books: List[Dict[str, Any]],
        relevant_books: List[str]
    ) -> Optional[float]:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1 / (첫 번째 관련 문서의 순위)

        Args:
            retrieved_books: List of retrieved book metadata
            relevant_books: List of relevant book titles

        Returns:
            MRR score (or None if no ground truth)
        """
        if not relevant_books:
            return None # Corrected: Return None if no ground truth

        for rank, book in enumerate(retrieved_books, start=1):
            title = book.get("상품명", "")
            if any(rel in title for rel in relevant_books):
                return 1.0 / rank

        return 0.0  # No relevant book found

    def evaluate(
        self,
        test_query: TestQuery,
        retrieved_books: List[Dict[str, Any]]
    ) -> Optional[RetrievalMetrics]:
        """
        Evaluate retrieval quality.

        Args:
            test_query: Test query with ground truth
            retrieved_books: List of retrieved book metadata

        Returns:
            RetrievalMetrics or None if no ground truth available
        """
        # Check if we have ground truth
        if not test_query.relevant_books:
            return None

        # Calculate metrics
        precision_at_k = self.calculate_precision_at_k(
            retrieved_books,
            test_query.relevant_books
        )

        recall_at_k = self.calculate_recall_at_k(
            retrieved_books,
            test_query.relevant_books
        )

        mrr = self.calculate_mrr(
            retrieved_books,
            test_query.relevant_books
        )

        # Calculate coverage
        all_titles = [book.get("상품명", "") for book in retrieved_books]
        covered = sum(
            1 for rel in test_query.relevant_books
            if any(rel in title for title in all_titles)
        )
        coverage = covered / len(test_query.relevant_books)

        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k={},  # TODO: Implement NDCG
            mrr=mrr or 0.0,
            coverage=coverage
        )


class SemanticEvaluator:
    """Evaluates semantic similarity between query and retrieved books."""

    def __init__(self, vectorstore_manager):
        """
        Initialize semantic evaluator.

        Args:
            vectorstore_manager: VectorStoreManager instance with embedding model
        """
        self.vectorstore_manager = vectorstore_manager

    def evaluate(
        self,
        query: str,
        retrieved_books: List[Dict[str, Any]]
    ) -> SemanticMetrics:
        """
        Calculate semantic similarity metrics.

        Args:
            query: User query string
            retrieved_books: List of retrieved book metadata

        Returns:
            SemanticMetrics object
        """
        if not retrieved_books or not self.vectorstore_manager.embeddings:
            return SemanticMetrics(0.0, 0.0, 0.0, [])

        try:
            # 1. Embed query
            query_embedding = self.vectorstore_manager.embeddings.embed_query(query)

            # 2. Embed retrieved books (combine title and description)
            book_texts = []
            for book in retrieved_books:
                title = book.get("상품명", "")
                desc = book.get("책소개", "")
                # Create a representative text for the book
                text = f"{title}. {desc}"[:1000] # Limit length
                book_texts.append(text)

            book_embeddings = self.vectorstore_manager.embeddings.embed_documents(book_texts)

            # 3. Calculate cosine similarity
            # Reshape query for scikit-learn
            query_vec = np.array(query_embedding).reshape(1, -1)
            book_vecs = np.array(book_embeddings)

            # Calculate cosine similarity (1 x N matrix)
            similarities = cosine_similarity(query_vec, book_vecs)[0]

            # 4. Compute metrics
            return SemanticMetrics(
                avg_similarity=float(np.mean(similarities)),
                max_similarity=float(np.max(similarities)),
                min_similarity=float(np.min(similarities)),
                similarity_at_k=[float(s) for s in similarities]
            )

        except Exception as e:
            print(f"Error in semantic evaluation: {str(e)}")
            return SemanticMetrics(0.0, 0.0, 0.0, [])


class ThemeEvaluator:
    """Evaluates theme matching between expected themes and retrieved books."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def _get_book_text(self, book_metadata: Dict[str, Any]) -> str:
        """
        Extract searchable text from book metadata.

        Args:
            book_metadata: Book metadata dictionary

        Returns:
            Combined text from title, description, etc.
        """
        title = book_metadata.get("상품명", "")
        description = book_metadata.get("책소개", "")
        author_intro = book_metadata.get("저자소개", "")

        return f"{title} {description} {author_intro}".lower()

    def evaluate(
        self,
        test_query: TestQuery,
        retrieved_books: List[Dict[str, Any]]
    ) -> Optional[ThemeMetrics]:
        """
        Evaluate theme matching.

        Args:
            test_query: Test query with expected themes
            retrieved_books: List of retrieved book metadata

        Returns:
            ThemeMetrics or None if no expected themes
        """
        # Check if we have expected themes
        if not test_query.expected_themes:
            return None

        expected_themes = [t.lower() for t in test_query.expected_themes]

        if not retrieved_books:
            return ThemeMetrics(
                theme_recall=0.0,
                theme_precision=0.0,
                matched_themes=[],
                missing_themes=test_query.expected_themes,
                theme_coverage_per_book=[]
            )

        # For each book, check which themes are found
        theme_coverage_per_book = []
        books_with_any_theme = 0

        matched_themes_set = set()

        for book in retrieved_books:
            book_text = self._get_book_text(book)
            matched_count = 0

            for theme in expected_themes:
                if theme in book_text:
                    matched_themes_set.add(theme)
                    matched_count += 1

            theme_coverage_per_book.append(matched_count)
            if matched_count > 0:
                books_with_any_theme += 1

        # Calculate metrics
        matched_themes = list(matched_themes_set)
        missing_themes = [t for t in test_query.expected_themes
                         if t.lower() not in matched_themes_set]

        # Theme recall: proportion of expected themes found in any book
        theme_recall = len(matched_themes) / len(expected_themes) if expected_themes else 0.0

        # Theme precision: proportion of books that match at least one theme
        theme_precision = books_with_any_theme / len(retrieved_books) if retrieved_books else 0.0

        return ThemeMetrics(
            theme_recall=theme_recall,
            theme_precision=theme_precision,
            matched_themes=matched_themes,
            missing_themes=missing_themes,
            theme_coverage_per_book=theme_coverage_per_book
        )


class EvaluationRunner:
    """Runs evaluation on the RAG system."""

    def __init__(
        self,
        vectorstore_manager,
        orchestrator=None,
        use_orchestrator: bool = False
    ):
        """
        Initialize evaluation runner.

        Args:
            vectorstore_manager: VectorStoreManager instance
            orchestrator: Optional orchestrator for end-to-end evaluation
            use_orchestrator: Whether to use orchestrator for retrieval
        """
        self.vectorstore_manager = vectorstore_manager
        self.orchestrator = orchestrator
        self.use_orchestrator = use_orchestrator

        self.genre_evaluator = GenreEvaluator()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.theme_evaluator = ThemeEvaluator()
        # Initialize semantic evaluator
        self.semantic_evaluator = SemanticEvaluator(vectorstore_manager)

    def evaluate_single_query(
        self,
        test_query: TestQuery,
        retrieval_config: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single test query.

        Args:
            test_query: Test query to evaluate
            retrieval_config: Dictionary with retrieval parameters
                - k: Number of documents
                - use_mmr: Use MMR
                - mmr_lambda: MMR lambda parameter
                - use_reranking: Use reranking
                - use_adaptive_k: Use adaptive k
                - etc.

        Returns:
            EvaluationResult
        """
        retrieval_config = retrieval_config or {}

        # Retrieve documents
        if self.use_orchestrator and self.orchestrator:
            # Use orchestrator for retrieval (includes query rewriting, clarification, etc.)
            result = self.orchestrator.process_query(
                test_query.query,
                include_links=False
            )
            retrieved_books = result.get("state").retrieved_books if result.get("state") else []
        else:
            # Direct retrieval from vector store
            retrieved_docs = self.vectorstore_manager.advanced_search(
                test_query.query,
                **retrieval_config
            )
            
            # Deduplicate by row to ensure metrics are based on unique books
            seen_rows = set()
            unique_books = []
            for doc in retrieved_docs:
                row = doc.metadata.get("row")
                if row is not None:
                    if row not in seen_rows:
                        seen_rows.add(row)
                        unique_books.append(doc.metadata)
                else:
                    # If no row ID, assume unique or keep
                    unique_books.append(doc.metadata)
            
            retrieved_books = unique_books

        # Evaluate genre
        genre_metrics = self.genre_evaluator.evaluate(test_query, retrieved_books)

        # Evaluate retrieval (if ground truth available)
        retrieval_metrics = self.retrieval_evaluator.evaluate(
            test_query,
            retrieved_books
        )

        # Evaluate semantic similarity
        semantic_metrics = self.semantic_evaluator.evaluate(
            test_query.query,
            retrieved_books
        )

        # Evaluate theme matching
        theme_metrics = self.theme_evaluator.evaluate(
            test_query,
            retrieved_books
        )

        return EvaluationResult(
            query_id=test_query.query_id,
            query=test_query.query,
            query_type=test_query.query_type,
            genre_metrics=genre_metrics,
            retrieval_metrics=retrieval_metrics,
            semantic_metrics=semantic_metrics,
            theme_metrics=theme_metrics,
            retrieved_books=retrieved_books,
            notes=test_query.notes or ""
        )

    def evaluate_dataset(
        self,
        test_queries: List[TestQuery],
        retrieval_config: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test queries.

        Args:
            test_queries: List of test queries
            retrieval_config: Dictionary with retrieval parameters

        Returns:
            List of EvaluationResult
        """
        results = []

        for i, test_query in enumerate(test_queries, 1):
            print(f"Evaluating query {i}/{len(test_queries)}: {test_query.query_id}")

            try:
                result = self.evaluate_single_query(test_query, retrieval_config=retrieval_config)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating query {test_query.query_id}: {str(e)}")
                continue

        return results

    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with aggregated metrics
        """
        if not results:
            return {
                "total_queries": 0,
                "genre_metrics": {
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "avg_f1": 0.0,
                    "avg_diversity": 0.0,
                },
                "semantic_metrics": {
                    "avg_similarity": 0.0,
                },
                "theme_metrics": {
                    "avg_recall": 0.0,
                    "avg_precision": 0.0,
                },
                "retrieval_metrics": {},
                "by_query_type": {}
            }

        # Genre metrics
        genre_precisions = [r.genre_metrics.genre_precision for r in results]
        genre_recalls = [r.genre_metrics.genre_recall for r in results]
        genre_f1s = [r.genre_metrics.genre_f1 for r in results]
        genre_diversities = [r.genre_metrics.genre_diversity for r in results]

        # Semantic metrics
        semantic_avgs = [r.semantic_metrics.avg_similarity for r in results if r.semantic_metrics]

        # Theme metrics
        theme_recalls = [r.theme_metrics.theme_recall for r in results if r.theme_metrics]
        theme_precisions = [r.theme_metrics.theme_precision for r in results if r.theme_metrics]

        # Retrieval metrics (P@k, R@k, MRR)
        results_with_retrieval = [r for r in results if r.retrieval_metrics]
        retrieval_metrics_agg = {}

        if results_with_retrieval:
            # Aggregate P@k
            k_values = [1, 2, 3, 5]
            for k in k_values:
                p_at_k_values = [r.retrieval_metrics.precision_at_k.get(k)
                                for r in results_with_retrieval
                                if r.retrieval_metrics.precision_at_k.get(k) is not None]
                if p_at_k_values:
                    retrieval_metrics_agg[f"avg_p@{k}"] = np.mean(p_at_k_values)

            # Aggregate R@k
            for k in k_values:
                r_at_k_values = [r.retrieval_metrics.recall_at_k.get(k)
                                for r in results_with_retrieval
                                if r.retrieval_metrics.recall_at_k.get(k) is not None]
                if r_at_k_values:
                    retrieval_metrics_agg[f"avg_r@{k}"] = np.mean(r_at_k_values)

            # Aggregate MRR
            mrr_values = [r.retrieval_metrics.mrr for r in results_with_retrieval
                         if r.retrieval_metrics.mrr is not None]
            if mrr_values:
                retrieval_metrics_agg["avg_mrr"] = np.mean(mrr_values)

            # Aggregate Coverage
            coverage_values = [r.retrieval_metrics.coverage for r in results_with_retrieval]
            retrieval_metrics_agg["avg_coverage"] = np.mean(coverage_values)

            retrieval_metrics_agg["queries_with_ground_truth"] = len(results_with_retrieval)

        aggregated = {
            "total_queries": len(results),
            "genre_metrics": {
                "avg_precision": np.mean(genre_precisions),
                "avg_recall": np.mean(genre_recalls),
                "avg_f1": np.mean(genre_f1s),
                "avg_diversity": np.mean(genre_diversities),
            },
            "semantic_metrics": {
                "avg_similarity": np.mean(semantic_avgs) if semantic_avgs else 0.0,
            },
            "theme_metrics": {
                "avg_recall": np.mean(theme_recalls) if theme_recalls else 0.0,
                "avg_precision": np.mean(theme_precisions) if theme_precisions else 0.0,
            },
            "retrieval_metrics": retrieval_metrics_agg if retrieval_metrics_agg else {},
            "by_query_type": {}
        }

        # Aggregate by query type
        from collections import defaultdict
        by_type = defaultdict(list)

        for result in results:
            by_type[result.query_type].append(result)

        for query_type, type_results in by_type.items():
            type_precisions = [r.genre_metrics.genre_precision for r in type_results]
            type_recalls = [r.genre_metrics.genre_recall for r in type_results]
            type_f1s = [r.genre_metrics.genre_f1 for r in type_results]

            # Type semantic
            type_semantic_avgs = [r.semantic_metrics.avg_similarity for r in type_results if r.semantic_metrics]

            # Type theme
            type_theme_recalls = [r.theme_metrics.theme_recall for r in type_results if r.theme_metrics]
            type_theme_precisions = [r.theme_metrics.theme_precision for r in type_results if r.theme_metrics]

            # Type retrieval metrics
            type_results_with_retrieval = [r for r in type_results if r.retrieval_metrics]
            type_retrieval_metrics = {}

            if type_results_with_retrieval:
                k_values = [1, 2, 3, 5]
                for k in k_values:
                    p_at_k_values = [r.retrieval_metrics.precision_at_k.get(k)
                                    for r in type_results_with_retrieval
                                    if r.retrieval_metrics.precision_at_k.get(k) is not None]
                    if p_at_k_values:
                        type_retrieval_metrics[f"avg_p@{k}"] = np.mean(p_at_k_values)

                for k in k_values:
                    r_at_k_values = [r.retrieval_metrics.recall_at_k.get(k)
                                    for r in type_results_with_retrieval
                                    if r.retrieval_metrics.recall_at_k.get(k) is not None]
                    if r_at_k_values:
                        type_retrieval_metrics[f"avg_r@{k}"] = np.mean(r_at_k_values)

                mrr_values = [r.retrieval_metrics.mrr for r in type_results_with_retrieval
                             if r.retrieval_metrics.mrr is not None]
                if mrr_values:
                    type_retrieval_metrics["avg_mrr"] = np.mean(mrr_values)

            aggregated["by_query_type"][query_type.value] = {
                "count": len(type_results),
                "avg_precision": np.mean(type_precisions),
                "avg_recall": np.mean(type_recalls),
                "avg_f1": np.mean(type_f1s),
                "avg_semantic_similarity": np.mean(type_semantic_avgs) if type_semantic_avgs else 0.0,
                "avg_theme_recall": np.mean(type_theme_recalls) if type_theme_recalls else 0.0,
                "avg_theme_precision": np.mean(type_theme_precisions) if type_theme_precisions else 0.0,
                "retrieval_metrics": type_retrieval_metrics
            }

        return aggregated


if __name__ == "__main__":
    # Test the evaluators
    from evaluation_dataset import EvaluationDataset
    from vector_store import VectorStoreManager # Import here for mock test

    dataset = EvaluationDataset()
    test_query = dataset.get_query_by_id("S001")

    # Mock retrieved books
    mock_books = [
        {"구분": "소설", "상품명": "프로젝트 헤일 메리", "책소개": "우주에서 홀로 깨어난 과학자의 생존기"},
        {"구분": "소설", "상품명": "삼체", "책소개": "중국 SF의 걸작"},
        {"구분": "에세이", "상품명": "달러구트 꿈 백화점", "책소개": "꿈을 사고 파는 백화점 이야기"},
    ]

    # Test genre evaluator
    evaluator = GenreEvaluator()
    metrics = evaluator.evaluate(test_query, mock_books)

    print("=== Genre Evaluation Test ===")
    print(f"Query: {test_query.query}")
    print(f"Expected genres: {test_query.expected_genres}")
    print(f"\nMetrics:")
    print(f"  Precision: {metrics.genre_precision:.2f}")
    print(f"  Recall: {metrics.genre_recall:.2f}")
    print(f"  F1: {metrics.genre_f1:.2f}")
    print(f"  Diversity: {metrics.genre_diversity:.2f}")
    print(f"  Matched genres: {metrics.matched_genres}")
    print(f"  Missing genres: {metrics.missing_genres}")
    
    # Note: SemanticEvaluator requires actual embeddings, skipping in this simple mock test
