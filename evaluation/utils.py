"""
Utility functions for RAG evaluation.
Extracted from run_evaluation.py for shared usage.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union

from evaluation.metrics import EvaluationResult


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_results(
    results: List[EvaluationResult],
    aggregated: Dict[str, Any],
    output_dir: Union[str, Path] = "evaluation_results",
    experiment_name: str = None
):
    """
    Save evaluation results to files.

    Args:
        results: List of evaluation results
        aggregated: Aggregated metrics
        output_dir: Directory to save results
        experiment_name: Optional name to prefix files or organize folder
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{experiment_name}_" if experiment_name else ""

    # Save detailed results
    detailed_file = output_path / f"{prefix}detailed_results_{timestamp}.json"
    detailed_data = []

    for result in results:
        data = {
            "query_id": result.query_id,
            "query": result.query,
            "query_type": result.query_type.value,
            "genre_metrics": {
                "precision": result.genre_metrics.genre_precision,
                "recall": result.genre_metrics.genre_recall,
                "f1": result.genre_metrics.genre_f1,
                "diversity": result.genre_metrics.genre_diversity,
                "unique_genres": result.genre_metrics.unique_genres,
                "genre_distribution": result.genre_metrics.genre_distribution,
                "expected_genres": result.genre_metrics.expected_genres,
                "retrieved_genres": result.genre_metrics.retrieved_genres,
                "matched_genres": result.genre_metrics.matched_genres,
                "missing_genres": result.genre_metrics.missing_genres,
            },
            "retrieved_books": [
                {
                    "title": str(book.get("상품명", "")),
                    "genre": str(book.get("구분", "")),
                    "rank": convert_to_serializable(book.get("순번/순위", ""))
                }
                for book in (result.retrieved_books or [])
            ],
            "notes": result.notes
        }

        # Add retrieval metrics if available
        if result.retrieval_metrics:
            data["retrieval_metrics"] = {
                "precision_at_k": result.retrieval_metrics.precision_at_k,
                "recall_at_k": result.retrieval_metrics.recall_at_k,
                "mrr": result.retrieval_metrics.mrr,
                "coverage": result.retrieval_metrics.coverage,
            }

        # Add semantic metrics if available
        if result.semantic_metrics:
            data["semantic_metrics"] = {
                "avg_similarity": result.semantic_metrics.avg_similarity,
                "max_similarity": result.semantic_metrics.max_similarity,
                "min_similarity": result.semantic_metrics.min_similarity,
                "similarity_at_k": result.semantic_metrics.similarity_at_k,
            }

        detailed_data.append(data)

    # Convert all data to serializable format
    detailed_data = convert_to_serializable(detailed_data)

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_data, f, ensure_ascii=False, indent=2)

    print(f"Detailed results saved to: {detailed_file}")

    # Save aggregated results
    aggregated_file = output_path / f"{prefix}aggregated_results_{timestamp}.json"
    aggregated = convert_to_serializable(aggregated)
    with open(aggregated_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"Aggregated results saved to: {aggregated_file}")

    # Save summary report
    summary_file = output_path / f"{prefix}summary_report_{timestamp}.txt"
    _write_summary_report(summary_file, timestamp, aggregated, results, experiment_name)
    print(f"Summary report saved to: {summary_file}")


def _write_summary_report(
    filepath: Path,
    timestamp: str,
    aggregated: Dict[str, Any],
    results: List[EvaluationResult],
    experiment_name: str = None
):
    """Helper to write summary report text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        title = f"EVALUATION REPORT: {experiment_name}" if experiment_name else "RAG BOOK RECOMMENDATION SYSTEM - EVALUATION REPORT"
        f.write(f"{title}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Queries: {aggregated['total_queries']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("OVERALL GENRE METRICS\n")
        f.write("-" * 80 + "\n")
        genre_metrics = aggregated["genre_metrics"]
        f.write(f"Average Precision:  {genre_metrics['avg_precision']:.3f}\n")
        f.write(f"Average Recall:     {genre_metrics['avg_recall']:.3f}\n")
        f.write(f"Average F1 Score:   {genre_metrics['avg_f1']:.3f}\n")
        f.write(f"Average Diversity:  {genre_metrics['avg_diversity']:.3f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("OVERALL SEMANTIC METRICS\n")
        f.write("-" * 80 + "\n")
        semantic_metrics = aggregated.get("semantic_metrics", {})
        f.write(f"Average Similarity: {semantic_metrics.get('avg_similarity', 0.0):.3f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("METRICS BY QUERY TYPE\n")
        f.write("-" * 80 + "\n\n")

        for qtype, metrics in aggregated["by_query_type"].items():
            f.write(f"{qtype.upper()}:\n")
            f.write(f"  Count:      {metrics['count']}\n")
            f.write(f"  Precision:  {metrics['avg_precision']:.3f}\n")
            f.write(f"  Recall:     {metrics['avg_recall']:.3f}\n")
            f.write(f"  F1 Score:   {metrics['avg_f1']:.3f}\n")
            f.write(f"  Semantic:   {metrics.get('avg_semantic_similarity', 0.0):.3f}\n\n")

        f.write("-" * 80 + "\n")
        f.write("TOP PERFORMING QUERIES (Genre F1)\n")
        f.write("-" * 80 + "\n\n")

        # Sort by F1 score
        sorted_results = sorted(
            results,
            key=lambda x: x.genre_metrics.genre_f1,
            reverse=True
        )

        for i, result in enumerate(sorted_results[:10], 1):
            f.write(f"{i}. [{result.query_id}] {result.query}\n")
            f.write(f"   F1: {result.genre_metrics.genre_f1:.3f}, ")
            f.write(f"P: {result.genre_metrics.genre_precision:.3f}, ")
            f.write(f"R: {result.genre_metrics.genre_recall:.3f}\n")
            if result.semantic_metrics:
                f.write(f"   Semantic Sim: {result.semantic_metrics.avg_similarity:.3f}\n")
            f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("LOW PERFORMING QUERIES (Genre F1)\n")
        f.write("-" * 80 + "\n\n")

        for i, result in enumerate(sorted_results[-10:][::-1], 1):
            f.write(f"{i}. [{result.query_id}] {result.query}\n")
            f.write(f"   F1: {result.genre_metrics.genre_f1:.3f}, ")
            f.write(f"P: {result.genre_metrics.genre_precision:.3f}, ")
            f.write(f"R: {result.genre_metrics.genre_recall:.3f}\n")
            if result.semantic_metrics:
                f.write(f"   Semantic Sim: {result.semantic_metrics.avg_similarity:.3f}\n")
            f.write(f"   Expected: {result.genre_metrics.expected_genres}\n")
            f.write(f"   Missing: {result.genre_metrics.missing_genres}\n\n")


def print_summary(aggregated: Dict[str, Any]):
    """
    Print summary to console.

    Args:
        aggregated: Aggregated metrics
    """
    print("EVALUATION SUMMARY")

    print(f"\nTotal Queries: {aggregated['total_queries']}")

    print("\n" + "-" * 80)
    print("OVERALL GENRE METRICS")
    print("-" * 80)

    genre_metrics = aggregated["genre_metrics"]
    print(f"Average Precision:  {genre_metrics['avg_precision']:.3f}")
    print(f"Average Recall:     {genre_metrics['avg_recall']:.3f}")
    print(f"Average F1 Score:   {genre_metrics['avg_f1']:.3f}")
    print(f"Average Diversity:  {genre_metrics['avg_diversity']:.3f}")

    print("\n" + "-" * 80)
    print("OVERALL SEMANTIC METRICS")
    print("-" * 80)

    semantic_metrics = aggregated.get("semantic_metrics", {})
    print(f"Average Similarity: {semantic_metrics.get('avg_similarity', 0.0):.3f}")

    print("\n" + "-" * 80)
    print("METRICS BY QUERY TYPE")
    print("-" * 80)

    for qtype, metrics in aggregated["by_query_type"].items():
        print(f"\n{qtype.upper()}:")
        print(f"  Count:      {metrics['count']}")
        print(f"  Precision:  {metrics['avg_precision']:.3f}")
        print(f"  Recall:     {metrics['avg_recall']:.3f}")
        print(f"  Semantic:   {metrics.get('avg_semantic_similarity', 0.0):.3f}")