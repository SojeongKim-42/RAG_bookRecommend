"""
Script to run evaluation on RAG book recommendation system.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from config import Config
from vector_store import VectorStoreManager
from evaluation_dataset import EvaluationDataset, QueryType
from evaluation_metrics import EvaluationRunner, EvaluationResult


def convert_to_serializable(obj):
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
    output_dir: str = "evaluation_results"
):
    """
    Save evaluation results to files.

    Args:
        results: List of evaluation results
        aggregated: Aggregated metrics
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    detailed_file = output_path / f"detailed_results_{timestamp}.json"
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
    aggregated_file = output_path / f"aggregated_results_{timestamp}.json"
    aggregated = convert_to_serializable(aggregated)
    with open(aggregated_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"Aggregated results saved to: {aggregated_file}")

    # Save summary report
    summary_file = output_path / f"summary_report_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAG BOOK RECOMMENDATION SYSTEM - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
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

    print(f"Summary report saved to: {summary_file}")


def print_summary(aggregated: Dict[str, Any]):
    """
    Print summary to console.

    Args:
        aggregated: Aggregated metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

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

    print("\n" + "=" * 80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG Book Recommendation System"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--query-type",
        type=str,
        choices=["all", "specific", "emotional", "situational", "vague", "multi_intent"],
        default="all",
        help="Type of queries to evaluate (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save results (default: evaluation_results)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Evaluate only a sample of N queries"
    )

    args = parser.parse_args()

    try:
        # Setup
        print("Setting up evaluation environment...")
        Config.setup_environment()

        # Load vector store
        print("Loading vector store...")
        vectorstore_manager = VectorStoreManager()
        if not vectorstore_manager.exists():
            raise FileNotFoundError(
                "Vector store not found. Please run main.py first to create it."
            )
        vectorstore_manager.load_vectorstore()

        # Load evaluation dataset
        print("Loading evaluation dataset...")
        dataset = EvaluationDataset()

        # Filter queries by type if specified
        if args.query_type == "all":
            test_queries = dataset.get_all_queries()
        else:
            query_type_map = {
                "specific": QueryType.SPECIFIC,
                "emotional": QueryType.EMOTIONAL,
                "situational": QueryType.SITUATIONAL,
                "vague": QueryType.VAGUE,
                "multi_intent": QueryType.MULTI_INTENT,
            }
            test_queries = dataset.get_queries_by_type(query_type_map[args.query_type])

        # Sample if requested
        if args.sample:
            import random
            test_queries = random.sample(test_queries, min(args.sample, len(test_queries)))

        print(f"Evaluating {len(test_queries)} queries with k={args.k}...")
        print()

        # Create evaluation runner
        runner = EvaluationRunner(vectorstore_manager)

        # Run evaluation
        results = runner.evaluate_dataset(test_queries, retrieval_config={"k": args.k})

        # Aggregate results
        print("\nAggregating results...")
        aggregated = runner.aggregate_results(results)

        # Print summary
        print_summary(aggregated)

        # Save results
        if not args.no_save:
            print("\nSaving results...")
            save_results(results, aggregated, args.output_dir)

        print("\n✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
