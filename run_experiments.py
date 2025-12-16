"""
Batch experiment runner for RAG evaluation.
Easily run multiple experiments with different configurations.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from config import Config
from vector_store import VectorStoreManager
from orchestrator import AmbiguityAwareOrchestrator
from evaluation_dataset import EvaluationDataset, QueryType
from evaluation_metrics import EvaluationRunner
from experiment_config import (
    ExperimentConfig,
    ExperimentPresets,
    RetrievalConfig,
    OrchestratorConfig,
)
from run_evaluation import save_results, print_summary, convert_to_serializable


class ExperimentBatchRunner:
    """Run multiple experiments in batch."""

    def __init__(
        self,
        output_base_dir: str = "experiment_results",
        verbose: bool = False
    ):
        """
        Initialize batch runner.

        Args:
            output_base_dir: Base directory for all experiment results
            verbose: Enable verbose logging
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.verbose = verbose

        # Setup environment
        Config.setup_environment()

        # Load vector store
        print("Loading vector store...")
        self.vectorstore_manager = VectorStoreManager()
        if not self.vectorstore_manager.exists():
            raise FileNotFoundError(
                "Vector store not found. Please run main.py first to create it."
            )
        self.vectorstore_manager.load_vectorstore()

        # Load evaluation dataset
        print("Loading evaluation dataset...")
        self.dataset = EvaluationDataset()

    def run_single_experiment(
        self,
        experiment: ExperimentConfig,
        test_queries: List = None
    ) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            experiment: Experiment configuration
            test_queries: Optional list of test queries (uses all if None)

        Returns:
            Dictionary with results and metadata
        """
        print(f"\n{'='*80}")
        print(f"Running Experiment: {experiment.name}")
        print(f"Description: {experiment.description}")
        print(f"{'='*80}\n")

        # Get test queries
        if test_queries is None:
            if experiment.query_types:
                test_queries = []
                for qtype_str in experiment.query_types:
                    qtype = QueryType[qtype_str.upper()]
                    test_queries.extend(self.dataset.get_queries_by_type(qtype))
            else:
                test_queries = self.dataset.get_all_queries()

        # Sample if needed
        if experiment.sample_size and experiment.sample_size < len(test_queries):
            import random
            test_queries = random.sample(test_queries, experiment.sample_size)

        print(f"Testing with {len(test_queries)} queries\n")

        # Initialize orchestrator if needed
        orchestrator = None
        if experiment.orchestrator.enabled:
            print("Initializing orchestrator...")
            orchestrator = AmbiguityAwareOrchestrator(
                vectorstore_manager=self.vectorstore_manager,
                model_name=experiment.orchestrator.model_name,
                chain_model_name=experiment.orchestrator.chain_model_name,
                verbose=experiment.orchestrator.verbose,
                use_rewriter=experiment.orchestrator.use_rewriter,
                use_quality_check=experiment.orchestrator.use_quality_check,
            )

        # Create evaluation runner
        runner = EvaluationRunner(
            vectorstore_manager=self.vectorstore_manager,
            orchestrator=orchestrator,
            use_orchestrator=experiment.orchestrator.enabled,
        )

        # Run evaluation
        retrieval_config = experiment.retrieval.to_dict()
        results = runner.evaluate_dataset(test_queries, retrieval_config=retrieval_config)

        # Aggregate results
        aggregated = runner.aggregate_results(results)

        # Add experiment metadata
        experiment_data = {
            "experiment": experiment.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(test_queries),
            "results": aggregated,
        }

        return {
            "experiment_data": experiment_data,
            "detailed_results": results,
            "aggregated": aggregated,
        }

    def run_experiments(
        self,
        experiments: List[ExperimentConfig],
        save_individual: bool = True
    ) -> Dict[str, Any]:
        """
        Run multiple experiments.

        Args:
            experiments: List of experiment configurations
            save_individual: Whether to save individual experiment results

        Returns:
            Dictionary with all results
        """
        all_results = {}
        comparison_data = []

        for i, experiment in enumerate(experiments, 1):
            print(f"\n\n{'#'*80}")
            print(f"# Experiment {i}/{len(experiments)}")
            print(f"{'#'*80}")

            try:
                result = self.run_single_experiment(experiment)

                # Store results
                exp_name = experiment.get_full_name()
                all_results[exp_name] = result

                # Add to comparison
                comparison_data.append({
                    "experiment_name": exp_name,
                    "config": experiment.to_dict(),
                    "metrics": result["aggregated"]["genre_metrics"],
                    "by_query_type": result["aggregated"]["by_query_type"],
                })

                # Save individual results if requested
                if save_individual:
                    exp_dir = self.output_base_dir / exp_name
                    exp_dir.mkdir(exist_ok=True)

                    # Save experiment config
                    config_file = exp_dir / "config.json"
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(
                            convert_to_serializable(experiment.to_dict()),
                            f,
                            ensure_ascii=False,
                            indent=2
                        )

                    # Save results using existing save_results function
                    save_results(
                        result["detailed_results"],
                        result["aggregated"],
                        str(exp_dir)
                    )

                    print(f"\n✅ Results saved to: {exp_dir}")

            except Exception as e:
                print(f"\n❌ Error in experiment {experiment.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save comparison data
        self._save_comparison(comparison_data)

        return {
            "all_results": all_results,
            "comparison": comparison_data,
        }

    def _save_comparison(self, comparison_data: List[Dict[str, Any]]):
        """Save comparison data across experiments."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.output_base_dir / f"comparison_{timestamp}.json"

        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(
                convert_to_serializable(comparison_data),
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"\n📊 Comparison data saved to: {comparison_file}")

        # Print comparison table
        self._print_comparison_table(comparison_data)

    def _print_comparison_table(self, comparison_data: List[Dict[str, Any]]):
        """Print comparison table to console."""
        print("\n" + "="*100)
        print("EXPERIMENT COMPARISON")
        print("="*100 + "\n")

        # Header
        print(f"{'Experiment':<40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Diversity':>10}")
        print("-"*100)

        # Sort by F1 score
        sorted_data = sorted(
            comparison_data,
            key=lambda x: x["metrics"]["avg_f1"],
            reverse=True
        )

        # Print rows
        for data in sorted_data:
            name = data["experiment_name"]
            if len(name) > 38:
                name = name[:35] + "..."

            metrics = data["metrics"]
            print(
                f"{name:<40} "
                f"{metrics['avg_precision']:>10.3f} "
                f"{metrics['avg_recall']:>10.3f} "
                f"{metrics['avg_f1']:>10.3f} "
                f"{metrics['avg_diversity']:>10.3f}"
            )

        print("\n" + "="*100 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments for RAG evaluation"
    )

    # Experiment selection
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "all",
            "ablation",
            "k_sweep",
            "lambda_sweep",
            "rerank_sweep",
            "baseline",
            "orchestrator",
            "improved_retrieval",
            "ablation_cross_encoder",
            "sales_vs_semantic",
            "comparison_orchestrator",
            "ablation_orchestrator",
        ],
        default="baseline",
        help="Predefined experiment set to run"
    )

    parser.add_argument(
        "--custom-config",
        type=str,
        help="Path to custom experiment config JSON file"
    )

    # Query filtering
    parser.add_argument(
        "--query-types",
        type=str,
        nargs="+",
        help="Limit to specific query types (e.g., emotional situational)"
    )

    parser.add_argument(
        "--sample",
        type=int,
        help="Sample N queries per experiment"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Base directory for experiment results"
    )

    parser.add_argument(
        "--no-save-individual",
        action="store_true",
        help="Don't save individual experiment results"
    )

    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        # Create batch runner
        runner = ExperimentBatchRunner(
            output_base_dir=args.output_dir,
            verbose=args.verbose
        )

        # Select experiments
        if args.custom_config:
            # Load custom config
            with open(args.custom_config, 'r') as f:
                config_data = json.load(f)
            # TODO: Parse custom config
            experiments = []
        elif args.preset == "all":
            experiments = ExperimentPresets.all_presets()
        elif args.preset == "ablation":
            experiments = ExperimentPresets.ablation_study()
        elif args.preset == "k_sweep":
            experiments = ExperimentPresets.k_sweep()
        elif args.preset == "lambda_sweep":
            experiments = ExperimentPresets.lambda_sweep()
        elif args.preset == "rerank_sweep":
            experiments = ExperimentPresets.rerank_weight_sweep()
        elif args.preset == "baseline":
            experiments = [ExperimentPresets.baseline()]
        elif args.preset == "orchestrator":
            experiments = [
                ExperimentPresets.baseline(),
                ExperimentPresets.orchestrator_enabled(),
            ]
        elif args.preset == "improved_retrieval":
            experiments = [ExperimentPresets.improved_retrieval()]
        elif args.preset == "ablation_cross_encoder":
            experiments = [ExperimentPresets.ablation_cross_encoder()]
        elif args.preset == "sales_vs_semantic":
            experiments = ExperimentPresets.comparison_sales_vs_semantic()
        elif args.preset == "comparison_orchestrator":
            experiments = [
                ExperimentPresets.improved_retrieval(), # Baseline (best retrieval)
                ExperimentPresets.orchestrator_enabled() # Orchestrator
            ]
            # Override query types to focus on VAGUE/MULTI_INTENT where orchestrator shines
            for exp in experiments:
                exp.query_types = ["vague", "multi_intent"]
        elif args.preset == "ablation_orchestrator":
            experiments = ExperimentPresets.ablation_orchestrator()
            # Focus on vague queries where rewriter matters most
            for exp in experiments:
                exp.query_types = ["vague", "multi_intent"]
        else:
            experiments = [ExperimentPresets.baseline()]

        # Apply query type filter if specified
        if args.query_types or args.sample:
            for exp in experiments:
                if args.query_types:
                    exp.query_types = args.query_types
                if args.sample:
                    exp.sample_size = args.sample

        print(f"\n🚀 Running {len(experiments)} experiment(s)...\n")

        # Run experiments
        results = runner.run_experiments(
            experiments,
            save_individual=not args.no_save_individual
        )

        print("\n✅ All experiments completed!")
        print(f"📁 Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiments interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
