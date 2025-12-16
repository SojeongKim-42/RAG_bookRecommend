"""
Unified Experiment and Evaluation Runner.
Supports running batch experiments (presets) and single ad-hoc evaluations.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to sys.path to allow importing from src
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import Config
from src.data.vector_store import VectorStoreManager
from src.core.orchestrator import AmbiguityAwareOrchestrator
from evaluation.dataset import EvaluationDataset, QueryType
from evaluation.metrics import EvaluationRunner
from evaluation.experiment_config import (
    ExperimentConfig,
    ExperimentPresets,
    RetrievalConfig,
    OrchestratorConfig,
)
from evaluation.utils import save_results, print_summary, convert_to_serializable


class ExperimentRunner:
    """Run experiments and evaluations."""

    def __init__(
        self,
        output_base_dir: str = "experiment_results",
        verbose: bool = False
    ):
        """
        Initialize runner.

        Args:
            output_base_dir: Base directory for results
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

        # Get test queries if not provided
        if test_queries is None:
            if experiment.query_types:
                test_queries = []
                for qtype_str in experiment.query_types:
                    # Handle both strings and direct QueryType matches if needed, but config usually has strings
                    if isinstance(qtype_str, str):
                        try:
                            qtype = QueryType[qtype_str.upper()]
                            test_queries.extend(self.dataset.get_queries_by_type(qtype))
                        except KeyError:
                            print(f"Warning: Unknown query type '{qtype_str}'")
                    else:
                         test_queries.extend(self.dataset.get_queries_by_type(qtype_str))
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

                # Add to comparison (only if aggregated results exist)
                if result["aggregated"]:
                    comparison_data.append({
                        "experiment_name": exp_name,
                        "config": experiment.to_dict(),
                        "metrics": result["aggregated"].get("genre_metrics", {}),
                        "by_query_type": result["aggregated"].get("by_query_type", {}),
                    })

                # Save individual results if requested
                if save_individual:
                    # Create unique directory for this run
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    exp_dir = self.output_base_dir / f"{exp_name}_{timestamp}"
                    exp_dir.mkdir(exist_ok=True, parents=True)

                    # Save experiment config
                    config_file = exp_dir / "config.json"
                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(
                            convert_to_serializable(experiment.to_dict()),
                            f,
                            ensure_ascii=False,
                            indent=2
                        )

                    # Save results using utility
                    save_results(
                        result["detailed_results"],
                        result["aggregated"],
                        str(exp_dir),
                        experiment_name=exp_name
                    )

                    print(f"\n✅ Results saved to: {exp_dir}")

            except Exception as e:
                print(f"\n❌ Error in experiment {experiment.name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Save comparison data if multiple experiments
        if len(experiments) > 1:
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


def create_custom_experiment(args) -> ExperimentConfig:
    """Create an experiment config from CLI arguments."""
    retrieval_config = RetrievalConfig(
        k=args.k,
        use_mmr=not args.no_mmr,
        mmr_lambda=args.mmr_lambda,
        use_reranking=True, # Default to true unless specified otherwise via new args (not implemented yet for simplicity)
        use_adaptive_k=True, # Default
        use_cross_encoder=args.use_cross_encoder
    )
    
    orchestrator_config = OrchestratorConfig(enabled=False)

    return ExperimentConfig(
        name="custom_experiment",
        description="Custom experiment from CLI arguments",
        retrieval=retrieval_config,
        orchestrator=orchestrator_config,
        query_types=[args.query_type] if args.query_type and args.query_type != "all" else None,
        sample_size=args.sample
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Unified Experiment and Evaluation Runner"
    )

    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--preset",
        type=str,
        choices=[
            "all", "ablation", "k_sweep", "lambda_sweep", "rerank_sweep",
            "baseline", "orchestrator", "improved_retrieval", 
            "ablation_cross_encoder", "sales_vs_semantic", 
            "comparison_orchestrator", "ablation_orchestrator"
        ],
        help="Run a predefined experiment preset"
    )
    group.add_argument(
        "--custom",
        action="store_true",
        help="Run a custom experiment using CLI arguments"
    )

    # Custom Experiment Arguments
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR")
    parser.add_argument("--mmr-lambda", type=float, default=0.8, help="MMR lambda value")
    parser.add_argument("--use-cross-encoder", action="store_true", help="Use Cross-Encoder")
    
    # Common Arguments
    parser.add_argument(
        "--query-type", 
        type=str, 
        default="all",
        help="Filter query type (specific, emotional, situational, vague, multi_intent)"
    )
    parser.add_argument("--sample", type=int, help="Sample N queries")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="experiment_results",
        help="Base directory for results"
    )
    parser.add_argument("--no-save-individual", action="store_true", help="Don't save individual results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--custom-config", type=str, help="Path to custom config JSON")

    args = parser.parse_args()

    try:
        runner = ExperimentRunner(
            output_base_dir=args.output_dir,
            verbose=args.verbose
        )

        experiments = []

        # 1. Preset Mode
        if args.preset:
            if args.preset == "all":
                experiments = ExperimentPresets.all_presets()
            elif args.preset == "baseline":
                experiments = [ExperimentPresets.baseline()]
            elif args.preset == "orchestrator":
                experiments = [ExperimentPresets.baseline(), ExperimentPresets.orchestrator_enabled()]
            elif args.preset == "ablation":
                experiments = ExperimentPresets.ablation_study()
            elif args.preset == "k_sweep":
                experiments = ExperimentPresets.k_sweep()
            elif args.preset == "lambda_sweep":
                experiments = ExperimentPresets.lambda_sweep()
            elif args.preset == "rerank_sweep":
                experiments = ExperimentPresets.rerank_weight_sweep()
            elif args.preset == "improved_retrieval":
                experiments = [ExperimentPresets.improved_retrieval()]
            elif args.preset == "ablation_cross_encoder":
                experiments = [ExperimentPresets.ablation_cross_encoder()]
            elif args.preset == "sales_vs_semantic":
                experiments = ExperimentPresets.comparison_sales_vs_semantic()
            elif args.preset == "comparison_orchestrator":
                experiments = [
                    ExperimentPresets.improved_retrieval(), 
                    ExperimentPresets.orchestrator_enabled()
                ]
                # Override types for comparison
                for exp in experiments:
                    exp.query_types = ["vague", "multi_intent"]
            elif args.preset == "ablation_orchestrator":
                 experiments = ExperimentPresets.ablation_orchestrator()
                 for exp in experiments:
                    exp.query_types = ["vague", "multi_intent"]
        
        # 2. Custom Config File
        elif args.custom_config:
            with open(args.custom_config, 'r') as f:
                # Logic to parse JSON to ExperimentConfig would go here
                # For now just placeholder
                print("Custom config parsing not fully implemented yet.")
                return

        # 3. Ad-hoc / Custom Mode (Default if no preset)
        else:
            # If no preset specified, treat as custom experiment (baseline or with args)
            print("Running custom ad-hoc experiment...")
            experiments = [create_custom_experiment(args)]

        # Apply overrides if specified
        if args.query_type and args.query_type != "all" and args.preset:
             # Only override if explicit preset + query type. 
             # For custom mode, create_custom_experiment handles it.
             for exp in experiments:
                 exp.query_types = [args.query_type]
        
        if args.sample:
            for exp in experiments:
                exp.sample_size = args.sample

        print(f"\n🚀 Running {len(experiments)} experiment(s)...\n")
        
        runner.run_experiments(
            experiments, 
            save_individual=not args.no_save_individual
        )
        
        print("\n✅ All experiments completed!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
