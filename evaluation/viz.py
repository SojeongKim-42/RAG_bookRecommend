"""
Visualization and analysis tools for evaluation results.
Supports three types of result files:
1. Detailed results: Individual query-level metrics
2. Aggregated results: Summary statistics
3. Comparison results: Cross-experiment comparisons
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import argparse


# Add project root to sys.path
import sys
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


class EvaluationVisualizer:
    """Visualize evaluation results."""

    def __init__(self, results_file: Optional[str] = None):
        """
        Initialize visualizer.

        Args:
            results_file: Path to results JSON file (detailed, aggregated, or comparison)
        """
        self.results_file = results_file
        self.results = None
        self.result_type = None  # 'detailed', 'aggregated', or 'comparison'
        self.df = None

        if results_file:
            self.load_results(results_file)

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def load_results(self, results_file: str):
        """
        Load results from JSON file and detect type.

        Args:
            results_file: Path to results file
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        # Detect result type
        self.result_type = self._detect_result_type(self.results)
        print(f"Detected result type: {self.result_type}")

        # Convert to DataFrame based on type
        if self.result_type == 'detailed':
            self.df = self._detailed_to_dataframe(self.results)
        elif self.result_type == 'comparison':
            self.df = self._comparison_to_dataframe(self.results)
        elif self.result_type == 'aggregated':
            print("Aggregated results loaded (no DataFrame conversion)")

    def _detect_result_type(self, results: Any) -> str:
        """Detect the type of results file."""
        if isinstance(results, list):
            if len(results) > 0:
                if 'query_id' in results[0]:
                    return 'detailed'
                elif 'experiment_name' in results[0]:
                    return 'comparison'
        elif isinstance(results, dict):
            if 'total_queries' in results and 'genre_metrics' in results:
                return 'aggregated'

        raise ValueError("Unknown result file format")

    def _detailed_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert detailed results to DataFrame."""
        records = []
        for result in results:
            gm = result.get('genre_metrics', {})
            sm = result.get('semantic_metrics', {})
            tm = result.get('theme_metrics', {})

            record = {
                'query_id': result['query_id'],
                'query': result['query'],
                'query_type': result['query_type'],
                'precision': gm.get('precision', 0),
                'recall': gm.get('recall', 0),
                'f1': gm.get('f1', 0),
                'diversity': gm.get('diversity', 0),
                'unique_genres': gm.get('unique_genres', 0),
            }

            # Add semantic metrics if available
            if sm:
                record['semantic_similarity'] = sm.get('avg_similarity', 0)

            # Add theme metrics if available
            if tm:
                record['theme_recall'] = tm.get('theme_recall', 0)
                record['theme_precision'] = tm.get('theme_precision', 0)

            records.append(record)

        return pd.DataFrame(records)

    def _comparison_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert comparison results to DataFrame."""
        records = []
        for result in results:
            metrics = result.get('metrics', {})

            record = {
                'experiment_name': result['experiment_name'],
                'precision': metrics.get('avg_precision', 0),
                'recall': metrics.get('avg_recall', 0),
                'f1': metrics.get('avg_f1', 0),
                'diversity': metrics.get('avg_diversity', 0),
            }

            records.append(record)

        return pd.DataFrame(records)

    def plot_metrics_by_query_type(self, save_path: str = None):
        """
        Plot metrics grouped by query type (only for detailed results).

        Args:
            save_path: Optional path to save figure
        """
        if self.result_type != 'detailed':
            print("This plot is only available for detailed results")
            return

        if self.df is None:
            raise ValueError("No results loaded")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics = ['precision', 'recall', 'f1', 'diversity']
        titles = ['Precision by Query Type', 'Recall by Query Type',
                  'F1 Score by Query Type', 'Diversity by Query Type']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            # Group by query type
            grouped = self.df.groupby('query_type')[metric].agg(['mean', 'std'])

            # Plot bars
            x = range(len(grouped))
            means = grouped['mean']
            stds = grouped['std']

            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

            # Color bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(grouped)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Labels
            ax.set_xticks(x)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(title)
            ax.set_ylim(0, 1.1)

            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02, f'{mean:.2f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def plot_experiment_comparison(self, save_path: str = None):
        """
        Plot comparison across experiments (only for comparison results).

        Args:
            save_path: Optional path to save figure
        """
        if self.result_type != 'comparison':
            print("This plot is only available for comparison results")
            return

        if self.df is None:
            raise ValueError("No results loaded")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        metrics = ['precision', 'recall', 'f1', 'diversity']
        titles = ['Precision Comparison', 'Recall Comparison',
                  'F1 Score Comparison', 'Diversity Comparison']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            # Sort by metric value
            sorted_df = self.df.sort_values(metric, ascending=False)

            # Plot bars
            x = range(len(sorted_df))
            values = sorted_df[metric]

            bars = ax.barh(x, values, alpha=0.7)

            # Color bars by value
            norm = plt.Normalize(vmin=values.min(), vmax=values.max())
            colors = plt.cm.RdYlGn(norm(values))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Labels
            ax.set_yticks(x)
            ax.set_yticklabels(sorted_df['experiment_name'], fontsize=9)
            ax.set_xlabel(metric.capitalize())
            ax.set_title(title)
            ax.set_xlim(0, max(1.0, values.max() * 1.1))

            # Add value labels
            for i, val in enumerate(values):
                ax.text(val + 0.01, i, f'{val:.3f}',
                       va='center', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def plot_genre_distribution(self, save_path: str = None):
        """
        Plot genre distribution (only for detailed results).

        Args:
            save_path: Optional path to save figure
        """
        if self.result_type != 'detailed':
            print("This plot is only available for detailed results")
            return

        if self.results is None:
            raise ValueError("No results loaded")

        # Collect all genres
        all_expected = []
        all_matched = []
        all_missing = []

        for result in self.results:
            gm = result.get('genre_metrics', {})
            all_expected.extend(gm.get('expected_genres', []))
            all_matched.extend(gm.get('matched_genres', []))
            all_missing.extend(gm.get('missing_genres', []))

        # Count occurrences
        from collections import Counter
        expected_counts = Counter(all_expected)
        matched_counts = Counter(all_matched)
        missing_counts = Counter(all_missing)

        # Create DataFrame
        genres = sorted(set(all_expected))
        data = {
            'Genre': genres,
            'Expected': [expected_counts[g] for g in genres],
            'Matched': [matched_counts[g] for g in genres],
            'Missing': [missing_counts[g] for g in genres]
        }
        df = pd.DataFrame(data)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(df))
        width = 0.35

        bars1 = ax.bar(x - width/2, df['Matched'], width, label='Matched', alpha=0.8)
        bars2 = ax.bar(x + width/2, df['Missing'], width, label='Missing', alpha=0.8)

        ax.set_xlabel('Genre')
        ax.set_ylabel('Count')
        ax.set_title('Genre Match/Miss Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Genre'], rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def plot_theme_metrics(self, save_path: str = None):
        """
        Plot theme recall and precision (only for detailed results with theme metrics).

        Args:
            save_path: Optional path to save figure
        """
        if self.result_type != 'detailed':
            print("This plot is only available for detailed results")
            return

        if 'theme_recall' not in self.df.columns:
            print("No theme metrics found in results")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Theme Recall by Query Type
        ax = axes[0]
        grouped = self.df.groupby('query_type')['theme_recall'].agg(['mean', 'std'])
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_ylabel('Theme Recall')
        ax.set_title('Theme Recall by Query Type')
        ax.set_ylim(0, 1.1)

        # Theme Precision by Query Type
        ax = axes[1]
        grouped = self.df.groupby('query_type')['theme_precision'].agg(['mean', 'std'])
        x = range(len(grouped))
        ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_ylabel('Theme Precision')
        ax.set_title('Theme Precision by Query Type')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

    def generate_all_plots(self, output_dir: str):
        """
        Generate all applicable plots based on result type.

        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating plots for {self.result_type} results...")

        if self.result_type == 'detailed':
            print("1. Metrics by Query Type...")
            self.plot_metrics_by_query_type(
                save_path=output_path / "metrics_by_query_type.png"
            )

            print("2. Genre Distribution...")
            self.plot_genre_distribution(
                save_path=output_path / "genre_distribution.png"
            )

            print("3. Theme Metrics...")
            self.plot_theme_metrics(
                save_path=output_path / "theme_metrics.png"
            )

        elif self.result_type == 'comparison':
            print("1. Experiment Comparison...")
            self.plot_experiment_comparison(
                save_path=output_path / "experiment_comparison.png"
            )

        print(f"\nAll plots saved to {output_dir}")

    def print_summary_statistics(self):
        """Print summary statistics."""
        if self.result_type == 'detailed':
            print("\n=== Summary Statistics (Detailed Results) ===")
            print(f"Total queries: {len(self.df)}")
            print(f"\nBy Query Type:")
            print(self.df.groupby('query_type').size())
            print(f"\nOverall Metrics:")
            print(self.df[['precision', 'recall', 'f1', 'diversity']].describe())

        elif self.result_type == 'comparison':
            print("\n=== Summary Statistics (Comparison Results) ===")
            print(f"Total experiments: {len(self.df)}")
            print(f"\nMetrics:")
            print(self.df[['precision', 'recall', 'f1', 'diversity']].describe())

        elif self.result_type == 'aggregated':
            print("\n=== Summary Statistics (Aggregated Results) ===")
            print(f"Total queries: {self.results.get('total_queries', 0)}")
            print(f"\nGenre Metrics:")
            for key, val in self.results.get('genre_metrics', {}).items():
                print(f"  {key}: {val:.4f}")
            print(f"\nTheme Metrics:")
            for key, val in self.results.get('theme_metrics', {}).items():
                print(f"  {key}: {val:.4f}")
            print(f"\nSemantic Metrics:")
            for key, val in self.results.get('semantic_metrics', {}).items():
                print(f"  {key}: {val:.4f}")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results"
    )
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to results JSON file (detailed, aggregated, or comparison)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, do not generate plots'
    )

    args = parser.parse_args()

    # Create visualizer
    viz = EvaluationVisualizer(args.results_file)

    # Print statistics
    viz.print_summary_statistics()

    # Generate plots
    if not args.stats_only:
        viz.generate_all_plots(args.output_dir)


if __name__ == "__main__":
    main()
