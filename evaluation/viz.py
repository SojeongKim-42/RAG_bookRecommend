"""
Visualization and analysis tools for evaluation results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


# Add project root to sys.path
import sys
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

class EvaluationVisualizer:
    """Visualize evaluation results."""

    def __init__(self, results_file: str = None):
        """
        Initialize visualizer.

        Args:
            results_file: Path to detailed results JSON file
        """
        self.results_file = results_file
        self.results = None
        self.df = None

        if results_file:
            self.load_results(results_file)

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def load_results(self, results_file: str):
        """
        Load results from JSON file.

        Args:
            results_file: Path to results file
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        # Convert to DataFrame
        records = []
        for result in self.results:
            record = {
                'query_id': result['query_id'],
                'query': result['query'],
                'query_type': result['query_type'],
                'precision': result['genre_metrics']['precision'],
                'recall': result['genre_metrics']['recall'],
                'f1': result['genre_metrics']['f1'],
                'diversity': result['genre_metrics']['diversity'],
                'unique_genres': result['genre_metrics']['unique_genres'],
            }
            records.append(record)

        self.df = pd.DataFrame(records)

    def plot_metrics_by_query_type(self, save_path: str = None):
        """
        Plot metrics grouped by query type.

        Args:
            save_path: Optional path to save figure
        """
        if self.df is None:
            raise ValueError("No results loaded. Call load_results() first.")

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

            ax.set_xlabel('Query Type', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_ylim(0, 1.0)

            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02, f'{mean:.2f}',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def plot_metric_distribution(self, metric: str = 'f1', save_path: str = None):
        """
        Plot distribution of a specific metric.

        Args:
            metric: Metric to plot ('precision', 'recall', 'f1', 'diversity')
            save_path: Optional path to save figure
        """
        if self.df is None:
            raise ValueError("No results loaded. Call load_results() first.")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        ax1 = axes[0]
        ax1.hist(self.df[metric], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.df[metric].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {self.df[metric].mean():.3f}')
        ax1.axvline(self.df[metric].median(), color='green', linestyle='--',
                   linewidth=2, label=f'Median: {self.df[metric].median():.3f}')
        ax1.set_xlabel(metric.capitalize(), fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'{metric.capitalize()} Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Box plot by query type
        ax2 = axes[1]
        self.df.boxplot(column=metric, by='query_type', ax=ax2)
        ax2.set_xlabel('Query Type', fontsize=12)
        ax2.set_ylabel(metric.capitalize(), fontsize=12)
        ax2.set_title(f'{metric.capitalize()} by Query Type', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def plot_genre_distribution(self, save_path: str = None):
        """
        Plot genre distribution from retrieved results.

        Args:
            save_path: Optional path to save figure
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        # Collect all retrieved genres
        genre_counts = defaultdict(int)

        for result in self.results:
            for genre in result['genre_metrics']['retrieved_genres']:
                genre_counts[genre] += 1

        # Sort by count
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        genres, counts = zip(*sorted_genres[:15])  # Top 15

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(range(len(genres)), counts, alpha=0.7)

        # Color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(genres)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_yticks(range(len(genres)))
        ax.set_yticklabels(genres, fontsize=11)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title('Top Retrieved Genres', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels
        for i, count in enumerate(counts):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def plot_correlation_matrix(self, save_path: str = None):
        """
        Plot correlation between metrics.

        Args:
            save_path: Optional path to save figure
        """
        if self.df is None:
            raise ValueError("No results loaded. Call load_results() first.")

        # Select numeric columns
        metrics_cols = ['precision', 'recall', 'f1', 'diversity', 'unique_genres']
        corr_matrix = self.df[metrics_cols].corr()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})

        ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def create_comprehensive_report(self, output_dir: str = "evaluation_results"):
        """
        Create comprehensive visualization report.

        Args:
            output_dir: Directory to save visualizations
        """
        if self.df is None:
            raise ValueError("No results loaded. Call load_results() first.")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("Generating comprehensive visualization report...")

        # 1. Metrics by query type
        print("  1/4 - Metrics by query type...")
        self.plot_metrics_by_query_type(
            save_path=str(output_path / "metrics_by_query_type.png")
        )
        plt.close()

        # 2. F1 distribution
        print("  2/4 - F1 score distribution...")
        self.plot_metric_distribution(
            metric='f1',
            save_path=str(output_path / "f1_distribution.png")
        )
        plt.close()

        # 3. Genre distribution
        print("  3/4 - Genre distribution...")
        self.plot_genre_distribution(
            save_path=str(output_path / "genre_distribution.png")
        )
        plt.close()

        # 4. Correlation matrix
        print("  4/4 - Correlation matrix...")
        self.plot_correlation_matrix(
            save_path=str(output_path / "correlation_matrix.png")
        )
        plt.close()

        print(f"\n✅ Comprehensive report generated in: {output_path}")

    def analyze_failures(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Analyze queries with low performance.

        Args:
            threshold: F1 threshold below which queries are considered failures

        Returns:
            DataFrame of failing queries
        """
        if self.df is None:
            raise ValueError("No results loaded. Call load_results() first.")

        failures = self.df[self.df['f1'] < threshold].copy()
        failures = failures.sort_values('f1')

        return failures

    def print_failure_analysis(self, threshold: float = 0.5):
        """
        Print analysis of low-performing queries.

        Args:
            threshold: F1 threshold for failures
        """
        failures = self.analyze_failures(threshold)

        if len(failures) == 0:
            print(f"✅ No queries with F1 < {threshold}")
            return

        print(f"\n⚠️  Found {len(failures)} queries with F1 < {threshold}")
        print("LOW PERFORMING QUERIES")

        for idx, row in failures.iterrows():
            # Get detailed info from results
            result = next(r for r in self.results if r['query_id'] == row['query_id'])

            print(f"\n[{row['query_id']}] {row['query']} ({row['query_type']})")
            print(f"  F1: {row['f1']:.3f}, P: {row['precision']:.3f}, R: {row['recall']:.3f}")
            print(f"  Expected genres: {result['genre_metrics']['expected_genres']}")
            print(f"  Missing genres: {result['genre_metrics']['missing_genres']}")
            print(f"  Retrieved books:")
            for book in result['retrieved_books'][:3]:
                print(f"    - [{book['genre']}] {book['title']}")


def main():
    """Main function for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to detailed results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.5,
        help="F1 threshold for failure analysis"
    )

    args = parser.parse_args()

    # Create visualizer
    viz = EvaluationVisualizer(args.results_file)

    # Generate comprehensive report
    viz.create_comprehensive_report(args.output_dir)

    # Analyze failures
    viz.print_failure_analysis(args.failure_threshold)


if __name__ == "__main__":
    main()
