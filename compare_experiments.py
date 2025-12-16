"""
Visualization tool for comparing multiple experiment results.
"""

import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class ExperimentComparator:
    """Compare and visualize multiple experiment results."""

    def __init__(self, comparison_file: str):
        """
        Initialize comparator.

        Args:
            comparison_file: Path to comparison JSON file
        """
        self.comparison_file = comparison_file
        self.data = None
        self.df = None

        self.load_data()

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def load_data(self):
        """Load comparison data from JSON."""
        with open(self.comparison_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Convert to DataFrame
        records = []
        for exp in self.data:
            record = {
                'experiment': exp['experiment_name'],
                'precision': exp['metrics']['avg_precision'],
                'recall': exp['metrics']['avg_recall'],
                'f1': exp['metrics']['avg_f1'],
                'diversity': exp['metrics']['avg_diversity'],
            }

            # Extract config info
            config = exp['config']
            if 'retrieval' in config:
                record['use_mmr'] = config['retrieval']['use_mmr']
                record['use_reranking'] = config['retrieval']['use_reranking']
                record['use_adaptive_k'] = config['retrieval']['use_adaptive_k']
                record['k'] = config['retrieval']['k']

            if 'orchestrator' in config:
                record['orchestrator'] = config['orchestrator']['enabled']

            records.append(record)

        self.df = pd.DataFrame(records)

    def plot_overall_comparison(self, save_path: str = None):
        """
        Plot overall metric comparison across experiments.

        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Prepare data
        experiments = self.df['experiment'].values
        x = np.arange(len(experiments))
        width = 0.2

        metrics = ['precision', 'recall', 'f1', 'diversity']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

        # Plot bars
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = self.df[metric].values
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=color, alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8, rotation=0)

        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Metric Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def plot_metric_ranking(self, metric: str = 'f1', save_path: str = None):
        """
        Plot ranking of experiments by a specific metric.

        Args:
            metric: Metric to rank by
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort by metric
        sorted_df = self.df.sort_values(metric, ascending=True)

        # Plot horizontal bars
        y = np.arange(len(sorted_df))
        bars = ax.barh(y, sorted_df[metric].values, alpha=0.7)

        # Color bars by value
        colors = plt.cm.RdYlGn(sorted_df[metric].values)
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_yticks(y)
        ax.set_yticklabels(sorted_df['experiment'].values, fontsize=10)
        ax.set_xlabel(metric.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'Experiment Ranking by {metric.capitalize()}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)

        # Add value labels
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            value = row[metric]
            ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def plot_feature_impact(self, save_path: str = None):
        """
        Plot impact of different features (MMR, Reranking, Adaptive K, Orchestrator).

        Args:
            save_path: Optional path to save figure
        """
        if 'use_mmr' not in self.df.columns:
            print("Feature columns not found in data. Skipping feature impact plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        features = []
        if 'use_mmr' in self.df.columns:
            features.append(('use_mmr', 'MMR'))
        if 'use_reranking' in self.df.columns:
            features.append(('use_reranking', 'Reranking'))
        if 'use_adaptive_k' in self.df.columns:
            features.append(('use_adaptive_k', 'Adaptive K'))
        if 'orchestrator' in self.df.columns:
            features.append(('orchestrator', 'Orchestrator'))

        for idx, (feature, label) in enumerate(features[:4]):
            ax = axes[idx]

            # Group by feature
            grouped = self.df.groupby(feature)['f1'].agg(['mean', 'std', 'count'])

            # Plot bars
            x = ['Disabled', 'Enabled']
            means = [grouped.loc[False, 'mean'] if False in grouped.index else 0,
                    grouped.loc[True, 'mean'] if True in grouped.index else 0]
            stds = [grouped.loc[False, 'std'] if False in grouped.index else 0,
                   grouped.loc[True, 'std'] if True in grouped.index else 0]

            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                         color=['#FF6B6B', '#4ECDC4'])

            ax.set_ylabel('F1 Score', fontsize=11)
            ax.set_title(f'Impact of {label}', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)

            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                if mean > 0:
                    ax.text(i, mean + std + 0.02, f'{mean:.3f}',
                           ha='center', va='bottom', fontsize=10)

        # Hide unused subplots
        for idx in range(len(features), 4):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def plot_tradeoff_analysis(self, save_path: str = None):
        """
        Plot precision-recall tradeoff and diversity-f1 tradeoff.

        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Precision-Recall tradeoff
        ax1 = axes[0]
        scatter = ax1.scatter(self.df['recall'], self.df['precision'],
                            s=200, alpha=0.6, c=self.df['f1'],
                            cmap='viridis', edgecolors='black', linewidths=1)

        # Add experiment labels
        for idx, row in self.df.iterrows():
            ax1.annotate(row['experiment'][:15],
                        (row['recall'], row['precision']),
                        fontsize=8, ha='center')

        ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax1.set_title('Precision-Recall Tradeoff', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 1.0)
        ax1.set_ylim(0, 1.0)

        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('F1 Score', fontsize=10)

        # Diversity-F1 tradeoff
        ax2 = axes[1]
        scatter2 = ax2.scatter(self.df['diversity'], self.df['f1'],
                             s=200, alpha=0.6, c=self.df['precision'],
                             cmap='plasma', edgecolors='black', linewidths=1)

        # Add experiment labels
        for idx, row in self.df.iterrows():
            ax2.annotate(row['experiment'][:15],
                        (row['diversity'], row['f1']),
                        fontsize=8, ha='center')

        ax2.set_xlabel('Diversity', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('Diversity-F1 Tradeoff', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 1.0)
        ax2.set_ylim(0, 1.0)

        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Precision', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

    def create_comprehensive_report(self, output_dir: str = None):
        """
        Create comprehensive comparison report.

        Args:
            output_dir: Directory to save visualizations
        """
        if output_dir is None:
            output_dir = Path(self.comparison_file).parent / "comparison_viz"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("Generating comprehensive comparison report...")

        # 1. Overall comparison
        print("  1/4 - Overall comparison...")
        self.plot_overall_comparison(
            save_path=str(output_path / "overall_comparison.png")
        )
        plt.close()

        # 2. F1 ranking
        print("  2/4 - F1 ranking...")
        self.plot_metric_ranking(
            metric='f1',
            save_path=str(output_path / "f1_ranking.png")
        )
        plt.close()

        # 3. Feature impact
        print("  3/4 - Feature impact analysis...")
        self.plot_feature_impact(
            save_path=str(output_path / "feature_impact.png")
        )
        plt.close()

        # 4. Tradeoff analysis
        print("  4/4 - Tradeoff analysis...")
        self.plot_tradeoff_analysis(
            save_path=str(output_path / "tradeoff_analysis.png")
        )
        plt.close()

        print(f"\n✅ Comprehensive comparison report generated in: {output_path}")

    def print_summary_table(self):
        """Print summary table to console."""
        print("\n" + "="*100)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*100 + "\n")

        # Sort by F1
        sorted_df = self.df.sort_values('f1', ascending=False)

        print(f"{'Rank':<6} {'Experiment':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Diversity':>10}")
        print("-"*100)

        for rank, (idx, row) in enumerate(sorted_df.iterrows(), 1):
            exp_name = row['experiment']
            if len(exp_name) > 33:
                exp_name = exp_name[:30] + "..."

            print(
                f"{rank:<6} {exp_name:<35} "
                f"{row['precision']:>10.3f} "
                f"{row['recall']:>10.3f} "
                f"{row['f1']:>10.3f} "
                f"{row['diversity']:>10.3f}"
            )

        print("\n" + "="*100 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare multiple experiment results"
    )
    parser.add_argument(
        "--comparison-file",
        type=str,
        required=True,
        help="Path to comparison JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save visualizations (default: same as comparison file)"
    )

    args = parser.parse_args()

    try:
        # Create comparator
        comparator = ExperimentComparator(args.comparison_file)

        # Print summary
        comparator.print_summary_table()

        # Generate visualizations
        comparator.create_comprehensive_report(args.output_dir)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
