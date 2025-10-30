"""Visualization tools for diversity analysis"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Dict, Optional
import os

from src.models import PatientRecord


class DiversityVisualizer:
    """Create visualizations for diversity metrics"""

    def __init__(self, output_dir: str = "outputs"):
        """
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)

    def plot_tsne(
        self,
        real_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray,
        perplexity: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Create t-SNE visualization of real vs synthetic embeddings

        Args:
            real_embeddings: Real data embeddings
            synthetic_embeddings: Synthetic data embeddings
            perplexity: t-SNE perplexity parameter
            save_path: Path to save the plot
        """
        print(f"Running t-SNE (perplexity={perplexity})...")

        # Combine embeddings
        combined = np.vstack([real_embeddings, synthetic_embeddings])
        labels = ['Real'] * len(real_embeddings) + ['Synthetic'] * len(synthetic_embeddings)

        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(combined)

        # Split back
        real_2d = embeddings_2d[:len(real_embeddings)]
        synthetic_2d = embeddings_2d[len(real_embeddings):]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.scatter(real_2d[:, 0], real_2d[:, 1],
                  c='blue', alpha=0.6, s=50, label='Real', edgecolors='navy', linewidth=0.5)
        ax.scatter(synthetic_2d[:, 0], synthetic_2d[:, 1],
                  c='red', alpha=0.6, s=50, label='Synthetic', edgecolors='darkred', linewidth=0.5)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE Visualization: Real vs Synthetic Patient Records', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "tsne_visualization.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
        plt.close()

    def plot_distribution_comparison(
        self,
        real_records: List[PatientRecord],
        synthetic_records: List[PatientRecord],
        save_path: Optional[str] = None
    ):
        """
        Create comparison plots for demographic distributions

        Args:
            real_records: Real patient records
            synthetic_records: Synthetic patient records
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Distribution Comparison: Real vs Synthetic', fontsize=16, fontweight='bold')

        # Age distribution
        real_ages = [r.demographics.age for r in real_records]
        synthetic_ages = [r.demographics.age for r in synthetic_records]

        axes[0, 0].hist(real_ages, bins=20, alpha=0.6, label='Real', color='blue', edgecolor='navy')
        axes[0, 0].hist(synthetic_ages, bins=20, alpha=0.6, label='Synthetic', color='red', edgecolor='darkred')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].legend()

        # Sex distribution
        real_sex = [r.demographics.sex.value for r in real_records]
        synthetic_sex = [r.demographics.sex.value for r in synthetic_records]

        sex_categories = list(set(real_sex + synthetic_sex))
        real_sex_counts = [real_sex.count(s) for s in sex_categories]
        synthetic_sex_counts = [synthetic_sex.count(s) for s in sex_categories]

        x = np.arange(len(sex_categories))
        width = 0.35

        axes[0, 1].bar(x - width/2, real_sex_counts, width, label='Real', color='blue', alpha=0.7)
        axes[0, 1].bar(x + width/2, synthetic_sex_counts, width, label='Synthetic', color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Sex')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Sex Distribution')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(sex_categories)
        axes[0, 1].legend()

        # Ethnicity distribution
        real_ethnicity = [r.demographics.ethnicity.value for r in real_records]
        synthetic_ethnicity = [r.demographics.ethnicity.value for r in synthetic_records]

        ethnicity_categories = list(set(real_ethnicity + synthetic_ethnicity))
        real_ethnicity_counts = [real_ethnicity.count(e) for e in ethnicity_categories]
        synthetic_ethnicity_counts = [synthetic_ethnicity.count(e) for e in ethnicity_categories]

        x = np.arange(len(ethnicity_categories))

        axes[0, 2].bar(x - width/2, real_ethnicity_counts, width, label='Real', color='blue', alpha=0.7)
        axes[0, 2].bar(x + width/2, synthetic_ethnicity_counts, width, label='Synthetic', color='red', alpha=0.7)
        axes[0, 2].set_xlabel('Ethnicity')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Ethnicity Distribution')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(ethnicity_categories, rotation=45, ha='right')
        axes[0, 2].legend()

        # Number of conditions
        real_num_conds = [len(r.conditions) for r in real_records]
        synthetic_num_conds = [len(r.conditions) for r in synthetic_records]

        axes[1, 0].hist(real_num_conds, bins=range(0, 6), alpha=0.6, label='Real', color='blue', edgecolor='navy')
        axes[1, 0].hist(synthetic_num_conds, bins=range(0, 6), alpha=0.6, label='Synthetic', color='red', edgecolor='darkred')
        axes[1, 0].set_xlabel('Number of Conditions')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Number of Conditions Distribution')
        axes[1, 0].legend()

        # Vital signs: Heart Rate
        real_hr = [r.vitals.heart_rate for r in real_records]
        synthetic_hr = [r.vitals.heart_rate for r in synthetic_records]

        axes[1, 1].hist(real_hr, bins=20, alpha=0.6, label='Real', color='blue', edgecolor='navy')
        axes[1, 1].hist(synthetic_hr, bins=20, alpha=0.6, label='Synthetic', color='red', edgecolor='darkred')
        axes[1, 1].set_xlabel('Heart Rate (bpm)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Heart Rate Distribution')
        axes[1, 1].legend()

        # Vital signs: Blood Pressure
        real_systolic = [r.vitals.systolic_bp for r in real_records]
        synthetic_systolic = [r.vitals.systolic_bp for r in synthetic_records]

        axes[1, 2].hist(real_systolic, bins=20, alpha=0.6, label='Real', color='blue', edgecolor='navy')
        axes[1, 2].hist(synthetic_systolic, bins=20, alpha=0.6, label='Synthetic', color='red', edgecolor='darkred')
        axes[1, 2].set_xlabel('Systolic BP (mmHg)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Systolic Blood Pressure Distribution')
        axes[1, 2].legend()

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "distribution_comparison.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution comparison plot saved to {save_path}")
        plt.close()

    def plot_entropy_comparison(
        self,
        entropy_comparison: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create bar plot comparing normalized entropy across dimensions

        Args:
            entropy_comparison: Output from ConceptEntropyCalculator.compare_entropy()
            save_path: Path to save the plot
        """
        entropy_diffs = entropy_comparison['entropy_differences']

        dimensions = list(entropy_diffs.keys())
        real_entropies = [entropy_diffs[d]['real'] for d in dimensions]
        synthetic_entropies = [entropy_diffs[d]['synthetic'] for d in dimensions]

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(dimensions))
        width = 0.35

        ax.bar(x - width/2, real_entropies, width, label='Real', color='blue', alpha=0.7)
        ax.bar(x + width/2, synthetic_entropies, width, label='Synthetic', color='red', alpha=0.7)

        ax.set_xlabel('Concept Dimension', fontsize=12)
        ax.set_ylabel('Normalized Entropy', fontsize=12)
        ax.set_title('Concept Entropy Comparison: Real vs Synthetic', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        # Add horizontal line at 1.0 (perfect entropy)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Balance')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "entropy_comparison.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Entropy comparison plot saved to {save_path}")
        plt.close()

    def plot_metrics_summary(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Create a summary visualization of key metrics

        Args:
            metrics: Dictionary of metric names and values
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in metric_values]

        ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Diversity Metrics Summary', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "metrics_summary.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics summary plot saved to {save_path}")
        plt.close()
