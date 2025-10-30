"""Generate comprehensive evaluation reports"""
import json
from datetime import datetime
from typing import Dict, Any
import os


class ReportGenerator:
    """Generate formatted reports from evaluation results"""

    @staticmethod
    def generate_text_report(results: Dict[str, Any], output_path: str = None) -> str:
        """
        Generate a human-readable text report

        Args:
            results: Complete evaluation results dictionary
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SYNTHETIC MEDICAL DATA EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Dataset info
        lines.append("DATASET INFORMATION")
        lines.append("-" * 80)
        lines.append(f"Real records: {results['dataset_info']['n_real']}")
        lines.append(f"Synthetic records: {results['dataset_info']['n_synthetic']}")
        lines.append("")

        # Embedding Coverage Metrics
        if 'embedding_metrics' in results and results['embedding_metrics'] is not None:
            lines.append("EMBEDDING COVERAGE METRICS (ClinicalBERT)")
            lines.append("-" * 80)
            metrics = results['embedding_metrics']['coverage_metrics']
            lines.append(f"Average Maximum Similarity: {metrics['avg_max_similarity']:.4f}")
            lines.append(f"Std Maximum Similarity: {metrics['std_max_similarity']:.4f}")
            lines.append(f"Coverage @ 0.7: {metrics['coverage@0.7']:.2%}")
            lines.append(f"Coverage @ 0.8: {metrics['coverage@0.8']:.2%}")
            lines.append(f"Coverage @ 0.9: {metrics['coverage@0.9']:.2%}")
            lines.append(f"Diversity Score: {metrics['diversity_score']:.4f}")
            lines.append(f"Centroid Distance: {metrics['centroid_distance']:.4f}")
            lines.append("")

            nn_metrics = results['embedding_metrics']['nearest_neighbor_metrics']
            lines.append(f"Average Top-5 Similarity: {nn_metrics['avg_top5_similarity']:.4f}")
            lines.append("")
        else:
            lines.append("EMBEDDING COVERAGE METRICS (ClinicalBERT)")
            lines.append("-" * 80)
            lines.append("Embeddings were not computed (skipped or failed)")
            lines.append("")

        # Distribution Metrics
        if 'distribution_metrics' in results:
            lines.append("DISTRIBUTION METRICS")
            lines.append("-" * 80)
            dist_metrics = results['distribution_metrics']
            lines.append(f"FID Score: {dist_metrics['fid_score']:.4f} (lower is better)")
            lines.append("")
            lines.append("KL Divergences by Feature:")
            for key in sorted(dist_metrics.keys()):
                if key.startswith('kl_div_'):
                    feature = key.replace('kl_div_', '')
                    lines.append(f"  {feature}: {dist_metrics[key]:.4f}")
            lines.append("")

        # Concept Entropy
        if 'entropy_comparison' in results:
            lines.append("CONCEPT ENTROPY ANALYSIS")
            lines.append("-" * 80)
            entropy = results['entropy_comparison']
            lines.append(f"Real Dataset Diversity Score: {entropy['real_diversity_score']:.4f}")
            lines.append(f"Synthetic Dataset Diversity Score: {entropy['synthetic_diversity_score']:.4f}")
            lines.append(f"Diversity Score Difference: {entropy['diversity_score_diff']:.4f}")
            lines.append("")

            lines.append("Normalized Entropy by Dimension:")
            for dim, values in sorted(entropy['entropy_differences'].items()):
                lines.append(f"  {dim}:")
                lines.append(f"    Real: {values['real']:.4f}")
                lines.append(f"    Synthetic: {values['synthetic']:.4f}")
                lines.append(f"    Difference: {values['difference']:.4f}")
            lines.append("")

        # Overall Assessment
        lines.append("OVERALL ASSESSMENT")
        lines.append("-" * 80)

        if 'embedding_metrics' in results and results['embedding_metrics'] is not None:
            diversity_score = results['embedding_metrics']['coverage_metrics']['diversity_score']
            if diversity_score > 0.7:
                assessment = "EXCELLENT - High internal diversity"
            elif diversity_score > 0.5:
                assessment = "GOOD - Moderate internal diversity"
            else:
                assessment = "POOR - Low internal diversity (too similar records)"
            lines.append(f"Internal Diversity: {assessment}")

        if 'distribution_metrics' in results:
            fid = results['distribution_metrics']['fid_score']
            if fid < 50:
                assessment = "EXCELLENT - Very close to real distribution"
            elif fid < 100:
                assessment = "GOOD - Reasonably close to real distribution"
            else:
                assessment = "POOR - Diverges from real distribution"
            lines.append(f"Distribution Match: {assessment}")

        if 'entropy_comparison' in results:
            diff = results['entropy_comparison']['diversity_score_diff']
            if diff < 0.1:
                assessment = "EXCELLENT - Very balanced like real data"
            elif diff < 0.2:
                assessment = "GOOD - Reasonably balanced"
            else:
                assessment = "POOR - Unbalanced representation"
            lines.append(f"Balance Quality: {assessment}")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Text report saved to {output_path}")

        return report

    @staticmethod
    def save_json_report(results: Dict[str, Any], output_path: str):
        """
        Save complete results as JSON

        Args:
            results: Complete evaluation results dictionary
            output_path: Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"JSON report saved to {output_path}")
