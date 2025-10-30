"""
Example: Evaluate diversity of synthetic data

This shows how to use individual evaluation components.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.generators import PatientRecordGenerator
from src.evaluators import ConceptEntropyCalculator, DistributionMetrics


def main():
    print("Diversity Evaluation Example")
    print("=" * 60)
    print()

    # Generate some records
    generator = PatientRecordGenerator(seed=42)

    print("Generating 30 patient records...")
    records = generator.generate_batch(n=30, diversity_constraints=True)
    print(f"Generated {len(records)} records\n")

    # =====================================================================
    # 1. Concept Entropy Analysis
    # =====================================================================
    print("=" * 60)
    print("CONCEPT ENTROPY ANALYSIS")
    print("=" * 60)

    entropy_calc = ConceptEntropyCalculator()
    entropy_metrics = entropy_calc.calculate_concept_entropy_metrics(records)

    print("\nNormalized Entropy by Dimension (0=concentrated, 1=balanced):")
    print("-" * 60)
    for dim, value in sorted(entropy_metrics['normalized_entropy'].items()):
        bar_length = int(value * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{dim:25s} {bar} {value:.3f}")

    print(f"\nOverall Diversity Score: {entropy_metrics['overall_diversity_score']:.3f}")

    if entropy_metrics['unbalanced_dimensions']:
        print("\nUnbalanced dimensions (entropy < 0.7):")
        for dim, value in entropy_metrics['unbalanced_dimensions'].items():
            print(f"  - {dim}: {value:.3f}")
    else:
        print("\nAll dimensions are well-balanced!")

    # =====================================================================
    # 2. Compare Two Datasets
    # =====================================================================
    print("\n" + "=" * 60)
    print("COMPARING TWO DATASETS")
    print("=" * 60)

    # Generate second dataset with different seed
    print("\nGenerating second dataset for comparison...")
    generator2 = PatientRecordGenerator(seed=999)
    records2 = generator2.generate_batch(n=30, diversity_constraints=False)  # Less diverse

    # Compare entropy
    comparison = entropy_calc.compare_entropy(records, records2)

    print(f"\nDataset 1 Diversity Score: {comparison['real_diversity_score']:.3f}")
    print(f"Dataset 2 Diversity Score: {comparison['synthetic_diversity_score']:.3f}")
    print(f"Difference: {comparison['diversity_score_diff']:.3f}")

    if comparison['diversity_score_diff'] < 0.1:
        print("\n✓ Datasets have similar diversity profiles")
    else:
        print("\n✗ Datasets have different diversity profiles")

    # =====================================================================
    # 3. Distribution Metrics (without embeddings)
    # =====================================================================
    print("\n" + "=" * 60)
    print("DISTRIBUTION METRICS")
    print("=" * 60)

    dist_calc = DistributionMetrics()
    kl_divs = dist_calc.compute_categorical_kl_divergences(records, records2)

    print("\nKL Divergence by Feature (lower = more similar):")
    print("-" * 60)
    for feature, value in sorted(kl_divs.items()):
        if feature != 'kl_div_average':
            feature_name = feature.replace('kl_div_', '')
            if value < 0.1:
                status = "✓ Very similar"
            elif value < 0.3:
                status = "~ Somewhat different"
            else:
                status = "✗ Quite different"
            print(f"{feature_name:20s}: {value:.4f}  {status}")

    print(f"\nAverage KL Divergence: {kl_divs['kl_div_average']:.4f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
