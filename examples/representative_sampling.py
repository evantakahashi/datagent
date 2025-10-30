"""
Example: Generate Representative, Non-Redundant Datasets

Shows how to use the IterativeGenerator to create datasets where
each record contributes unique value.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.generators import IterativeGenerator
from src.evaluators import CoverageAnalyzer, RedundancyChecker


def main():
    print("=" * 60)
    print("REPRESENTATIVE SAMPLING EXAMPLE")
    print("=" * 60)
    print()

    # =====================================================================
    # Basic Usage: Generate Representative Dataset
    # =====================================================================
    print("BASIC USAGE: Generating 30 representative records...")
    print("-" * 60)

    # Initialize generator (no embeddings for faster demo)
    generator = IterativeGenerator(
        embedding_analyzer=None,  # Set to EmbeddingAnalyzer() for better quality
        redundancy_threshold=0.90,
        seed=42
    )

    # Generate representative dataset
    records, report = generator.generate_representative_dataset(
        target_size=30,
        max_iterations=3,
        selection_method="greedy_diversity",
        use_embeddings=False  # Set to True with ClinicalBERT
    )

    print(f"\n✓ Generated {len(records)} representative records")
    print(f"  Coverage score: {report['final_coverage_score']:.3f}")
    print(f"  Iterations used: {report['n_iterations']}")

    # =====================================================================
    # Analyze Quality
    # =====================================================================
    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS")
    print("=" * 60)

    # Check coverage
    coverage_analyzer = CoverageAnalyzer()
    coverage = coverage_analyzer.analyze_coverage(records)

    print(f"\nCoverage Analysis:")
    print(f"  Overall score: {coverage['overall_coverage_score']:.3f}")
    print(f"  Gaps found: {len(coverage['gaps'])}")
    print(f"  Strata covered: {coverage['strata_analysis']['n_strata']}")

    print(f"\nDimension-Level Coverage:")
    for dim, score in coverage['coverage_scores'].items():
        status = "✓" if score > 0.7 else "✗"
        print(f"  {status} {dim:20s}: {score:.3f}")

    # Check redundancy
    redundancy_checker = RedundancyChecker()
    redundancy = redundancy_checker.analyze_redundancy(records)

    print(f"\nRedundancy Analysis:")
    print(f"  Exact duplicates: {redundancy['n_exact_duplicates']}")
    print(f"  Near-duplicate pairs: {redundancy['n_near_duplicate_pairs']}")

    if redundancy['n_exact_duplicates'] == 0 and redundancy['n_near_duplicate_pairs'] == 0:
        print("  ✓ No redundancy detected!")

    # =====================================================================
    # Show Sample Records
    # =====================================================================
    print("\n" + "=" * 60)
    print("SAMPLE RECORDS")
    print("=" * 60)

    for i, record in enumerate(records[:5], 1):
        print(f"\n{i}. {record.record_id}")
        print(f"   Demographics: {record.demographics.age}y {record.demographics.sex.value} {record.demographics.ethnicity.value}")
        print(f"   Conditions: {', '.join([c.name for c in record.conditions])}")
        print(f"   Medications: {len(record.medications)} | Labs: {len(record.lab_results)}")

    # =====================================================================
    # Compare Different Selection Methods
    # =====================================================================
    print("\n" + "=" * 60)
    print("COMPARING SELECTION METHODS")
    print("=" * 60)
    print()

    methods_to_compare = ["greedy_diversity", "coverage_based", "cluster_based"]

    for method in methods_to_compare:
        print(f"\nTesting {method}...")
        records_method, report_method = generator.generate_representative_dataset(
            target_size=30,
            max_iterations=2,
            selection_method=method,
            use_embeddings=False
        )

        coverage_method = coverage_analyzer.analyze_coverage(records_method)
        print(f"  Coverage score: {coverage_method['overall_coverage_score']:.3f}")
        print(f"  Gaps: {len(coverage_method['gaps'])}")
        print(f"  Iterations: {report_method['n_iterations']}")

    # =====================================================================
    # Missing Combinations Detection
    # =====================================================================
    print("\n" + "=" * 60)
    print("MISSING COMBINATIONS ANALYSIS")
    print("=" * 60)

    missing = coverage_analyzer.identify_missing_combinations(
        records,
        dimensions=['age_group', 'sex', 'ethnicity']
    )

    if missing:
        print(f"\nFound {len(missing)} missing demographic combinations:")
        for combo in missing[:10]:  # Show first 10
            print(f"  - {combo}")

        print(f"\n💡 To fill these gaps, generate records with these specific constraints")
    else:
        print("\n✓ All demographic combinations are covered!")

    # =====================================================================
    # Generation Recommendations
    # =====================================================================
    print("\n" + "=" * 60)
    print("GENERATION RECOMMENDATIONS")
    print("=" * 60)

    recommendations = coverage_analyzer.get_generation_recommendations(
        records,
        target_size=50  # If we wanted to expand to 50
    )

    if recommendations:
        print(f"\nTo expand from {len(records)} to 50 records, generate:")
        for rec in recommendations[:5]:
            print(f"\n  Priority: {rec.get('priority', 'N/A')}")
            print(f"  Constraints: {rec.get('constraints', {})}")
            print(f"  Quantity: {rec.get('n_records', 1)}")
    else:
        print("\nDataset is already well-balanced!")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
