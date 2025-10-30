"""
Demo: Representative vs Random Sampling

This demonstrates the difference between:
1. Random generation (potentially redundant, with gaps)
2. Representative generation (non-redundant, full coverage)
"""

import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.generators import PatientRecordGenerator, IterativeGenerator
from src.evaluators import (
    EmbeddingAnalyzer,
    RedundancyChecker,
    CoverageAnalyzer,
    DiversityVisualizer,
    ConceptEntropyCalculator
)


def main():
    print("=" * 80)
    print("REPRESENTATIVE vs RANDOM SAMPLING COMPARISON")
    print("=" * 80)
    print()

    TARGET_SIZE = 50
    OUTPUT_DIR = "outputs/representative_comparison"
    USE_EMBEDDINGS = False  # Set to True to use ClinicalBERT (slower but more accurate)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # METHOD 1: Random Generation (Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: RANDOM GENERATION (BASELINE)")
    print("=" * 80)
    print()

    print(f"Generating {TARGET_SIZE} records randomly...")
    random_generator = PatientRecordGenerator(seed=42)
    random_records = random_generator.generate_batch(TARGET_SIZE, diversity_constraints=False)
    print(f"Generated {len(random_records)} records")

    # Analyze random dataset
    print("\nAnalyzing random dataset...")

    # Coverage analysis
    coverage_analyzer = CoverageAnalyzer()
    random_coverage = coverage_analyzer.analyze_coverage(random_records)

    print(f"\nRandom Dataset - Coverage Analysis:")
    print(f"  Overall coverage score: {random_coverage['overall_coverage_score']:.3f}")
    print(f"  Number of gaps: {len(random_coverage['gaps'])}")
    print(f"  Strata covered: {random_coverage['strata_analysis']['n_strata']}")
    print(f"  Strata with gaps: {random_coverage['strata_analysis']['strata_with_gaps']}")

    # Redundancy analysis
    redundancy_checker = RedundancyChecker(similarity_threshold=0.90)
    random_redundancy = redundancy_checker.analyze_redundancy(random_records)

    print(f"\nRandom Dataset - Redundancy Analysis:")
    print(f"  Exact duplicates: {random_redundancy['n_exact_duplicates']}")
    print(f"  Near-duplicate pairs: {random_redundancy['n_near_duplicate_pairs']}")

    # Entropy analysis
    entropy_calc = ConceptEntropyCalculator()
    random_entropy = entropy_calc.calculate_concept_entropy_metrics(random_records)

    print(f"\nRandom Dataset - Entropy Analysis:")
    print(f"  Overall diversity score: {random_entropy['overall_diversity_score']:.3f}")
    print(f"  Mean normalized entropy: {random_entropy['summary']['mean_normalized_entropy']:.3f}")

    # =========================================================================
    # METHOD 2: Representative Generation (Our Approach)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: REPRESENTATIVE GENERATION (OUR APPROACH)")
    print("=" * 80)
    print()

    # Initialize with embedding analyzer if requested
    embedding_analyzer = None
    if USE_EMBEDDINGS:
        print("Initializing ClinicalBERT embeddings...")
        embedding_analyzer = EmbeddingAnalyzer()

    iterative_gen = IterativeGenerator(
        embedding_analyzer=embedding_analyzer,
        redundancy_threshold=0.90,
        seed=42
    )

    # Generate representative dataset
    representative_records, generation_report = iterative_gen.generate_representative_dataset(
        target_size=TARGET_SIZE,
        max_iterations=3,
        overgenerate_factor=1.5,
        selection_method="greedy_diversity",
        use_embeddings=USE_EMBEDDINGS
    )

    # Analyze representative dataset
    print("\nAnalyzing representative dataset...")

    rep_coverage = coverage_analyzer.analyze_coverage(representative_records)
    rep_redundancy = redundancy_checker.analyze_redundancy(representative_records)
    rep_entropy = entropy_calc.calculate_concept_entropy_metrics(representative_records)

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    comparison_table = [
        ("Metric", "Random", "Representative", "Improvement"),
        ("-" * 20, "-" * 15, "-" * 15, "-" * 15),
        (
            "Coverage Score",
            f"{random_coverage['overall_coverage_score']:.3f}",
            f"{rep_coverage['overall_coverage_score']:.3f}",
            f"{((rep_coverage['overall_coverage_score'] - random_coverage['overall_coverage_score']) / random_coverage['overall_coverage_score'] * 100):.1f}%"
        ),
        (
            "Diversity Score",
            f"{random_entropy['overall_diversity_score']:.3f}",
            f"{rep_entropy['overall_diversity_score']:.3f}",
            f"{((rep_entropy['overall_diversity_score'] - random_entropy['overall_diversity_score']) / random_entropy['overall_diversity_score'] * 100):.1f}%"
        ),
        (
            "Coverage Gaps",
            f"{len(random_coverage['gaps'])}",
            f"{len(rep_coverage['gaps'])}",
            f"{len(random_coverage['gaps']) - len(rep_coverage['gaps'])} fewer"
        ),
        (
            "Exact Duplicates",
            f"{random_redundancy['n_exact_duplicates']}",
            f"{rep_redundancy['n_exact_duplicates']}",
            f"{random_redundancy['n_exact_duplicates'] - rep_redundancy['n_exact_duplicates']} fewer"
        ),
        (
            "Near-Dup Pairs",
            f"{random_redundancy['n_near_duplicate_pairs']}",
            f"{rep_redundancy['n_near_duplicate_pairs']}",
            f"{random_redundancy['n_near_duplicate_pairs'] - rep_redundancy['n_near_duplicate_pairs']} fewer"
        ),
        (
            "Strata Covered",
            f"{random_coverage['strata_analysis']['n_strata']}",
            f"{rep_coverage['strata_analysis']['n_strata']}",
            f"+{rep_coverage['strata_analysis']['n_strata'] - random_coverage['strata_analysis']['n_strata']}"
        ),
    ]

    for row in comparison_table:
        print(f"{row[0]:20s} | {row[1]:15s} | {row[2]:15s} | {row[3]:15s}")

    # =========================================================================
    # DETAILED ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()

    print("RANDOM DATASET - Coverage Gaps:")
    if random_coverage['gaps']:
        for dimension, gap_info in random_coverage['gaps'].items():
            print(f"\n  {dimension}:")
            if isinstance(gap_info, dict) and 'gaps' in gap_info:
                for gap in gap_info['gaps']:
                    if isinstance(gap, dict):
                        print(f"    - {gap.get('value', 'N/A')}: {gap.get('count', 0)} (expected: {gap.get('expected', 0):.1f})")
    else:
        print("  No significant gaps")

    print("\n\nREPRESENTATIVE DATASET - Coverage Gaps:")
    if rep_coverage['gaps']:
        for dimension, gap_info in rep_coverage['gaps'].items():
            print(f"\n  {dimension}:")
            if isinstance(gap_info, dict) and 'gaps' in gap_info:
                for gap in gap_info['gaps']:
                    if isinstance(gap, dict):
                        print(f"    - {gap.get('value', 'N/A')}: {gap.get('count', 0)} (expected: {gap.get('expected', 0):.1f})")
    else:
        print("  No significant gaps ✓")

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    visualizer = DiversityVisualizer(output_dir=OUTPUT_DIR)

    # Distribution comparison
    print("Creating distribution comparison plots...")
    visualizer.plot_distribution_comparison(
        random_records,
        representative_records,
        save_path=os.path.join(OUTPUT_DIR, "random_vs_representative_distributions.png")
    )

    # Entropy comparison
    print("Creating entropy comparison plots...")
    entropy_comparison = entropy_calc.compare_entropy(random_records, representative_records)
    visualizer.plot_entropy_comparison(
        entropy_comparison,
        save_path=os.path.join(OUTPUT_DIR, "random_vs_representative_entropy.png")
    )

    # Summary metrics
    print("Creating summary metrics...")
    summary_metrics = {
        'Random Coverage': random_coverage['overall_coverage_score'],
        'Representative Coverage': rep_coverage['overall_coverage_score'],
        'Random Diversity': random_entropy['overall_diversity_score'],
        'Representative Diversity': rep_entropy['overall_diversity_score'],
    }
    visualizer.plot_metrics_summary(
        summary_metrics,
        save_path=os.path.join(OUTPUT_DIR, "comparison_summary.png")
    )

    # =========================================================================
    # KEY INSIGHTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    insights = []

    # Coverage improvement
    coverage_improvement = rep_coverage['overall_coverage_score'] - random_coverage['overall_coverage_score']
    if coverage_improvement > 0.1:
        insights.append(
            f"✓ Representative sampling improved coverage by {coverage_improvement:.1%}, "
            f"ensuring better representation across all demographic and clinical dimensions."
        )

    # Redundancy reduction
    redundancy_reduction = random_redundancy['n_near_duplicate_pairs'] - rep_redundancy['n_near_duplicate_pairs']
    if redundancy_reduction > 0:
        insights.append(
            f"✓ Eliminated {redundancy_reduction} near-duplicate pairs, "
            f"ensuring each record contributes unique information."
        )

    # Diversity improvement
    diversity_improvement = rep_entropy['overall_diversity_score'] - random_entropy['overall_diversity_score']
    if diversity_improvement > 0.05:
        insights.append(
            f"✓ Increased diversity score by {diversity_improvement:.1%}, "
            f"providing more balanced representation across categories."
        )

    # Gap reduction
    gap_reduction = len(random_coverage['gaps']) - len(rep_coverage['gaps'])
    if gap_reduction > 0:
        insights.append(
            f"✓ Filled {gap_reduction} coverage gaps, "
            f"ensuring no underrepresented groups."
        )

    # Strata improvement
    strata_improvement = rep_coverage['strata_analysis']['n_strata'] - random_coverage['strata_analysis']['n_strata']
    if strata_improvement > 0:
        insights.append(
            f"✓ Covered {strata_improvement} additional demographic strata, "
            f"improving representation of rare combinations."
        )

    if insights:
        for i, insight in enumerate(insights, 1):
            # Replace Unicode checkmarks for Windows console compatibility
            console_insight = insight.replace('✓', '[OK]')
            try:
                print(f"{i}. {console_insight}\n")
            except UnicodeEncodeError:
                # Fallback if still issues
                print(f"{i}. {console_insight.encode('ascii', 'replace').decode('ascii')}\n")
    else:
        print("Datasets are comparable in quality.")

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("For maximum data quality and representativeness:")
    print()
    print("1. USE REPRESENTATIVE GENERATION when:")
    print("   - You need maximum information density")
    print("   - Dataset size is limited")
    print("   - Coverage across all strata is critical")
    print("   - You want to avoid redundant/duplicate records")
    print()
    print("2. Consider random generation when:")
    print("   - Speed is more important than quality")
    print("   - You have unlimited data budget")
    print("   - Simple random sampling is sufficient")
    print()
    print("3. For best results:")
    print("   - Enable ClinicalBERT embeddings (set USE_EMBEDDINGS=True)")
    print("   - Use iterative refinement with 3-5 iterations")
    print("   - Monitor coverage gaps and fill systematically")
    print()

    # =========================================================================
    # SAVE DETAILED REPORT
    # =========================================================================
    print("=" * 80)
    print("Saving detailed comparison report...")
    print("=" * 80)
    print()

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPRESENTATIVE vs RANDOM SAMPLING - DETAILED REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target size: {TARGET_SIZE}\n")
        f.write("\n")

        f.write("COMPARISON TABLE:\n")
        f.write("-" * 80 + "\n")
        for row in comparison_table:
            f.write(f"{row[0]:20s} | {row[1]:15s} | {row[2]:15s} | {row[3]:15s}\n")

        f.write("\n\nKEY INSIGHTS:\n")
        f.write("-" * 80 + "\n")
        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n\n")

    print(f"Report saved to: {report_path}")
    print()
    print("Generated visualizations:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'random_vs_representative_distributions.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'random_vs_representative_entropy.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'comparison_summary.png')}")
    print()

    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
