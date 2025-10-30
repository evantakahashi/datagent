"""Iterative Generator with Redundancy Elimination and Coverage Optimization"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from src.models import PatientRecord
from src.generators import PatientRecordGenerator
from src.evaluators.redundancy_checker import RedundancyChecker
from src.evaluators.coverage_analyzer import CoverageAnalyzer
from src.evaluators.active_selector import ActiveDataSelector


class IterativeGenerator:
    """
    Generate representative, non-redundant datasets through iterative refinement

    Process:
    1. Generate initial batch (over-generate)
    2. Remove redundant records
    3. Analyze coverage gaps
    4. Generate targeted records to fill gaps
    5. Repeat until target size with good coverage
    """

    def __init__(
        self,
        embedding_analyzer=None,
        redundancy_threshold: float = 0.90,
        seed: int = None
    ):
        """
        Args:
            embedding_analyzer: Optional EmbeddingAnalyzer for semantic similarity
            redundancy_threshold: Similarity threshold for redundancy
            seed: Random seed for reproducibility
        """
        self.generator = PatientRecordGenerator(seed=seed)
        self.redundancy_checker = RedundancyChecker(similarity_threshold=redundancy_threshold)
        self.coverage_analyzer = CoverageAnalyzer()
        self.active_selector = ActiveDataSelector(redundancy_threshold=redundancy_threshold)
        self.embedding_analyzer = embedding_analyzer

    def generate_representative_dataset(
        self,
        target_size: int,
        max_iterations: int = 5,
        overgenerate_factor: float = 1.5,
        selection_method: str = "greedy_diversity",
        use_embeddings: bool = True
    ) -> Tuple[List[PatientRecord], Dict]:
        """
        Generate a representative, non-redundant dataset

        Args:
            target_size: Desired final dataset size
            max_iterations: Maximum refinement iterations
            overgenerate_factor: Generate this many times target_size initially
            selection_method: "greedy_diversity", "coverage_based", "cluster_based"
            use_embeddings: Use ClinicalBERT embeddings (slower but more accurate)

        Returns:
            (final_records, generation_report)
        """
        print("=" * 80)
        print("ITERATIVE REPRESENTATIVE DATASET GENERATION")
        print("=" * 80)
        print(f"Target size: {target_size}")
        print(f"Selection method: {selection_method}")
        print(f"Max iterations: {max_iterations}")
        print()

        iteration_reports = []
        current_records = []

        for iteration in range(max_iterations):
            print(f"\n{'=' * 80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'=' * 80}\n")

            if iteration == 0:
                # Initial generation: over-generate
                n_to_generate = int(target_size * overgenerate_factor)
                print(f"Generating {n_to_generate} initial records (over-generation)...")
                current_records = self.generator.generate_batch(
                    n=n_to_generate,
                    diversity_constraints=True
                )
                print(f"Generated {len(current_records)} records")
            else:
                # Subsequent iterations: targeted generation for gaps
                print(f"Current dataset size: {len(current_records)}")

                # Analyze coverage gaps
                print("Analyzing coverage gaps...")
                recommendations = self.coverage_analyzer.get_generation_recommendations(
                    current_records,
                    target_size
                )

                if not recommendations:
                    print("No significant gaps found. Dataset is representative!")
                    break

                print(f"Found {len(recommendations)} generation recommendations")

                # Generate records to fill gaps
                n_to_generate = min(target_size - len(current_records), len(recommendations))
                print(f"Generating {n_to_generate} targeted records...")

                new_records = []
                for rec in recommendations[:n_to_generate]:
                    constraints = rec.get('constraints', {})
                    try:
                        new_record = self.generator.generate_record(constraints=constraints)
                        new_records.append(new_record)
                    except Exception as e:
                        print(f"Warning: Failed to generate record with constraints {constraints}: {e}")

                current_records.extend(new_records)
                print(f"Added {len(new_records)} new records. Total: {len(current_records)}")

            # Remove redundancy
            print("\nChecking for redundancy...")
            embeddings = None

            if use_embeddings and self.embedding_analyzer is not None:
                print("Computing embeddings for redundancy detection...")
                embeddings = self.embedding_analyzer.embed_records(current_records, show_progress=True)

            redundancy_analysis = self.redundancy_checker.analyze_redundancy(
                current_records,
                embeddings
            )

            print(f"Redundancy Analysis:")
            print(f"  Exact duplicates: {redundancy_analysis['n_exact_duplicates']}")
            print(f"  Near-duplicate pairs: {redundancy_analysis['n_near_duplicate_pairs']}")
            if redundancy_analysis['avg_pairwise_similarity'] is not None:
                print(f"  Avg pairwise similarity: {redundancy_analysis['avg_pairwise_similarity']:.3f}")

            # Deduplicate if needed
            if redundancy_analysis['n_exact_duplicates'] > 0 or redundancy_analysis['n_near_duplicate_pairs'] > 0:
                print("\nRemoving redundant records...")
                method = "embedding" if embeddings is not None else "feature"
                current_records, kept_indices, dedup_stats = self.redundancy_checker.deduplicate(
                    current_records,
                    embeddings,
                    method=method
                )
                print(f"Removed {dedup_stats['n_removed']} redundant records")
                print(f"Remaining: {len(current_records)} records")

                # Update embeddings
                if embeddings is not None:
                    embeddings = embeddings[kept_indices]

            # Analyze coverage
            print("\nAnalyzing coverage...")
            coverage_report = self.coverage_analyzer.analyze_coverage(current_records)

            print(f"Coverage Analysis:")
            print(f"  Overall coverage score: {coverage_report['overall_coverage_score']:.3f}")
            print(f"  Gaps identified: {len(coverage_report['gaps'])}")

            # Select final subset if we have too many
            if len(current_records) > target_size:
                print(f"\nSelecting best {target_size} records using {selection_method}...")

                if embeddings is None and self.embedding_analyzer is not None:
                    print("Computing embeddings for selection...")
                    embeddings = self.embedding_analyzer.embed_records(current_records, show_progress=True)

                if selection_method == "greedy_diversity" and embeddings is not None:
                    selected_records, _, selection_stats = self.active_selector.greedy_diversity_selection(
                        current_records,
                        embeddings,
                        target_size
                    )
                elif selection_method == "coverage_based" and embeddings is not None:
                    selected_records, _, selection_stats = self.active_selector.coverage_based_selection(
                        current_records,
                        embeddings,
                        target_size
                    )
                elif selection_method == "cluster_based" and embeddings is not None:
                    selected_records, _, selection_stats = self.active_selector.cluster_based_selection(
                        current_records,
                        embeddings,
                        target_size
                    )
                else:
                    # Fallback: entropy maximizing (doesn't need embeddings)
                    selected_records, _, selection_stats = self.active_selector.entropy_maximizing_selection(
                        current_records,
                        target_size
                    )

                current_records = selected_records
                print(f"Selected {len(current_records)} most representative records")

            # Store iteration report
            iteration_reports.append({
                'iteration': iteration + 1,
                'n_records': len(current_records),
                'coverage_score': coverage_report['overall_coverage_score'],
                'redundancy_analysis': redundancy_analysis,
                'coverage_report': coverage_report,
            })

            # Check if we're done
            if (len(current_records) == target_size and
                coverage_report['overall_coverage_score'] > 0.8 and
                len(coverage_report['gaps']) == 0):
                print("\n✓ Target achieved! Dataset is representative and non-redundant.")
                break

        # Final report
        final_coverage = self.coverage_analyzer.analyze_coverage(current_records)

        generation_report = {
            'target_size': target_size,
            'final_size': len(current_records),
            'n_iterations': len(iteration_reports),
            'selection_method': selection_method,
            'final_coverage_score': final_coverage['overall_coverage_score'],
            'iteration_reports': iteration_reports,
            'final_coverage_analysis': final_coverage,
        }

        print("\n" + "=" * 80)
        print("GENERATION COMPLETE")
        print("=" * 80)
        print(f"Final dataset size: {len(current_records)}")
        print(f"Final coverage score: {final_coverage['overall_coverage_score']:.3f}")
        print(f"Iterations used: {len(iteration_reports)}")

        return current_records, generation_report

    def compare_methods(
        self,
        target_size: int,
        methods: List[str] = None
    ) -> Dict:
        """
        Compare different selection methods

        Args:
            target_size: Target dataset size
            methods: List of methods to compare

        Returns:
            Comparison report
        """
        if methods is None:
            methods = ["greedy_diversity", "coverage_based", "cluster_based"]

        print("=" * 80)
        print("COMPARING SELECTION METHODS")
        print("=" * 80)
        print()

        results = {}

        for method in methods:
            print(f"\nTesting method: {method}")
            print("-" * 80)

            records, report = self.generate_representative_dataset(
                target_size=target_size,
                max_iterations=3,
                selection_method=method,
                use_embeddings=(self.embedding_analyzer is not None)
            )

            results[method] = {
                'records': records,
                'report': report,
                'coverage_score': report['final_coverage_score'],
                'n_iterations': report['n_iterations'],
            }

        # Print comparison
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print()

        for method, result in results.items():
            print(f"{method}:")
            print(f"  Coverage Score: {result['coverage_score']:.3f}")
            print(f"  Iterations: {result['n_iterations']}")

        # Find best method
        best_method = max(results.keys(), key=lambda m: results[m]['coverage_score'])
        print(f"\nBest method: {best_method} (coverage: {results[best_method]['coverage_score']:.3f})")

        return results
