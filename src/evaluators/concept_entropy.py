"""Concept Entropy Calculator - Measures balance across conditions, demographics, and modalities"""
import numpy as np
from typing import List, Dict
from collections import Counter
from scipy.stats import entropy as scipy_entropy

from src.models import PatientRecord


class ConceptEntropyCalculator:
    """Calculate entropy metrics to measure diversity balance"""

    @staticmethod
    def calculate_entropy(distribution: List[any]) -> float:
        """
        Calculate Shannon entropy for a distribution

        Higher entropy = more balanced/diverse
        Lower entropy = concentrated on few values

        Args:
            distribution: List of values (can be any hashable type)

        Returns:
            Entropy value (in bits)
        """
        if not distribution:
            return 0.0

        # Count occurrences
        counts = Counter(distribution)
        total = len(distribution)

        # Calculate probabilities
        probabilities = np.array([count / total for count in counts.values()])

        # Calculate Shannon entropy
        return float(scipy_entropy(probabilities, base=2))

    def calculate_max_entropy(self, n_unique_values: int) -> float:
        """
        Calculate maximum possible entropy for n unique values
        (uniform distribution)

        Args:
            n_unique_values: Number of unique values

        Returns:
            Maximum entropy value
        """
        if n_unique_values <= 1:
            return 0.0

        uniform_prob = 1.0 / n_unique_values
        uniform_dist = np.array([uniform_prob] * n_unique_values)
        return float(scipy_entropy(uniform_dist, base=2))

    def calculate_normalized_entropy(self, distribution: List[any]) -> float:
        """
        Calculate normalized entropy (0 to 1)

        1.0 = perfectly balanced (maximum entropy)
        0.0 = completely concentrated on one value

        Args:
            distribution: List of values

        Returns:
            Normalized entropy (0-1)
        """
        if not distribution:
            return 0.0

        entropy_val = self.calculate_entropy(distribution)
        n_unique = len(set(distribution))
        max_entropy = self.calculate_max_entropy(n_unique)

        if max_entropy == 0:
            return 0.0

        return entropy_val / max_entropy

    def extract_concepts(self, records: List[PatientRecord]) -> Dict[str, List]:
        """
        Extract all concept dimensions from patient records

        Args:
            records: List of patient records

        Returns:
            Dictionary mapping concept names to value lists
        """
        concepts = {
            'age_decades': [],
            'age_groups': [],
            'sex': [],
            'ethnicity': [],
            'location': [],
            'conditions': [],
            'condition_names': [],
            'condition_severities': [],
            'condition_statuses': [],
            'num_conditions': [],
            'medications': [],
            'num_medications': [],
            'lab_tests': [],
            'num_lab_tests': [],
            'abnormal_labs': [],
        }

        for record in records:
            # Age dimensions
            concepts['age_decades'].append(record.demographics.age // 10)
            if record.demographics.age < 30:
                age_group = 'young_adult'
            elif record.demographics.age < 50:
                age_group = 'middle_age'
            elif record.demographics.age < 65:
                age_group = 'older_adult'
            else:
                age_group = 'geriatric'
            concepts['age_groups'].append(age_group)

            # Demographics
            concepts['sex'].append(record.demographics.sex.value)
            concepts['ethnicity'].append(record.demographics.ethnicity.value)
            concepts['location'].append(record.demographics.location)

            # Conditions
            concepts['num_conditions'].append(len(record.conditions))
            for condition in record.conditions:
                concepts['conditions'].append(condition.icd10_code)
                concepts['condition_names'].append(condition.name)
                concepts['condition_severities'].append(condition.severity)
                concepts['condition_statuses'].append(condition.status)

            # Medications
            concepts['num_medications'].append(len(record.medications))
            for medication in record.medications:
                concepts['medications'].append(medication.name)

            # Lab tests
            concepts['num_lab_tests'].append(len(record.lab_results))
            for lab in record.lab_results:
                concepts['lab_tests'].append(lab.test_name)
                if lab.abnormal:
                    concepts['abnormal_labs'].append(lab.test_name)

        return concepts

    def calculate_concept_entropy_metrics(self, records: List[PatientRecord]) -> Dict[str, any]:
        """
        Calculate entropy metrics for all concept dimensions

        Args:
            records: List of patient records

        Returns:
            Dictionary containing entropy metrics and analysis
        """
        concepts = self.extract_concepts(records)

        metrics = {
            'entropy': {},
            'normalized_entropy': {},
            'unique_counts': {},
            'total_counts': {},
        }

        # Calculate entropy for each concept dimension
        for concept_name, values in concepts.items():
            if not values:
                continue

            # Raw entropy
            ent = self.calculate_entropy(values)
            metrics['entropy'][concept_name] = float(ent)

            # Normalized entropy
            norm_ent = self.calculate_normalized_entropy(values)
            metrics['normalized_entropy'][concept_name] = float(norm_ent)

            # Counts
            metrics['unique_counts'][concept_name] = len(set(values))
            metrics['total_counts'][concept_name] = len(values)

        # Summary statistics
        all_entropies = list(metrics['entropy'].values())
        all_normalized_entropies = list(metrics['normalized_entropy'].values())

        metrics['summary'] = {
            'mean_entropy': float(np.mean(all_entropies)),
            'std_entropy': float(np.std(all_entropies)),
            'mean_normalized_entropy': float(np.mean(all_normalized_entropies)),
            'std_normalized_entropy': float(np.std(all_normalized_entropies)),
            'min_normalized_entropy': float(np.min(all_normalized_entropies)),
            'max_normalized_entropy': float(np.max(all_normalized_entropies)),
        }

        # Identify low-entropy (unbalanced) dimensions
        threshold = 0.7  # Consider anything below 0.7 as "unbalanced"
        unbalanced = {
            name: metrics['normalized_entropy'][name]
            for name in metrics['normalized_entropy']
            if metrics['normalized_entropy'][name] < threshold
        }
        metrics['unbalanced_dimensions'] = unbalanced

        # Overall diversity score (average normalized entropy across key dimensions)
        key_dimensions = ['age_groups', 'sex', 'ethnicity', 'condition_names', 'condition_severities']
        key_entropies = [
            metrics['normalized_entropy'][dim]
            for dim in key_dimensions
            if dim in metrics['normalized_entropy']
        ]
        if key_entropies:
            metrics['overall_diversity_score'] = float(np.mean(key_entropies))
        else:
            metrics['overall_diversity_score'] = 0.0

        return metrics

    def compare_entropy(
        self,
        real_records: List[PatientRecord],
        synthetic_records: List[PatientRecord]
    ) -> Dict[str, any]:
        """
        Compare entropy between real and synthetic datasets

        Args:
            real_records: Real patient records
            synthetic_records: Synthetic patient records

        Returns:
            Comparison metrics
        """
        print("Calculating entropy for real records...")
        real_metrics = self.calculate_concept_entropy_metrics(real_records)

        print("Calculating entropy for synthetic records...")
        synthetic_metrics = self.calculate_concept_entropy_metrics(synthetic_records)

        # Compare normalized entropies
        comparison = {
            'real_diversity_score': real_metrics['overall_diversity_score'],
            'synthetic_diversity_score': synthetic_metrics['overall_diversity_score'],
            'diversity_score_diff': abs(
                real_metrics['overall_diversity_score'] -
                synthetic_metrics['overall_diversity_score']
            ),
            'entropy_differences': {},
        }

        # Calculate differences for each dimension
        for concept_name in real_metrics['normalized_entropy']:
            if concept_name in synthetic_metrics['normalized_entropy']:
                real_ent = real_metrics['normalized_entropy'][concept_name]
                synth_ent = synthetic_metrics['normalized_entropy'][concept_name]
                diff = abs(real_ent - synth_ent)
                comparison['entropy_differences'][concept_name] = {
                    'real': real_ent,
                    'synthetic': synth_ent,
                    'difference': diff,
                }

        comparison['real_metrics'] = real_metrics
        comparison['synthetic_metrics'] = synthetic_metrics

        return comparison
