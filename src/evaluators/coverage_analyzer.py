"""Coverage Analysis - Ensure all important dimensions are represented"""
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
from scipy.stats import entropy as scipy_entropy

from src.models import PatientRecord


class CoverageAnalyzer:
    """
    Analyze and report on dataset coverage across important dimensions

    Identifies gaps and underrepresented areas in the dataset.
    """

    def __init__(self):
        """Initialize coverage analyzer"""
        self.critical_dimensions = [
            'age_group',
            'sex',
            'ethnicity',
            'condition_category',
            'severity',
            'comorbidity_count',
        ]

    def analyze_coverage(self, records: List[PatientRecord]) -> Dict:
        """
        Comprehensive coverage analysis

        Args:
            records: List of patient records

        Returns:
            Coverage report with gaps and recommendations
        """
        # Extract features
        features = self._extract_features(records)

        # Analyze each dimension
        dimension_coverage = {}
        gaps = {}
        recommendations = []

        for dimension in self.critical_dimensions:
            if dimension not in features:
                continue

            coverage = self._analyze_dimension_coverage(dimension, features[dimension])
            dimension_coverage[dimension] = coverage

            # Identify gaps
            if coverage['has_gaps']:
                gaps[dimension] = coverage['gaps']
                recommendations.append({
                    'dimension': dimension,
                    'issue': coverage['issue'],
                    'action': coverage['recommendation'],
                })

        # Compute coverage scores
        coverage_scores = self._compute_coverage_scores(dimension_coverage)

        # Overall assessment
        overall_score = np.mean([score for score in coverage_scores.values()])

        report = {
            'n_records': len(records),
            'overall_coverage_score': float(overall_score),
            'dimension_coverage': dimension_coverage,
            'coverage_scores': coverage_scores,
            'gaps': gaps,
            'recommendations': recommendations,
            'strata_analysis': self._analyze_strata_coverage(records),
        }

        return report

    def identify_missing_combinations(
        self,
        records: List[PatientRecord],
        dimensions: List[str] = None
    ) -> List[Dict]:
        """
        Identify missing combinations of features

        Args:
            records: List of patient records
            dimensions: Dimensions to check (default: all critical)

        Returns:
            List of missing combinations
        """
        if dimensions is None:
            dimensions = ['age_group', 'sex', 'ethnicity']

        features = self._extract_features(records)

        # Get all unique values per dimension
        unique_values = {}
        for dim in dimensions:
            if dim in features:
                unique_values[dim] = sorted(set(features[dim]))

        # Generate all possible combinations
        from itertools import product
        all_combinations = list(product(*[unique_values[dim] for dim in dimensions]))

        # Find which combinations exist in the data
        existing_combinations = set()
        for i in range(len(records)):
            combo = tuple(features[dim][i] for dim in dimensions)
            existing_combinations.add(combo)

        # Find missing combinations
        missing = []
        for combo in all_combinations:
            if combo not in existing_combinations:
                missing_dict = {dimensions[i]: combo[i] for i in range(len(dimensions))}
                missing.append(missing_dict)

        return missing

    def compute_coverage_gaps(
        self,
        records: List[PatientRecord],
        target_distribution: Dict[str, Dict] = None
    ) -> Dict:
        """
        Compare actual distribution to target distribution

        Args:
            records: List of patient records
            target_distribution: Expected distribution per dimension
                e.g., {'sex': {'male': 0.48, 'female': 0.50, 'other': 0.02}}

        Returns:
            Gap analysis report
        """
        if target_distribution is None:
            # Use uniform distribution as default
            target_distribution = self._get_default_target_distribution()

        features = self._extract_features(records)
        gaps = {}

        for dimension, target_dist in target_distribution.items():
            if dimension not in features:
                continue

            # Compute actual distribution
            counts = Counter(features[dimension])
            total = len(features[dimension])
            actual_dist = {k: counts[k] / total for k in counts}

            # Compute gaps
            dimension_gaps = {}
            for value, target_proportion in target_dist.items():
                actual_proportion = actual_dist.get(value, 0.0)
                gap = target_proportion - actual_proportion

                if abs(gap) > 0.05:  # Significant gap (>5%)
                    dimension_gaps[value] = {
                        'target': target_proportion,
                        'actual': actual_proportion,
                        'gap': gap,
                        'status': 'underrepresented' if gap > 0 else 'overrepresented',
                    }

            if dimension_gaps:
                gaps[dimension] = dimension_gaps

        return gaps

    def get_generation_recommendations(
        self,
        records: List[PatientRecord],
        target_size: int
    ) -> List[Dict]:
        """
        Recommend what types of records to generate next to improve coverage

        Args:
            records: Existing patient records
            target_size: Target total dataset size

        Returns:
            List of generation recommendations with constraints
        """
        n_current = len(records)
        n_to_generate = target_size - n_current

        if n_to_generate <= 0:
            return []

        # Analyze current coverage
        coverage_report = self.analyze_coverage(records)
        gaps = coverage_report['gaps']

        recommendations = []

        # Prioritize filling gaps
        for dimension, gap_info in gaps.items():
            if isinstance(gap_info, dict) and 'underrepresented' in str(gap_info):
                # Find underrepresented values
                for value, details in gap_info.items():
                    if isinstance(details, dict) and details.get('status') == 'underrepresented':
                        n_needed = int(details['gap'] * target_size)
                        if n_needed > 0:
                            recommendations.append({
                                'dimension': dimension,
                                'value': value,
                                'n_records': n_needed,
                                'constraints': self._create_constraints(dimension, value),
                                'priority': 'high',
                            })

        # Fill missing combinations
        missing_combos = self.identify_missing_combinations(records)
        for combo in missing_combos[:n_to_generate // 2]:  # Don't generate too many
            recommendations.append({
                'type': 'missing_combination',
                'constraints': combo,
                'n_records': 1,
                'priority': 'medium',
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 3))

        return recommendations[:n_to_generate]

    # Helper methods

    def _extract_features(self, records: List[PatientRecord]) -> Dict[str, List]:
        """Extract all relevant features for coverage analysis"""
        features = defaultdict(list)

        for record in records:
            # Age groups
            age = record.demographics.age
            if age < 18:
                age_group = 'pediatric'
            elif age < 30:
                age_group = 'young_adult'
            elif age < 50:
                age_group = 'middle_age'
            elif age < 65:
                age_group = 'older_adult'
            else:
                age_group = 'geriatric'
            features['age_group'].append(age_group)

            # Demographics
            features['sex'].append(record.demographics.sex.value)
            features['ethnicity'].append(record.demographics.ethnicity.value)

            # Conditions
            features['comorbidity_count'].append(str(len(record.conditions)))

            if record.conditions:
                # Primary condition
                primary = record.conditions[0]
                features['primary_condition'].append(primary.name)
                features['severity'].append(primary.severity)

                # Condition categories
                condition_category = self._categorize_condition(primary.name)
                features['condition_category'].append(condition_category)
            else:
                features['primary_condition'].append('none')
                features['severity'].append('none')
                features['condition_category'].append('none')

        return dict(features)

    def _categorize_condition(self, condition_name: str) -> str:
        """Categorize condition into broad category"""
        condition_name = condition_name.lower()

        if 'diabetes' in condition_name:
            return 'endocrine'
        elif 'hypertension' in condition_name or 'coronary' in condition_name:
            return 'cardiovascular'
        elif 'asthma' in condition_name or 'copd' in condition_name:
            return 'respiratory'
        elif 'kidney' in condition_name or 'renal' in condition_name:
            return 'renal'
        else:
            return 'other'

    def _analyze_dimension_coverage(self, dimension: str, values: List) -> Dict:
        """Analyze coverage for a single dimension"""
        counts = Counter(values)
        total = len(values)

        # Compute entropy (higher = better balance)
        probs = np.array([count / total for count in counts.values()])
        ent = scipy_entropy(probs, base=2)

        # Maximum possible entropy for this dimension
        n_unique = len(counts)
        max_ent = np.log2(n_unique) if n_unique > 1 else 1.0

        # Normalized entropy (0-1)
        norm_ent = ent / max_ent if max_ent > 0 else 0.0

        # Check for gaps
        has_gaps = False
        gaps = []
        issue = None
        recommendation = None

        # If entropy is low, some values are underrepresented
        if norm_ent < 0.7:
            has_gaps = True
            # Find underrepresented values
            avg_count = total / n_unique
            for value, count in counts.items():
                if count < avg_count * 0.5:  # Less than 50% of average
                    gaps.append({
                        'value': value,
                        'count': count,
                        'expected': avg_count,
                        'deficit': avg_count - count,
                    })

            issue = f"Unbalanced distribution (entropy: {norm_ent:.2f})"
            recommendation = f"Generate more records for underrepresented {dimension} values"

        # Check for missing expected values
        expected_values = self._get_expected_values(dimension)
        missing_values = set(expected_values) - set(counts.keys())

        if missing_values:
            has_gaps = True
            if issue is None:
                issue = f"Missing values: {', '.join(missing_values)}"
            if recommendation is None:
                recommendation = f"Add records with {dimension} in {missing_values}"

        return {
            'n_unique': n_unique,
            'entropy': float(ent),
            'normalized_entropy': float(norm_ent),
            'distribution': dict(counts),
            'has_gaps': has_gaps,
            'gaps': gaps,
            'missing_values': list(missing_values),
            'issue': issue,
            'recommendation': recommendation,
        }

    def _get_expected_values(self, dimension: str) -> List[str]:
        """Get expected values for a dimension"""
        expected = {
            'sex': ['male', 'female', 'other'],
            'ethnicity': ['caucasian', 'african_american', 'hispanic', 'asian', 'other'],
            'age_group': ['pediatric', 'young_adult', 'middle_age', 'older_adult', 'geriatric'],
            'severity': ['mild', 'moderate', 'severe'],
            'condition_category': ['endocrine', 'cardiovascular', 'respiratory', 'renal', 'other'],
        }
        return expected.get(dimension, [])

    def _compute_coverage_scores(self, dimension_coverage: Dict) -> Dict[str, float]:
        """Compute coverage scores per dimension"""
        scores = {}
        for dimension, coverage in dimension_coverage.items():
            # Score based on normalized entropy and missing values
            ent_score = coverage['normalized_entropy']
            missing_penalty = len(coverage.get('missing_values', [])) * 0.1
            score = max(0.0, ent_score - missing_penalty)
            scores[dimension] = float(score)
        return scores

    def _analyze_strata_coverage(self, records: List[PatientRecord]) -> Dict:
        """Analyze coverage across combined dimensions (strata)"""
        # Key strata: age_group x sex x ethnicity
        strata = defaultdict(int)

        for record in records:
            age = record.demographics.age
            if age < 30:
                age_group = 'young'
            elif age < 50:
                age_group = 'middle'
            elif age < 65:
                age_group = 'older'
            else:
                age_group = 'geriatric'

            stratum = f"{age_group}|{record.demographics.sex.value}|{record.demographics.ethnicity.value}"
            strata[stratum] += 1

        # Compute statistics
        counts = list(strata.values())
        return {
            'n_strata': len(strata),
            'avg_per_stratum': float(np.mean(counts)),
            'std_per_stratum': float(np.std(counts)),
            'min_per_stratum': int(np.min(counts)),
            'max_per_stratum': int(np.max(counts)),
            'strata_with_gaps': sum(1 for c in counts if c < np.mean(counts) * 0.5),
        }

    def _get_default_target_distribution(self) -> Dict[str, Dict]:
        """Get default target distributions based on US population"""
        return {
            'sex': {
                'male': 0.48,
                'female': 0.50,
                'other': 0.02,
            },
            'ethnicity': {
                'caucasian': 0.40,
                'african_american': 0.25,
                'hispanic': 0.20,
                'asian': 0.12,
                'other': 0.03,
            },
            'age_group': {
                'young_adult': 0.20,
                'middle_age': 0.30,
                'older_adult': 0.30,
                'geriatric': 0.20,
            },
        }

    def _create_constraints(self, dimension: str, value: str) -> Dict:
        """Create generation constraints for a specific dimension/value"""
        if dimension == 'age_group':
            age_ranges = {
                'pediatric': (0, 17),
                'young_adult': (18, 29),
                'middle_age': (30, 49),
                'older_adult': (50, 64),
                'geriatric': (65, 85),
            }
            age_min, age_max = age_ranges.get(value, (30, 50))
            return {'age': np.random.randint(age_min, age_max + 1)}

        elif dimension == 'sex':
            return {'sex': value}

        elif dimension == 'ethnicity':
            return {'ethnicity': value}

        return {}
