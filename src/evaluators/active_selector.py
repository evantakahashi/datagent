"""Active Selection for Representative, Non-Redundant Datasets"""
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
import random

from src.models import PatientRecord


class ActiveDataSelector:
    """
    Select the most representative, non-redundant subset of patient records

    Strategies:
    1. Greedy diversity maximization
    2. Coverage-based selection
    3. Cluster-based sampling
    4. Entropy-maximizing selection
    """

    def __init__(self, redundancy_threshold: float = 0.90):
        """
        Args:
            redundancy_threshold: Similarity threshold for redundancy detection
        """
        self.redundancy_threshold = redundancy_threshold

    def greedy_diversity_selection(
        self,
        records: List[PatientRecord],
        embeddings: np.ndarray,
        target_size: int,
        diversity_weight: float = 0.7
    ) -> Tuple[List[PatientRecord], List[int], Dict]:
        """
        Greedily select records that maximize diversity

        Algorithm:
        1. Start with the record closest to the centroid (most representative)
        2. Iteratively add the record that is most dissimilar to already selected records
        3. Balance between diversity and coverage

        Args:
            records: List of patient records
            embeddings: Record embeddings
            target_size: Desired number of records
            diversity_weight: Weight for diversity vs representativeness (0-1)

        Returns:
            (selected_records, selected_indices, stats)
        """
        n = len(records)
        target_size = min(target_size, n)

        selected_indices = []
        remaining_indices = set(range(n))

        # Step 1: Select most representative record (closest to centroid)
        centroid = embeddings.mean(axis=0)
        distances_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
        first_idx = np.argmin(distances_to_centroid)

        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        print(f"Selecting {target_size} records via greedy diversity maximization...")

        # Step 2: Iteratively select most diverse records
        while len(selected_indices) < target_size and remaining_indices:
            best_idx = None
            best_score = -float('inf')

            for idx in remaining_indices:
                # Compute minimum similarity to already selected records
                similarities = cosine_similarity(
                    embeddings[idx].reshape(1, -1),
                    embeddings[selected_indices]
                )
                min_similarity = similarities.min()

                # Compute distance to centroid (representativeness)
                distance_to_centroid = distances_to_centroid[idx]

                # Combined score: maximize diversity, minimize distance to centroid
                diversity_score = 1 - min_similarity
                representativeness_score = 1 - (distance_to_centroid / distances_to_centroid.max())

                score = (diversity_weight * diversity_score +
                        (1 - diversity_weight) * representativeness_score)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        selected_records = [records[i] for i in selected_indices]

        # Compute statistics
        selected_embeddings = embeddings[selected_indices]
        avg_min_distance = self._compute_avg_min_distance(selected_embeddings)

        stats = {
            'method': 'greedy_diversity',
            'target_size': target_size,
            'actual_size': len(selected_records),
            'avg_min_pairwise_distance': float(avg_min_distance),
            'diversity_weight': diversity_weight,
        }

        return selected_records, selected_indices, stats

    def coverage_based_selection(
        self,
        records: List[PatientRecord],
        embeddings: np.ndarray,
        target_size: int,
        coverage_dimensions: List[str] = None
    ) -> Tuple[List[PatientRecord], List[int], Dict]:
        """
        Select records to maximize coverage across key dimensions

        Ensures representation across:
        - Age groups
        - Sex
        - Ethnicity
        - Medical conditions
        - Severity levels

        Args:
            records: List of patient records
            embeddings: Record embeddings
            target_size: Desired number of records
            coverage_dimensions: Specific dimensions to ensure coverage

        Returns:
            (selected_records, selected_indices, stats)
        """
        if coverage_dimensions is None:
            coverage_dimensions = ['age_group', 'sex', 'ethnicity', 'primary_condition']

        # Extract feature values for each dimension
        feature_map = self._extract_coverage_features(records)

        # Count representation needed per dimension
        dimension_counts = {}
        for dim in coverage_dimensions:
            if dim in feature_map:
                dimension_counts[dim] = Counter(feature_map[dim])

        # Calculate how many records we need per stratum
        strata = self._create_strata(records, coverage_dimensions)

        # Proportional stratified sampling
        selected_indices = []

        # Ensure at least one record per unique stratum
        stratum_to_indices = {}
        for idx, record in enumerate(records):
            stratum_key = self._get_stratum_key(record, coverage_dimensions)
            if stratum_key not in stratum_to_indices:
                stratum_to_indices[stratum_key] = []
            stratum_to_indices[stratum_key].append(idx)

        print(f"Found {len(stratum_to_indices)} unique strata")

        # First pass: ensure at least one record per stratum
        for stratum_key, indices in stratum_to_indices.items():
            if len(selected_indices) >= target_size:
                break
            # Select the most central record from this stratum
            stratum_embeddings = embeddings[indices]
            stratum_centroid = stratum_embeddings.mean(axis=0)
            distances = np.linalg.norm(stratum_embeddings - stratum_centroid, axis=1)
            best_idx_in_stratum = indices[np.argmin(distances)]
            selected_indices.append(best_idx_in_stratum)

        # Second pass: fill remaining slots proportionally
        remaining_slots = target_size - len(selected_indices)
        if remaining_slots > 0:
            # Allocate remaining slots proportionally to stratum size
            stratum_sizes = {k: len(v) for k, v in stratum_to_indices.items()}
            total_size = sum(stratum_sizes.values())

            for stratum_key, indices in stratum_to_indices.items():
                # How many more from this stratum?
                proportion = stratum_sizes[stratum_key] / total_size
                n_additional = int(proportion * remaining_slots)

                # Add diverse records from this stratum
                available = [i for i in indices if i not in selected_indices]
                if available:
                    # Select diverse ones
                    n_to_add = min(n_additional, len(available))
                    if n_to_add > 0:
                        # Use embeddings to select diverse records
                        if len(selected_indices) > 0:
                            for _ in range(n_to_add):
                                if not available:
                                    break
                                # Find most dissimilar to already selected
                                sims = cosine_similarity(
                                    embeddings[available],
                                    embeddings[selected_indices]
                                )
                                min_sims = sims.min(axis=1)
                                best_idx_pos = np.argmin(min_sims)  # Most dissimilar
                                selected_indices.append(available[best_idx_pos])
                                available.pop(best_idx_pos)
                        else:
                            selected_indices.extend(available[:n_to_add])

        selected_records = [records[i] for i in selected_indices[:target_size]]
        selected_indices = selected_indices[:target_size]

        stats = {
            'method': 'coverage_based',
            'target_size': target_size,
            'actual_size': len(selected_records),
            'n_strata': len(stratum_to_indices),
            'strata_covered': len(set(self._get_stratum_key(r, coverage_dimensions) for r in selected_records)),
        }

        return selected_records, selected_indices, stats

    def cluster_based_selection(
        self,
        records: List[PatientRecord],
        embeddings: np.ndarray,
        target_size: int,
        n_clusters: int = None
    ) -> Tuple[List[PatientRecord], List[int], Dict]:
        """
        Select records using clustering to ensure diversity

        1. Cluster records into K groups
        2. Sample proportionally from each cluster
        3. Select most representative records from each cluster

        Args:
            records: List of patient records
            embeddings: Record embeddings
            target_size: Desired number of records
            n_clusters: Number of clusters (default: sqrt(n))

        Returns:
            (selected_records, selected_indices, stats)
        """
        n = len(records)

        if n_clusters is None:
            n_clusters = max(int(np.sqrt(n)), target_size // 3)

        n_clusters = min(n_clusters, n, target_size)

        print(f"Clustering {n} records into {n_clusters} clusters...")

        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Select records from each cluster
        selected_indices = []

        # Calculate how many to sample from each cluster (proportional)
        cluster_counts = Counter(cluster_labels)
        samples_per_cluster = {}

        for cluster_id in range(n_clusters):
            proportion = cluster_counts[cluster_id] / n
            n_samples = max(1, int(proportion * target_size))
            samples_per_cluster[cluster_id] = n_samples

        # Adjust to match target size exactly
        total_allocated = sum(samples_per_cluster.values())
        if total_allocated < target_size:
            # Add remaining to largest clusters
            diff = target_size - total_allocated
            largest_clusters = sorted(cluster_counts.keys(), key=lambda x: cluster_counts[x], reverse=True)
            for i in range(diff):
                samples_per_cluster[largest_clusters[i % len(largest_clusters)]] += 1
        elif total_allocated > target_size:
            # Remove from largest clusters
            diff = total_allocated - target_size
            largest_clusters = sorted(cluster_counts.keys(), key=lambda x: samples_per_cluster[x], reverse=True)
            for i in range(diff):
                cluster_id = largest_clusters[i % len(largest_clusters)]
                if samples_per_cluster[cluster_id] > 1:
                    samples_per_cluster[cluster_id] -= 1

        # Select records from each cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            n_samples = samples_per_cluster[cluster_id]

            if len(cluster_indices) <= n_samples:
                # Take all
                selected_indices.extend(cluster_indices.tolist())
            else:
                # Select most representative (closest to cluster center)
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_embeddings = embeddings[cluster_indices]
                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                closest_indices = cluster_indices[np.argsort(distances)[:n_samples]]
                selected_indices.extend(closest_indices.tolist())

        selected_records = [records[i] for i in selected_indices]

        stats = {
            'method': 'cluster_based',
            'target_size': target_size,
            'actual_size': len(selected_records),
            'n_clusters': n_clusters,
            'samples_per_cluster': samples_per_cluster,
        }

        return selected_records, selected_indices, stats

    def entropy_maximizing_selection(
        self,
        records: List[PatientRecord],
        target_size: int
    ) -> Tuple[List[PatientRecord], List[int], Dict]:
        """
        Select records to maximize entropy across all dimensions

        Ensures maximum diversity in categorical features.

        Args:
            records: List of patient records
            target_size: Desired number of records

        Returns:
            (selected_records, selected_indices, stats)
        """
        # Extract all categorical features
        features = self._extract_all_categorical_features(records)

        selected_indices = []
        remaining_indices = set(range(len(records)))

        # Greedy selection to maximize entropy
        while len(selected_indices) < target_size and remaining_indices:
            best_idx = None
            best_entropy = -float('inf')

            for idx in remaining_indices:
                # Compute entropy if we add this record
                trial_indices = selected_indices + [idx]
                entropy = self._compute_combined_entropy(records, trial_indices, features)

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        selected_records = [records[i] for i in selected_indices]

        final_entropy = self._compute_combined_entropy(records, selected_indices, features)

        stats = {
            'method': 'entropy_maximizing',
            'target_size': target_size,
            'actual_size': len(selected_records),
            'final_entropy': float(final_entropy),
        }

        return selected_records, selected_indices, stats

    # Helper methods

    def _compute_avg_min_distance(self, embeddings: np.ndarray) -> float:
        """Compute average minimum pairwise distance"""
        if len(embeddings) < 2:
            return 0.0

        similarities = cosine_similarity(embeddings)
        np.fill_diagonal(similarities, -np.inf)  # Ignore self-similarity

        # For each record, find its maximum similarity to another
        max_sims = similarities.max(axis=1)
        avg_min_distance = (1 - max_sims).mean()

        return avg_min_distance

    def _extract_coverage_features(self, records: List[PatientRecord]) -> Dict[str, List]:
        """Extract features for coverage analysis"""
        features = {
            'age_group': [],
            'sex': [],
            'ethnicity': [],
            'primary_condition': [],
            'severity': [],
        }

        for record in records:
            # Age groups
            age = record.demographics.age
            if age < 30:
                age_group = 'young_adult'
            elif age < 50:
                age_group = 'middle_age'
            elif age < 65:
                age_group = 'older_adult'
            else:
                age_group = 'geriatric'
            features['age_group'].append(age_group)

            features['sex'].append(record.demographics.sex.value)
            features['ethnicity'].append(record.demographics.ethnicity.value)

            if record.conditions:
                features['primary_condition'].append(record.conditions[0].name)
                features['severity'].append(record.conditions[0].severity)
            else:
                features['primary_condition'].append('none')
                features['severity'].append('none')

        return features

    def _create_strata(self, records: List[PatientRecord], dimensions: List[str]) -> Dict:
        """Create strata based on specified dimensions"""
        strata = {}
        for record in records:
            key = self._get_stratum_key(record, dimensions)
            if key not in strata:
                strata[key] = []
            strata[key].append(record)
        return strata

    def _get_stratum_key(self, record: PatientRecord, dimensions: List[str]) -> str:
        """Get stratum key for a record"""
        key_parts = []

        if 'age_group' in dimensions:
            age = record.demographics.age
            if age < 30:
                key_parts.append('young')
            elif age < 50:
                key_parts.append('middle')
            elif age < 65:
                key_parts.append('older')
            else:
                key_parts.append('geriatric')

        if 'sex' in dimensions:
            key_parts.append(record.demographics.sex.value)

        if 'ethnicity' in dimensions:
            key_parts.append(record.demographics.ethnicity.value)

        if 'primary_condition' in dimensions and record.conditions:
            key_parts.append(record.conditions[0].name)

        return '|'.join(key_parts)

    def _extract_all_categorical_features(self, records: List[PatientRecord]) -> Dict:
        """Extract all categorical features for entropy computation"""
        return self._extract_coverage_features(records)

    def _compute_combined_entropy(
        self,
        records: List[PatientRecord],
        indices: List[int],
        features: Dict[str, List]
    ) -> float:
        """Compute combined entropy across all features"""
        from scipy.stats import entropy as scipy_entropy

        if not indices:
            return 0.0

        entropies = []

        for feature_name, all_values in features.items():
            # Get values for selected indices
            selected_values = [all_values[i] for i in indices]

            # Count
            counts = Counter(selected_values)
            total = len(selected_values)
            probs = np.array([counts[v] / total for v in counts])

            ent = scipy_entropy(probs, base=2)
            entropies.append(ent)

        return np.mean(entropies)
