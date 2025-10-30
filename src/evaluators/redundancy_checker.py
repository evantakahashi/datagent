"""Redundancy Detection and Elimination"""
import numpy as np
from typing import List, Dict, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import hashlib

from src.models import PatientRecord


class RedundancyChecker:
    """Detect and eliminate redundant patient records"""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Args:
            similarity_threshold: Cosine similarity above which records are considered redundant
        """
        self.similarity_threshold = similarity_threshold

    def compute_feature_hash(self, record: PatientRecord) -> str:
        """
        Compute a hash of key features for exact duplicate detection

        Args:
            record: Patient record

        Returns:
            Hash string
        """
        # Create a string representation of key features
        features = [
            str(record.demographics.age),
            record.demographics.sex.value,
            record.demographics.ethnicity.value,
            str(sorted([c.name for c in record.conditions])),
            str(sorted([m.name for m in record.medications])),
            str(len(record.lab_results)),
        ]
        feature_str = "|".join(features)
        return hashlib.md5(feature_str.encode()).hexdigest()

    def find_exact_duplicates(self, records: List[PatientRecord]) -> Dict[str, List[int]]:
        """
        Find exact duplicate records based on feature hash

        Args:
            records: List of patient records

        Returns:
            Dictionary mapping hash to list of record indices
        """
        hash_to_indices = defaultdict(list)

        for idx, record in enumerate(records):
            record_hash = self.compute_feature_hash(record)
            hash_to_indices[record_hash].append(idx)

        # Filter to only duplicates
        duplicates = {k: v for k, v in hash_to_indices.items() if len(v) > 1}

        return duplicates

    def find_near_duplicates_embedding(
        self,
        records: List[PatientRecord],
        embeddings: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicate records using embedding similarity

        Args:
            records: List of patient records
            embeddings: Record embeddings (n_records, embedding_dim)

        Returns:
            List of (idx1, idx2, similarity) tuples for near-duplicates
        """
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Find pairs above threshold (excluding self-similarity)
        near_duplicates = []

        n = len(records)
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle
                sim = similarities[i, j]
                if sim >= self.similarity_threshold:
                    near_duplicates.append((i, j, sim))

        # Sort by similarity (highest first)
        near_duplicates.sort(key=lambda x: x[2], reverse=True)

        return near_duplicates

    def find_near_duplicates_features(
        self,
        records: List[PatientRecord]
    ) -> List[Tuple[int, int, float]]:
        """
        Find near-duplicate records using feature-based similarity

        This is faster than embeddings but less semantically aware.

        Args:
            records: List of patient records

        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        near_duplicates = []

        n = len(records)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_feature_similarity(records[i], records[j])
                if sim >= self.similarity_threshold:
                    near_duplicates.append((i, j, sim))

        near_duplicates.sort(key=lambda x: x[2], reverse=True)
        return near_duplicates

    def _compute_feature_similarity(
        self,
        record1: PatientRecord,
        record2: PatientRecord
    ) -> float:
        """
        Compute similarity between two records based on discrete features

        Args:
            record1: First patient record
            record2: Second patient record

        Returns:
            Similarity score (0-1)
        """
        similarities = []

        # Age similarity (normalized by max possible difference)
        age_diff = abs(record1.demographics.age - record2.demographics.age)
        age_sim = 1 - (age_diff / 120.0)  # Max age is 120
        similarities.append(age_sim)

        # Sex match
        sex_sim = 1.0 if record1.demographics.sex == record2.demographics.sex else 0.0
        similarities.append(sex_sim)

        # Ethnicity match
        eth_sim = 1.0 if record1.demographics.ethnicity == record2.demographics.ethnicity else 0.0
        similarities.append(eth_sim)

        # Condition overlap (Jaccard similarity)
        cond1 = set([c.name for c in record1.conditions])
        cond2 = set([c.name for c in record2.conditions])
        if cond1 or cond2:
            cond_sim = len(cond1 & cond2) / len(cond1 | cond2)
        else:
            cond_sim = 1.0
        similarities.append(cond_sim * 2)  # Weight conditions more heavily

        # Medication overlap
        med1 = set([m.name for m in record1.medications])
        med2 = set([m.name for m in record2.medications])
        if med1 or med2:
            med_sim = len(med1 & med2) / len(med1 | med2)
        else:
            med_sim = 1.0
        similarities.append(med_sim)

        # Lab test overlap
        lab1 = set([l.test_name for l in record1.lab_results])
        lab2 = set([l.test_name for l in record2.lab_results])
        if lab1 or lab2:
            lab_sim = len(lab1 & lab2) / len(lab1 | lab2)
        else:
            lab_sim = 1.0
        similarities.append(lab_sim)

        # Weighted average
        return np.mean(similarities)

    def deduplicate(
        self,
        records: List[PatientRecord],
        embeddings: np.ndarray = None,
        method: str = "embedding"
    ) -> Tuple[List[PatientRecord], List[int], Dict]:
        """
        Remove redundant records from a dataset

        Args:
            records: List of patient records
            embeddings: Optional embeddings for similarity computation
            method: "embedding" (more accurate) or "feature" (faster)

        Returns:
            Tuple of (deduplicated_records, kept_indices, stats)
        """
        n_original = len(records)

        # Step 1: Remove exact duplicates
        exact_dupes = self.find_exact_duplicates(records)
        exact_dupe_indices = set()
        for indices in exact_dupes.values():
            # Keep first, mark rest as duplicates
            exact_dupe_indices.update(indices[1:])

        print(f"Found {len(exact_dupe_indices)} exact duplicates")

        # Step 2: Find near-duplicates
        if method == "embedding" and embeddings is not None:
            near_dupes = self.find_near_duplicates_embedding(records, embeddings)
        else:
            near_dupes = self.find_near_duplicates_features(records)

        print(f"Found {len(near_dupes)} near-duplicate pairs")

        # Step 3: Greedy selection - keep one from each near-duplicate pair
        removed_indices = set(exact_dupe_indices)

        for idx1, idx2, sim in near_dupes:
            # If both are still in the dataset, remove one
            if idx1 not in removed_indices and idx2 not in removed_indices:
                # Remove the one with fewer unique conditions (less informative)
                if len(records[idx1].conditions) >= len(records[idx2].conditions):
                    removed_indices.add(idx2)
                else:
                    removed_indices.add(idx1)

        # Step 4: Create deduplicated list
        kept_indices = [i for i in range(n_original) if i not in removed_indices]
        deduplicated_records = [records[i] for i in kept_indices]

        stats = {
            'n_original': n_original,
            'n_deduplicated': len(deduplicated_records),
            'n_removed': len(removed_indices),
            'n_exact_duplicates': len(exact_dupe_indices),
            'n_near_duplicates': len(removed_indices) - len(exact_dupe_indices),
            'reduction_rate': len(removed_indices) / n_original,
        }

        return deduplicated_records, kept_indices, stats

    def analyze_redundancy(
        self,
        records: List[PatientRecord],
        embeddings: np.ndarray = None
    ) -> Dict:
        """
        Analyze redundancy in a dataset without removing records

        Args:
            records: List of patient records
            embeddings: Optional embeddings

        Returns:
            Redundancy analysis report
        """
        exact_dupes = self.find_exact_duplicates(records)

        if embeddings is not None:
            near_dupes = self.find_near_duplicates_embedding(records, embeddings)
        else:
            near_dupes = self.find_near_duplicates_features(records)

        # Compute average pairwise similarity
        if embeddings is not None:
            similarities = cosine_similarity(embeddings)
            # Exclude diagonal
            mask = np.ones_like(similarities, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_similarity = similarities[mask].mean()
            min_similarity = similarities[mask].min()
            max_similarity = similarities[mask].max()
        else:
            avg_similarity = None
            min_similarity = None
            max_similarity = None

        return {
            'n_records': len(records),
            'n_exact_duplicates': sum(len(v) - 1 for v in exact_dupes.values()),
            'n_near_duplicate_pairs': len(near_dupes),
            'avg_pairwise_similarity': float(avg_similarity) if avg_similarity is not None else None,
            'min_pairwise_similarity': float(min_similarity) if min_similarity is not None else None,
            'max_pairwise_similarity': float(max_similarity) if max_similarity is not None else None,
            'near_duplicate_pairs': [(i, j, float(s)) for i, j, s in near_dupes[:10]],  # Top 10
        }
