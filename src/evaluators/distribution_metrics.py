"""Distribution-based diversity metrics (KL-divergence, FID)"""
import numpy as np
from typing import List, Dict, Tuple
from scipy import linalg
from scipy.stats import entropy
from collections import Counter

from src.models import PatientRecord


class DistributionMetrics:
    """Compute distribution-based diversity metrics"""

    @staticmethod
    def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute KL divergence between two distributions

        Args:
            p: True distribution
            q: Approximating distribution
            epsilon: Small value to avoid log(0)

        Returns:
            KL divergence value
        """
        # Ensure distributions sum to 1
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        p = p / p.sum()
        q = q / q.sum()

        # Add epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon

        # Renormalize
        p = p / p.sum()
        q = q / q.sum()

        return entropy(p, q)

    @staticmethod
    def compute_fid(real_embeddings: np.ndarray, synthetic_embeddings: np.ndarray) -> float:
        """
        Compute Fréchet Inception Distance (FID) between real and synthetic embeddings

        FID measures the distance between two multivariate Gaussians fitted to
        the real and synthetic data in embedding space.

        Args:
            real_embeddings: Real data embeddings (n_real, embedding_dim)
            synthetic_embeddings: Synthetic data embeddings (n_synthetic, embedding_dim)

        Returns:
            FID score (lower is better, 0 = identical distributions)
        """
        # Compute mean and covariance
        mu_real = np.mean(real_embeddings, axis=0)
        mu_synthetic = np.mean(synthetic_embeddings, axis=0)

        sigma_real = np.cov(real_embeddings, rowvar=False)
        sigma_synthetic = np.cov(synthetic_embeddings, rowvar=False)

        # Compute squared distance between means
        diff = mu_real - mu_synthetic
        mean_dist = np.dot(diff, diff)

        # Compute sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_synthetic, disp=False)

        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Compute FID
        fid = mean_dist + np.trace(sigma_real + sigma_synthetic - 2 * covmean)

        return float(fid)

    @staticmethod
    def extract_categorical_features(records: List[PatientRecord]) -> Dict[str, List]:
        """Extract categorical features from patient records"""
        features = {
            'age_groups': [],
            'sex': [],
            'ethnicity': [],
            'conditions': [],
            'num_conditions': [],
            'severities': [],
        }

        for record in records:
            # Age groups
            age = record.demographics.age
            if age < 30:
                age_group = '0-29'
            elif age < 50:
                age_group = '30-49'
            elif age < 65:
                age_group = '50-64'
            else:
                age_group = '65+'
            features['age_groups'].append(age_group)

            # Demographics
            features['sex'].append(record.demographics.sex.value)
            features['ethnicity'].append(record.demographics.ethnicity.value)

            # Conditions
            features['num_conditions'].append(len(record.conditions))
            for condition in record.conditions:
                features['conditions'].append(condition.name)
                features['severities'].append(condition.severity)

        return features

    def compute_categorical_kl_divergences(
        self,
        real_records: List[PatientRecord],
        synthetic_records: List[PatientRecord]
    ) -> Dict[str, float]:
        """Compute KL divergence for all categorical features"""

        real_features = self.extract_categorical_features(real_records)
        synthetic_features = self.extract_categorical_features(synthetic_records)

        kl_divergences = {}

        for feature_name in ['age_groups', 'sex', 'ethnicity', 'num_conditions', 'severities']:
            # Get value counts
            real_counts = Counter(real_features[feature_name])
            synthetic_counts = Counter(synthetic_features[feature_name])

            # Get all unique values
            all_values = sorted(set(list(real_counts.keys()) + list(synthetic_counts.keys())))

            # Create probability distributions
            total_real = sum(real_counts.values())
            total_synthetic = sum(synthetic_counts.values())

            p = np.array([real_counts.get(v, 0) / total_real for v in all_values])
            q = np.array([synthetic_counts.get(v, 0) / total_synthetic for v in all_values])

            # Compute KL divergence
            kl = self.compute_kl_divergence(p, q)
            kl_divergences[f'kl_div_{feature_name}'] = float(kl)

        # Average KL divergence
        kl_divergences['kl_div_average'] = float(np.mean(list(kl_divergences.values())))

        return kl_divergences

    def compute_distribution_metrics(
        self,
        real_records: List[PatientRecord],
        synthetic_records: List[PatientRecord],
        real_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all distribution-based metrics

        Args:
            real_records: List of real patient records
            synthetic_records: List of synthetic patient records
            real_embeddings: Real record embeddings
            synthetic_embeddings: Synthetic record embeddings

        Returns:
            Dictionary of distribution metrics
        """
        metrics = {}

        # FID score
        print("Computing FID score...")
        fid_score = self.compute_fid(real_embeddings, synthetic_embeddings)
        metrics['fid_score'] = fid_score

        # KL divergences for categorical features
        print("Computing KL divergences...")
        kl_divergences = self.compute_categorical_kl_divergences(real_records, synthetic_records)
        metrics.update(kl_divergences)

        # Continuous feature statistics (vitals, labs)
        print("Computing continuous feature statistics...")
        continuous_metrics = self._compute_continuous_metrics(real_records, synthetic_records)
        metrics.update(continuous_metrics)

        return metrics

    def _compute_continuous_metrics(
        self,
        real_records: List[PatientRecord],
        synthetic_records: List[PatientRecord]
    ) -> Dict[str, float]:
        """Compare distributions of continuous features (vitals)"""
        metrics = {}

        vital_names = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature',
                      'respiratory_rate', 'oxygen_saturation']

        for vital_name in vital_names:
            real_values = [getattr(r.vitals, vital_name) for r in real_records]
            synthetic_values = [getattr(r.vitals, vital_name) for r in synthetic_records]

            # Mean difference
            mean_diff = abs(np.mean(real_values) - np.mean(synthetic_values))
            metrics[f'{vital_name}_mean_diff'] = float(mean_diff)

            # Std difference
            std_diff = abs(np.std(real_values) - np.std(synthetic_values))
            metrics[f'{vital_name}_std_diff'] = float(std_diff)

        return metrics
