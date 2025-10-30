"""Embedding-based diversity analysis using ClinicalBERT"""
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.models import PatientRecord


class EmbeddingAnalyzer:
    """Analyze diversity using ClinicalBERT embeddings"""

    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", device: str = None):
        """
        Args:
            model_name: HuggingFace model name (default: Bio_ClinicalBERT)
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading ClinicalBERT model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze()

    def embed_records(self, records: List[PatientRecord], show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of patient records"""
        embeddings = []

        iterator = tqdm(records, desc="Generating embeddings") if show_progress else records

        for record in iterator:
            text = record.to_text()
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.array(embeddings)

    def compute_coverage_metrics(
        self,
        real_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute embedding coverage metrics between real and synthetic data

        Args:
            real_embeddings: Embeddings from real patient records
            synthetic_embeddings: Embeddings from synthetic patient records

        Returns:
            Dictionary with coverage metrics
        """
        # Compute pairwise cosine similarities
        similarities = cosine_similarity(synthetic_embeddings, real_embeddings)

        # Coverage metrics
        metrics = {}

        # 1. Average maximum similarity (how close is each synthetic to nearest real)
        max_similarities = similarities.max(axis=1)
        metrics['avg_max_similarity'] = float(np.mean(max_similarities))
        metrics['std_max_similarity'] = float(np.std(max_similarities))

        # 2. Coverage at different thresholds
        for threshold in [0.7, 0.8, 0.9]:
            coverage = (max_similarities >= threshold).mean()
            metrics[f'coverage@{threshold}'] = float(coverage)

        # 3. Diversity within synthetic data (lower similarity = more diverse)
        synthetic_self_sim = cosine_similarity(synthetic_embeddings, synthetic_embeddings)
        # Exclude diagonal (self-similarity)
        mask = np.ones_like(synthetic_self_sim, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_internal_sim = synthetic_self_sim[mask].mean()
        metrics['avg_internal_similarity'] = float(avg_internal_sim)
        metrics['diversity_score'] = float(1 - avg_internal_sim)  # Higher = more diverse

        # 4. Embedding space statistics
        metrics['real_embedding_mean_norm'] = float(np.linalg.norm(real_embeddings.mean(axis=0)))
        metrics['synthetic_embedding_mean_norm'] = float(np.linalg.norm(synthetic_embeddings.mean(axis=0)))

        # 5. Centroid distance
        real_centroid = real_embeddings.mean(axis=0)
        synthetic_centroid = synthetic_embeddings.mean(axis=0)
        centroid_distance = np.linalg.norm(real_centroid - synthetic_centroid)
        metrics['centroid_distance'] = float(centroid_distance)

        return metrics

    def compute_nearest_neighbor_metrics(
        self,
        real_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray,
        k: int = 5
    ) -> Dict[str, float]:
        """Compute k-nearest neighbor based metrics"""
        similarities = cosine_similarity(synthetic_embeddings, real_embeddings)

        metrics = {}

        # Get top-k similarities for each synthetic sample
        top_k_similarities = np.sort(similarities, axis=1)[:, -k:]

        metrics[f'avg_top{k}_similarity'] = float(top_k_similarities.mean())
        metrics[f'min_top{k}_similarity'] = float(top_k_similarities.min())
        metrics[f'max_top{k}_similarity'] = float(top_k_similarities.max())

        return metrics

    def analyze_diversity(
        self,
        real_records: List[PatientRecord],
        synthetic_records: List[PatientRecord]
    ) -> Dict[str, any]:
        """
        Complete diversity analysis comparing real and synthetic records

        Returns:
            Dictionary containing all metrics and embeddings
        """
        print("Embedding real records...")
        real_embeddings = self.embed_records(real_records)

        print("Embedding synthetic records...")
        synthetic_embeddings = self.embed_records(synthetic_records)

        print("Computing coverage metrics...")
        coverage_metrics = self.compute_coverage_metrics(real_embeddings, synthetic_embeddings)

        print("Computing nearest neighbor metrics...")
        nn_metrics = self.compute_nearest_neighbor_metrics(real_embeddings, synthetic_embeddings)

        return {
            'coverage_metrics': coverage_metrics,
            'nearest_neighbor_metrics': nn_metrics,
            'real_embeddings': real_embeddings,
            'synthetic_embeddings': synthetic_embeddings,
        }
