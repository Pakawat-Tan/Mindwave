"""
Topic Data Module

Topic representation based on unsupervised clustering.
Labels are NOT predefined — they emerge from the model after training.

A TopicData represents one cluster learned by the model:
    cluster_id    : Integer ID assigned by the clustering algorithm
    top_keywords  : Most representative terms for this cluster
    coherence     : Cluster coherence score (how tight the cluster is)
    label         : Human-readable label — None until model assigns it
    embedding     : Centroid vector of the cluster (optional)
    document_count: Number of documents assigned to this cluster
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import json
import math


# ============================================================================
# TOPIC DATA
# ============================================================================

@dataclass
class TopicData:
    """
    One topic cluster learned by an unsupervised model.

    Attributes:
        cluster_id     : Integer ID from the clustering algorithm
        top_keywords   : Ranked list of most representative terms
        coherence      : Cluster coherence (0.0 = loose, 1.0 = tight)
        label          : Emerged label string — None if not yet assigned
        embedding      : Centroid vector (None if not computed)
        document_count : Number of documents in this cluster
    """
    cluster_id:     int
    top_keywords:   List[str]
    coherence:      float
    label:          Optional[str]  = None
    embedding:      Optional[List[float]] = field(default=None, repr=False)
    document_count: int            = 0

    def __post_init__(self) -> None:
        self.coherence      = max(0.0, min(1.0, self.coherence))
        self.document_count = max(0, self.document_count)
        # Normalize keywords: strip + lowercase + deduplicate
        seen, clean = set(), []
        for kw in self.top_keywords:
            kw = kw.strip().lower()
            if kw and kw not in seen:
                seen.add(kw)
                clean.append(kw)
        self.top_keywords = clean

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def has_label(self) -> bool:
        """True once the model has assigned a label."""
        return self.label is not None

    @property
    def is_coherent(self) -> bool:
        """Cluster is considered coherent when score >= 0.5"""
        return self.coherence >= 0.5

    @property
    def top_keyword(self) -> Optional[str]:
        """Most representative keyword (first in list), or None if empty."""
        return self.top_keywords[0] if self.top_keywords else None

    # ------------------------------------------------------------------
    # Similarity — cosine on embedding centroids
    # ------------------------------------------------------------------

    def cosine_similarity(self, other: "TopicData") -> Optional[float]:
        """
        Cosine similarity between cluster centroids.
        Returns None if either topic has no embedding.
        """
        if self.embedding is None or other.embedding is None:
            return None
        a, b = self.embedding, other.embedding
        if len(a) != len(b):
            raise ValueError(f"Embedding dimensions differ: {len(a)} vs {len(b)}")
        dot    = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def keyword_overlap(self, other: "TopicData") -> float:
        """
        Jaccard overlap of top_keywords between two clusters.
        Returns 0.0 if both lists are empty.
        """
        s, o = set(self.top_keywords), set(other.top_keywords)
        if not s and not o:
            return 0.0
        return len(s & o) / len(s | o)

    # ------------------------------------------------------------------
    # Label assignment (called by model after unsupervised training)
    # ------------------------------------------------------------------

    def assign_label(self, label: str) -> None:
        """Assign an emerged label to this cluster."""
        if not label or not label.strip():
            raise ValueError("Label must be a non-empty string.")
        self.label = label.strip()

    def clear_label(self) -> None:
        """Remove assigned label (revert to unlabeled state)."""
        self.label = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id":     self.cluster_id,
            "top_keywords":   self.top_keywords,
            "coherence":      self.coherence,
            "label":          self.label,
            "embedding":      self.embedding,
            "document_count": self.document_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicData":
        return cls(
            cluster_id     = data["cluster_id"],
            top_keywords   = data.get("top_keywords", []),
            coherence      = data["coherence"],
            label          = data.get("label"),
            embedding      = data.get("embedding"),
            document_count = data.get("document_count", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TopicData":
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        label_str = f'"{self.label}"' if self.label else "unlabeled"
        kws = ", ".join(self.top_keywords[:5])
        return (
            f"TopicData(cluster={self.cluster_id}, "
            f"label={label_str}, "
            f"coherence={self.coherence:.2f}, "
            f"keywords=[{kws}], "
            f"docs={self.document_count})"
        )


# ============================================================================
# FACTORY
# ============================================================================

def create_topic(
    cluster_id:     int,
    top_keywords:   List[str],
    coherence:      float,
    label:          Optional[str]        = None,
    embedding:      Optional[List[float]] = None,
    document_count: int                  = 0,
) -> TopicData:
    """
    Convenience factory.
    In production, cluster_id / top_keywords / coherence come from model output.
    """
    return TopicData(
        cluster_id     = cluster_id,
        top_keywords   = top_keywords,
        coherence      = coherence,
        label          = label,
        embedding      = embedding,
        document_count = document_count,
    )