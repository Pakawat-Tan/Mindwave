"""
Topic Clustering — จัดกลุ่ม topics อัตโนมัติ

Features:
  1. auto clustering      — จัดกลุ่ม topic จาก interactions
  2. similarity detection — หา topic ที่คล้ายกัน
  3. topic merging        — รวม topic ที่คล้ายกัน
  4. topic suggestion     — แนะนำ topic ที่เหมาะสม
  5. cluster evolution    — cluster เปลี่ยนแปลงตามเวลา
  6. cluster stats        — สถิติของแต่ละ cluster

Algorithm:
  - ใช้ simple text-based similarity (Jaccard, edit distance)
  - clustering ด้วย threshold-based grouping
  - evolve clusters ตาม interaction frequency
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger("mindwave.topic_clustering")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity ระหว่าง 2 strings (word-based)"""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance"""
    if len(a) < len(b):
        return edit_distance(b, a)
    if len(b) == 0:
        return len(a)

    previous_row = range(len(b) + 1)
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """Edit distance normalized เป็น 0.0-1.0"""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return 1.0 - (edit_distance(a, b) / max_len)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicCluster:
    """Cluster ของ topics ที่คล้ายกัน"""
    cluster_id:   int
    topics:       Set[str]          # topics ใน cluster
    centroid:     str               # topic แทน cluster (ที่พบบ่อยสุด)
    size:         int               # จำนวน topics
    frequency:    int               # รวม interactions ทั้งหมด
    created_at:   float = field(default_factory=time.time)
    updated_at:   float = field(default_factory=time.time)

    def add_topic(self, topic: str) -> None:
        if topic not in self.topics:
            self.topics.add(topic)
            self.size = len(self.topics)
            self.updated_at = time.time()

    def remove_topic(self, topic: str) -> None:
        if topic in self.topics:
            self.topics.discard(topic)
            self.size = len(self.topics)
            self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "topics":     list(self.topics),
            "centroid":   self.centroid,
            "size":       self.size,
            "frequency":  self.frequency,
        }


@dataclass(frozen=True)
class SimilarityPair:
    """คู่ของ topics ที่คล้ายกัน"""
    topic_a:    str
    topic_b:    str
    similarity: float
    method:     str  # "jaccard" / "edit_distance"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_a":    self.topic_a,
            "topic_b":    self.topic_b,
            "similarity": round(self.similarity, 3),
            "method":     self.method,
        }


@dataclass(frozen=True)
class TopicSuggestion:
    """แนะนำ topic"""
    suggested_topic: str
    confidence:      float
    reason:          str
    similar_to:      List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggested_topic": self.suggested_topic,
            "confidence":      round(self.confidence, 3),
            "reason":          self.reason,
            "similar_to":      self.similar_to,
        }


@dataclass(frozen=True)
class ClusterEvolution:
    """การเปลี่ยนแปลงของ cluster"""
    cluster_id:   int
    timestamp:    float
    change_type:  str  # "created" / "merged" / "split" / "topic_added" / "topic_removed"
    before_size:  int
    after_size:   int
    description:  str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id":  self.cluster_id,
            "timestamp":   self.timestamp,
            "change_type": self.change_type,
            "before_size": self.before_size,
            "after_size":  self.after_size,
            "description": self.description,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TopicClustering
# ─────────────────────────────────────────────────────────────────────────────

class TopicClustering:
    """
    จัดกลุ่ม topics อัตโนมัติ

    Flow:
      interactions → extract topics → cluster → evolve
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        merge_threshold:      float = 0.7,
    ):
        self._sim_threshold   = similarity_threshold
        self._merge_threshold = merge_threshold

        self._clusters:   Dict[int, TopicCluster] = {}
        self._next_id:    int = 0
        self._topic_freq: Dict[str, int] = defaultdict(int)  # interaction count per topic
        self._evolutions: List[ClusterEvolution] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Auto Clustering
    # ─────────────────────────────────────────────────────────────────────────

    def cluster_topics(self, topics: List[str]) -> List[TopicCluster]:
        """
        จัดกลุ่ม topics อัตโนมัติ

        Algorithm:
          1. หา similarity ทุกคู่
          2. ถ้า similarity > threshold → ใส่ cluster เดียวกัน
          3. ถ้ายังไม่มี cluster → สร้างใหม่

        Args:
            topics: list ของ topic names

        Returns:
            List[TopicCluster] ที่สร้างใหม่
        """
        if not topics:
            return []

        # update frequency
        for topic in topics:
            self._topic_freq[topic] += 1

        new_clusters = []
        unassigned = set(topics)

        # พยายามใส่เข้า cluster เก่าก่อน
        for topic in topics:
            if topic not in unassigned:
                continue

            assigned = False
            for cluster in self._clusters.values():
                # ตรวจว่า topic คล้ายกับ centroid หรือไม่
                sim = self._calculate_similarity(topic, cluster.centroid)
                if sim >= self._sim_threshold:
                    cluster.add_topic(topic)
                    cluster.frequency += self._topic_freq[topic]
                    unassigned.discard(topic)
                    assigned = True
                    self._record_evolution(
                        cluster.cluster_id, "topic_added",
                        cluster.size - 1, cluster.size,
                        f"Added '{topic}' to cluster"
                    )
                    break

            # ถ้ายังไม่ได้ assign → สร้าง cluster ใหม่
            if not assigned and topic in unassigned:
                cluster = self._create_cluster([topic])
                new_clusters.append(cluster)
                unassigned.discard(topic)

        if new_clusters:
            logger.info(
                f"[TopicClustering] CLUSTER {len(new_clusters)} new clusters"
            )
        return new_clusters

    def _create_cluster(self, topics: List[str]) -> TopicCluster:
        """สร้าง cluster ใหม่"""
        cluster_id = self._next_id
        self._next_id += 1

        # centroid = topic ที่พบบ่อยสุด
        centroid = max(topics, key=lambda t: self._topic_freq[t])
        frequency = sum(self._topic_freq[t] for t in topics)

        cluster = TopicCluster(
            cluster_id = cluster_id,
            topics     = set(topics),
            centroid   = centroid,
            size       = len(topics),
            frequency  = frequency,
        )
        self._clusters[cluster_id] = cluster

        self._record_evolution(
            cluster_id, "created", 0, len(topics),
            f"Created cluster with {len(topics)} topics"
        )
        return cluster

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Similarity Detection
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_topics(
        self,
        topic: str,
        threshold: Optional[float] = None,
        method: str = "jaccard",
    ) -> List[SimilarityPair]:
        """
        หา topics ที่คล้ายกับ topic ที่ให้มา

        Args:
            topic: topic ที่ต้องการหาคู่
            threshold: similarity threshold (default ใช้ self._sim_threshold)
            method: "jaccard" / "edit_distance"

        Returns:
            List[SimilarityPair] เรียงจากคล้ายสุด → ไม่คล้าย
        """
        threshold = threshold or self._sim_threshold
        pairs = []

        all_topics = set(self._topic_freq.keys())
        for other in all_topics:
            if other == topic:
                continue

            if method == "jaccard":
                sim = jaccard_similarity(topic, other)
            else:  # edit_distance
                sim = normalized_edit_distance(topic, other)

            if sim >= threshold:
                pairs.append(SimilarityPair(
                    topic_a    = topic,
                    topic_b    = other,
                    similarity = sim,
                    method     = method,
                ))

        # sort by similarity descending
        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs

    def _calculate_similarity(self, a: str, b: str) -> float:
        """คำนวณ similarity (ใช้ jaccard เป็น default)"""
        return jaccard_similarity(a, b)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Topic Merging
    # ─────────────────────────────────────────────────────────────────────────

    def merge_similar_clusters(
        self,
        threshold: Optional[float] = None,
    ) -> int:
        """
        รวม clusters ที่คล้ายกัน

        Args:
            threshold: merge threshold (default ใช้ self._merge_threshold)

        Returns:
            จำนวน clusters ที่ถูก merge
        """
        threshold = threshold or self._merge_threshold
        merged_count = 0

        cluster_list = list(self._clusters.values())
        i = 0
        while i < len(cluster_list):
            cluster_a = cluster_list[i]
            j = i + 1
            while j < len(cluster_list):
                cluster_b = cluster_list[j]

                # ตรวจว่า centroid คล้ายกันไหม
                sim = self._calculate_similarity(
                    cluster_a.centroid,
                    cluster_b.centroid,
                )

                if sim >= threshold:
                    # merge B into A
                    before_a = cluster_a.size
                    before_b = cluster_b.size

                    for topic in cluster_b.topics:
                        cluster_a.add_topic(topic)
                    cluster_a.frequency += cluster_b.frequency

                    # remove cluster B
                    del self._clusters[cluster_b.cluster_id]
                    cluster_list.pop(j)

                    self._record_evolution(
                        cluster_a.cluster_id, "merged",
                        before_a, cluster_a.size,
                        f"Merged cluster {cluster_b.cluster_id} (sim={sim:.2f})"
                    )
                    merged_count += 1
                else:
                    j += 1
            i += 1

        if merged_count:
            logger.info(
                f"[TopicClustering] MERGE {merged_count} clusters"
            )
        return merged_count

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Topic Suggestion
    # ─────────────────────────────────────────────────────────────────────────

    def suggest_topic(
        self,
        context: str,
        current_topics: List[str] = None,
    ) -> TopicSuggestion:
        """
        แนะนำ topic ที่เหมาะสม

        Logic:
          - ถ้ามี cluster ที่ centroid คล้าย context → แนะนำ centroid
          - ถ้าไม่มี → แนะนำ topic ที่พบบ่อยสุด

        Args:
            context: context string
            current_topics: topics ที่ใช้อยู่แล้ว (optional)

        Returns:
            TopicSuggestion
        """
        current_topics = current_topics or []

        # หา cluster ที่ centroid คล้าย context
        best_cluster = None
        best_sim = 0.0

        for cluster in self._clusters.values():
            sim = self._calculate_similarity(context, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster and best_sim >= self._sim_threshold:
            return TopicSuggestion(
                suggested_topic = best_cluster.centroid,
                confidence      = best_sim,
                reason          = f"Similar to cluster centroid (sim={best_sim:.2f})",
                similar_to      = list(best_cluster.topics)[:3],
            )

        # ถ้าไม่เจอ → แนะนำ topic ที่พบบ่อยสุด (ที่ยังไม่ใช้)
        freq_topics = sorted(
            self._topic_freq.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for topic, freq in freq_topics:
            if topic not in current_topics:
                return TopicSuggestion(
                    suggested_topic = topic,
                    confidence      = min(0.7, freq / 100),  # cap at 0.7
                    reason          = f"Most frequent topic (freq={freq})",
                    similar_to      = [],
                )

        # fallback
        return TopicSuggestion(
            suggested_topic = "general",
            confidence      = 0.3,
            reason          = "No suitable topic found",
            similar_to      = [],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Cluster Evolution
    # ─────────────────────────────────────────────────────────────────────────

    def _record_evolution(
        self,
        cluster_id:  int,
        change_type: str,
        before_size: int,
        after_size:  int,
        description: str,
    ) -> None:
        """บันทึก evolution event"""
        evolution = ClusterEvolution(
            cluster_id  = cluster_id,
            timestamp   = time.time(),
            change_type = change_type,
            before_size = before_size,
            after_size  = after_size,
            description = description,
        )
        self._evolutions.append(evolution)

    @property
    def evolutions(self) -> List[ClusterEvolution]:
        """ประวัติการเปลี่ยนแปลง clusters"""
        return list(self._evolutions)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Cluster Stats
    # ─────────────────────────────────────────────────────────────────────────

    def cluster_stats(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """สถิติของ cluster"""
        cluster = self._clusters.get(cluster_id)
        if cluster is None:
            return None

        return {
            "cluster_id": cluster_id,
            "size":       cluster.size,
            "topics":     list(cluster.topics),
            "centroid":   cluster.centroid,
            "frequency":  cluster.frequency,
            "created_at": cluster.created_at,
            "updated_at": cluster.updated_at,
            "age_seconds": time.time() - cluster.created_at,
        }

    def stats(self) -> Dict[str, Any]:
        """สถิติรวม"""
        if not self._clusters:
            return {
                "total_clusters": 0,
                "total_topics":   0,
                "evolutions":     0,
            }

        total_topics = sum(c.size for c in self._clusters.values())
        avg_cluster_size = total_topics / len(self._clusters)

        largest = max(self._clusters.values(), key=lambda c: c.size)
        most_freq = max(self._clusters.values(), key=lambda c: c.frequency)

        return {
            "total_clusters":    len(self._clusters),
            "total_topics":      total_topics,
            "avg_cluster_size":  round(avg_cluster_size, 2),
            "largest_cluster":   largest.cluster_id,
            "largest_size":      largest.size,
            "most_frequent":     most_freq.cluster_id,
            "most_freq_count":   most_freq.frequency,
            "evolutions":        len(self._evolutions),
            "similarity_threshold": self._sim_threshold,
            "merge_threshold":   self._merge_threshold,
        }

    @property
    def clusters(self) -> List[TopicCluster]:
        return list(self._clusters.values())

    def get_cluster(self, cluster_id: int) -> Optional[TopicCluster]:
        return self._clusters.get(cluster_id)