"""
WeightArchive.py
Archives snapshots of brain weights for rollback and comparison.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json
import copy


@dataclass
class WeightSnapshot:
    """Represents a weight snapshot"""
    snapshot_id: str
    timestamp: float = field(default_factory=time.time)
    weights: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""


class WeightArchive:
    """Archives weight snapshots for rollback, comparison, and recovery."""
    
    def __init__(self, max_snapshots: int = 50):
        """Initialize weight archive.
        
        Parameters
        ----------
        max_snapshots : int
            Maximum number of snapshots to keep
        """
        self.max_snapshots = max_snapshots
        self.snapshots: Dict[str, WeightSnapshot] = {}
        self.snapshot_history: List[str] = []
        self.current_snapshot: Optional[str] = None
    
    def create_snapshot(self, snapshot_id: str, weights: Dict[str, Any],
                       metadata: Optional[Dict[str, Any]] = None,
                       performance_metrics: Optional[Dict[str, float]] = None,
                       description: str = "") -> bool:
        """Create a snapshot of current weights.
        
        Parameters
        ----------
        snapshot_id : str
            Unique snapshot ID
        weights : Dict[str, Any]
            Weight dictionary
        metadata : Optional[Dict[str, Any]]
            Additional metadata
        performance_metrics : Optional[Dict[str, float]]
            Performance metrics at snapshot time
        description : str
            Description of the snapshot
            
        Returns
        -------
        bool
            Success
        """
        if len(self.snapshots) >= self.max_snapshots:
            self._evict_snapshot()
        
        snapshot = WeightSnapshot(
            snapshot_id=snapshot_id,
            weights=copy.deepcopy(weights),
            metadata=metadata or {},
            performance_metrics=performance_metrics or {},
            description=description
        )
        
        self.snapshots[snapshot_id] = snapshot
        self.snapshot_history.append(snapshot_id)
        self.current_snapshot = snapshot_id
        
        return True
    
    def rollback_to_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Rollback weights to a previous snapshot.
        
        Parameters
        ----------
        snapshot_id : str
            Snapshot ID to rollback to
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Weights from snapshot or None
        """
        if snapshot_id not in self.snapshots:
            return None
        
        snapshot = self.snapshots[snapshot_id]
        self.current_snapshot = snapshot_id
        
        return copy.deepcopy(snapshot.weights)
    
    def compare_snapshots(self, snapshot_id_1: str, snapshot_id_2: str) -> Optional[Dict[str, Any]]:
        """Compare two weight snapshots.
        
        Parameters
        ----------
        snapshot_id_1 : str
            First snapshot ID
        snapshot_id_2 : str
            Second snapshot ID
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Comparison results
        """
        if snapshot_id_1 not in self.snapshots or snapshot_id_2 not in self.snapshots:
            return None
        
        snap1 = self.snapshots[snapshot_id_1]
        snap2 = self.snapshots[snapshot_id_2]
        
        comparison = {
            "snapshot_1": snapshot_id_1,
            "snapshot_2": snapshot_id_2,
            "timestamp_1": snap1.timestamp,
            "timestamp_2": snap2.timestamp,
            "time_difference_seconds": snap2.timestamp - snap1.timestamp,
            "weights_changed": 0,
            "weight_changes": {},
            "metric_changes": {}
        }
        
        # Compare weights
        all_keys = set(snap1.weights.keys()) | set(snap2.weights.keys())
        for key in all_keys:
            val1 = snap1.weights.get(key)
            val2 = snap2.weights.get(key)
            
            if val1 != val2:
                comparison["weights_changed"] += 1
                comparison["weight_changes"][key] = {
                    "before": val1,
                    "after": val2
                }
        
        # Compare metrics
        all_metric_keys = set(snap1.performance_metrics.keys()) | set(snap2.performance_metrics.keys())
        for key in all_metric_keys:
            val1 = snap1.performance_metrics.get(key, 0.0)
            val2 = snap2.performance_metrics.get(key, 0.0)
            
            if val1 != val2:
                comparison["metric_changes"][key] = {
                    "before": val1,
                    "after": val2,
                    "delta": val2 - val1
                }
        
        return comparison
    
    def get_weight_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the history of weight snapshots.
        
        Parameters
        ----------
        limit : int
            Maximum number of snapshots to return
            
        Returns
        -------
        List[Dict[str, Any]]
            Snapshot history
        """
        history = []
        
        for snapshot_id in self.snapshot_history[-limit:]:
            if snapshot_id in self.snapshots:
                snap = self.snapshots[snapshot_id]
                history.append({
                    "snapshot_id": snapshot_id,
                    "timestamp": snap.timestamp,
                    "description": snap.description,
                    "num_weights": len(snap.weights),
                    "performance_metrics": snap.performance_metrics
                })
        
        return history
    
    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific snapshot.
        
        Parameters
        ----------
        snapshot_id : str
            Snapshot ID
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Snapshot information
        """
        if snapshot_id not in self.snapshots:
            return None
        
        snap = self.snapshots[snapshot_id]
        
        return {
            "snapshot_id": snapshot_id,
            "timestamp": snap.timestamp,
            "description": snap.description,
            "num_weights": len(snap.weights),
            "metadata": snap.metadata,
            "performance_metrics": snap.performance_metrics
        }
    
    def get_recovery_snapshots(self, performance_metric: str, threshold: float) -> List[str]:
        """Get snapshots that met a performance threshold.
        
        Parameters
        ----------
        performance_metric : str
            Metric name
        threshold : float
            Threshold value
            
        Returns
        -------
        List[str]
            Snapshot IDs meeting threshold
        """
        results = []
        
        for snapshot_id, snap in self.snapshots.items():
            metric_value = snap.performance_metrics.get(performance_metric)
            if metric_value is not None and metric_value >= threshold:
                results.append(snapshot_id)
        
        return results
    
    def get_best_snapshot(self, performance_metric: str) -> Optional[str]:
        """Get snapshot with best performance on metric.
        
        Parameters
        ----------
        performance_metric : str
            Metric name
            
        Returns
        -------
        Optional[str]
            Best snapshot ID
        """
        best_id = None
        best_value = float('-inf')
        
        for snapshot_id, snap in self.snapshots.items():
            metric_value = snap.performance_metrics.get(performance_metric)
            if metric_value is not None and metric_value > best_value:
                best_value = metric_value
                best_id = snapshot_id
        
        return best_id
    
    def _evict_snapshot(self) -> Optional[str]:
        """Evict oldest snapshot.
        
        Returns
        -------
        Optional[str]
            Evicted snapshot ID
        """
        if not self.snapshot_history:
            return None
        
        evicted_id = self.snapshot_history.pop(0)
        if evicted_id in self.snapshots:
            del self.snapshots[evicted_id]
        
        return evicted_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get weight archive status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "max_snapshots": self.max_snapshots,
            "current_snapshots": len(self.snapshots),
            "utilization_percent": (len(self.snapshots) / self.max_snapshots * 100) if self.max_snapshots > 0 else 0,
            "current_snapshot": self.current_snapshot,
            "oldest_snapshot": self.snapshot_history[0] if self.snapshot_history else None,
            "newest_snapshot": self.snapshot_history[-1] if self.snapshot_history else None,
            "total_snapshots_created": len(self.snapshot_history)
        }
