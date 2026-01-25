"""
PerformanceMonitor.py
ติดตาม performance ระยะยาว
Monitors long-term performance.
"""

from typing import Dict, Any, List
import time
import statistics


class PerformanceMonitor:
    """Monitors long-term system performance."""

    def __init__(self):
        self.performance_metrics: Dict[str, List[float]] = {}
        self.historical_data: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []

    # -------------------------------------------------

    def record_performance(self, metric_name: str, value: float):
        """Record a performance metric."""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []

        self.performance_metrics[metric_name].append(value)

        self.historical_data.append({
            "metric": metric_name,
            "value": value,
            "timestamp": time.time()
        })

    # -------------------------------------------------

    def compute_trend(self, metric_name: str, window_size: int = 100) -> Dict[str, float]:
        """Compute performance trend using moving averages."""

        values = self.performance_metrics.get(metric_name, [])
        if len(values) < 2:
            return {"trend": 0.0, "current": None, "previous": None}

        recent = values[-window_size:]
        mid = max(1, len(recent) // 2)

        first_half = recent[:mid]
        second_half = recent[mid:]

        prev_avg = statistics.mean(first_half)
        curr_avg = statistics.mean(second_half)

        trend = curr_avg - prev_avg

        return {
            "trend": trend,
            "previous": prev_avg,
            "current": curr_avg
        }

    # -------------------------------------------------

    def detect_performance_degradation(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect metrics that are degrading beyond threshold.
        threshold = acceptable drop
        """

        degradations = []

        for metric in self.performance_metrics:
            trend_info = self.compute_trend(metric)

            if trend_info["current"] is None:
                continue

            if trend_info["trend"] < -threshold:
                alert = {
                    "metric": metric,
                    "drop": trend_info["trend"],
                    "current": trend_info["current"],
                    "previous": trend_info["previous"],
                    "timestamp": time.time()
                }
                degradations.append(alert)
                self.alerts.append(alert)

        return degradations

    # -------------------------------------------------

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify potential bottlenecks based on variance and drops.
        """

        bottlenecks = {}

        for metric, values in self.performance_metrics.items():
            if len(values) < 5:
                continue

            variance = statistics.variance(values)
            trend = self.compute_trend(metric)["trend"]

            if variance > 0.05 or trend < 0:
                bottlenecks[metric] = {
                    "variance": variance,
                    "trend": trend,
                    "latest": values[-1]
                }

        return bottlenecks

    # -------------------------------------------------

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""

        report = {
            "metrics": {},
            "alerts": self.alerts[-10:],  # last alerts
            "bottlenecks": self.identify_bottlenecks(),
            "timestamp": time.time()
        }

        for metric, values in self.performance_metrics.items():
            if not values:
                continue

            report["metrics"][metric] = {
                "latest": values[-1],
                "average": sum(values) / len(values),
                "trend": self.compute_trend(metric)
            }

        return report
