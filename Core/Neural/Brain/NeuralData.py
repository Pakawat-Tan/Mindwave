"""
Data structures สำหรับ Neural module

Spec:
  - proposal-only: Neural เสนอการเปลี่ยน weight ผ่าน Proposal system
  - ไม่ approve เอง
  - Gradient monitoring: ตรวจ explode/NaN/Inf
  - Rollback: ย้อน weight กลับได้
  - Evolution tracking: เก็บประวัติทุก mutation
  - Usage tracking: นับจำนวนครั้งที่ใช้ weight แต่ละตัว

NeuralEvolution.json Rule (SYSTEM):
  - ห้าม uncontrolled_growth
  - ห้าม self_modify_core
  - หยุดเมื่อ gradient explode/NaN/Inf
  - ห้าม block rollback
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import math
import time
import uuid


# ============================================================================
# ENUMS
# ============================================================================

class GradientStatus(Enum):
    """สถานะของ gradient"""
    OK       = "ok"
    EXPLODE  = "explode"   # |grad| > threshold
    VANISH   = "vanish"    # |grad| < min_threshold
    NAN      = "nan"       # math.isnan
    INF      = "inf"       # math.isinf

    def __str__(self): return self.value.upper()

    @property
    def is_healthy(self) -> bool:
        return self == GradientStatus.OK

    @property
    def is_critical(self) -> bool:
        return self in (GradientStatus.NAN, GradientStatus.INF,
                        GradientStatus.EXPLODE)


class ConflictType(Enum):
    """ประเภทของ knowledge conflict ที่ Neural ตรวจพบ"""
    NONE            = "none"
    WEIGHT_DIVERGE  = "weight_diverge"   # weights แตกต่างมากเกินไป
    KNOWLEDGE_GAP   = "knowledge_gap"    # ความรู้ขัดแย้งกัน
    GRADIENT_UNSAFE = "gradient_unsafe"  # gradient ไม่ปลอดภัย
    EVOLUTION_LOOP  = "evolution_loop"   # วน evolution ซ้ำ

    def __str__(self): return self.value.upper()


class WeightUpdateSource(Enum):
    """แหล่งที่มาของการ update weight"""
    PROPOSAL_APPROVED = "proposal_approved"  # ผ่าน Proposal → approved
    ROLLBACK          = "rollback"           # ย้อนกลับ
    INIT              = "init"               # ค่าเริ่มต้น
    FORCE             = "force"              # admin force (reviewer)

    def __str__(self): return self.value


# ============================================================================
# WEIGHT DATA
# ============================================================================

@dataclass
class WeightData:
    """
    Neural weight สำหรับ topic/domain หนึ่ง

    - ไม่ใช้ frozen — แต่เปลี่ยนได้เฉพาะผ่าน NeuralController
    - เก็บ snapshot history สำหรับ rollback
    - นับ usage_count ทุกครั้งที่ถูกใช้
    """
    weight_id:   str
    domain:      str           # ชื่อ domain/topic
    value:       float         # current weight value
    min_value:   float = 0.0
    max_value:   float = 1.0
    _history:    List[Tuple[float, float, str]] = field(
        default_factory=list, init=False, repr=False
    )  # [(value, timestamp, source)]
    _usage_count: int = field(default=0, init=False, repr=False)
    created_ts:  float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.weight_id or not self.weight_id.strip():
            raise ValueError("WeightData: weight_id must not be empty")
        if not self.domain or not self.domain.strip():
            raise ValueError("WeightData: domain must not be empty")
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError(
                f"WeightData: value {self.value} out of range "
                f"[{self.min_value}, {self.max_value}]"
            )
        # เก็บ initial snapshot
        self._history.append((self.value, self.created_ts, WeightUpdateSource.INIT.value))

    @property
    def usage_count(self) -> int:
        return self._usage_count

    @property
    def history(self) -> List[Tuple[float, float, str]]:
        return list(self._history)

    @property
    def previous_value(self) -> Optional[float]:
        """ค่าก่อนหน้า (สำหรับ rollback)"""
        return self._history[-2][0] if len(self._history) >= 2 else None

    def record_usage(self) -> None:
        """นับ usage ทุกครั้งที่ weight ถูกใช้"""
        self._usage_count += 1

    def update(self, new_value: float, source: str = "") -> float:
        """
        อัปเดต value — เรียกจาก NeuralController เท่านั้น

        Returns:
            old value
        """
        if not (self.min_value <= new_value <= self.max_value):
            raise ValueError(
                f"WeightData.update: {new_value} out of range "
                f"[{self.min_value}, {self.max_value}]"
            )
        old = self.value
        self.value = round(new_value, 6)
        self._history.append((self.value, time.time(), source))
        return old

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_id":   self.weight_id,
            "domain":      self.domain,
            "value":       self.value,
            "min_value":   self.min_value,
            "max_value":   self.max_value,
            "usage_count": self._usage_count,
            "history_len": len(self._history),
        }

    def __str__(self) -> str:
        return (
            f"Weight[{self.domain}] "
            f"val={self.value:.6f} "
            f"usage={self._usage_count}"
        )


# ============================================================================
# GRADIENT SNAPSHOT
# ============================================================================

@dataclass(frozen=True)
class GradientSnapshot:
    """
    บันทึกสถานะ gradient ณ เวลาหนึ่ง

    Thresholds (default):
        explode : |grad| > 100.0
        vanish  : |grad| < 1e-7
    """
    domain:     str
    gradient:   float
    status:     GradientStatus
    threshold_explode: float = 100.0
    threshold_vanish:  float = 1e-7
    timestamp:  float = field(default_factory=time.time)
    snap_id:    str   = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @classmethod
    def evaluate(
        cls,
        domain:   str,
        gradient: float,
        threshold_explode: float = 100.0,
        threshold_vanish:  float = 1e-7,
    ) -> GradientSnapshot:
        """ตรวจสอบ gradient และสร้าง snapshot"""
        if math.isnan(gradient):
            status = GradientStatus.NAN
        elif math.isinf(gradient):
            status = GradientStatus.INF
        elif abs(gradient) > threshold_explode:
            status = GradientStatus.EXPLODE
        elif abs(gradient) < threshold_vanish:
            status = GradientStatus.VANISH
        else:
            status = GradientStatus.OK
        return cls(
            domain=domain, gradient=gradient, status=status,
            threshold_explode=threshold_explode,
            threshold_vanish=threshold_vanish,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snap_id":   self.snap_id,
            "domain":    self.domain,
            "gradient":  self.gradient,
            "status":    self.status.value,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        return (
            f"Gradient[{self.domain}] "
            f"grad={self.gradient:.6f} → {self.status}"
        )


# ============================================================================
# CONFLICT DATA
# ============================================================================

@dataclass(frozen=True)
class ConflictData:
    """
    Knowledge conflict ที่ Neural ตรวจพบ

    เมื่อพบ conflict → Neural สร้าง Proposal เพื่อขอแก้ไข
    ไม่แก้เอง
    """
    conflict_id:   str   = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: ConflictType = ConflictType.NONE
    domain:        str   = ""
    description:   str   = ""
    severity:      float = 0.0   # 0.0–1.0
    timestamp:     float = field(default_factory=time.time)
    resolved:      bool  = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id":   self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "domain":        self.domain,
            "description":   self.description,
            "severity":      self.severity,
            "timestamp":     self.timestamp,
            "resolved":      self.resolved,
        }

    def __str__(self) -> str:
        return (
            f"Conflict[{self.conflict_type}] "
            f"domain={self.domain} "
            f"severity={self.severity:.2f} "
            f"{'resolved' if self.resolved else 'open'}"
        )


# ============================================================================
# EVOLUTION RECORD
# ============================================================================

@dataclass(frozen=True)
class EvolutionRecord:
    """
    บันทึก evolution event หนึ่งครั้ง

    Trigger: proposal approved / rollback / conflict resolved
    """
    record_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    event_type:   str   = ""      # "weight_update", "rollback", "conflict_resolved"
    domain:       str   = ""
    old_value:    Optional[float] = None
    new_value:    Optional[float] = None
    triggered_by: str   = ""      # proposal_id / reviewer_id
    description:  str   = ""
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id":    self.record_id,
            "event_type":   self.event_type,
            "domain":       self.domain,
            "old_value":    self.old_value,
            "new_value":    self.new_value,
            "triggered_by": self.triggered_by,
            "description":  self.description,
            "timestamp":    self.timestamp,
        }

    def __str__(self) -> str:
        val_str = (
            f"{self.old_value:.6f} → {self.new_value:.6f}"
            if self.old_value is not None and self.new_value is not None
            else self.event_type
        )
        return (
            f"Evolution[{self.event_type}] "
            f"domain={self.domain} {val_str} "
            f"by={self.triggered_by}"
        )