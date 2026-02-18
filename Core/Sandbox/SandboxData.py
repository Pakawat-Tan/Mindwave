"""
Core/Sandbox/SandboxData.py

Data structures สำหรับ Sandbox system

SandboxAtom     — ผลการเรียนรู้ใน Sandbox (ยังไม่ใช่ production)
ExperimentState — state ที่แลกเปลี่ยนระหว่าง instances ผ่าน SCL
SandboxWorld    — container สำหรับ ≥1 instances
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class SandboxStatus(Enum):
    ACTIVE    = "active"    # กำลัง experiment อยู่
    PROMOTED  = "promoted"  # ผ่าน Reviewer แล้ว → production
    EXPIRED   = "expired"   # หมดอายุ
    REJECTED  = "rejected"  # Reviewer reject


class AtomType(Enum):
    LEARNING  = "learning"  # เรียนรู้จาก interaction
    CONFLICT  = "conflict"  # บันทึก conflict ระหว่าง instances
    HYPOTHESIS = "hypothesis" # hypothesis ใหม่จาก conflict


# ─────────────────────────────────────────────────────────────────────────────
# SandboxAtom
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SandboxAtom:
    """
    ผลการเรียนรู้ใน Sandbox

    ยังไม่ใช่ production atom — ต้องผ่าน Reviewer ก่อน promote
    แต่ละ instance มี SandboxAtom แยกกันโดยสมบูรณ์
    """
    atom_id:        str   = field(default_factory=lambda: str(uuid.uuid4()))
    instance_id:    str   = ""
    atom_type:      AtomType = AtomType.LEARNING
    context:        str   = ""
    payload:        bytes = b""
    source:         str   = ""          # "sandbox_response_{context}"
    confidence:     float = 0.0
    status:         SandboxStatus = SandboxStatus.ACTIVE
    created_ts:     float = field(default_factory=time.time)
    expiry_ts:      Optional[float] = None
    tags:           List[str] = field(default_factory=list)
    metadata:       Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expiry_ts is None:
            return False
        return time.time() > self.expiry_ts

    @property
    def is_promotable(self) -> bool:
        return (
            self.status == SandboxStatus.ACTIVE
            and not self.is_expired
            and self.confidence >= 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id":     self.atom_id,
            "instance_id": self.instance_id,
            "atom_type":   self.atom_type.value,
            "context":     self.context,
            "confidence":  self.confidence,
            "status":      self.status.value,
            "created_ts":  self.created_ts,
            "tags":        self.tags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentState
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentState:
    """
    State ที่แลกเปลี่ยนระหว่าง instances ผ่าน SCL

    ห้ามส่ง: identity จริง, immortal term, production atom,
             decision system, belief โดยตรง, knowlet ที่ยังไม่ผ่าน reviewer

    source_instance ถูก hash เสมอ — ไม่เปิดเผย identity จริง
    """
    experiment_id:   str   = field(default_factory=lambda: str(uuid.uuid4()))
    source_instance: str   = ""         # hashed instance id
    hypothesis:      str   = ""
    stimulus_ref:    str   = ""         # SandboxAtom.atom_id (reference only)
    outcome:         str   = ""         # ผลการทดลอง
    confidence_delta: float = 0.0       # confidence เปลี่ยนแปลง
    timestamp:       float = field(default_factory=time.time)
    expiry_ts:       float = field(
        default_factory=lambda: time.time() + 86400  # 24h default
    )
    tags:            List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        instance_id:     str,
        hypothesis:      str,
        outcome:         str,
        stimulus_ref:    str   = "",
        confidence_delta: float = 0.0,
        tags:            List[str] = None,
        ttl_seconds:     int   = 86400,
    ) -> "ExperimentState":
        """สร้าง ExperimentState พร้อม hash instance_id"""
        hashed = hashlib.sha256(instance_id.encode()).hexdigest()[:16]
        return cls(
            source_instance  = hashed,
            hypothesis       = hypothesis,
            stimulus_ref     = stimulus_ref,
            outcome          = outcome,
            confidence_delta = confidence_delta,
            tags             = tags or [],
            expiry_ts        = time.time() + ttl_seconds,
        )

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expiry_ts

    def conflicts_with(self, other: "ExperimentState") -> bool:
        """ตรวจว่า hypothesis ขัดแย้งกันไหม"""
        if self.source_instance == other.source_instance:
            return False
        if self.hypothesis == other.hypothesis:
            return (self.confidence_delta * other.confidence_delta) < 0
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id":   self.experiment_id,
            "source_instance": self.source_instance,
            "hypothesis":      self.hypothesis,
            "outcome":         self.outcome,
            "confidence_delta": self.confidence_delta,
            "timestamp":       self.timestamp,
            "expiry_ts":       self.expiry_ts,
            "tags":            self.tags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SandboxWorld
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SandboxWorld:
    """
    Container สำหรับ ≥1 Sandbox instances

    1 World = หลาย instances ที่อยู่ใน experiment เดียวกัน
    Memory ของแต่ละ instance แยกกันโดยสมบูรณ์

    Sandbox World:
      ├── Instance A → SandboxController A
      ├── Instance B → SandboxController B
      └── Instance C → SandboxController C
    """
    world_id:    str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:        str  = "default_world"
    created_ts:  float = field(default_factory=time.time)
    instance_ids: List[str] = field(default_factory=list)

    def register_instance(self, instance_id: str) -> None:
        if instance_id not in self.instance_ids:
            self.instance_ids.append(instance_id)

    def remove_instance(self, instance_id: str) -> None:
        self.instance_ids = [i for i in self.instance_ids if i != instance_id]

    @property
    def instance_count(self) -> int:
        return len(self.instance_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id":       self.world_id,
            "name":           self.name,
            "instance_count": self.instance_count,
            "created_ts":     self.created_ts,
        }