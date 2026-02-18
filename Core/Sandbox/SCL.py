"""
Core/Sandbox/SCL.py

Shared Collaboration Layer — แลกเปลี่ยน ExperimentState ระหว่าง instances

กฎเหล็ก:
  ✅ แลกเปลี่ยนได้: ExperimentState (hypothesis + outcome + confidence_delta)
  ❌ ห้ามส่ง: identity จริง, immortal term, production atom,
              decision system, belief โดยตรง, knowlet ที่ยังไม่ผ่าน reviewer

SCL ไม่ resolve conflict เอง — แต่ละ instance ตีความด้วย Neural ของตัวเอง
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from Core.Sandbox.SandboxData import ExperimentState, SandboxAtom, AtomType

logger = logging.getLogger("mindwave.scl")


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConflictRecord:
    """
    บันทึก conflict ระหว่าง ExperimentStates

    SCL ไม่ resolve — แค่บันทึกและแจ้ง instances ที่เกี่ยวข้อง
    instance จะสร้าง SandboxAtom(AtomType.CONFLICT) เพื่อ
    ตั้ง hypothesis ใหม่รอบถัดไป
    """
    conflict_id:  str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state_a_id:   str   = ""
    state_b_id:   str   = ""
    hypothesis:   str   = ""
    description:  str   = ""
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "conflict_id": self.conflict_id,
            "state_a_id":  self.state_a_id,
            "state_b_id":  self.state_b_id,
            "hypothesis":  self.hypothesis,
            "timestamp":   self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCL
# ─────────────────────────────────────────────────────────────────────────────

class SCL:
    """
    Shared Collaboration Layer

    Flow:
      Instance A ทดลอง → publish(ExperimentState A)
      Instance B ทดลอง → publish(ExperimentState B)
      SCL รวม states → แต่ละ instance อ่านผ่าน read_for(instance_id)
      ถ้า conflict → บันทึก ConflictRecord → แจ้ง instance

    Instance อ่าน state ของ instance อื่น แล้วตีความด้วย Neural ของตัวเอง
    → เกิด Emergent Knowledge ที่แยกกัน
    """

    # สิ่งที่ห้ามส่งผ่าน SCL
    _FORBIDDEN_TAGS = {
        "identity", "immortal", "production_atom",
        "decision", "belief", "core_value",
        "knowlet_unreviewed",
    }

    def __init__(self, world_id: str = ""):
        self._world_id    = world_id or str(uuid.uuid4())[:8]
        self._states:     Dict[str, ExperimentState]  = {}
        self._conflicts:  List[ConflictRecord]         = []
        self._registered: List[str]                    = []   # instance ids

    # ─────────────────────────────────────────────────────────────────────────
    # Instance registry
    # ─────────────────────────────────────────────────────────────────────────

    def register(self, instance_id: str) -> None:
        """ลงทะเบียน instance เข้า SCL"""
        if instance_id not in self._registered:
            self._registered.append(instance_id)
            logger.info(f"[SCL] REGISTERED instance={instance_id[:8]} world={self._world_id}")

    def unregister(self, instance_id: str) -> None:
        self._registered = [i for i in self._registered if i != instance_id]

    @property
    def instance_count(self) -> int:
        return len(self._registered)

    # ─────────────────────────────────────────────────────────────────────────
    # Publish / Read
    # ─────────────────────────────────────────────────────────────────────────

    def publish(
        self,
        instance_id: str,
        state:       ExperimentState,
    ) -> bool:
        """
        Instance ส่ง ExperimentState เข้า SCL

        Validation:
          - instance ต้อง registered
          - ห้ามมี forbidden tags
          - ห้าม state หมดอายุ
        """
        if instance_id not in self._registered:
            raise PermissionError(
                f"[SCL] instance '{instance_id[:8]}' not registered"
            )

        # ตรวจ forbidden tags
        forbidden = set(state.tags) & self._FORBIDDEN_TAGS
        if forbidden:
            raise ValueError(
                f"[SCL] FORBIDDEN tags in ExperimentState: {forbidden}"
            )

        if state.is_expired:
            logger.warning(f"[SCL] SKIP expired state {state.experiment_id[:8]}")
            return False

        # ลบ states เก่าของ instance นี้ก่อน publish ใหม่
        self._states = {
            eid: s for eid, s in self._states.items()
            if s.source_instance != state.source_instance
            or eid == state.experiment_id
        }

        self._states[state.experiment_id] = state
        logger.info(
            f"[SCL] PUBLISHED instance={instance_id[:8]} "
            f"exp={state.experiment_id[:8]} "
            f"hypothesis='{state.hypothesis[:40]}'"
        )

        # ตรวจ conflict กับ states อื่น
        self._detect_conflicts(state)
        return True

    def read_for(
        self,
        instance_id: str,
        exclude_own: bool = True,
    ) -> List[ExperimentState]:
        """
        Instance อ่าน ExperimentStates จาก instances อื่น

        exclude_own=True (default) — ไม่อ่าน state ของตัวเอง
        กรอง expired states ออกอัตโนมัติ
        """
        if instance_id not in self._registered:
            raise PermissionError(
                f"[SCL] instance '{instance_id[:8]}' not registered"
            )

        import hashlib
        own_hash = hashlib.sha256(instance_id.encode()).hexdigest()[:16]

        result = []
        for state in self._states.values():
            if state.is_expired:
                continue
            if exclude_own and state.source_instance == own_hash:
                continue
            result.append(state)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Conflict
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_conflicts(self, new_state: ExperimentState) -> None:
        """
        ตรวจ conflict กับทุก state ที่มีอยู่

        SCL ไม่ resolve เอง — แค่บันทึก ConflictRecord
        Instance จะอ่าน conflict และตั้ง hypothesis ใหม่
        """
        for eid, existing in self._states.items():
            if eid == new_state.experiment_id:
                continue
            if existing.conflicts_with(new_state):
                record = ConflictRecord(
                    state_a_id  = existing.experiment_id,
                    state_b_id  = new_state.experiment_id,
                    hypothesis  = new_state.hypothesis,
                    description = (
                        f"Conflicting outcomes: "
                        f"'{existing.outcome}' vs '{new_state.outcome}'"
                    ),
                )
                self._conflicts.append(record)
                logger.warning(
                    f"[SCL] CONFLICT detected "
                    f"A={existing.experiment_id[:8]} "
                    f"B={new_state.experiment_id[:8]} "
                    f"hypothesis='{new_state.hypothesis[:40]}'"
                )

    def conflicts_for(self, instance_id: str) -> List[ConflictRecord]:
        """
        คืน conflicts ที่เกี่ยวข้องกับ instance นี้
        Instance จะสร้าง SandboxAtom(CONFLICT) จาก conflicts เหล่านี้
        """
        import hashlib
        own_hash = hashlib.sha256(instance_id.encode()).hexdigest()[:16]

        result = []
        for conflict in self._conflicts:
            for eid, state in self._states.items():
                if (eid in (conflict.state_a_id, conflict.state_b_id)
                        and state.source_instance == own_hash):
                    result.append(conflict)
                    break
        return result

    def purge_expired(self) -> int:
        """ลบ states ที่หมดอายุ คืนจำนวนที่ลบ"""
        before = len(self._states)
        self._states = {
            eid: s for eid, s in self._states.items()
            if not s.is_expired
        }
        removed = before - len(self._states)
        if removed:
            logger.info(f"[SCL] PURGED {removed} expired states")
        return removed

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        active = sum(1 for s in self._states.values() if not s.is_expired)
        return {
            "world_id":        self._world_id,
            "registered":      len(self._registered),
            "states_total":    len(self._states),
            "states_active":   active,
            "conflicts_total": len(self._conflicts),
        }