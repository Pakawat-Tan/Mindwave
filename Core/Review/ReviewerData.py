"""
Data structures สำหรับ Reviewer module

ReviewDecision  — บันทึกการตัดสินใจแต่ละครั้ง (approve/reject)
AuditEvent      — audit trail ทุก action ที่ reviewer ทำ
RollbackRecord  — บันทึกการ rollback
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time
import uuid


class ReviewAction(Enum):
    APPROVE  = "approve"
    REJECT   = "reject"
    ROLLBACK = "rollback"
    QUEUE    = "queue"
    DEQUEUE  = "dequeue"

    def __str__(self): return self.value.upper()


class ReviewerRole(Enum):
    """
    ระดับของ reviewer

    STANDARD : approve STANDARD proposals เท่านั้น
    CREATOR  : approve ได้ทุกอย่าง รวม SYSTEM authority
    """
    STANDARD = "standard"
    CREATOR  = "creator"

    def __str__(self): return self.value.upper()


# ============================================================================
# REVIEW DECISION
# ============================================================================

@dataclass(frozen=True)
class ReviewDecision:
    """
    บันทึกการตัดสินใจ approve/reject Proposal หนึ่งรายการ
    """
    decision_id:  str   = field(default_factory=lambda: str(uuid.uuid4()))
    proposal_id:  str   = ""
    action:       ReviewAction = ReviewAction.APPROVE
    reviewer_id:  str   = ""
    reviewer_role: ReviewerRole = ReviewerRole.STANDARD
    note:         str   = ""
    timestamp:    float = field(default_factory=time.time)
    # snapshot ของ proposal payload ตอน decide (สำหรับ rollback)
    payload_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id":   self.decision_id,
            "proposal_id":   self.proposal_id,
            "action":        self.action.value,
            "reviewer_id":   self.reviewer_id,
            "reviewer_role": self.reviewer_role.value,
            "note":          self.note,
            "timestamp":     self.timestamp,
        }

    def __str__(self) -> str:
        return (
            f"Decision[{self.action}] "
            f"proposal={self.proposal_id[:8]} "
            f"by='{self.reviewer_id}' "
            f"note='{self.note}'"
        )


# ============================================================================
# AUDIT EVENT
# ============================================================================

@dataclass(frozen=True)
class AuditEvent:
    """
    Audit trail — บันทึกทุก action ที่ reviewer ทำ

    Audit-ready 100% ตาม Phase 3 spec
    """
    event_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    action:      ReviewAction = ReviewAction.APPROVE
    reviewer_id: str   = ""
    target_id:   str   = ""        # proposal_id หรือ decision_id
    target_type: str   = ""        # "proposal", "rollback"
    detail:      str   = ""
    success:     bool  = True
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":    self.event_id,
            "action":      self.action.value,
            "reviewer_id": self.reviewer_id,
            "target_id":   self.target_id,
            "target_type": self.target_type,
            "detail":      self.detail,
            "success":     self.success,
            "timestamp":   self.timestamp,
        }

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"Audit[{self.action}]{status} "
            f"by='{self.reviewer_id}' "
            f"target={self.target_id[:8]} "
            f"| {self.detail}"
        )


# ============================================================================
# ROLLBACK RECORD
# ============================================================================

@dataclass(frozen=True)
class RollbackRecord:
    """
    บันทึกการ rollback — ย้อน approved proposal

    เก็บ decision_id ที่ถูก rollback เพื่อ trace กลับได้
    """
    rollback_id:  str   = field(default_factory=lambda: str(uuid.uuid4()))
    decision_id:  str   = ""     # decision ที่ถูก rollback
    proposal_id:  str   = ""
    rolled_back_by: str = ""
    reason:       str   = ""
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id":    self.rollback_id,
            "decision_id":    self.decision_id,
            "proposal_id":    self.proposal_id,
            "rolled_back_by": self.rolled_back_by,
            "reason":         self.reason,
            "timestamp":      self.timestamp,
        }

    def __str__(self) -> str:
        return (
            f"Rollback[{self.rollback_id[:8]}] "
            f"decision={self.decision_id[:8]} "
            f"by='{self.rolled_back_by}' "
            f"reason='{self.reason}'"
        )