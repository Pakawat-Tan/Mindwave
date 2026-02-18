"""
Proposal — กลไกที่โมเดลใช้เสนอการเพิ่ม/แก้ไข/ลบ Rule และ Policy

Flow:
    Model/System  → submit_proposal()   → status=PENDING
    Reviewer      → approve() / reject()
        STANDARD rule + Policy  → reviewer_id พอ
        SYSTEM rule             → creator_id เท่านั้น

Framework guarantee:
    "การเปลี่ยนแปลงใด ๆ ต้องถูกระบุเป็น PROPOSAL ก่อนใช้งาน"
    ไม่มี proposal ที่ approved → ไม่มีการเปลี่ยน Rule/Policy
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import json
import uuid
import time


# ============================================================================
# ENUMS
# ============================================================================

class RuleAuthority(Enum):
    """
    ระดับอำนาจของ Rule

    SYSTEM   : กฎหมายระดับระบบ — approve ได้เฉพาะ creator_id
               เช่น ห้ามเปลี่ยน Identity, ห้ามข้าม Governance Hierarchy
    STANDARD : กฎทั่วไป — approve ได้ด้วย reviewer_id
               เช่น content filter, topic block
    """
    SYSTEM   = "system"
    STANDARD = "standard"

    def __str__(self): return self.value.upper()


class ProposalAction(Enum):
    ADD    = "add"
    MODIFY = "modify"
    REMOVE = "remove"

    def __str__(self): return self.value.upper()


class ProposalTarget(Enum):
    RULE   = "rule"
    POLICY = "policy"

    def __str__(self): return self.value.upper()


class ProposalStatus(Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

    def __str__(self): return self.value.upper()


# ============================================================================
# PROPOSAL DATA
# ============================================================================

@dataclass
class ProposalData:
    """
    คำเสนอการเปลี่ยนแปลง Rule หรือ Policy

    สร้างโดยโมเดล (เมื่อเรียนรู้และต้องการปรับ)
    หรือโดย user ที่มีสิทธิ์

    Attributes:
        proposal_id   : unique id
        proposed_by   : ผู้เสนอ (model name, user id, ระบบ)
        action        : ADD / MODIFY / REMOVE
        target_type   : RULE / POLICY
        authority     : SYSTEM / STANDARD (เฉพาะ RULE target)
        payload       : dict ของ Rule หรือ Policy ที่เสนอ
        reason        : เหตุผลที่เสนอ (จากโมเดล)
        status        : PENDING / APPROVED / REJECTED
        created_ts    : timestamp ที่สร้าง
        reviewed_by   : ผู้ approve/reject
        reviewed_ts   : timestamp ที่ review
        review_note   : หมายเหตุจาก reviewer
    """
    proposed_by:  str
    action:       ProposalAction
    target_type:  ProposalTarget
    payload:      Dict[str, Any]
    reason:       str
    authority:    RuleAuthority      = RuleAuthority.STANDARD
    proposal_id:  str                = field(default_factory=lambda: str(uuid.uuid4()))
    status:       ProposalStatus     = ProposalStatus.PENDING
    created_ts:   float              = field(default_factory=time.time)
    reviewed_by:  Optional[str]      = None
    reviewed_ts:  Optional[float]    = None
    review_note:  str                = ""

    def __post_init__(self) -> None:
        if not self.proposed_by or not self.proposed_by.strip():
            raise ValueError("ProposalData: proposed_by must not be empty")
        if not self.reason or not self.reason.strip():
            raise ValueError("ProposalData: reason must not be empty")
        if not self.payload:
            raise ValueError("ProposalData: payload must not be empty")
        # POLICY target ไม่มี authority concept
        if self.target_type == ProposalTarget.POLICY:
            object.__setattr__(self, 'authority', RuleAuthority.STANDARD) \
                if hasattr(self, '__dataclass_params__') and self.__dataclass_params__.frozen \
                else setattr(self, 'authority', RuleAuthority.STANDARD)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def is_pending(self)  -> bool: return self.status == ProposalStatus.PENDING
    @property
    def is_approved(self) -> bool: return self.status == ProposalStatus.APPROVED
    @property
    def is_rejected(self) -> bool: return self.status == ProposalStatus.REJECTED

    @property
    def requires_creator(self) -> bool:
        """True ถ้า proposal นี้ต้องการ creator_id ในการ approve"""
        return (
            self.target_type == ProposalTarget.RULE and
            self.authority   == RuleAuthority.SYSTEM
        )

    # ------------------------------------------------------------------
    # State transitions (เรียกโดย ConditionController เท่านั้น)
    # ------------------------------------------------------------------

    def _approve(self, reviewed_by: str, note: str = "") -> None:
        """Internal — เรียกโดย ConditionController"""
        self.status      = ProposalStatus.APPROVED
        self.reviewed_by = reviewed_by
        self.reviewed_ts = time.time()
        self.review_note = note

    def _reject(self, reviewed_by: str, note: str = "") -> None:
        """Internal — เรียกโดย ConditionController"""
        self.status      = ProposalStatus.REJECTED
        self.reviewed_by = reviewed_by
        self.reviewed_ts = time.time()
        self.review_note = note

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id":  self.proposal_id,
            "proposed_by":  self.proposed_by,
            "action":       self.action.value,
            "target_type":  self.target_type.value,
            "authority":    self.authority.value,
            "payload":      self.payload,
            "reason":       self.reason,
            "status":       self.status.value,
            "created_ts":   self.created_ts,
            "reviewed_by":  self.reviewed_by,
            "reviewed_ts":  self.reviewed_ts,
            "review_note":  self.review_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProposalData:
        p = cls(
            proposal_id  = data["proposal_id"],
            proposed_by  = data["proposed_by"],
            action       = ProposalAction(data["action"]),
            target_type  = ProposalTarget(data["target_type"]),
            authority    = RuleAuthority(data["authority"]),
            payload      = data["payload"],
            reason       = data["reason"],
            status       = ProposalStatus(data["status"]),
            created_ts   = data.get("created_ts", time.time()),
            reviewed_by  = data.get("reviewed_by"),
            reviewed_ts  = data.get("reviewed_ts"),
            review_note  = data.get("review_note", ""),
        )
        return p

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> ProposalData:
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        auth = f"|{self.authority}" if self.target_type == ProposalTarget.RULE else ""
        return (
            f"Proposal[{self.action}|{self.target_type}{auth}|{self.status}] "
            f"by='{self.proposed_by}' "
            f"id={self.proposal_id[:8]}"
        )


# ============================================================================
# FACTORY
# ============================================================================

def create_proposal(
    proposed_by: str,
    action:      ProposalAction,
    target_type: ProposalTarget,
    payload:     Dict[str, Any],
    reason:      str,
    authority:   RuleAuthority = RuleAuthority.STANDARD,
) -> ProposalData:
    """
    สร้าง ProposalData — ใช้โดยโมเดลหรือ user ที่ต้องการเสนอการเปลี่ยนแปลง
    """
    return ProposalData(
        proposed_by = proposed_by,
        action      = action,
        target_type = target_type,
        payload     = payload,
        reason      = reason,
        authority   = authority,
    )