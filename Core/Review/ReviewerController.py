"""
Reviewer — gate สำหรับทุก Proposal ในระบบ

ครอบคลุม Proposals จากทุก module:
  Rule, Policy, Neural, Brain (Continuous Learning)

Rules:
  - ต้องมี reviewer_id เสมอ → PermissionError ถ้าว่าง
  - SYSTEM authority → ต้องเป็น CREATOR role
  - STANDARD authority → STANDARD หรือ CREATOR approve ได้
  - rollback ได้เฉพาะ decision ที่ approve ไปแล้ว
  - ทุก action → AuditEvent

Phase 3 spec:
  approve/reject/rollback + audit trail + permission check + queue
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from Core.Review.Proposal import (
    ProposalData, ProposalStatus, RuleAuthority
)
from Core.Review.ReviewerData import (
    ReviewDecision, AuditEvent, RollbackRecord,
    ReviewAction, ReviewerRole
)


class ReviewerController:

    def __init__(self):
        self._reviewers:  Dict[str, ReviewerRole]  = {}
        self._queue:      List[ProposalData]        = []   # pending proposals
        self._decisions:  Dict[str, ReviewDecision] = {}   # decision_id → decision
        self._rollbacks:  List[RollbackRecord]      = []
        self._audit:      List[AuditEvent]          = []
        self._logger = logging.getLogger("mindwave.reviewer")

    # ─────────────────────────────────────────────────────────────
    # Reviewer Registry
    # ─────────────────────────────────────────────────────────────

    def register_reviewer(
        self,
        reviewer_id: str,
        role:        ReviewerRole = ReviewerRole.STANDARD,
    ) -> None:
        """ลงทะเบียน reviewer"""
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("reviewer_id must not be empty")
        self._reviewers[reviewer_id] = role
        self._logger.info(
            f"[ReviewerController] REGISTERED '{reviewer_id}' role={role}"
        )

    def get_role(self, reviewer_id: str) -> Optional[ReviewerRole]:
        return self._reviewers.get(reviewer_id)

    def is_registered(self, reviewer_id: str) -> bool:
        return reviewer_id in self._reviewers

    # ─────────────────────────────────────────────────────────────
    # Queue Management
    # ─────────────────────────────────────────────────────────────

    def enqueue(self, proposal: ProposalData) -> None:
        """เพิ่ม proposal เข้า queue"""
        if not proposal.is_pending:
            raise ValueError(
                f"[ReviewerController] enqueue: "
                f"proposal {proposal.proposal_id[:8]} "
                f"is not pending (status={proposal.status})"
            )
        # ไม่ enqueue ซ้ำ
        if any(p.proposal_id == proposal.proposal_id for p in self._queue):
            return
        self._queue.append(proposal)
        self._record_audit(
            action      = ReviewAction.QUEUE,
            reviewer_id = "system",
            target_id   = proposal.proposal_id,
            target_type = "proposal",
            detail      = f"enqueued proposal from '{proposal.proposed_by}'",
        )

    def dequeue(self, proposal_id: str) -> Optional[ProposalData]:
        """เอา proposal ออกจาก queue"""
        for i, p in enumerate(self._queue):
            if p.proposal_id == proposal_id:
                self._queue.pop(i)
                self._record_audit(
                    action      = ReviewAction.DEQUEUE,
                    reviewer_id = "system",
                    target_id   = proposal_id,
                    target_type = "proposal",
                    detail      = "dequeued",
                )
                return p
        return None

    @property
    def queue(self) -> List[ProposalData]:
        return list(self._queue)

    def queue_size(self) -> int:
        return len(self._queue)

    def pending_by_authority(self, authority: RuleAuthority) -> List[ProposalData]:
        """กรอง queue ตาม authority"""
        return [p for p in self._queue if p.authority == authority]

    # ─────────────────────────────────────────────────────────────
    # Permission Check
    # ─────────────────────────────────────────────────────────────

    def _check_permission(
        self,
        reviewer_id: str,
        proposal:    ProposalData,
    ) -> ReviewerRole:
        """
        ตรวจสิทธิ์ reviewer

        Rules:
          - reviewer_id ต้องไม่ว่าง
          - ต้อง registered
          - SYSTEM authority → ต้องเป็น CREATOR
          - STANDARD authority → STANDARD หรือ CREATOR

        Returns:
            ReviewerRole ของ reviewer

        Raises:
            PermissionError ถ้าไม่มีสิทธิ์
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[ReviewerController] reviewer_id must not be empty"
            )
        if reviewer_id not in self._reviewers:
            raise PermissionError(
                f"[ReviewerController] '{reviewer_id}' is not a registered reviewer"
            )
        role = self._reviewers[reviewer_id]

        if proposal.authority == RuleAuthority.SYSTEM:
            if role != ReviewerRole.CREATOR:
                raise PermissionError(
                    f"[ReviewerController] SYSTEM authority requires CREATOR role, "
                    f"'{reviewer_id}' is {role}"
                )
        return role

    # ─────────────────────────────────────────────────────────────
    # Approve
    # ─────────────────────────────────────────────────────────────

    def approve(
        self,
        proposal:    ProposalData,
        reviewer_id: str,
        note:        str = "",
    ) -> ReviewDecision:
        """
        Approve proposal

        ตรวจสิทธิ์ → approve ProposalData → บันทึก decision → audit

        Returns:
            ReviewDecision
        """
        role = self._check_permission(reviewer_id, proposal)

        if not proposal.is_pending:
            raise ValueError(
                f"[ReviewerController] proposal {proposal.proposal_id[:8]} "
                f"is not pending (status={proposal.status})"
            )

        # approve ใน ProposalData
        proposal._approve(reviewer_id, note)

        decision = ReviewDecision(
            proposal_id      = proposal.proposal_id,
            action           = ReviewAction.APPROVE,
            reviewer_id      = reviewer_id,
            reviewer_role    = role,
            note             = note,
            payload_snapshot = dict(proposal.payload),
        )
        self._decisions[decision.decision_id] = decision

        # เอาออกจาก queue
        self.dequeue(proposal.proposal_id)

        self._record_audit(
            action      = ReviewAction.APPROVE,
            reviewer_id = reviewer_id,
            target_id   = proposal.proposal_id,
            target_type = "proposal",
            detail      = f"approved by '{reviewer_id}' | {note}",
            success     = True,
        )
        self._logger.warning(
            f"[ReviewerController] APPROVED {proposal.proposal_id[:8]} "
            f"by='{reviewer_id}'"
        )
        return decision

    # ─────────────────────────────────────────────────────────────
    # Reject
    # ─────────────────────────────────────────────────────────────

    def reject(
        self,
        proposal:    ProposalData,
        reviewer_id: str,
        note:        str = "",
    ) -> ReviewDecision:
        """
        Reject proposal — ต้องมี note/เหตุผล

        Returns:
            ReviewDecision
        """
        role = self._check_permission(reviewer_id, proposal)

        if not proposal.is_pending:
            raise ValueError(
                f"[ReviewerController] proposal {proposal.proposal_id[:8]} "
                f"is not pending"
            )

        proposal._reject(reviewer_id, note)

        decision = ReviewDecision(
            proposal_id      = proposal.proposal_id,
            action           = ReviewAction.REJECT,
            reviewer_id      = reviewer_id,
            reviewer_role    = role,
            note             = note,
            payload_snapshot = dict(proposal.payload),
        )
        self._decisions[decision.decision_id] = decision

        self.dequeue(proposal.proposal_id)

        self._record_audit(
            action      = ReviewAction.REJECT,
            reviewer_id = reviewer_id,
            target_id   = proposal.proposal_id,
            target_type = "proposal",
            detail      = f"rejected by '{reviewer_id}' | {note}",
            success     = True,
        )
        self._logger.warning(
            f"[ReviewerController] REJECTED {proposal.proposal_id[:8]} "
            f"by='{reviewer_id}' reason='{note}'"
        )
        return decision

    # ─────────────────────────────────────────────────────────────
    # Rollback
    # ─────────────────────────────────────────────────────────────

    def rollback(
        self,
        decision_id: str,
        reviewer_id: str,
        reason:      str = "",
    ) -> RollbackRecord:
        """
        ย้อน decision ที่ approve ไปแล้ว

        ใช้ payload_snapshot เพื่อ trace กลับว่า apply อะไรไป
        ต้องมี reviewer_id ที่ registered

        Returns:
            RollbackRecord

        Note:
            Rollback บันทึกเจตนา — การ undo จริงทำโดย Controller
            ที่รับผิดชอบ (NeuralController.rollback, RuleController etc.)
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[ReviewerController] rollback requires reviewer_id"
            )
        if reviewer_id not in self._reviewers:
            raise PermissionError(
                f"[ReviewerController] '{reviewer_id}' is not registered"
            )

        decision = self._decisions.get(decision_id)
        if decision is None:
            raise KeyError(
                f"[ReviewerController] decision '{decision_id[:8]}' not found"
            )
        if decision.action != ReviewAction.APPROVE:
            raise ValueError(
                f"[ReviewerController] can only rollback APPROVE decisions, "
                f"got {decision.action}"
            )

        record = RollbackRecord(
            decision_id    = decision_id,
            proposal_id    = decision.proposal_id,
            rolled_back_by = reviewer_id,
            reason         = reason or "rollback requested",
        )
        self._rollbacks.append(record)

        self._record_audit(
            action      = ReviewAction.ROLLBACK,
            reviewer_id = reviewer_id,
            target_id   = decision_id,
            target_type = "rollback",
            detail      = f"rollback decision {decision_id[:8]} | {reason}",
            success     = True,
        )
        self._logger.warning(
            f"[ReviewerController] ROLLBACK decision={decision_id[:8]} "
            f"proposal={decision.proposal_id[:8]} "
            f"by='{reviewer_id}'"
        )
        return record

    def get_rollback_snapshot(self, decision_id: str) -> Optional[dict]:
        """คืน payload snapshot ของ decision — ใช้ตอน undo จริง"""
        d = self._decisions.get(decision_id)
        return dict(d.payload_snapshot) if d else None

    # ─────────────────────────────────────────────────────────────
    # Audit
    # ─────────────────────────────────────────────────────────────

    def audit_log(self, n: int = 50) -> List[AuditEvent]:
        """คืน n รายการล่าสุดของ audit trail"""
        return list(self._audit[-n:])

    def audit_by_reviewer(self, reviewer_id: str) -> List[AuditEvent]:
        return [e for e in self._audit if e.reviewer_id == reviewer_id]

    def audit_by_proposal(self, proposal_id: str) -> List[AuditEvent]:
        return [e for e in self._audit if e.target_id == proposal_id]

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        approvals = sum(
            1 for d in self._decisions.values()
            if d.action == ReviewAction.APPROVE
        )
        rejections = sum(
            1 for d in self._decisions.values()
            if d.action == ReviewAction.REJECT
        )
        return {
            "reviewers_registered": len(self._reviewers),
            "queue_size":           len(self._queue),
            "decisions_total":      len(self._decisions),
            "approvals":            approvals,
            "rejections":           rejections,
            "rollbacks":            len(self._rollbacks),
            "audit_events":         len(self._audit),
        }

    # ─────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────

    def _record_audit(
        self,
        action:      ReviewAction,
        reviewer_id: str,
        target_id:   str,
        target_type: str,
        detail:      str  = "",
        success:     bool = True,
    ) -> AuditEvent:
        event = AuditEvent(
            action      = action,
            reviewer_id = reviewer_id,
            target_id   = target_id,
            target_type = target_type,
            detail      = detail,
            success     = success,
        )
        self._audit.append(event)
        return event