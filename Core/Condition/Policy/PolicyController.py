"""
จัดการ Policy ทั้งหมด — add/remove, model proposals, evaluation

Policy เกิดจาก 2 แหล่ง:
  1. ระบบ/admin เพิ่มโดยตรงผ่าน add()
  2. โมเดลเรียนรู้แล้วเสนอผ่าน submit_proposal() → รอ approve

API:
  add(policy) / remove(policy_id)
  get_numeric(scope, key, default)   → float
  get_modifier(scope, text, cluster_id) → float
  submit_proposal(proposal)
  approve_proposal(proposal_id, reviewer_id)
  reject_proposal(proposal_id, reviewer_id)
  save() / load()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from Core.Condition.Policy.PolicyData import (
    NumericPolicy, BehavioralPolicy, PolicyScope,
    PolicyResult, AnyPolicy, policy_from_dict,
    NEUTRAL_POLICY_RESULT, MatchType
)
from ..Proposal import (
    ProposalData, ProposalAction, ProposalTarget, ProposalStatus
)


class PolicyController:

    def __init__(self, data_path: str):
        self._path    = Path(data_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._logger  = logging.getLogger("mindwave.condition.policy")
        self._policies:  Dict[str, AnyPolicy]    = {}
        self._proposals: Dict[str, ProposalData] = {}
        self.load()

    # ─────────────────────────────────────────────────────────────
    # Policy management (direct)
    # ─────────────────────────────────────────────────────────────

    def add(self, policy: AnyPolicy) -> None:
        self._policies[policy.policy_id] = policy
        self._logger.info(
            f"[PolicyController] ADD {policy.policy_id[:8]} {policy}"
        )

    def remove(self, policy_id: str) -> bool:
        if policy_id in self._policies:
            del self._policies[policy_id]
            self._logger.info(
                f"[PolicyController] REMOVE {policy_id[:8]}"
            )
            return True
        return False

    def get(self, policy_id: str) -> Optional[AnyPolicy]:
        return self._policies.get(policy_id)

    def list(
        self,
        scope:       Optional[PolicyScope] = None,
        active_only: bool                  = True,
    ) -> List[AnyPolicy]:
        policies = list(self._policies.values())
        if scope is not None:
            policies = [p for p in policies if p.scope == scope]
        if active_only:
            policies = [p for p in policies if p.is_active]
        return policies

    # ─────────────────────────────────────────────────────────────
    # Policy evaluation
    # ─────────────────────────────────────────────────────────────

    def get_numeric(
        self,
        scope:   PolicyScope,
        key:     str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        """ดึงค่า NumericPolicy ตาม scope + key"""
        for p in self.list(scope=scope):
            if isinstance(p, NumericPolicy) and p.key == key:
                return p.value
        return default

    def get_modifier(
        self,
        scope:      PolicyScope,
        text:       str           = "",
        cluster_id: Optional[int] = None,
    ) -> float:
        """คืน final weight modifier จาก BehavioralPolicy ที่ match"""
        return self._evaluate_behavioral(scope, text, cluster_id).final_weight

    def get_result(
        self,
        scope:      PolicyScope,
        text:       str           = "",
        cluster_id: Optional[int] = None,
    ) -> PolicyResult:
        """คืน PolicyResult เต็ม"""
        return self._evaluate_behavioral(scope, text, cluster_id)

    def get_system_numeric(self) -> List[NumericPolicy]:
        return [
            p for p in self.list(scope=PolicyScope.SYSTEM)
            if isinstance(p, NumericPolicy)
        ]

    # ─────────────────────────────────────────────────────────────
    # Proposal system (model → propose Policy)
    # ─────────────────────────────────────────────────────────────

    def submit_proposal(self, proposal: ProposalData) -> str:
        """
        โมเดลเสนอเพิ่ม/แก้/ลบ Policy
        ต้องเป็น POLICY target เท่านั้น
        """
        if proposal.target_type != ProposalTarget.POLICY:
            raise ValueError(
                f"[PolicyController] proposal target must be POLICY, "
                f"got {proposal.target_type}"
            )
        if not proposal.is_pending:
            raise ValueError(
                f"[PolicyController] proposal must be PENDING, "
                f"got {proposal.status}"
            )
        self._proposals[proposal.proposal_id] = proposal
        self._logger.info(
            f"[PolicyController] PROPOSAL_SUBMITTED {proposal.proposal_id[:8]} "
            f"{proposal.action} by='{proposal.proposed_by}'"
        )
        return proposal.proposal_id

    def approve_proposal(
        self,
        proposal_id: str,
        reviewer_id: str,
        note:        str = "",
    ) -> bool:
        """
        Policy proposals ทั้งหมด approve ด้วย reviewer_id เท่านั้น
        (ไม่มี SYSTEM authority สำหรับ Policy)
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[PolicyController] approve_proposal requires reviewer_id"
            )
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        if not proposal.is_pending:
            raise ValueError(
                f"[PolicyController] proposal {proposal_id[:8]} "
                f"is already {proposal.status}"
            )
        self._apply_proposal(proposal)
        proposal._approve(reviewer_id, note)
        self._logger.info(
            f"[PolicyController] PROPOSAL_APPROVED {proposal_id[:8]} "
            f"by='{reviewer_id}'"
        )
        return True

    def reject_proposal(
        self,
        proposal_id: str,
        reviewer_id: str,
        note:        str = "",
    ) -> bool:
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[PolicyController] reject_proposal requires reviewer_id"
            )
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        if not proposal.is_pending:
            raise ValueError(
                f"[PolicyController] proposal {proposal_id[:8]} "
                f"is already {proposal.status}"
            )
        proposal._reject(reviewer_id, note)
        self._logger.info(
            f"[PolicyController] PROPOSAL_REJECTED {proposal_id[:8]} "
            f"by='{reviewer_id}'"
        )
        return True

    def get_proposal(self, proposal_id: str) -> Optional[ProposalData]:
        return self._proposals.get(proposal_id)

    def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
    ) -> List[ProposalData]:
        proposals = list(self._proposals.values())
        if status is not None:
            proposals = [p for p in proposals if p.status == status]
        return sorted(proposals, key=lambda p: p.created_ts, reverse=True)

    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────

    def save(self) -> None:
        (self._path / "policies.json").write_text(
            json.dumps([p.to_dict() for p in self._policies.values()], indent=2),
            encoding="utf-8"
        )
        (self._path / "policy_proposals.json").write_text(
            json.dumps([p.to_dict() for p in self._proposals.values()], indent=2),
            encoding="utf-8"
        )

    def load(self) -> None:
        policies_path  = self._path / "policies.json"
        proposals_path = self._path / "policy_proposals.json"
        if policies_path.exists():
            for d in json.loads(policies_path.read_text(encoding="utf-8")):
                p = policy_from_dict(d)
                self._policies[p.policy_id] = p
        if proposals_path.exists():
            for d in json.loads(proposals_path.read_text(encoding="utf-8")):
                p = ProposalData.from_dict(d)
                self._proposals[p.proposal_id] = p

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "policies_numeric":    sum(1 for p in self._policies.values() if isinstance(p, NumericPolicy)),
            "policies_behavioral": sum(1 for p in self._policies.values() if isinstance(p, BehavioralPolicy)),
            "policies_total":      len(self._policies),
            "proposals_pending":   sum(1 for p in self._proposals.values() if p.is_pending),
        }

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _apply_proposal(self, proposal: ProposalData) -> None:
        if proposal.action == ProposalAction.ADD:
            policy = policy_from_dict(proposal.payload)
            self._policies[policy.policy_id] = policy
        elif proposal.action == ProposalAction.REMOVE:
            self._policies.pop(proposal.payload.get("policy_id", ""), None)
        elif proposal.action == ProposalAction.MODIFY:
            policy = policy_from_dict(proposal.payload)
            self._policies[policy.policy_id] = policy

    def _evaluate_behavioral(
        self,
        scope:      PolicyScope,
        text:       str,
        cluster_id: Optional[int],
    ) -> PolicyResult:
        matched = sorted(
            [
                p for p in self.list(scope=scope)
                if isinstance(p, BehavioralPolicy) and
                   p.matches(text=text, cluster_id=cluster_id)
            ],
            key=lambda p: p.priority, reverse=True
        )
        if not matched:
            return NEUTRAL_POLICY_RESULT
        final_weight = 1.0
        for p in matched:
            final_weight *= p.weight
        return PolicyResult(
            final_weight     = final_weight,
            applied_policies = tuple(matched),
            reason           = ", ".join(
                p.description or p.policy_id[:8] for p in matched
            ),
        )