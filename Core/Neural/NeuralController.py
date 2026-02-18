"""
จัดการ Neural weights — proposal-only, ไม่ approve เอง

ลำดับการทำงาน:
  1. monitor_gradient()   → ตรวจ gradient ก่อนเสมอ
  2. detect_conflict()    → หา knowledge conflict
  3. propose_update()     → เสนอผ่าน Proposal system (ไม่ apply เอง)
  4. apply_approved()     → รับ approved proposal → apply weight
  5. rollback()           → ย้อน weight กลับ

Rules (จาก NeuralEvolution.json — SYSTEM authority):
  - ห้าม uncontrolled_growth
  - ห้าม self_modify_core
  - หยุดเมื่อ gradient explode/NaN/Inf
  - ห้าม block rollback
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Optional, List, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from Core.Neural.Brain.NeuralData import (
    WeightData, GradientSnapshot, GradientStatus,
    ConflictData, ConflictType, EvolutionRecord, WeightUpdateSource
)
from Core.Review.Proposal import (
    ProposalData, ProposalAction, ProposalTarget,
    ProposalStatus, create_proposal, RuleAuthority
)


class NeuralController:

    def __init__(
        self,
        gradient_explode_threshold: float = 100.0,
        gradient_vanish_threshold:  float = 1e-7,
    ):
        self._weights:    Dict[str, WeightData]     = {}
        self._conflicts:  List[ConflictData]        = []
        self._evolutions: List[EvolutionRecord]     = []
        self._gradients:  List[GradientSnapshot]    = []
        self._proposals:  Dict[str, ProposalData]   = {}

        self._explode_threshold = gradient_explode_threshold
        self._vanish_threshold  = gradient_vanish_threshold
        self._logger = logging.getLogger("mindwave.neural")

    # ─────────────────────────────────────────────────────────────
    # Weight Registry
    # ─────────────────────────────────────────────────────────────

    def register_weight(
        self,
        domain:    str,
        value:     float = 0.5,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> WeightData:
        """ลงทะเบียน weight domain ใหม่"""
        if domain in self._weights:
            return self._weights[domain]

        w = WeightData(
            weight_id = str(uuid.uuid4())[:8],
            domain    = domain,
            value     = value,
            min_value = min_value,
            max_value = max_value,
        )
        self._weights[domain] = w
        self._logger.info(
            f"[NeuralController] REGISTER '{domain}' val={value:.6f}"
        )
        return w

    def get_weight(self, domain: str) -> Optional[WeightData]:
        return self._weights.get(domain)

    def use_weight(self, domain: str) -> Optional[float]:
        """
        ดึง weight value สำหรับใช้งาน — นับ usage ด้วย
        """
        w = self._weights.get(domain)
        if w is None:
            return None
        w.record_usage()
        return w.value

    def list_weights(self) -> List[WeightData]:
        return list(self._weights.values())

    # ─────────────────────────────────────────────────────────────
    # Gradient Monitoring
    # ─────────────────────────────────────────────────────────────

    def monitor_gradient(
        self, domain: str, gradient: float
    ) -> GradientSnapshot:
        """
        ตรวจ gradient — หยุดทันทีถ้า critical

        Returns:
            GradientSnapshot

        Raises:
            RuntimeError: ถ้า gradient เป็น NaN/Inf/Explode
                          (ตาม NeuralEvolution Rule)
        """
        snap = GradientSnapshot.evaluate(
            domain    = domain,
            gradient  = gradient,
            threshold_explode = self._explode_threshold,
            threshold_vanish  = self._vanish_threshold,
        )
        self._gradients.append(snap)

        if snap.status.is_critical:
            self._logger.error(
                f"[NeuralController] GRADIENT_CRITICAL {snap}"
            )
            raise RuntimeError(
                f"[NeuralController] gradient {snap.status} detected "
                f"in domain='{domain}' value={gradient} — "
                f"stopping as per NeuralEvolution Rule"
            )

        if snap.status == GradientStatus.VANISH:
            self._logger.warning(
                f"[NeuralController] GRADIENT_VANISH {snap}"
            )

        self._logger.debug(f"[NeuralController] GRADIENT {snap}")
        return snap

    def last_gradient(self, domain: str) -> Optional[GradientSnapshot]:
        """gradient snapshot ล่าสุดของ domain"""
        snaps = [g for g in self._gradients if g.domain == domain]
        return snaps[-1] if snaps else None

    # ─────────────────────────────────────────────────────────────
    # Conflict Detection
    # ─────────────────────────────────────────────────────────────

    def detect_conflict(
        self,
        domain:      str,
        description: str  = "",
        severity:    float = 0.5,
        conflict_type: ConflictType = ConflictType.KNOWLEDGE_GAP,
    ) -> ConflictData:
        """
        บันทึก conflict ที่ตรวจพบ

        หลังตรวจพบ → Neural ควร propose_update() เพื่อแก้ไข
        ไม่แก้เอง
        """
        conflict = ConflictData(
            conflict_type = conflict_type,
            domain        = domain,
            description   = description or f"conflict detected in {domain}",
            severity      = max(0.0, min(1.0, severity)),
        )
        self._conflicts.append(conflict)
        self._logger.warning(
            f"[NeuralController] CONFLICT_DETECTED {conflict}"
        )
        return conflict

    def open_conflicts(self) -> List[ConflictData]:
        return [c for c in self._conflicts if not c.resolved]

    def resolve_conflict(self, conflict_id: str) -> bool:
        """mark conflict เป็น resolved"""
        for i, c in enumerate(self._conflicts):
            if c.conflict_id == conflict_id:
                # frozen → ต้องสร้างใหม่
                resolved = ConflictData(
                    conflict_id   = c.conflict_id,
                    conflict_type = c.conflict_type,
                    domain        = c.domain,
                    description   = c.description,
                    severity      = c.severity,
                    timestamp     = c.timestamp,
                    resolved      = True,
                )
                self._conflicts[i] = resolved
                self._record_evolution(
                    event_type   = "conflict_resolved",
                    domain       = c.domain,
                    triggered_by = conflict_id,
                    description  = f"conflict {c.conflict_type} resolved",
                )
                return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Proposal Submission (proposal-only — ไม่ approve เอง)
    # ─────────────────────────────────────────────────────────────

    def propose_weight_update(
        self,
        domain:      str,
        new_value:   float,
        proposed_by: str,
        reason:      str = "",
    ) -> ProposalData:
        """
        เสนอการเปลี่ยน weight ผ่าน Proposal system

        Neural ไม่ apply เอง — ต้องรอ approve จาก reviewer
        """
        if domain not in self._weights:
            raise KeyError(
                f"[NeuralController] domain '{domain}' not registered"
            )
        w = self._weights[domain]
        if not (w.min_value <= new_value <= w.max_value):
            raise ValueError(
                f"[NeuralController] new_value {new_value} out of range "
                f"[{w.min_value}, {w.max_value}]"
            )

        proposal = create_proposal(
            proposed_by = proposed_by,
            action      = ProposalAction.MODIFY,
            target_type = ProposalTarget.RULE,   # ใช้ RULE target สำหรับ weight changes
            authority   = RuleAuthority.STANDARD,
            payload     = {
                "domain":    domain,
                "old_value": w.value,
                "new_value": new_value,
                "weight_id": w.weight_id,
            },
            reason = reason or f"neural proposes weight update: {domain}",
        )
        self._proposals[proposal.proposal_id] = proposal
        self._logger.info(
            f"[NeuralController] PROPOSAL_SUBMITTED {proposal.proposal_id[:8]} "
            f"domain='{domain}' {w.value:.6f} → {new_value:.6f} "
            f"by='{proposed_by}'"
        )
        return proposal

    def apply_approved_proposal(
        self,
        proposal_id: str,
        reviewer_id: str,
    ) -> Optional[EvolutionRecord]:
        """
        Apply weight จาก approved proposal

        เรียกโดย reviewer หลัง approve แล้ว
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[NeuralController] apply_approved_proposal requires reviewer_id"
            )
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        if not proposal.is_approved:
            raise ValueError(
                f"[NeuralController] proposal {proposal_id[:8]} "
                f"is not approved (status={proposal.status})"
            )

        payload = proposal.payload
        domain    = payload["domain"]
        new_value = payload["new_value"]

        w = self._weights.get(domain)
        if w is None:
            return None

        old_value = w.update(
            new_value,
            source = WeightUpdateSource.PROPOSAL_APPROVED.value
        )

        record = self._record_evolution(
            event_type   = "weight_update",
            domain       = domain,
            old_value    = old_value,
            new_value    = new_value,
            triggered_by = proposal_id,
            description  = f"approved by {reviewer_id}",
        )
        self._logger.warning(
            f"[NeuralController] WEIGHT_UPDATED '{domain}' "
            f"{old_value:.6f} → {new_value:.6f} "
            f"by='{reviewer_id}'"
        )
        return record

    def pending_proposals(self) -> List[ProposalData]:
        return [p for p in self._proposals.values() if p.is_pending]

    def get_proposal(self, proposal_id: str) -> Optional[ProposalData]:
        return self._proposals.get(proposal_id)

    # ─────────────────────────────────────────────────────────────
    # Rollback
    # ─────────────────────────────────────────────────────────────

    def rollback(
        self,
        domain:      str,
        reviewer_id: str,
        reason:      str = "",
    ) -> Optional[EvolutionRecord]:
        """
        ย้อน weight กลับ 1 step

        ต้องมี reviewer_id — ห้าม block rollback (NeuralEvolution Rule)
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[NeuralController] rollback requires reviewer_id"
            )
        w = self._weights.get(domain)
        if w is None:
            raise KeyError(
                f"[NeuralController] domain '{domain}' not found"
            )
        prev = w.previous_value
        if prev is None:
            self._logger.warning(
                f"[NeuralController] ROLLBACK '{domain}': no previous value"
            )
            return None

        current = w.value
        w.update(prev, source=WeightUpdateSource.ROLLBACK.value)

        record = self._record_evolution(
            event_type   = "rollback",
            domain       = domain,
            old_value    = current,
            new_value    = prev,
            triggered_by = reviewer_id,
            description  = reason or "rollback",
        )
        self._logger.warning(
            f"[NeuralController] ROLLBACK '{domain}' "
            f"{current:.6f} → {prev:.6f} "
            f"by='{reviewer_id}'"
        )
        return record

    # ─────────────────────────────────────────────────────────────
    # Evolution History
    # ─────────────────────────────────────────────────────────────

    def evolution_history(
        self,
        domain: Optional[str] = None,
        n:      int           = 20,
    ) -> List[EvolutionRecord]:
        history = self._evolutions
        if domain:
            history = [r for r in history if r.domain == domain]
        return list(history[-n:])

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        critical_grads = sum(
            1 for g in self._gradients if g.status.is_critical
        )
        return {
            "weights_total":     len(self._weights),
            "conflicts_open":    len(self.open_conflicts()),
            "conflicts_total":   len(self._conflicts),
            "proposals_pending": len(self.pending_proposals()),
            "evolutions_total":  len(self._evolutions),
            "gradients_checked": len(self._gradients),
            "gradients_critical": critical_grads,
            "total_usage":       sum(
                w.usage_count for w in self._weights.values()
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _record_evolution(
        self,
        event_type:   str,
        domain:       str,
        old_value:    Optional[float] = None,
        new_value:    Optional[float] = None,
        triggered_by: str             = "",
        description:  str             = "",
    ) -> EvolutionRecord:
        record = EvolutionRecord(
            event_type   = event_type,
            domain       = domain,
            old_value    = old_value,
            new_value    = new_value,
            triggered_by = triggered_by,
            description  = description,
        )
        self._evolutions.append(record)
        return record