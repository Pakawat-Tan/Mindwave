"""
Orchestrator — เหมือน MemoryController ที่เรียก Tiers

เรียกผ่าน:
    self._rule   → RuleController
    self._policy → PolicyController

Layer order (ไม่เปลี่ยน):
    Rule Layer   → ตรวจก่อน — BLOCK = หยุดทันที
        ↓ ผ่านแล้ว
    Policy Layer → ปรับ behavior
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict

import sys, os
from Core.Condition.Rule.RuleData    import RuleData, RuleScope, RuleAuthority, RuleResult
from Core.Condition.Policy.PolicyData  import NumericPolicy, BehavioralPolicy, PolicyScope, PolicyResult, AnyPolicy
from Core.Review.Proposal import ProposalData, ProposalStatus, ProposalTarget

from .Rule.RuleController     import RuleController
from .Policy.PolicyController import PolicyController


class ConditionController:

    def __init__(self, base_path: str = "Core/Condition/Data"):
        self._base   = Path(base_path)
        self._logger = logging.getLogger("mindwave.condition")

        self._rule   = RuleController  (str(self._base / "rule"))
        self._policy = PolicyController(str(self._base / "policy"))

        self._logger.info("[ConditionController] initialized")

    # ═══════════════════════════════════════════════════════════════
    # RULE LAYER
    # ═══════════════════════════════════════════════════════════════

    # ── Governance ────────────────────────────────────────────────

    def governance_add_rule(
        self,
        rule:        RuleData,
        reviewer_id: Optional[str] = None,
        creator_id:  Optional[str] = None,
    ) -> None:
        self._rule.governance_add(rule, reviewer_id=reviewer_id, creator_id=creator_id)

    def governance_remove_rule(
        self,
        rule_id:     str,
        reviewer_id: Optional[str] = None,
        creator_id:  Optional[str] = None,
    ) -> bool:
        return self._rule.governance_remove(rule_id, reviewer_id=reviewer_id, creator_id=creator_id)

    def get_rule(self, rule_id: str) -> Optional[RuleData]:
        return self._rule.get(rule_id)

    def list_rules(self, scope: Optional[RuleScope] = None) -> List[RuleData]:
        return self._rule.list(scope=scope)

    # ── Check API ─────────────────────────────────────────────────

    def check_input(self,  text="", cluster_id=None) -> RuleResult:
        return self._rule.check(RuleScope.INPUT,  text, cluster_id)

    def check_output(self, text="", cluster_id=None) -> RuleResult:
        return self._rule.check(RuleScope.OUTPUT, text, cluster_id)

    def is_input_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ INPUT — BrainController ใช้ก่อนรับ input"""
        return self._gate(RuleScope.INPUT, text, cluster_id)

    def is_output_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ OUTPUT — ก่อนส่ง response ออก"""
        return self._gate(RuleScope.OUTPUT, text, cluster_id)

    def is_memory_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ Memory — ก่อน recall/write"""
        return self._gate(RuleScope.MEMORY, text, cluster_id)

    def is_skill_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ Skill — ก่อน arbitrate/grow"""
        return self._gate(RuleScope.SKILL, text, cluster_id)

    def is_neural_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ Neural — ก่อน observe/evolve"""
        return self._gate(RuleScope.NEURAL, text, cluster_id)

    def is_personality_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ Personality — ก่อน apply profile"""
        return self._gate(RuleScope.PERSONALITY, text, cluster_id)

    def is_confidence_allowed(self, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Gate สำหรับ Confidence — ก่อน evaluate"""
        return self._gate(RuleScope.CONFIDENCE, text, cluster_id)

    def _gate(self, scope: RuleScope, text: str = "", cluster_id=None) -> tuple[bool, str]:
        """Internal gate — ใช้ร่วมกันทุก is_X_allowed()"""
        result  = self._rule.check(scope, text, cluster_id)
        # ถ้าไม่ match text/topic ลอง check scope-level block (ไม่ส่ง text)
        if not result.is_blocked and text == "" and cluster_id is None:
            result = self._rule.check(scope, "", None)
        allowed = not result.is_blocked
        reason  = (result.triggered_rule.rule_id if result.triggered_rule else "")
        return allowed, reason

    def check_memory(self, text="", cluster_id=None) -> RuleResult:
        return self._rule.check(RuleScope.MEMORY, text, cluster_id)

    def check_system(self, text="", cluster_id=None) -> RuleResult:
        return self._rule.check(RuleScope.SYSTEM, text, cluster_id)

    # ── Rule Proposals ────────────────────────────────────────────

    def submit_rule_proposal(self, proposal: ProposalData) -> str:
        return self._rule.submit_proposal(proposal)

    def approve_rule_proposal(
        self,
        proposal_id: str,
        reviewer_id: Optional[str] = None,
        creator_id:  Optional[str] = None,
        note:        str           = "",
    ) -> bool:
        return self._rule.approve_proposal(
            proposal_id, reviewer_id=reviewer_id,
            creator_id=creator_id, note=note
        )

    def reject_rule_proposal(
        self, proposal_id: str, reviewer_id: str, note: str = ""
    ) -> bool:
        return self._rule.reject_proposal(proposal_id, reviewer_id, note)

    # ── Default Rules loader ──────────────────────────────────────

    def load_default_rules(
        self,
        creator_id:   str,
        reviewer_id:  str           = "default_system",
        defaults_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """โหลด Default Rules จาก JSON files ทั้งหมด"""
        d = defaults_dir or str(Path(__file__).parent / "Rule" / "Defaults")
        return self._rule.load_defaults(creator_id, reviewer_id, d)

    # ═══════════════════════════════════════════════════════════════
    # POLICY LAYER
    # ═══════════════════════════════════════════════════════════════

    # ── Direct management ─────────────────────────────────────────

    def add_policy(self, policy: AnyPolicy) -> None:
        self._policy.add(policy)

    def remove_policy(self, policy_id: str) -> bool:
        return self._policy.remove(policy_id)

    def get_policy(self, policy_id: str) -> Optional[AnyPolicy]:
        return self._policy.get(policy_id)

    def list_policies(
        self,
        scope:       Optional[PolicyScope] = None,
        active_only: bool                  = True,
    ) -> List[AnyPolicy]:
        return self._policy.list(scope=scope, active_only=active_only)

    # ── Numeric Policy ────────────────────────────────────────────

    def get_numeric_value(
        self,
        scope:   PolicyScope,
        key:     str,
        default: Optional[float] = None,
    ) -> Optional[float]:
        return self._policy.get_numeric(scope, key, default)

    def get_system_numeric_policies(self) -> List[NumericPolicy]:
        return self._policy.get_system_numeric()

    # ── Behavioral Policy ─────────────────────────────────────────

    def get_behavior_modifier(
        self,
        scope:      PolicyScope,
        text:       str           = "",
        cluster_id: Optional[int] = None,
    ) -> float:
        return self._policy.get_modifier(scope, text, cluster_id)

    def get_memory_modifier(
        self,
        cluster_id: Optional[int] = None,
        text:       str           = "",
    ) -> float:
        return self._policy.get_modifier(PolicyScope.MEMORY, text, cluster_id)

    def get_policy_result(
        self,
        scope:      PolicyScope,
        text:       str           = "",
        cluster_id: Optional[int] = None,
    ) -> PolicyResult:
        return self._policy.get_result(scope, text, cluster_id)

    # ── Policy Proposals (model → propose) ───────────────────────

    def submit_policy_proposal(self, proposal: ProposalData) -> str:
        return self._policy.submit_proposal(proposal)

    def approve_policy_proposal(
        self,
        proposal_id: str,
        reviewer_id: str,
        note:        str = "",
    ) -> bool:
        return self._policy.approve_proposal(proposal_id, reviewer_id, note)

    def reject_policy_proposal(
        self, proposal_id: str, reviewer_id: str, note: str = ""
    ) -> bool:
        return self._policy.reject_proposal(proposal_id, reviewer_id, note)

    # ═══════════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════════

    def save(self) -> None:
        self._rule.save()
        self._policy.save()
        self._logger.info("[ConditionController] saved")

    # ═══════════════════════════════════════════════════════════════
    # Stats
    # ═══════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        r = self._rule.stats()
        p = self._policy.stats()
        return {**r, **p}