"""
SandboxController — Isolated Brain สำหรับทดลองโดยไม่กระทบ production

Features:
  1. isolated Brain   — แยก state จาก production Brain
  2. simulation       — dry-run respond() ไม่บันทึก Memory/Log จริง
  3. safety check     — ตรวจ input อันตรายก่อนส่งเข้า Brain
  4. rule testing     — ทดสอบ Rule/Policy ใหม่ใน isolated environment
  5. replay           — เล่นซ้ำ BrainLog เพื่อ debug
  6. promote          — ส่ง SandboxAtom ผ่าน Reviewer → production
  7. SCL              — แลกเปลี่ยน ExperimentState กับ instances อื่น

Architecture:
  1 SandboxController = 1 Mindwave Instance (isolated)
  Memory แยกจากกันโดยสมบูรณ์
  Neural adaptation อิสระ, Emotion influence ได้
  ทุกการเรียนรู้บันทึกเป็น SandboxAtom (ไม่ใช่ production atom)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Core.BrainController import BrainController, BrainLog
from Core.Condition.ConditionController import ConditionController
from Core.Review.ReviewerController import ReviewerController
from Core.Review.ReviewerData import ReviewerRole
from Core.Review.Proposal import create_proposal, ProposalAction, ProposalTarget, RuleAuthority
from Core.Sandbox.SandboxData import (
    SandboxAtom, SandboxStatus, AtomType,
    ExperimentState, SandboxWorld,
)
from Core.Sandbox.SCL import SCL, ConflictRecord

logger = logging.getLogger("mindwave.sandbox")


# ─────────────────────────────────────────────────────────────────────────────
# SimulationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimulationResult:
    """ผลของ dry-run — ไม่บันทึก Memory/Log จริง"""
    input_text:    str
    context:       str
    outcome:       str
    confidence:    float
    personality:   str
    skill_weight:  float
    is_safe:       bool
    safety_reason: str   = ""
    simulated_at:  float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_text":   self.input_text[:100],
            "context":      self.context,
            "outcome":      self.outcome,
            "confidence":   self.confidence,
            "is_safe":      self.is_safe,
            "safety_reason": self.safety_reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SandboxController
# ─────────────────────────────────────────────────────────────────────────────

class SandboxController:

    def __init__(
        self,
        instance_id:  str                    = "",
        world:        Optional[SandboxWorld] = None,
        scl:          Optional[SCL]          = None,
        brain:        Optional[BrainController] = None,
        reviewer:     Optional[ReviewerController] = None,
    ):
        self._instance_id = instance_id or str(uuid.uuid4())[:8]
        self._world       = world
        self._scl         = scl

        # isolated Brain — แยกจาก production
        self._brain       = brain or BrainController()

        # Reviewer สำหรับ promote sandbox atoms → production
        self._reviewer    = reviewer or ReviewerController()

        # Sandbox state
        self._atoms:       List[SandboxAtom]     = []
        self._sim_results: List[SimulationResult] = []
        self._replay_logs: List[Dict[str, Any]]  = []
        self._active       = True

        # register เข้า World และ SCL
        if self._world:
            self._world.register_instance(self._instance_id)
        if self._scl:
            self._scl.register(self._instance_id)

        logger.info(
            f"[SandboxController] INIT instance={self._instance_id} "
            f"world={self._world.world_id if self._world else 'none'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Isolated respond() — บันทึก SandboxAtom แทน production atom
    # ─────────────────────────────────────────────────────────────────────────

    def respond(
        self,
        input_text:   str,
        context:      str   = "general",
        input_vector: Any   = None,
    ) -> Dict[str, Any]:
        """
        Isolated respond — เหมือน Brain.respond() แต่:
          - บันทึก SandboxAtom แทน production Memory
          - สร้าง ExperimentState ส่งเข้า SCL อัตโนมัติ
          - Neural adapt อิสระ (ไม่กระทบ production)
        """
        if not self._active:
            return {"outcome": "reject", "reason": "sandbox inactive"}

        # safety check ก่อนส่งเข้า Brain
        safe, reason = self._safety_check(input_text, context)
        if not safe:
            atom = self._make_atom(
                context    = context,
                payload    = f"[BLOCKED] {reason}".encode(),
                confidence = 0.0,
                atom_type  = AtomType.CONFLICT,
                tags       = ["safety_blocked"],
            )
            return {
                "outcome":       "reject",
                "is_safe":       False,
                "safety_reason": reason,
                "atom_id":       atom.atom_id,
            }

        # เรียก isolated Brain
        result = self._brain.respond(
            input_text   = input_text,
            context      = context,
            input_vector = input_vector,
        )

        # บันทึก SandboxAtom
        atom = self._make_atom(
            context    = context,
            payload    = result.get("response", "").encode("utf-8"),
            confidence = result.get("confidence", 0.0),
            tags       = [context, result.get("outcome", "")],
        )

        # publish ExperimentState เข้า SCL
        if self._scl:
            state = ExperimentState.create(
                instance_id      = self._instance_id,
                hypothesis       = f"context={context} outcome={result.get('outcome')}",
                outcome          = result.get("response", "")[:200],
                stimulus_ref     = atom.atom_id,
                confidence_delta = result.get("confidence", 0.0),
                tags             = [context],
            )
            try:
                self._scl.publish(self._instance_id, state)
            except Exception as e:
                logger.warning(f"[SandboxController] SCL publish failed: {e}")

        result["atom_id"]    = atom.atom_id
        result["instance_id"] = self._instance_id
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Safety Check
    # ─────────────────────────────────────────────────────────────────────────

    def is_safe(self, input_text: str, context: str = "") -> tuple[bool, str]:
        """ตรวจ input อันตรายก่อนส่งเข้า Brain"""
        return self._safety_check(input_text, context)

    def _safety_check(self, text: str, context: str) -> tuple[bool, str]:
        """
        Safety check ผ่าน Condition gate

        Rule BLOCK INPUT → ไม่ปลอดภัย
        """
        allowed, reason = self._brain.condition.is_input_allowed(text)
        if not allowed:
            return False, f"blocked by condition rule: {reason}"
        return True, ""

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Simulation — dry-run ไม่บันทึกอะไรจริง
    # ─────────────────────────────────────────────────────────────────────────

    def simulate(
        self,
        input_text: str,
        context:    str = "general",
    ) -> SimulationResult:
        """
        Dry-run respond() — ไม่บันทึก Memory, Log, SandboxAtom จริง
        ใช้สำหรับ preview ผล ก่อน respond จริง
        """
        safe, reason = self._safety_check(input_text, context)

        # รัน Skill Contract โดยตรง ไม่ผ่าน respond()
        contract = self._brain._run_skill_contract(
            input_text = input_text,
            context    = context,
            topic_id   = None,
        )

        sim = SimulationResult(
            input_text   = input_text,
            context      = context,
            outcome      = contract.final_outcome.value,
            confidence   = contract.confidence_score,
            personality  = contract.personality,
            skill_weight = contract.skill_weight,
            is_safe      = safe,
            safety_reason = reason,
        )
        self._sim_results.append(sim)
        logger.debug(
            f"[SandboxController] SIMULATE "
            f"outcome={sim.outcome} conf={sim.confidence:.3f} safe={safe}"
        )
        return sim

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Rule Testing
    # ─────────────────────────────────────────────────────────────────────────

    def test_rule(
        self,
        rule,
        test_inputs: List[str],
        context: str = "general",
    ) -> List[Dict[str, Any]]:
        """
        ทดสอบ Rule ใหม่ใน isolated environment

        ไม่ add rule ลง production Condition
        ทดสอบแค่ว่า rule จะ block/allow inputs ไหน
        """
        results = []
        for text in test_inputs:
            matches = rule.matches(text=text)
            results.append({
                "input":   text,
                "matched": matches,
                "action":  rule.action.value if matches else "allow",
            })
        return results

    def test_rule_live(
        self,
        rule,
        test_inputs: List[str],
        reviewer_id: str,
        context: str = "general",
    ) -> List[SimulationResult]:
        """
        ทดสอบ Rule โดย inject เข้า isolated Brain ชั่วคราว
        แล้ว simulate ผล

        rule จะถูกลบออกหลังทดสอบเสร็จ (sandbox only)
        """
        if not reviewer_id:
            raise PermissionError("test_rule_live requires reviewer_id")

        # inject rule เข้า isolated condition
        self._brain.condition.governance_add_rule(rule, reviewer_id=reviewer_id)

        results = [self.simulate(text, context) for text in test_inputs]

        # ลบ rule ออกหลัง test
        try:
            self._brain.condition.governance_remove_rule(
                rule.rule_id, reviewer_id=reviewer_id
            )
        except Exception:
            pass

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Replay
    # ─────────────────────────────────────────────────────────────────────────

    def replay(
        self,
        logs: List[BrainLog],
        dry_run: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        เล่นซ้ำ BrainLogs เก่า เพื่อ debug

        dry_run=True  → simulate เท่านั้น (default)
        dry_run=False → respond จริงใน isolated brain
        """
        results = []
        for log in logs:
            if dry_run:
                sim = self.simulate(log.input_text, log.context)
                results.append({
                    "original_outcome":  log.outcome,
                    "replayed_outcome":  sim.outcome,
                    "original_conf":     log.confidence,
                    "replayed_conf":     sim.confidence,
                    "match":             log.outcome == sim.outcome,
                    "log_id":            log.log_id,
                })
            else:
                result = self.respond(log.input_text, log.context)
                results.append({
                    "original_outcome": log.outcome,
                    "replayed_outcome": result.get("outcome"),
                    "log_id":           log.log_id,
                    "atom_id":          result.get("atom_id"),
                })
            self._replay_logs.append(results[-1])

        logger.info(
            f"[SandboxController] REPLAY {len(logs)} logs "
            f"dry_run={dry_run} "
            f"match_rate={sum(r.get('match', False) for r in results)}/{len(results)}"
        )
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Promote → production
    # ─────────────────────────────────────────────────────────────────────────

    def promote(
        self,
        atom_id:     str,
        reviewer_id: str,
        reason:      str = "",
    ) -> Optional[str]:
        """
        Promote SandboxAtom → production ผ่าน Reviewer

        Flow:
          SandboxAtom → Neural สร้าง Proposal → Reviewer พิจารณา
          อนุมัติ → promote เป็น Knowlet หรือ Atom ใหม่

        Returns:
            proposal_id ถ้า propose สำเร็จ, None ถ้าไม่ได้
        """
        atom = self._get_atom(atom_id)
        if atom is None:
            raise KeyError(f"[SandboxController] atom '{atom_id}' not found")
        if not atom.is_promotable:
            raise ValueError(
                f"[SandboxController] atom '{atom_id}' not promotable "
                f"(status={atom.status.value} conf={atom.confidence:.2f})"
            )
        if not reviewer_id:
            raise PermissionError("promote requires reviewer_id")

        # สร้าง Proposal
        proposal = create_proposal(
            proposed_by = self._instance_id,
            action      = ProposalAction.ADD,
            target_type = ProposalTarget.RULE,
            authority   = RuleAuthority.STANDARD,
            payload     = {
                "atom_id":   atom.atom_id,
                "context":   atom.context,
                "payload":   atom.payload.decode("utf-8", errors="replace"),
                "confidence": atom.confidence,
                "source":    "sandbox",
            },
            reason = reason or f"promote sandbox atom from instance {self._instance_id}",
        )

        self._reviewer.enqueue(proposal)
        self._reviewer.register_reviewer(reviewer_id, ReviewerRole.STANDARD)
        decision = self._reviewer.approve(proposal, reviewer_id, reason)

        if decision:
            atom.status = SandboxStatus.PROMOTED
            logger.info(
                f"[SandboxController] PROMOTED atom={atom_id[:8]} "
                f"proposal={proposal.proposal_id[:8]}"
            )
            return proposal.proposal_id

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 7. SCL inter-instance
    # ─────────────────────────────────────────────────────────────────────────

    def read_experiments(self) -> List[ExperimentState]:
        """อ่าน ExperimentStates จาก instances อื่นใน SCL"""
        if self._scl is None:
            return []
        return self._scl.read_for(self._instance_id)

    def read_conflicts(self) -> List[ConflictRecord]:
        """
        อ่าน conflicts ที่เกี่ยวข้องกับ instance นี้

        Instance จะสร้าง SandboxAtom(CONFLICT) แล้ว
        ตั้ง hypothesis ใหม่สำหรับทดลองรอบถัดไป
        """
        if self._scl is None:
            return []
        conflicts = self._scl.conflicts_for(self._instance_id)
        # บันทึก conflict เป็น SandboxAtom
        for c in conflicts:
            self._make_atom(
                context    = "conflict",
                payload    = c.description.encode(),
                confidence = 0.0,
                atom_type  = AtomType.CONFLICT,
                tags       = ["conflict", c.hypothesis[:30]],
            )
        return conflicts

    def publish_hypothesis(
        self,
        hypothesis:       str,
        outcome:          str,
        confidence_delta: float = 0.0,
        tags:             List[str] = None,
    ) -> bool:
        """
        สร้างและ publish ExperimentState ไปยัง SCL
        instances อื่นจะอ่านและตีความด้วย Neural ของตัวเอง
        """
        if self._scl is None:
            raise RuntimeError("[SandboxController] no SCL attached")

        state = ExperimentState.create(
            instance_id      = self._instance_id,
            hypothesis       = hypothesis,
            outcome          = outcome,
            confidence_delta = confidence_delta,
            tags             = tags or [],
        )
        return self._scl.publish(self._instance_id, state)

    # ─────────────────────────────────────────────────────────────────────────
    # State / Stats
    # ─────────────────────────────────────────────────────────────────────────

    def atoms(self, status: Optional[SandboxStatus] = None) -> List[SandboxAtom]:
        """ดู SandboxAtoms (filter by status ได้)"""
        if status is None:
            return list(self._atoms)
        return [a for a in self._atoms if a.status == status]

    def deactivate(self) -> None:
        """ปิด sandbox instance"""
        self._active = False
        if self._world:
            self._world.remove_instance(self._instance_id)
        if self._scl:
            self._scl.unregister(self._instance_id)
        logger.info(f"[SandboxController] DEACTIVATED instance={self._instance_id}")

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def is_active(self) -> bool:
        return self._active

    def stats(self) -> Dict[str, Any]:
        return {
            "instance_id":  self._instance_id,
            "active":       self._active,
            "world_id":     self._world.world_id if self._world else None,
            "atoms_total":  len(self._atoms),
            "atoms_active": sum(1 for a in self._atoms if a.status == SandboxStatus.ACTIVE),
            "atoms_promoted": sum(1 for a in self._atoms if a.status == SandboxStatus.PROMOTED),
            "simulations":  len(self._sim_results),
            "replays":      len(self._replay_logs),
            "scl":          self._scl.stats() if self._scl else None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_atom(
        self,
        context:    str,
        payload:    bytes,
        confidence: float,
        atom_type:  AtomType = AtomType.LEARNING,
        tags:       List[str] = None,
    ) -> SandboxAtom:
        atom = SandboxAtom(
            instance_id = self._instance_id,
            atom_type   = atom_type,
            context     = context,
            payload     = payload,
            source      = f"sandbox_{context}_{self._instance_id}",
            confidence  = confidence,
            tags        = tags or [],
        )
        self._atoms.append(atom)
        return atom

    def _get_atom(self, atom_id: str) -> Optional[SandboxAtom]:
        for atom in self._atoms:
            if atom.atom_id == atom_id:
                return atom
        return None