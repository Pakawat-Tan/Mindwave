"""
BrainController — ศูนย์กลางของทุก Module

หน้าที่:
  1. รับ input จากภายนอก
  2. เรียก Skill Contract (Rule → Confidence → Skill → Personality → Emotion)
  3. ตัดสินใจ outcome
  4. เรียนรู้จาก interaction (Continuous Learning)
  5. คืน response พร้อม outcome

กฎเหล็ก:
  - ไม่มี logic ของตัวเอง — เป็นแค่ relay + orchestrator
  - ห้าม relay ตรงไปหา IO (ต้องผ่าน module)
  - Skill Contract ต้องเรียกทุกครั้งก่อนตอบ
  - บันทึก brain log ทุก interaction
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Core.Condition.ConditionController import ConditionController
from Core.Confidence.ConfidenceController import ConfidenceController
from Core.Confidence.ConfidenceData import ConfidenceOutcome, ConfidenceResult
from Core.Skill.SkillController import SkillController
from Core.Personality.PersonalityController import PersonalityController
from Core.Neural.NeuralController import NeuralController
from Core.Neural.Brain.BrainStructure import BrainStructure
from Core.Review.ReviewerController import ReviewerController
from Core.Memory.MemoryController import MemoryController
from Core.Brain.MetaCognition import MetaCognition
from Core.Brain.PatternRecognition import PatternRecognition
from Core.Brain.TopicClustering import TopicClustering
from Core.Brain.EmotionInference import EmotionInference, Emotion
from Core.Brain.DistributedSystem import DistributedSystem
from Core.Brain.FeedbackInference import FeedbackInference, FeedbackType
from Core.Brain.ResponseEngine import ResponseEngine
from Core.Brain.LearnMode import LearnMode
from Core.Brain.BeliefSystem import BeliefSystem
from Core.Brain.NeuralTrainer import NeuralTrainer


logger = logging.getLogger("mindwave.brain")


# ============================================================================
# BRAIN LOG — บันทึกทุก interaction
# ============================================================================

@dataclass(frozen=True)
class BrainLog:
    log_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_text:   str   = ""
    context:      str   = ""
    outcome:      str   = ""          # commit/conditional/ask/silence/reject
    confidence:   float = 0.0
    skill_weight: float = 0.0
    personality:  str   = ""
    learned:      bool  = False
    response:     str   = ""
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_id":       self.log_id,
            "input_text":   self.input_text[:100],
            "context":      self.context,
            "outcome":      self.outcome,
            "confidence":   self.confidence,
            "skill_weight": self.skill_weight,
            "personality":  self.personality,
            "learned":      self.learned,
            "timestamp":    self.timestamp,
        }


# ============================================================================
# SKILL CONTRACT RESULT
# ============================================================================

@dataclass(frozen=True)
class ContractResult:
    """
    ผลของ Skill Contract — ใช้ตัดสินใจ outcome

    Execution Priority:
      Rule → Confidence → Skill → Personality → Emotion
    """
    rule_passed:      bool              = True
    rule_reason:      str               = ""
    confidence_score: float             = 0.0
    confidence_outcome: ConfidenceOutcome = ConfidenceOutcome.COMMIT
    skill_weight:     float             = 0.0
    personality:      str               = ""
    final_outcome:    ConfidenceOutcome = ConfidenceOutcome.COMMIT

    @property
    def can_respond(self) -> bool:
        return self.final_outcome in (
            ConfidenceOutcome.COMMIT,
            ConfidenceOutcome.CONDITIONAL,
        )


# ============================================================================
# BRAIN CONTROLLER
# ============================================================================

class BrainController:

    # IO ห้าม relay ตรง (hard-coded)
    _RESTRICTED_TARGETS = {"io", "IO", "input_output"}

    def __init__(
        self,
        condition:   Optional[ConditionController]   = None,
        confidence:  Optional[ConfidenceController]  = None,
        skill:       Optional[SkillController]       = None,
        personality: Optional[PersonalityController] = None,
        memory:      Optional[MemoryController]      = None,
        neural:      Optional[NeuralController]      = None,
        brain_structure: Optional[BrainStructure]    = None,
        reviewer:    Optional[ReviewerController]    = None,
        metacognition:      Optional[MetaCognition]      = None,
        pattern_recognition: Optional[PatternRecognition] = None,
        topic_clustering:   Optional[TopicClustering]    = None,
        emotion_inference:  Optional[EmotionInference]   = None,
        distributed_system: Optional[DistributedSystem]  = None,
        feedback:           Optional[FeedbackInference]  = None,
        response_engine:    Optional[ResponseEngine]     = None,
        learn_mode:         Optional[LearnMode]          = None,
        belief_system:      Optional[BeliefSystem]       = None,
    ):
        # ── Modules ──────────────────────────────────────────────
        self._condition   = condition   or ConditionController()

        # inject condition เข้าทุก module ตาม Skill Contract priority
        self._confidence  = confidence  or ConfidenceController(
            condition = self._condition
        )
        self._skill       = skill       or SkillController(
            condition = self._condition
        )
        self._personality = personality or PersonalityController(
            condition = self._condition
        )
        self._memory      = memory      or MemoryController(
            condition = self._condition
        )
        self._neural      = neural      or NeuralController()
        self._brain_struct = brain_structure or BrainStructure(
            condition = self._condition
        )
        # build neural network structure ถ้ายังไม่มี nodes
        if self._brain_struct is not None and len(self._brain_struct.nodes) == 0:
            try:
                self._brain_struct.build_structure()
            except Exception:
                pass
        self._reviewer    = reviewer    or ReviewerController()

        # ── Phase 4 Modules ───────────────────────────────────────
        self._metacognition   = metacognition      or MetaCognition()
        self._pattern         = pattern_recognition or PatternRecognition()
        self._topic           = topic_clustering    or TopicClustering()
        self._emotion         = emotion_inference   or EmotionInference()
        self._distributed     = distributed_system  or DistributedSystem(
            instance_id = str(uuid.uuid4())[:8]
        )
        self._feedback        = feedback            or FeedbackInference()
        self._response_engine = response_engine    or ResponseEngine()
        self._learn_mode      = learn_mode         or LearnMode()
        self._belief_system   = belief_system      or BeliefSystem(
            learning_rate = 0.3,
            persist_path  = "Core/Data/beliefs.json",
        )
        
        # ── Neural Trainer ────────────────────────────────────────
        self._neural_trainer = NeuralTrainer(
            brain_struct      = self._brain_struct,
            learning_rate     = 0.01,
            activation        = "sigmoid",
            enable_evolution  = True,   # เปิด auto-evolution
            evolve_every      = 50,     # evolve ทุก 50 samples
        )

        # ── State ─────────────────────────────────────────────────
        self._logs:         List[BrainLog]   = []
        self._mode:         str              = "active"
        self._instance_id:  str             = str(uuid.uuid4())[:8]
        self._prev_context: str              = ""  # track context เปลี่ยนไหม

        # Phase 4 config
        self._metacog_interval  = 5   # reflect ทุก 5 logs
        self._pattern_interval  = 10  # หา patterns ทุก 10 logs
        self._topic_interval    = 3   # cluster topics ทุก 3 logs

        # init personality ถ้ายังไม่มี
        if not self._personality.is_initialized():
            self._personality.init()

        # register instance ใน distributed system
        self._distributed.heartbeat(self._instance_id)

        logger.info(
            f"[BrainController] INIT instance={self._instance_id} "
            f"personality={self._personality.profile_name}"
        )

    # ─────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────

    def respond(
        self,
        input_text:    str,
        context:       str   = "general",
        reviewer_id:   str   = "",
        topic_id:      Optional[int]  = None,
        input_vector:  Optional[Any]  = None,
    ) -> Dict[str, Any]:
        """
        รับ input → เรียก Skill Contract → เรียนรู้ Realtime → คืน response

        Args:
            input_text   : ข้อความ input จาก user
            context      : topic/domain ของ input
            reviewer_id  : สำหรับ lock/unlock และ reviewer operations
                           (ไม่จำเป็นสำหรับ learning แล้ว — Realtime)
            topic_id     : topic id สำหรับ Skill arbitration
            input_vector : numpy array สำหรับ BrainStructure.observe()

        Returns:
            dict ของ response result
        """
        if self._mode == "locked":
            return self._make_response(
                input_text, context,
                outcome  = "reject",
                response = "[LOCKED] Brain is in locked mode",
                learned  = False,
                contract = None,
            )

        # ── 1. Memory recall ─────────────────────────────────────
        memory_context_score = 0.5   # default ถ้าไม่มี memory
        memory_rep           = 1     # default topic repetition

        try:
            mem_atoms = self._memory.read_for_response(
                atom_ids = [context],
                limit    = 5,
            )
            if mem_atoms:
                # context_score มาจาก avg weight ของ memory ที่ recall ได้
                memory_context_score = min(1.0, sum(
                    a.weight for a in mem_atoms
                ) / len(mem_atoms))
                # topic_repetition มาจากจำนวน memory ที่ตรงกับ context
                memory_rep = len(mem_atoms)
        except Exception:
            pass  # ถ้า memory ยังว่าง → ใช้ default

        # ── 2. Skill Contract ─────────────────────────────────────
        contract = self._run_skill_contract(
            input_text, context, topic_id,
            memory_context_score = memory_context_score,
            memory_rep           = memory_rep,
        )

        # ── 2b. Implicit Feedback — สังเกตพฤติกรรม ───────────────
        prev_log = self._logs[-1] if self._logs else None
        fb_signal = self._feedback.infer(
            current_text = input_text,
            context      = context,
            prev_log     = prev_log,
            prev_context = self._prev_context,
        )
        # apply immediate effect
        if fb_signal:
            effect = self._feedback.get_immediate_effect(fb_signal)
            # ปรับ MetaCognition bias ทันที
            self._metacognition._confidence_bias = max(-0.5, min(0.5,
                self._metacognition._confidence_bias + effect.confidence_delta
            ))
            # ปรับ Skill ถ้ามี effect แรงพอ
            if abs(effect.skill_delta) > 0.01:
                self._skill.try_grow(
                    skill_name       = context,
                    delta            = effect.skill_delta,
                    topic_repetition = 1,
                    avg_confidence   = contract.confidence_score,
                    reason           = f"feedback:{fb_signal.signal_type.value}",
                )
        self._prev_context = context

        # ── 2. ตัดสินใจ outcome ──────────────────────────────────
        outcome_str = contract.final_outcome.value

        # ── 3. สร้าง response text ────────────────────────────────
        response_text = self._build_response(contract, input_text, context)

        # ── 4. Realtime Learning — ทุก interaction เรียนรู้ทันที ───
        learned = False
        if input_vector is not None and self._brain_struct is not None:
            try:
                obs = self._brain_struct.observe(
                    input_vector  = input_vector,
                    context_label = context,
                    confidence    = contract.confidence_score,
                )
                learned = obs.get("learned", False)

                # Skill grow ทุกครั้งที่ learned
                if learned:
                    self._skill.try_grow(
                        skill_name       = context,
                        delta            = 0.05 * (1 + memory_rep),
                        topic_repetition = memory_rep + 1,
                        avg_confidence   = contract.confidence_score,
                        reason           = "realtime learning",
                    )
            except RuntimeError as e:
                logger.error(f"[BrainController] LEARN_ERROR: {e}")

        # ── 4b. Auto Belief Learning — เรียนรู้จาก input ทุกครั้งโดยอัตโนมัติ ─
        try:
            self._learn_mode.learn(f"{context}:{input_text}")

            # BeliefSystem — อัปเดต probabilistic belief พร้อมกัน
            # input_value = confidence ของ Brain ต่อ input นี้
            input_value = contract.confidence_score
            self._belief_system.update(
                subject      = input_text[:60],
                input_value  = input_value,
                context      = context,
                source       = "auto",
            )
            # อัปเดต belief ระดับ context ด้วย
            self._belief_system.update(
                subject      = f"[ctx:{context}]",
                input_value  = contract.skill_weight,
                context      = context,
                source       = "skill",
            )
        except Exception:
            pass

        # ── 5. Store response ใน Memory (ผ่าน MemoryController เท่านั้น) ─
        if contract.can_respond and response_text:
            try:
                # importance = ความสำคัญจริง ≠ confidence
                # - commit + high skill   → important
                # - conditional           → moderate
                # - repeat/confusion      → less important
                base_importance = contract.confidence_score * 0.5
                if outcome_str == "commit":
                    base_importance += contract.skill_weight * 0.3
                    if memory_rep >= 2:
                        base_importance += 0.1   # ถามซ้ำ = สำคัญ
                elif outcome_str == "conditional":
                    base_importance *= 0.7
                importance = max(0.1, min(0.94, base_importance))

                self._memory.write_response(
                    text       = response_text,
                    context    = context,
                    importance = importance,
                )
            except Exception:
                pass  # memory write failure ไม่ block response

        # ── 6. Log ────────────────────────────────────────────────
        result = self._make_response(
            input_text = input_text,
            context    = context,
            outcome    = outcome_str,
            response   = response_text,
            learned    = learned,
            contract   = contract,
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # Skill Contract
    # ─────────────────────────────────────────────────────────────

    def _run_skill_contract(
        self,
        input_text:          str,
        context:             str,
        topic_id:            Optional[int],
        memory_context_score: float = 0.5,
        memory_rep:          int   = 1,
    ) -> ContractResult:
        """
        Skill Contract — ตรวจทุก check ก่อนตอบ

        Execution Priority:
          Rule → Confidence → Skill → Personality → Emotion
        """
        # ── 1. Rule check ─────────────────────────────────────────
        rule_passed, rule_reason = self._condition.is_input_allowed(
            text = input_text,
        )

        # ── 2. Confidence evaluation ──────────────────────────────
        conf_result = self._confidence.evaluate(
            rule_score      = 0.0 if not rule_passed else 1.0,
            context_score   = memory_context_score,   # ← จาก Memory recall
            skill_score     = 0.5,
            identity_score  = 1.0,
            rule_blocked    = not rule_passed,
        )
        
        # Apply MetaCognition bias to confidence score
        # (adjust score while preserving level and outcome)
        adjusted_confidence = max(0.0, min(1.0, 
            conf_result.score - self._metacognition.confidence_bias
        ))
        conf_result = ConfidenceResult(
            score   = adjusted_confidence,
            level   = conf_result.level,
            outcome = conf_result.outcome,
        )

        # ── 2b. Emotion influence ─────────────────────────────────
        # detect emotion จาก input + apply influence
        emotion_score = self._emotion.detect_emotion(input_text)
        state = self._emotion.get_emotional_state()
        inf   = self._emotion.get_influence(state.primary_emotion)
        if inf != 0.0:
            emotion_adjusted = max(0.0, min(1.0, conf_result.score + inf))
            conf_result = ConfidenceResult(
                score   = emotion_adjusted,
                level   = conf_result.level,
                outcome = conf_result.outcome,
            )

        # ── 3. Skill arbitration + grow จาก memory repetition ────
        arb = self._skill.arbitrate(topic_id=topic_id)
        skill_weight = arb.weight if arb.has_skills else 0.5

        # memory rep ช่วย Skill grow
        if memory_rep >= 2:
            self._skill.try_grow(
                skill_name       = context,
                delta            = 0.05 * memory_rep,
                topic_repetition = memory_rep,
                avg_confidence   = conf_result.score,
                reason           = f"memory_recall rep={memory_rep}",
            )

        # ── 4. Personality ────────────────────────────────────────
        personality_name = self._personality.profile_name or "Balanced"

        # ── 5. Final outcome ──────────────────────────────────────
        final_outcome = conf_result.outcome

        return ContractResult(
            rule_passed        = rule_passed,
            rule_reason        = rule_reason,
            confidence_score   = conf_result.score,
            confidence_outcome = conf_result.outcome,
            skill_weight       = skill_weight,
            personality        = personality_name,
            final_outcome      = final_outcome,
        )

    # ─────────────────────────────────────────────────────────────
    # Relay (ห้าม relay ไป IO ตรง)
    # ─────────────────────────────────────────────────────────────

    def relay(self, target: str, message: Any) -> Any:
        """
        Relay message ไปหา module

        กฎ: ห้าม relay ตรงไปหา IO
        """
        if target in self._RESTRICTED_TARGETS:
            logger.error(
                f"[BrainController] DENIED relay→{target}: "
                f"BrainController cannot relay directly to IO"
            )
            raise PermissionError(
                f"[BrainController] cannot relay directly to '{target}'"
            )
        logger.debug(f"[BrainController] RELAY →{target}")
        return message

    # ─────────────────────────────────────────────────────────────
    # Mode
    # ─────────────────────────────────────────────────────────────

    def lock(self, reviewer_id: str) -> None:
        """ล็อค Brain — ปฏิเสธทุก input"""
        if not reviewer_id:
            raise PermissionError("lock requires reviewer_id")
        self._mode = "locked"
        logger.warning(f"[BrainController] LOCKED by='{reviewer_id}'")

    def unlock(self, reviewer_id: str) -> None:
        if not reviewer_id:
            raise PermissionError("unlock requires reviewer_id")
        self._mode = "active"
        logger.warning(f"[BrainController] UNLOCKED by='{reviewer_id}'")

    @property
    def mode(self) -> str:
        return self._mode

    # ─────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        return {
            "instance_id":  self._instance_id,
            "mode":         self._mode,
            "personality":  self._personality.profile_name,
            "skill_count":  len(self._skill.list()),
            "logs_total":   len(self._logs),
            "modules": {
                "condition":    True,
                "confidence":   True,
                "skill":        True,
                "personality":  self._personality.is_initialized(),
                "memory":       True,
                "neural":       True,
                "brain_struct": self._brain_struct is not None,
                "reviewer":     True,
                "metacognition":  True,
                "pattern":        True,
                "topic":          True,
                "emotion":        True,
                "distributed":    True,
                "feedback":       True,
                "response_engine": True,
            }
        }

    @property
    def logs(self) -> List[BrainLog]:
        return list(self._logs)

    def last_log(self) -> Optional[BrainLog]:
        return self._logs[-1] if self._logs else None

    # ─────────────────────────────────────────────────────────────
    # Module access (read-only)
    # ─────────────────────────────────────────────────────────────

    @property
    def condition(self)   -> ConditionController:   return self._condition
    @property
    def confidence(self)  -> ConfidenceController:  return self._confidence
    @property
    def skill(self)       -> SkillController:       return self._skill
    @property
    def personality(self) -> PersonalityController: return self._personality
    @property
    def neural(self)      -> NeuralController:      return self._neural
    @property
    def memory(self)      -> MemoryController:      return self._memory
    @property
    def reviewer(self)    -> ReviewerController:    return self._reviewer
    @property
    def metacognition(self)   -> MetaCognition:      return self._metacognition
    @property
    def pattern(self)         -> PatternRecognition: return self._pattern
    @property
    def topic(self)           -> TopicClustering:    return self._topic
    @property
    def emotion(self)         -> EmotionInference:   return self._emotion
    @property
    def distributed(self)     -> DistributedSystem:  return self._distributed
    @property
    def feedback(self)        -> FeedbackInference:  return self._feedback
    @property
    def response_engine(self) -> ResponseEngine:     return self._response_engine
    @property
    def learn_mode(self)      -> LearnMode:          return self._learn_mode
    @property
    def belief_system(self)   -> BeliefSystem:       return self._belief_system
    @property
    def neural_trainer(self)  -> NeuralTrainer:      return self._neural_trainer

    def learn(self, text: str) -> Dict[str, Any]:
        """
        /learn mode — structured intentional learning

        ต่างจาก respond():
          - ไม่ผ่าน Skill Contract / Emotion
          - learning_rate สูงกว่า
          - เข้า long-term memory โดยตรงถ้า consolidate
          - ไม่มี emotional heuristics
        """
        session = self._learn_mode.learn(text)

        # ถ้า consolidate → write ลง long-term memory
        if session.consolidated:
            try:
                self._memory.write_response(
                    text       = text,
                    context    = session.subject,
                    importance = 0.85,   # high → long tier โดยตรง
                )
            except Exception:
                pass

        logger.info(
            f"[BrainController] LEARN subject='{session.subject}' "
            f"consolidated={session.consolidated}"
        )

        # สร้าง response สำหรับ learn mode
        belief = self._learn_mode.get_belief(session.subject)
        if session.consolidated:
            response = (
                f"✓ เรียนรู้และบันทึกลง long-term แล้ว: '{session.subject}'\n"
                f"  confidence={belief.confidence_score:.2f} "
                f"(เรียนซ้ำ {belief.update_count} ครั้ง)"
            )
        elif belief and belief.conflict_rate > 0.2:
            response = (
                f"⚠ รับทราบ แต่พบข้อมูลขัดแย้ง: '{session.subject}'\n"
                f"  กำลังสะสมหลักฐานเพิ่ม "
                f"(variance={belief.belief_variance:.2f})"
            )
        else:
            response = (
                f"~ บันทึกไว้ชั่วคราว: '{session.subject}'\n"
                f"  ต้องการหลักฐานซ้ำอีก "
                f"{max(0, 3 - belief.update_count)} ครั้งจึงจะเสถียร"
            )

        return {
            "response":     response,
            "outcome":      "learn",
            "subject":      session.subject,
            "consolidated": session.consolidated,
            "confidence":   belief.confidence_score if belief else 0.0,
            "variance":     belief.belief_variance  if belief else 1.0,
        }
    
    def train_neural(self, text: str, target_response: str, importance: float = 0.7) -> Dict[str, Any]:
        """
        Train neural network โดยตรง
        
        Args:
            text: input text
            target_response: expected response
            importance: 0.0–1.0 (ยิ่งสูงยิ่งส่งผลต่อ target vector)
        
        Returns:
            {loss, accuracy, nodes_used}
        """
        try:
            from Core.Brain.NeuralTrainer import TrainingBatch
            
            # แปลง text → input vector (simple encoding)
            inputs = self._encode_text(text)
            
            # แปลง target_response → target vector
            targets = self._encode_response(target_response, importance)
            
            # create batch
            batch = TrainingBatch(
                inputs=inputs,
                targets=targets,
                importance=importance,
            )
            
            # train — returns tuple (loss, accuracy)
            loss, accuracy = self._neural_trainer.train_batch(batch)
            
            return {
                "loss": float(loss),
                "accuracy": float(accuracy),
                "nodes_used": len(self._neural_trainer._node_outputs),
            }
        except Exception as e:
            import traceback
            logger.error(f"[BrainController] train_neural ERROR: {e}")
            logger.error(traceback.format_exc())
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "nodes_used": 0,
            }
    
    def _encode_text(self, text: str, max_len: int = 9) -> List[float]:
        """
        แปลง text → input vector
        
        Simple encoding:
        - word count / 10
        - char count / 100
        - has question mark
        - has Thai
        - has English
        - avg char per word
        - has number
        - has punctuation
        - importance (0.5 default)
        """
        words = text.split()
        chars = len(text)
        
        features = [
            min(1.0, len(words) / 10.0),           # word count
            min(1.0, chars / 100.0),               # char count
            1.0 if "?" in text else 0.0,           # question
            1.0 if any(ord(c) >= 3584 for c in text) else 0.0,  # Thai
            1.0 if any(c.isalpha() and ord(c) < 128 for c in text) else 0.0,  # English
            min(1.0, (chars / max(1, len(words))) / 10.0),  # avg char/word
            1.0 if any(c.isdigit() for c in text) else 0.0,     # number
            1.0 if any(c in ".,!?;:" for c in text) else 0.0,   # punctuation
            0.5,  # importance placeholder
        ]
        
        # pad or truncate to max_len
        while len(features) < max_len:
            features.append(0.0)
        result = features[:max_len]
        
        # validate — ต้องเป็น float ทุกตัว
        return [float(x) for x in result]
    
    def _encode_response(self, response: str, importance: float) -> List[float]:
        """
        แปลง response → target vector (output nodes)
        
        Output nodes (15 nodes):
        - confidence (0–1)
        - outcome: commit/conditional/ask/silence/reject (one-hot 5 nodes)
        - emotion: neutral/curious/confident/uncertain/frustrated/satisfied (one-hot 6 nodes)
        - importance
        - length_category (0=short, 0.5=medium, 1=long)
        - has_thai
        """
        # parse outcome from response (heuristic)
        outcome_map = {
            "commit": [1, 0, 0, 0, 0],
            "conditional": [0, 1, 0, 0, 0],
            "ask": [0, 0, 1, 0, 0],
            "silence": [0, 0, 0, 1, 0],
            "reject": [0, 0, 0, 0, 1],
        }
        
        # detect outcome from response keywords
        if any(w in response for w in ["ไม่", "ห้าม", "อย่า"]):
            outcome = "reject"
        elif any(w in response for w in ["ไหม", "หรือ", "?"]):
            outcome = "ask"
        elif any(w in response for w in ["บางที", "อาจจะ", "ไม่แน่ใจ"]):
            outcome = "conditional"
        else:
            outcome = "commit"
        
        # emotion (simple: neutral=1)
        emotion_vec = [1, 0, 0, 0, 0, 0]  # neutral
        
        # length category
        word_count = len(response.split())
        length_cat = 0.0 if word_count < 5 else (0.5 if word_count < 20 else 1.0)
        
        # has Thai
        has_thai = 1.0 if any(ord(c) >= 3584 for c in response) else 0.0
        
        target = [
            importance,  # confidence
            *outcome_map[outcome],  # outcome (5 nodes)
            *emotion_vec,  # emotion (6 nodes)
            importance,  # importance
            length_cat,  # length
            has_thai,  # language
        ]
        
        # validate — ต้อง 15 nodes พอดี
        result = target[:15]
        while len(result) < 15:
            result.append(0.0)
        
        return [float(x) for x in result]

    def seal_session(self, silence: bool = True) -> None:
        """ปิด session — บันทึก beliefs ลง disk"""
        self._feedback.seal_session(silence_reward=silence)
        self._belief_system.save()
        logger.info(f"[BrainController] SESSION_SEALED silence={silence}")

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _build_response(
        self,
        contract:   ContractResult,
        input_text: str,
        context:    str,
    ) -> str:
        """สร้าง response จริงผ่าน ResponseEngine"""
        outcome = contract.final_outcome.value

        # recall memory atoms สำหรับ context นี้
        try:
            memory_atoms = self._memory.read_for_response(
                atom_ids = [context], limit = 3
            )
        except Exception:
            memory_atoms = []

        # success patterns จาก PatternRecognition
        success_patterns = self._pattern.successes

        return self._response_engine.generate(
            input_text       = input_text,
            context          = context,
            outcome          = outcome,
            confidence       = contract.confidence_score,
            personality      = contract.personality,
            memory_atoms     = memory_atoms,
            success_patterns = success_patterns,
            belief_system    = self._belief_system,
            learn_mode       = self._learn_mode,
        )

    def _make_response(
        self,
        input_text: str,
        context:    str,
        outcome:    str,
        response:   str,
        learned:    bool,
        contract:   Optional[ContractResult],
    ) -> Dict[str, Any]:
        log = BrainLog(
            input_text   = input_text,
            context      = context,
            outcome      = outcome,
            confidence   = contract.confidence_score if contract else 0.0,
            skill_weight = contract.skill_weight     if contract else 0.0,
            personality  = contract.personality      if contract else "",
            learned      = learned,
            response     = response,
        )
        self._logs.append(log)
        
        # ── MetaCognition analysis (ทุก N logs) ───────────────────
        if len(self._logs) % self._metacog_interval == 0:
            recent = self._logs[-self._metacog_interval:]
            self._metacognition.reflect(recent)
            self._metacognition.calibrate_confidence(recent)
            self._metacognition.detect_errors(recent)
            self._metacognition.track_learning(recent)

        # ── PatternRecognition (ทุก N logs) ───────────────────────
        if len(self._logs) % self._pattern_interval == 0:
            recent = self._logs[-self._pattern_interval:]
            self._pattern.detect_sequences(recent)
            self._pattern.detect_behavior(recent)
            self._pattern.detect_errors(recent)
            self._pattern.detect_success(recent)

        # ── TopicClustering (ทุก interaction) ────────────────────
        if len(self._logs) % self._topic_interval == 0:
            recent_topics = [l.context for l in self._logs[-self._topic_interval:]]
            self._topic.cluster_topics(recent_topics)

        # ── DistributedSystem heartbeat ───────────────────────────
        self._distributed.heartbeat(self._instance_id)

        # ── Long-term Feedback — apply ทุก 10 logs ────────────────
        if len(self._logs) % 10 == 0:
            conf_d, skill_d = self._feedback.get_long_term_delta()
            if abs(conf_d) > 0.001:
                self._metacognition._confidence_bias = max(-0.5, min(0.5,
                    self._metacognition._confidence_bias + conf_d
                ))

        logger.info(
            f"[BrainController] RESPOND "
            f"outcome={outcome} "
            f"conf={log.confidence:.3f} "
            f"learned={learned}"
        )
        return {
            "response":     response,
            "outcome":      outcome,
            "confidence":   log.confidence,
            "skill_weight": log.skill_weight,
            "personality":  log.personality,
            "learned":      learned,
            "log_id":       log.log_id,
        }

    def evolution_stats(self) -> dict:
        """สรุปสถิติ evolution ของ BrainStructure"""
        if self._brain_struct is None:
            return {"error": "no BrainStructure attached"}
        return self._brain_struct.evolution_stats()

    def set_evolve_every(self, n: int) -> None:
        """ตั้ง interval ของ auto-evolve"""
        if self._brain_struct is not None:
            self._brain_struct.set_evolve_every(n)