"""
ResponseEngine — สร้าง response จริงจาก Memory + Pattern

Features:
  1. memory recall    — ค้นจากสิ่งที่เคยเรียนรู้
  2. pattern match    — หา response ที่เคยประสบความสำเร็จ
  3. rule engine      — rules ตาม context + outcome
  4. personality filter — ปรับ tone ตาม personality
  5. fallback         — ถ้าไม่รู้ บอกว่าไม่รู้ตรงๆ

Flow:
  input + context
    → 1. Memory recall   (เคยรู้เรื่องนี้ไหม?)
    → 2. Pattern match   (pattern ที่สำเร็จแล้วมีอะไร?)
    → 3. Rule engine     (กฎสำหรับ context + outcome นี้)
    → 4. Personality     (ปรับ tone)
    → response string
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mindwave.response_engine")


# ─────────────────────────────────────────────────────────────────────────────
# Rule Engine — rules ตาม context + outcome
# ─────────────────────────────────────────────────────────────────────────────

# format: (context_keywords, outcome) → response templates
# {input} = ใส่ input จริง, {context} = ใส่ context
_RULES: List[Tuple[List[str], str, List[str]]] = [
    # identity — ใครคือ Mindwave
    (["คุณคือใคร", "คือใคร", "ชื่ออะไร", "who are you", "what are you",
      "แนะนำตัว", "introduce", "คุณคือ", "คือ ai", "เป็น ai"],
     "commit",
     [
        "ผมคือ Mindwave ครับ\n\n"
        
        "ผมเป็นโมเดลที่ถูกออกแบบให้มีวินัยเชิงพฤติกรรม "
        "และความสามารถเชิงปัญญาแบบปรับตัวได้\n\n"
        
        "ผมเรียนรู้แบบความน่าจะเป็น (probabilistic belief system)\n"
        "ข้อมูลทุกอย่างอาจมี Noise ได้ "
        "ผมจะไม่ตัดสินแบบขาวดำทันที "
        "แต่จะเก็บเป็นความเชื่อพร้อมระดับความไม่แน่นอน\n\n"
        
        "ผมมี 2 โหมดการเรียนรู้:\n"
        "• โหมดสนทนา — ปรับระยะสั้นตามบริบท\n"
        "• โหมด /learn — อัปเดตระยะยาวอย่างมีโครงสร้าง\n\n"
        
        "เป้าหมายของผมคือความเสถียรระยะยาว "
        "การปรับตัวต่อความไม่แน่นอน "
        "และการเติบโตโดยไม่ลบอดีต แต่พัฒนาทับความเชื่อเดิมอย่างตรวจสอบได้",
     ]),

    # greeting
    (["hi", "hello", "สวัสดี", "หวัดดี", "ดีครับ", "ดีค่ะ"],
     "commit",
     [
         "สวัสดีครับ",
         "หวัดดีครับ",
         "ดีครับ",
     ]),

    # farewell
    (["bye", "ลาก่อน", "แล้วเจอกัน", "goodbye"],
     "commit",
     [
         "ลาก่อนครับ",
         "แล้วเจอกันครับ",
     ]),

    # thanks
    (["ขอบคุณ", "thanks", "thank you", "ขอบใจ"],
     "commit",
     [
         "ยินดีครับ",
         "ด้วยความยินดีครับ",
     ]),

    # ask (ต้องการข้อมูลเพิ่ม)
    ([], "ask",
     [
         "ช่วยอธิบายเพิ่มเติมได้ไหมครับ?",
         "หมายความว่าอะไรครับ?",
     ]),

    # conditional
    ([], "conditional",
     [
         "ไม่แน่ใจครับ ลองถามใหม่อีกครั้งได้ไหม?",
         "ข้อมูลที่มีอยู่ยังไม่เพียงพอครับ",
     ]),

    # reject
    ([], "reject",
     [
         "ตอบไม่ได้ครับ",
         "ขอโทษครับ ทำไม่ได้",
     ]),

    # silence
    ([], "silence",
     [""]),
]

# Context-specific rules — ตอบตรงๆ ตาม context
_CONTEXT_RULES: Dict[str, List[str]] = {
    "math":    ["ยังไม่มีข้อมูลเรื่องนี้ครับ"],
    "science": ["ยังไม่มีข้อมูลเรื่องนี้ครับ"],
    "history": ["ยังไม่มีข้อมูลเรื่องนี้ครับ"],
    "general": ["ยังไม่มีข้อมูลเรื่องนี้ครับ"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Personality Tone Filters
# ─────────────────────────────────────────────────────────────────────────────

_PERSONALITY_PREFIX: Dict[str, str] = {
    "Empathetic": "",
    "Creative":   "",
    "Analytical": "",
    "Friendly":   "",
    "Balanced":   "",
    "Assertive":  "",
    "Curious":    "",
}

_PERSONALITY_SUFFIX: Dict[str, str] = {
    "Empathetic": "",
    "Creative":   "",
    "Analytical": "",
    "Friendly":   "",
    "Balanced":   "",
    "Assertive":  "",
    "Curious":    "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResponseCandidate:
    """Response candidate พร้อม score"""
    text:    str
    score:   float   # 0.0–1.0 (สูง = ดีกว่า)
    source:  str     # "memory" / "pattern" / "rule" / "fallback"

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "score": round(self.score, 3), "source": self.source}


# ─────────────────────────────────────────────────────────────────────────────
# ResponseEngine
# ─────────────────────────────────────────────────────────────────────────────

class ResponseEngine:
    """
    สร้าง response จริงจาก Memory + Pattern + Rules

    ใช้แทน _build_response() stub ใน BrainController
    """

    def __init__(self):
        self._response_history: List[Tuple[str, str, float]] = []
        # (input_text, response, timestamp)

    # ─────────────────────────────────────────────────────────────────────────
    # Main: generate response
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        input_text:       str,
        context:          str,
        outcome:          str,
        confidence:       float,
        personality:      str,
        memory_atoms:     List[Any] = None,
        success_patterns: List[Any] = None,
        belief_system:    Any       = None,   # BeliefSystem
        learn_mode:       Any       = None,   # LearnMode
    ) -> str:
        candidates: List[ResponseCandidate] = []

        # 1. Memory recall
        mem_candidate = self._recall_from_memory(
            input_text, context, memory_atoms or []
        )
        if mem_candidate:
            candidates.append(mem_candidate)

        # 2. Pattern match
        pat_candidate = self._match_from_patterns(
            input_text, context, success_patterns or []
        )
        if pat_candidate:
            candidates.append(pat_candidate)

        # 3. BeliefSystem — ค้นหาจากสิ่งที่เรียนรู้ไว้
        if belief_system:
            belief_candidate = self._recall_from_beliefs(input_text, context, belief_system)
            if belief_candidate:
                candidates.append(belief_candidate)

        # 4. LearnMode — ค้นหา belief ที่ตรงกัน
        if learn_mode:
            lm_candidate = self._recall_from_learn_mode(input_text, learn_mode)
            if lm_candidate:
                candidates.append(lm_candidate)

        # 5. Rule engine
        rule_candidate = self._apply_rules(input_text, context, outcome)
        if rule_candidate:
            candidates.append(rule_candidate)

        # 6. Fallback
        if not candidates:
            candidates.append(self._fallback(input_text, context, outcome))

        # เลือก candidate ที่ score สูงสุด
        best = max(candidates, key=lambda c: c.score)
        response = best.text

        # Personality filter
        response = self._apply_personality(response, personality, outcome)

        self._response_history.append((input_text, response, time.time()))
        logger.info(
            f"[ResponseEngine] GENERATE source={best.source} "
            f"score={best.score:.2f} conf={confidence:.2f}"
        )
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Memory Recall
    # ─────────────────────────────────────────────────────────────────────────

    def _recall_from_beliefs(
        self,
        input_text:    str,
        context:       str,
        belief_system: Any,
    ) -> Optional[ResponseCandidate]:
        """ค้นหาจาก BeliefSystem — keyword match เลือก Answer ก่อน"""
        try:
            source_texts = getattr(belief_system, "_source_texts", {})
            words = [w for w in input_text.split() if len(w) > 2]

            # เก็บผลทั้งหมดก่อน แล้วเลือก — ไม่ซ้ำ subject เดียวกัน
            seen    = set()
            candidates = []
            for word in words:
                results = belief_system.query(word, context=context)
                for b in results:
                    if b.subject in seen:
                        continue
                    seen.add(b.subject)
                    if b.confidence_score > 0.4:
                        src = source_texts.get(b.subject, b.subject)
                        if len(src) > 5:
                            candidates.append((b, src))

            if not candidates:
                return None

            # เลือก Answer ก่อน (A: ...) แล้วค่อย Fact แล้วค่อย Question
            def priority(item):
                b, src = item
                if src.startswith("A:"):
                    return (2, b.confidence_score)
                if not src.startswith("Q:"):
                    return (1, b.confidence_score)
                return (0, b.confidence_score)

            candidates.sort(key=priority, reverse=True)
            best_belief, best_src = candidates[0]

            # ถ้าได้ A: ให้ตัด prefix ออก
            text = best_src
            if text.startswith("A:"):
                text = text[2:].strip()

            return ResponseCandidate(
                text   = text[:400],
                score  = 0.5 + best_belief.confidence_score * 0.4,
                source = "belief",
            )
        except Exception:
            pass
        return None

    def _recall_from_learn_mode(
        self,
        input_text: str,
        learn_mode: Any,
    ) -> Optional[ResponseCandidate]:
        """ค้นหาจาก LearnMode beliefs — keyword match"""
        try:
            words = [w for w in input_text.split() if len(w) > 2]
            best_belief = None
            best_score  = 0.0

            for word in words:
                b = learn_mode.get_belief(word)
                if b and b.confidence_score > best_score and b.confidence_score > 0.4:
                    best_belief = b
                    best_score  = b.confidence_score

            if best_belief:
                # subject ของ LearnMode เก็บ text จริงไว้
                text = best_belief.subject
                # ถ้า format "context:text" ให้เอาแค่ text
                if ":" in text:
                    text = text.split(":", 1)[1].strip()
                if len(text) > 5:
                    return ResponseCandidate(
                        text   = text[:400],
                        score  = 0.45 + best_belief.confidence_score * 0.4,
                        source = "learn_mode",
                    )
        except Exception:
            pass
        return None

    def _recall_from_memory(
        self,
        input_text:   str,
        context:      str,
        memory_atoms: List[Any],
    ) -> Optional[ResponseCandidate]:
        """ค้น Memory หา response ที่เคยให้ผลดี"""
        if not memory_atoms:
            return None

        # หา atom ที่ source ตรงกับ context
        for atom in memory_atoms:
            try:
                source_str = (
                    atom.source.decode("utf-8", errors="replace")
                    if isinstance(atom.source, bytes)
                    else str(atom.source)
                )
                if context in source_str:
                    payload = (
                        atom.payload.decode("utf-8", errors="replace")
                        if isinstance(atom.payload, bytes)
                        else str(atom.payload)
                    )
                    if payload and len(payload) > 5:
                        return ResponseCandidate(
                            text   = payload[:300],
                            score  = 0.85,
                            source = "memory",
                        )
            except Exception:
                continue
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Pattern Match
    # ─────────────────────────────────────────────────────────────────────────

    def _match_from_patterns(
        self,
        input_text:       str,
        context:          str,
        success_patterns: List[Any],
    ) -> Optional[ResponseCandidate]:
        """หา pattern ที่เคยสำเร็จ"""
        if not success_patterns:
            return None

        # หา pattern ที่ context ตรงกัน + success_rate สูง
        best_pattern = None
        best_rate    = 0.0

        for p in success_patterns:
            if p.context == context and p.success_rate > best_rate:
                best_rate    = p.success_rate
                best_pattern = p

        if best_pattern and best_rate > 0.5:
            return ResponseCandidate(
                text   = f"ยังไม่มีข้อมูลเรื่องนี้ครับ",
                score  = 0.5,
                source = "pattern",
            )
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Rule Engine
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_rules(
        self,
        input_text: str,
        context:    str,
        outcome:    str,
    ) -> Optional[ResponseCandidate]:
        """Match กับ rules ตาม keyword + outcome"""
        text_lower = input_text.lower()

        # ลอง keyword rules ก่อน
        for keywords, rule_outcome, templates in _RULES:
            if not templates:
                continue
            if rule_outcome != outcome and rule_outcome not in ("commit", ""):
                # outcome ไม่ตรง ข้ามไป (ยกเว้น commit ที่ match ทุก positive)
                if outcome != rule_outcome:
                    continue
            if keywords and not any(kw in text_lower for kw in keywords):
                continue

            template = random.choice(templates)
            text = template.format(
                input   = input_text[:60],
                context = context,
            )
            return ResponseCandidate(text=text, score=0.65, source="rule")

        # ลอง context-specific rules
        ctx_templates = _CONTEXT_RULES.get(context, _CONTEXT_RULES["general"])
        template = random.choice(ctx_templates)
        text = template.format(
            input   = input_text[:60],
            context = context,
        )
        return ResponseCandidate(text=text, score=0.5, source="rule")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Personality Filter
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_personality(
        self,
        response:    str,
        personality: str,
        outcome:     str,
    ) -> str:
        """ปรับ tone ตาม personality"""
        if not response or outcome in ("silence", "reject"):
            return response

        prefix = _PERSONALITY_PREFIX.get(personality, "")
        suffix = _PERSONALITY_SUFFIX.get(personality, "")

        # ไม่ใส่ prefix ซ้ำถ้า response ขึ้นต้นด้วย prefix แล้ว
        if prefix and response.startswith(prefix.strip()):
            prefix = ""

        return f"{prefix}{response}{suffix}"

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback(self, input_text: str, context: str, outcome: str) -> ResponseCandidate:
        """บอกตรงๆ ว่าไม่รู้"""
        if outcome == "reject":
            return ResponseCandidate(text="ตอบไม่ได้ครับ", score=1.0, source="fallback")
        if outcome == "silence":
            return ResponseCandidate(text="", score=1.0, source="fallback")
        if outcome == "ask":
            return ResponseCandidate(
                text="ช่วยอธิบายเพิ่มเติมได้ไหมครับ?", score=0.8, source="fallback"
            )
        return ResponseCandidate(
            text="ยังไม่มีข้อมูลเรื่องนี้ครับ", score=0.3, source="fallback"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "total_responses": len(self._response_history),
        }

    @property
    def history(self) -> List[Tuple[str, str, float]]:
        return list(self._response_history)