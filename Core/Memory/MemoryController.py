"""
Core/Memory/MemoryController.py  — Version 3

การเปลี่ยนแปลงจาก v2:
  - write()  : ถอด emotion ออก → (data, topic, importance, tier)
  - Emotion  : ไม่เก็บใน atom — เป็น inference-layer signal เท่านั้น
  - เพิ่ม read_for_response(topic, emotion, limit)
              ดึง atoms ตาม topic แล้ว weight ด้วย VAD formula

VAD Weighting Formula:
    v_norm  = (valence + 1) / 2          # normalize -1..1 → 0..1
    blended = (1-v_norm)×importance + v_norm×coherence
              negative valence → เน้น importance (ความจำที่หนักแน่น)
              positive valence → เน้น coherence  (ความจำที่ชัดเจน)
    arousal_boost = 1.0 + arousal × 0.5  # 1.0..1.5  (urgency)
    tier_factor   = 1.0 + (dominance-0.5) × (tier_rank/4)
                    dominance สูง → ดึงจาก deep tier ด้วย
                    dominance ต่ำ → focused, เน้น shallow tier
    score = blended × arousal_boost × tier_factor
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Core.Condition.ConditionController import ConditionController

import json
import logging
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from .Structure.AtomStructure import AtomData
from .Structure.AtomRepair import quick_check, auto_repair
from .Structure.KnowletStructure import ShardPath, SHARD_DEPTH_MIN
from .Tiers.Short_term    import Short_term,    SHORT_TERM_PROMOTION_THRESHOLD
from .Tiers.Middle_term   import Middle_term,   MIDDLE_TERM_PROMOTION_THRESHOLD
from .Tiers.Long_term     import Long_term,     LONG_TERM_PROMOTION_THRESHOLD
from .Tiers.Immortal_term import Immortal_term
from .KnowletController import KnowletController
from .Emotion import EmotionData, NEUTRAL_EMOTION
from .Topic   import TopicData


# ─────────────────────────────────────────────────────────────────────────────
# AtomContext
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AtomContext:
    """Atom พร้อม context ที่ deserialize แล้ว"""
    atom_id: str
    data:    AtomData
    topic:   Optional[TopicData] = None
    tier:    Optional[str]       = None


@dataclass
class WeightedAtom:
    """
    Atom พร้อม emotion-weighted score สำหรับ response generation

    Attributes:
        context      : AtomContext ที่ดึงมา
        score        : VAD-weighted score (สูง = ควรใช้ก่อน)
        importance   : importance ดั้งเดิมของ atom
    """
    context:    AtomContext
    score:      float
    importance: float


# ─────────────────────────────────────────────────────────────────────────────
# Tier rank map (ใช้ใน weighting formula)
# ─────────────────────────────────────────────────────────────────────────────

_TIER_RANK = {"short": 1, "middle": 2, "long": 3, "immortal": 4}


# ─────────────────────────────────────────────────────────────────────────────
# MemoryController
# ─────────────────────────────────────────────────────────────────────────────

class MemoryController:

    def __init__(self, base_path: str = "Core/Data", condition=None):
        self._base   = Path(base_path)
        self._condition = condition
        self._logger = logging.getLogger("mindwave.memory.controller")

        self._short    = Short_term   (str(self._base / "production" / "short"))
        self._middle   = Middle_term  (str(self._base / "production" / "middle"))
        self._long     = Long_term    (str(self._base / "production" / "long"))
        self._immortal = Immortal_term(str(self._base / "production" / "immortal"))
        self._knowlet  = KnowletController(str(self._base))

        self._logger.info("[MemoryController] initialized")

    # ─────────────────────────────────────────
    # Write  (emotion ไม่เกี่ยวข้อง)
    # ─────────────────────────────────────────

    def write(
        self,
        data:       AtomData,
        topic:      TopicData,
        importance: float         = 0.3,
        tier:       Optional[str] = None,
    ) -> Optional[str]:
        """
        เขียน Atom พร้อม TopicData

        category / primary สำหรับ path derive จาก topic:
            category = topic.label  หรือ  "cluster_{cluster_id}"
            primary  = topic.top_keyword  หรือ  "unknown"
        """
        if importance < 0.3:
            self._logger.debug(
                f"[MemoryController] SKIP importance too low ({importance})"
            )
            return None

        category = topic.label if topic.has_label else f"cluster_{topic.cluster_id}"
        primary  = topic.top_keyword or "unknown"
        target_tier = tier or self._select_tier(importance)

        raw_payload_preview = (
            data.payload[:32].decode("utf-8", errors="replace")
            if isinstance(data.payload, bytes)
            else str(data.payload)[:32]
        )
        raw     = f"{category}:{primary}:{raw_payload_preview}:{time.time()}".encode()
        atom_id = hashlib.sha256(raw).hexdigest()

        meta = {
            "category":   category,
            "primary":    primary,
            "importance": importance,
            "tier":       target_tier,
            "topic":      topic.to_dict(),
        }
        enriched = AtomData(
            payload       = data.payload,
            metadata      = json.dumps(meta).encode("utf-8"),
            source        = data.source,
            flags         = data.flags,
            created_ts_ms = data.created_ts_ms,
        )

        success = self._get_tier(target_tier).write(atom_id, enriched)

        if success:
            self._logger.info(
                f"[MemoryController] WRITE {atom_id[:8]} "
                f"[{category}/{primary}] tier:{target_tier} imp:{importance}"
            )
            return atom_id
        return None

    def write_response(
        self,
        text:       str,
        context:    str,
        importance: float = 0.5,
    ) -> Optional[str]:
        """
        BrainController ใช้ method นี้เพื่อบันทึก response

        BrainController ไม่ต้องรู้จัก AtomData หรือ TopicData โดยตรง
        MemoryController จัดการสร้าง Atom เอง

        Args:
            text       : response text ที่จะบันทึก
            context    : topic/domain (เช่น "math", "general")
            importance : ความสำคัญ 0.0–1.0

        Returns:
            atom_id ถ้าบันทึกสำเร็จ, None ถ้าไม่บันทึก
        """
        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_memory_allowed(text)
            if not _allowed:
                self._logger.warning(
                    f"[MemoryController] write_response BLOCKED reason={_reason}"
                )
                return None

        atom = AtomData(
            payload       = text.encode("utf-8") if isinstance(text, str) else text,
            source        = f"brain_response_{context}".encode("utf-8"),
        )
        topic = TopicData(
            cluster_id   = abs(hash(context)) % 10000,
            top_keywords = [context],
            coherence    = min(1.0, importance),
            label        = context,
        )
        return self.write(atom, topic, importance=importance)

    # ─────────────────────────────────────────
    # Read
    # ─────────────────────────────────────────

    def read(self, atom_id: str, tier: Optional[str] = None) -> Optional[AtomData]:
        """อ่าน raw AtomData"""
        if tier:
            return self._get_tier(tier).read(atom_id)
        for t in ["immortal", "long", "middle", "short"]:
            result = self._get_tier(t).read(atom_id)
            if result:
                return result
        return None

    def read_with_context(
        self,
        atom_id: str,
        tier:    Optional[str] = None,
    ) -> Optional[AtomContext]:
        """อ่าน Atom พร้อม deserialize TopicData จาก metadata"""
        found_tier = tier
        data       = None

        if tier:
            data = self._get_tier(tier).read(atom_id)
        else:
            for t in ["immortal", "long", "middle", "short"]:
                data = self._get_tier(t).read(atom_id)
                if data:
                    found_tier = t
                    break

        if data is None:
            return None

        topic_obj = None
        try:
            meta = json.loads(data.metadata.decode("utf-8"))
            if "topic" in meta:
                topic_obj = TopicData.from_dict(meta["topic"])
        except Exception as e:
            self._logger.warning(
                f"[MemoryController] metadata parse error {atom_id[:8]}: {e}"
            )

        return AtomContext(atom_id=atom_id, data=data, topic=topic_obj, tier=found_tier)

    def read_for_response(
        self,
        atom_ids: List[str],
        emotion:  Optional[EmotionData] = None,
        limit:    int                   = 5,
    ) -> List[WeightedAtom]:
        """
        ดึง atoms และ weight ด้วย VAD emotion signal

        BrainController ใช้ method นี้เพื่อขอความจำที่เหมาะสม
        กับอารมณ์ของ user ณ ขณะนั้น

        Args:
            atom_ids : list ของ atom_id ที่ต้องการ (หา topology ไว้ก่อน)
            emotion  : EmotionData จาก inference layer
                       None → ใช้ NEUTRAL_EMOTION
            limit    : จำนวน atom สูงสุดที่คืน

        Returns:
            List[WeightedAtom] เรียงจาก score สูง → ต่ำ
        """
        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_memory_allowed()
            if not _allowed:
                self._logger.warning(
                    f"[MemoryController] read_for_response BLOCKED reason={_reason}"
                )
                return []

        eff_emotion = emotion if emotion is not None else NEUTRAL_EMOTION
        results: List[WeightedAtom] = []

        for atom_id in atom_ids:
            ctx = self.read_with_context(atom_id)
            if ctx is None:
                continue

            try:
                meta       = json.loads(ctx.data.metadata.decode("utf-8"))
                importance = float(meta.get("importance", 0.3))
                coherence  = float(ctx.topic.coherence) if ctx.topic else 0.5
                tier_rank  = _TIER_RANK.get(ctx.tier or "short", 1)
                score      = self._emotion_weight(
                    importance, coherence, tier_rank, eff_emotion
                )
                results.append(WeightedAtom(
                    context    = ctx,
                    score      = score,
                    importance = importance,
                ))
            except Exception as e:
                self._logger.warning(
                    f"[MemoryController] weight error {atom_id[:8]}: {e}"
                )

        results.sort(key=lambda w: w.score, reverse=True)
        return results[:limit]

    def exists(self, atom_id: str, tier: Optional[str] = None) -> bool:
        if tier:
            return self._get_tier(tier).exists(atom_id)
        return any(
            self._get_tier(t).exists(atom_id)
            for t in ["short", "middle", "long", "immortal"]
        )

    # ─────────────────────────────────────────
    # Promote
    # ─────────────────────────────────────────

    def promote(
        self,
        atom_id:     str,
        from_tier:   str,
        reviewer_id: Optional[str] = None,
    ) -> bool:
        to_tier = self._next_tier(from_tier)
        if not to_tier:
            return False
        if to_tier == "immortal" and not reviewer_id:
            raise PermissionError(
                "[MemoryController] promote to immortal requires reviewer_id"
            )
        data = self._get_tier(from_tier).read(atom_id)
        if data is None:
            return False
        if not self._get_tier(to_tier).write(atom_id, data):
            return False
        self._get_tier(from_tier).delete(atom_id)
        self._logger.info(
            f"[MemoryController] PROMOTE {atom_id[:8]} {from_tier} → {to_tier}"
        )
        return True

    def auto_promote(self) -> dict:
        summary = {"short_to_middle": 0, "middle_to_long": 0}
        for atom_id in self._short.list_promotable():
            if self.promote(atom_id, "short"):
                summary["short_to_middle"] += 1
        for atom_id in self._middle.list_promotable():
            if self.promote(atom_id, "middle"):
                summary["middle_to_long"] += 1
        return summary

    # ─────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────

    def cleanup(self) -> dict:
        summary = {"short": 0, "middle": 0, "long": 0}
        for atom_id in self._middle.list_expired():
            if self._middle.delete(atom_id): summary["middle"] += 1
        for atom_id in self._long.list_expired():
            if self._long.delete(atom_id):   summary["long"] += 1
        promotable = set(self._short.list_promotable())
        for atom_id in self._short.list_stale():
            if atom_id not in promotable:
                if self._short.delete(atom_id): summary["short"] += 1
        return summary

    def clear_session(self) -> int:
        return self._short.clear()

    # ─────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "short":    self._short.count(),
            "middle":   self._middle.count(),
            "long":     self._long.count(),
            "immortal": self._immortal.count(),
        }

    # ─────────────────────────────────────────
    # VAD Weighting Formula
    # ─────────────────────────────────────────

    @staticmethod
    def _emotion_weight(
        importance: float,
        coherence:  float,
        tier_rank:  int,          # short=1, middle=2, long=3, immortal=4
        emotion:    EmotionData,
    ) -> float:
        """
        คำนวณ emotion-weighted score

        Valence  : negative → เน้น importance (ความจำที่หนักแน่น)
                   positive → เน้น coherence  (ความจำที่ชัดเจน)
        Arousal  : สูง → urgent → boost ทุก score × 1.0–1.5
        Dominance: สูง → ดึงจาก deep tier ด้วย
                   ต่ำ → focused เน้น shallow tier
        """
        # Valence blend: 0 = all-importance, 1 = all-coherence
        v_norm   = (emotion.valence + 1.0) / 2.0          # -1..1 → 0..1
        blended  = (1.0 - v_norm) * importance + v_norm * coherence

        # Arousal boost: calm=×1.0, full arousal=×1.5
        arousal_boost = 1.0 + emotion.arousal * 0.5

        # Dominance tier factor
        # dominance=1.0 + tier_rank=4 → factor = 1.0 + 0.5×1.0 = 1.5
        # dominance=0.0 + tier_rank=1 → factor = 1.0 + (-0.5)×0.25 = 0.875
        tier_factor = 1.0 + (emotion.dominance - 0.5) * (tier_rank / 4.0)

        return blended * arousal_boost * tier_factor

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _select_tier(self, importance: float) -> str:
        """
        เลือก tier ตาม importance จริงๆ

        short    < 0.4  — interaction ทั่วไป ไม่ต้องจำนาน
        middle   < 0.6  — น่าสนใจ จำระยะกลาง
        long     < 0.9  — สำคัญ จำนาน
        immortal >= 0.9 — สำคัญมาก จำตลอด
        """
        if importance >= 0.9:  return "immortal"
        if importance >= 0.6:  return "long"
        if importance >= 0.4:  return "middle"
        return "short"

    def _next_tier(self, tier: str) -> Optional[str]:
        order = ["short", "middle", "long", "immortal"]
        idx   = order.index(tier) if tier in order else -1
        return order[idx + 1] if 0 <= idx < len(order) - 1 else None

    def _get_tier(self, tier: str):
        return {
            "short":    self._short,
            "middle":   self._middle,
            "long":     self._long,
            "immortal": self._immortal,
        }[tier]