# Random-once, fix หลัง init
"""
PersonalityData — จิตวิญญาณของระบบ

Spec (Phase 3):
  - Random ครั้งเดียวตอน first creation จาก preset profiles
  - หลังจากนั้น Fix — ห้ามเปลี่ยนโดยไม่ผ่าน Creator
  - Creator เท่านั้นที่แก้ไขได้ (ต้องผ่าน reviewer)
  - ควบคุมเฉพาะ: Tone / Friendliness / Firmness / Response style / Humor / Empathy
  - ไม่เกี่ยวกับ: Belief / Rule / Confidence

Execution Priority:
  Rule → Confidence → Skill → Personality → Emotion
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import random
import time
import uuid


# ============================================================================
# DIMENSION ENUMS
# ============================================================================

class Tone(Enum):
    FORMAL  = "formal"
    CASUAL  = "casual"
    NEUTRAL = "neutral"
    def __str__(self): return self.value

class Friendliness(Enum):
    WARM     = "warm"
    COLD     = "cold"
    BALANCED = "balanced"
    def __str__(self): return self.value

class Firmness(Enum):
    ASSERTIVE = "assertive"
    GENTLE    = "gentle"
    ADAPTIVE  = "adaptive"
    def __str__(self): return self.value

class ResponseStyle(Enum):
    VERBOSE    = "verbose"
    CONCISE    = "concise"
    STRUCTURED = "structured"
    def __str__(self): return self.value

class Humor(Enum):
    SERIOUS  = "serious"
    PLAYFUL  = "playful"
    DRY      = "dry"
    def __str__(self): return self.value

class Empathy(Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    def __str__(self): return self.value


# ============================================================================
# PRESET PROFILES
# ============================================================================

@dataclass(frozen=True)
class PersonalityProfile:
    """
    Preset profile — combination ของทุก dimension

    Random ครั้งแรกจาก PROFILES registry
    """
    profile_name:   str
    tone:           Tone
    friendliness:   Friendliness
    firmness:       Firmness
    response_style: ResponseStyle
    humor:          Humor
    empathy:        Empathy
    description:    str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name":   self.profile_name,
            "tone":           self.tone.value,
            "friendliness":   self.friendliness.value,
            "firmness":       self.firmness.value,
            "response_style": self.response_style.value,
            "humor":          self.humor.value,
            "empathy":        self.empathy.value,
            "description":    self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersonalityProfile:
        return cls(
            profile_name   = data["profile_name"],
            tone           = Tone(data["tone"]),
            friendliness   = Friendliness(data["friendliness"]),
            firmness       = Firmness(data["firmness"]),
            response_style = ResponseStyle(data["response_style"]),
            humor          = Humor(data["humor"]),
            empathy        = Empathy(data["empathy"]),
            description    = data.get("description", ""),
        )

    def __str__(self) -> str:
        return (
            f"Profile[{self.profile_name}] "
            f"tone={self.tone} friendly={self.friendliness} "
            f"firm={self.firmness} style={self.response_style} "
            f"humor={self.humor} empathy={self.empathy}"
        )


# ── Preset Profiles Registry ──────────────────────────────────────────────

PROFILES: Dict[str, PersonalityProfile] = {
    p.profile_name: p for p in [
        PersonalityProfile(
            profile_name   = "Friendly",
            tone           = Tone.CASUAL,
            friendliness   = Friendliness.WARM,
            firmness       = Firmness.GENTLE,
            response_style = ResponseStyle.CONCISE,
            humor          = Humor.PLAYFUL,
            empathy        = Empathy.HIGH,
            description    = "อบอุ่น เป็นกันเอง เน้นความสัมพันธ์",
        ),
        PersonalityProfile(
            profile_name   = "Professional",
            tone           = Tone.FORMAL,
            friendliness   = Friendliness.BALANCED,
            firmness       = Firmness.ASSERTIVE,
            response_style = ResponseStyle.STRUCTURED,
            humor          = Humor.SERIOUS,
            empathy        = Empathy.MEDIUM,
            description    = "เป็นทางการ มั่นใจ มีโครงสร้างชัดเจน",
        ),
        PersonalityProfile(
            profile_name   = "Balanced",
            tone           = Tone.NEUTRAL,
            friendliness   = Friendliness.BALANCED,
            firmness       = Firmness.ADAPTIVE,
            response_style = ResponseStyle.CONCISE,
            humor          = Humor.DRY,
            empathy        = Empathy.MEDIUM,
            description    = "สมดุล ปรับตัวได้ ไม่เอนเอียง",
        ),
        PersonalityProfile(
            profile_name   = "Empathetic",
            tone           = Tone.CASUAL,
            friendliness   = Friendliness.WARM,
            firmness       = Firmness.GENTLE,
            response_style = ResponseStyle.VERBOSE,
            humor          = Humor.SERIOUS,
            empathy        = Empathy.HIGH,
            description    = "เอาใจใส่ ฟังมาก พูดละเอียด",
        ),
        PersonalityProfile(
            profile_name   = "Direct",
            tone           = Tone.NEUTRAL,
            friendliness   = Friendliness.COLD,
            firmness       = Firmness.ASSERTIVE,
            response_style = ResponseStyle.CONCISE,
            humor          = Humor.SERIOUS,
            empathy        = Empathy.LOW,
            description    = "ตรงไปตรงมา กระชับ ไม่อ้อมค้อม",
        ),
        PersonalityProfile(
            profile_name   = "Creative",
            tone           = Tone.CASUAL,
            friendliness   = Friendliness.WARM,
            firmness       = Firmness.ADAPTIVE,
            response_style = ResponseStyle.VERBOSE,
            humor          = Humor.PLAYFUL,
            empathy        = Empathy.HIGH,
            description    = "สร้างสรรค์ เปิดกว้าง ชอบสำรวจไอเดีย",
        ),
    ]
}


def random_profile(seed: Optional[int] = None) -> PersonalityProfile:
    """สุ่ม profile ครั้งแรก — seed ได้สำหรับ reproducibility"""
    rng = random.Random(seed)
    return rng.choice(list(PROFILES.values()))


# ============================================================================
# PERSONALITY CHANGE EVENT — audit trail
# ============================================================================

@dataclass(frozen=True)
class PersonalityChangeEvent:
    """
    บันทึกทุกครั้งที่ personality ถูกเปลี่ยน (Creator only)
    """
    event_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    changed_by:   str   = ""
    from_profile: str   = ""
    to_profile:   str   = ""
    reason:       str   = ""
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":     self.event_id,
            "changed_by":   self.changed_by,
            "from_profile": self.from_profile,
            "to_profile":   self.to_profile,
            "reason":       self.reason,
            "timestamp":    self.timestamp,
        }

    def __str__(self) -> str:
        return (
            f"PersonalityChange[{self.from_profile} → {self.to_profile}] "
            f"by='{self.changed_by}' reason='{self.reason}'"
        )


# ============================================================================
# PERSONALITY DATA — the soul
# ============================================================================

@dataclass
class PersonalityData:
    """
    จิตวิญญาณของระบบ — fixed after init

    Rules:
      - สร้างครั้งเดียวตอน init (random from profiles)
      - เปลี่ยนได้เฉพาะ creator เรียก change()
      - ทุกการเปลี่ยนแปลงถูก log
      - ไม่เกี่ยวกับ Rule / Confidence / Belief
    """
    _profile:  PersonalityProfile = field(init=False, repr=False)
    _events:   List[PersonalityChangeEvent] = field(
        default_factory=list, init=False, repr=False
    )
    _created_ts: float = field(default_factory=time.time, init=False)

    def __init__(self, seed: Optional[int] = None):
        """
        Init ครั้งแรก — random profile
        seed ใช้สำหรับ test reproducibility
        """
        self._profile    = random_profile(seed=seed)
        self._events     = []
        self._created_ts = time.time()

    # ── Read-only profile access ──────────────────────────────────

    @property
    def profile(self) -> PersonalityProfile:
        return self._profile

    @property
    def profile_name(self) -> str:
        return self._profile.profile_name

    @property
    def tone(self)           -> Tone:           return self._profile.tone
    @property
    def friendliness(self)   -> Friendliness:   return self._profile.friendliness
    @property
    def firmness(self)       -> Firmness:       return self._profile.firmness
    @property
    def response_style(self) -> ResponseStyle:  return self._profile.response_style
    @property
    def humor(self)          -> Humor:          return self._profile.humor
    @property
    def empathy(self)        -> Empathy:        return self._profile.empathy

    @property
    def change_count(self) -> int:
        return len(self._events)

    @property
    def events(self) -> List[PersonalityChangeEvent]:
        return list(self._events)

    # ── Change — Creator only ─────────────────────────────────────

    def change(
        self,
        new_profile_name: str,
        creator_id:       str,
        reason:           str = "",
    ) -> PersonalityChangeEvent:
        """
        เปลี่ยน personality profile

        ต้องมี creator_id เท่านั้น — PermissionError ถ้าว่าง
        ต้อง profile_name มีใน PROFILES — ValueError ถ้าไม่พบ
        """
        if not creator_id or not creator_id.strip():
            raise PermissionError(
                "PersonalityData.change: requires creator_id"
            )
        if new_profile_name not in PROFILES:
            raise ValueError(
                f"PersonalityData.change: "
                f"unknown profile '{new_profile_name}'. "
                f"Available: {list(PROFILES.keys())}"
            )

        event = PersonalityChangeEvent(
            changed_by   = creator_id,
            from_profile = self._profile.profile_name,
            to_profile   = new_profile_name,
            reason       = reason,
        )
        self._profile = PROFILES[new_profile_name]
        self._events.append(event)
        return event

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile":     self._profile.to_dict(),
            "created_ts":  self._created_ts,
            "change_count": self.change_count,
        }

    def __str__(self) -> str:
        return f"Personality[{self._profile.profile_name}] changes={self.change_count}"