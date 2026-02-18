"""
Policy — แนวทางปฏิบัติ (Operational Strategy)

แยกเป็น 2 ประเภท:
  NumericPolicy   : value-based parameter
                    เช่น timeout=3600, max_depth=3, threshold=0.5
  BehavioralPolicy: weight-based modifier
                    เช่น ×1.5 boost, ×0.5 suppress

คุณลักษณะ:
  - ปรับได้ผ่าน governance process
  - ต้องไม่ขัดแย้งกับ Rule
  - Rule ละเมิด → Policy ไม่มีผล
  - Policy ตอบ "ควรทำอย่างไร" ไม่ใช่ "อนุญาตหรือไม่"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Union
import json
import re
import uuid


# ============================================================================
# ENUMS
# ============================================================================

class PolicyScope(Enum):
    INPUT  = "input"
    OUTPUT = "output"
    MEMORY = "memory"
    SYSTEM = "system"
    def __str__(self): return self.value.upper()


class PolicyType(Enum):
    NUMERIC    = "numeric"     # key-value parameter (timeout, threshold, limit)
    BEHAVIORAL = "behavioral"  # weight modifier (boost / suppress)
    def __str__(self): return self.value.upper()


class MatchType(Enum):
    PATTERN = "pattern"
    TOPIC   = "topic"
    BOTH    = "both"
    def __str__(self): return self.value.upper()


# ============================================================================
# NUMERIC POLICY  — operational parameters
# ============================================================================

@dataclass
class NumericPolicy:
    """
    Value-based operational parameter

    ตัวอย่าง:
        NumericPolicy(scope=SYSTEM, key="resume_timeout",   value=3600.0)
        NumericPolicy(scope=MEMORY, key="max_depth",        value=3.0)
        NumericPolicy(scope=OUTPUT, key="confidence_threshold", value=0.7)

    Attributes:
        policy_id  : unique id
        scope      : INPUT / OUTPUT / MEMORY / SYSTEM
        key        : ชื่อ parameter
        value      : ค่า parameter (float)
        description: อธิบาย
        is_active  : เปิด/ปิด
    """
    scope:       PolicyScope
    key:         str
    value:       float
    policy_id:   str  = field(default_factory=lambda: str(uuid.uuid4()))
    description: str  = ""
    is_active:   bool = True

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("NumericPolicy: key must not be empty")

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.NUMERIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_type": PolicyType.NUMERIC.value,
            "policy_id":   self.policy_id,
            "scope":       self.scope.value,
            "key":         self.key,
            "value":       self.value,
            "description": self.description,
            "is_active":   self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NumericPolicy:
        return cls(
            policy_id   = data["policy_id"],
            scope       = PolicyScope(data["scope"]),
            key         = data["key"],
            value       = data["value"],
            description = data.get("description", ""),
            is_active   = data.get("is_active", True),
        )

    def to_json(self) -> str: return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return (
            f"NumericPolicy[{self.scope}] "
            f"{self.key}={self.value} "
            f"'{self.description or self.policy_id[:8]}'"
        )


# ============================================================================
# BEHAVIORAL POLICY  — weight modifier
# ============================================================================

@dataclass
class BehavioralPolicy:
    """
    Weight-based behavior modifier

    ตัวอย่าง:
        BehavioralPolicy(scope=MEMORY, match_type=TOPIC,
                         topic_cluster_id=42, weight=1.5)  # boost
        BehavioralPolicy(scope=OUTPUT, match_type=PATTERN,
                         pattern="spoiler", weight=0.3)    # suppress

    Attributes:
        policy_id         : unique id
        scope             : INPUT / OUTPUT / MEMORY / SYSTEM
        match_type        : PATTERN / TOPIC / BOTH
        weight            : modifier (>1 = boost, <1 = suppress, 1.0 = neutral)
        pattern           : keyword หรือ regex
        topic_cluster_id  : cluster_id เป้าหมาย
        priority          : สูง = apply ก่อน
        is_active         : เปิด/ปิด
        use_regex         : True = regex
        description       : อธิบาย
    """
    scope:            PolicyScope
    match_type:       MatchType
    weight:           float
    policy_id:        str           = field(default_factory=lambda: str(uuid.uuid4()))
    pattern:          Optional[str] = None
    topic_cluster_id: Optional[int] = None
    priority:         int           = 0
    is_active:        bool          = True
    use_regex:        bool          = False
    description:      str           = ""

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("BehavioralPolicy: weight must be > 0")
        if self.match_type in (MatchType.PATTERN, MatchType.BOTH):
            if not self.pattern:
                raise ValueError(
                    f"BehavioralPolicy: match_type={self.match_type} requires 'pattern'"
                )
        if self.match_type in (MatchType.TOPIC, MatchType.BOTH):
            if self.topic_cluster_id is None:
                raise ValueError(
                    f"BehavioralPolicy: match_type={self.match_type} requires 'topic_cluster_id'"
                )

    @property
    def policy_type(self) -> PolicyType:
        return PolicyType.BEHAVIORAL

    @property
    def is_boosted(self)    -> bool: return self.weight > 1.0
    @property
    def is_suppressed(self) -> bool: return self.weight < 1.0
    @property
    def is_neutral(self)    -> bool: return abs(self.weight - 1.0) < 1e-6

    def matches_text(self, text: str) -> bool:
        if not self.pattern:
            return False
        if self.use_regex:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        return self.pattern.lower() in text.lower()

    def matches_topic(self, cluster_id: Optional[int]) -> bool:
        if self.topic_cluster_id is None or cluster_id is None:
            return False
        return self.topic_cluster_id == cluster_id

    def matches(self, text: str = "", cluster_id: Optional[int] = None) -> bool:
        if not self.is_active:
            return False
        if self.match_type == MatchType.PATTERN:
            return self.matches_text(text)
        if self.match_type == MatchType.TOPIC:
            return self.matches_topic(cluster_id)
        if self.match_type == MatchType.BOTH:
            return self.matches_text(text) and self.matches_topic(cluster_id)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_type":      PolicyType.BEHAVIORAL.value,
            "policy_id":        self.policy_id,
            "scope":            self.scope.value,
            "match_type":       self.match_type.value,
            "weight":           self.weight,
            "pattern":          self.pattern,
            "topic_cluster_id": self.topic_cluster_id,
            "priority":         self.priority,
            "is_active":        self.is_active,
            "use_regex":        self.use_regex,
            "description":      self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BehavioralPolicy:
        return cls(
            policy_id        = data["policy_id"],
            scope            = PolicyScope(data["scope"]),
            match_type       = MatchType(data["match_type"]),
            weight           = data["weight"],
            pattern          = data.get("pattern"),
            topic_cluster_id = data.get("topic_cluster_id"),
            priority         = data.get("priority", 0),
            is_active        = data.get("is_active", True),
            use_regex        = data.get("use_regex", False),
            description      = data.get("description", ""),
        )

    def to_json(self) -> str: return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return (
            f"BehavioralPolicy[{self.scope}|{self.match_type}] "
            f"weight={self.weight:.2f} "
            f"'{self.description or self.policy_id[:8]}'"
        )


# ============================================================================
# UNION TYPE
# ============================================================================

AnyPolicy = Union[NumericPolicy, BehavioralPolicy]


def policy_from_dict(data: Dict[str, Any]) -> AnyPolicy:
    """Deserialize ไม่ว่าจะเป็น Numeric หรือ Behavioral"""
    t = data.get("policy_type", PolicyType.BEHAVIORAL.value)
    if t == PolicyType.NUMERIC.value:
        return NumericPolicy.from_dict(data)
    return BehavioralPolicy.from_dict(data)


# ============================================================================
# POLICY RESULT
# ============================================================================

@dataclass(frozen=True)
class PolicyResult:
    """
    ผลลัพธ์จาก BehavioralPolicy evaluation

    Attributes:
        final_weight     : product ของทุก weight ที่ match
        applied_policies : รายการ BehavioralPolicy ที่ trigger
        reason           : สรุป
    """
    final_weight:     float
    applied_policies: tuple = field(default_factory=tuple)
    reason:           str   = ""

    @property
    def is_boosted(self)    -> bool: return self.final_weight > 1.0
    @property
    def is_suppressed(self) -> bool: return self.final_weight < 1.0
    @property
    def is_neutral(self)    -> bool: return abs(self.final_weight - 1.0) < 1e-6

    def __str__(self) -> str:
        n = len(self.applied_policies)
        return f"PolicyResult[weight={self.final_weight:.3f} from {n} policies]: {self.reason}"


# Neutral sentinel
NEUTRAL_POLICY_RESULT = PolicyResult(final_weight=1.0, reason="no behavioral policy applied")