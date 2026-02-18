"""
Rule — กฎหมายเชิงโครงสร้าง (Immutable Structural Law)

คุณลักษณะ:
  - โหลดจาก JSON ตอน startup เท่านั้น
  - ไม่สามารถเปลี่ยนแปลงได้ที่ runtime โดยทั่วไป
  - เพิ่ม/ลบได้ผ่าน governance API ที่ต้องใช้ reviewer_id เท่านั้น
  - ละเมิด → หยุดทันที ไม่มีเงื่อนไข

Match Types:
  PATTERN  — keyword / regex ในข้อความ
  TOPIC    — ตรง TopicData.cluster_id
  BOTH     — ต้องตรงทั้ง pattern และ topic
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import json
import re
import uuid

# ============================================================================
# ENUMS
# ============================================================================

class RuleAuthority(Enum):
    """
    ระดับอำนาจของ Rule

    SYSTEM   : กฎหมายระดับระบบ — approve ได้เฉพาะ creator_id
    STANDARD : กฎทั่วไป — approve ได้ด้วย reviewer_id
    """
    SYSTEM   = "system"
    STANDARD = "standard"
    def __str__(self): return self.value.upper()


class RuleScope(Enum):
    INPUT       = "input"
    OUTPUT      = "output"
    MEMORY      = "memory"
    SYSTEM      = "system"
    SKILL       = "skill"
    NEURAL      = "neural"
    PERSONALITY = "personality"
    CONFIDENCE  = "confidence"
    def __str__(self): return self.value.upper()


class RuleAction(Enum):
    ALLOW    = "allow"
    BLOCK    = "block"
    REDIRECT = "redirect"
    def __str__(self): return self.value.upper()


class MatchType(Enum):
    PATTERN = "pattern"
    TOPIC   = "topic"
    BOTH    = "both"
    ANY     = "any"    # match ทุกอย่าง — block ทุก request ใน scope
    def __str__(self): return self.value.upper()


# ============================================================================
# RULE DATA
# ============================================================================

@dataclass(frozen=True)
class RuleData:
    """
    Hard constraint — กฎหมายเชิงโครงสร้างที่ไม่ต่อรอง

    frozen=True — ป้องกันการแก้ไข field ใด ๆ หลัง construction
    การเพิ่ม/ลบ Rule ต้องผ่าน governance_add / governance_remove
    ที่ต้องใช้ reviewer_id เท่านั้น

    Attributes:
        rule_id           : unique id
        scope             : INPUT / OUTPUT / MEMORY / SYSTEM
        action            : ALLOW / BLOCK / REDIRECT
        match_type        : PATTERN / TOPIC / BOTH
        pattern           : keyword หรือ regex string
        topic_cluster_id  : cluster_id เป้าหมาย
        priority          : สูง = ตรวจก่อน
        description       : อธิบาย rule
        use_regex         : True = regex, False = substring
    """
    scope:            RuleScope
    action:           RuleAction
    match_type:       MatchType
    authority:        RuleAuthority  = RuleAuthority.STANDARD
    rule_id:          str            = field(default_factory=lambda: str(uuid.uuid4()))
    pattern:          Optional[str]  = None
    topic_cluster_id: Optional[int]  = None
    priority:         int            = 0
    description:      str            = ""
    use_regex:        bool           = False

    def __post_init__(self) -> None:
        if self.match_type == MatchType.ANY:
            pass   # ANY ไม่ต้องการ pattern หรือ topic_cluster_id
        else:
            if self.match_type in (MatchType.PATTERN, MatchType.BOTH):
                if not self.pattern:
                    raise ValueError(
                        f"RuleData: match_type={self.match_type} requires 'pattern'"
                    )
            if self.match_type in (MatchType.TOPIC, MatchType.BOTH):
                if self.topic_cluster_id is None:
                    raise ValueError(
                        f"RuleData: match_type={self.match_type} requires 'topic_cluster_id'"
                    )

    # ------------------------------------------------------------------
    # Match Logic
    # ------------------------------------------------------------------

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
        """
        PATTERN : text ตรง pattern
        TOPIC   : cluster_id ตรง
        BOTH    : ต้องตรงทั้งสอง
        """
        if self.match_type == MatchType.ANY:
            return True   # block ทุกอย่างใน scope
        if self.match_type == MatchType.PATTERN:
            return self.matches_text(text)
        if self.match_type == MatchType.TOPIC:
            return self.matches_topic(cluster_id)
        if self.match_type == MatchType.BOTH:
            return self.matches_text(text) and self.matches_topic(cluster_id)
        return False

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id":          self.rule_id,
            "scope":            self.scope.value,
            "action":           self.action.value,
            "match_type":       self.match_type.value,
            "authority":        self.authority.value,
            "pattern":          self.pattern,
            "topic_cluster_id": self.topic_cluster_id,
            "priority":         self.priority,
            "description":      self.description,
            "use_regex":        self.use_regex,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuleData:
        return cls(
            rule_id          = data["rule_id"],
            scope            = RuleScope(data["scope"]),
            action           = RuleAction(data["action"]),
            match_type       = MatchType(data["match_type"]),
            authority        = RuleAuthority(data.get("authority", "standard")),
            pattern          = data.get("pattern"),
            topic_cluster_id = data.get("topic_cluster_id"),
            priority         = data.get("priority", 0),
            description      = data.get("description", ""),
            use_regex        = data.get("use_regex", False),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> RuleData:
        return cls.from_dict(json.loads(s))

    def __str__(self) -> str:
        return (
            f"Rule[{self.scope}|{self.action}|{self.match_type}] "
            f"'{self.description or self.rule_id[:8]}' priority={self.priority}"
        )


# ============================================================================
# RULE RESULT
# ============================================================================

@dataclass(frozen=True)
class RuleResult:
    """
    ผลลัพธ์จากการตรวจ Rule

    Attributes:
        action         : ALLOW / BLOCK / REDIRECT
        triggered_rule : Rule ที่ trigger (None = ไม่มี rule trigger)
        reason         : อธิบายเหตุผล
    """
    action:         RuleAction
    triggered_rule: Optional[RuleData] = None
    reason:         str                = ""

    @property
    def is_blocked(self)    -> bool: return self.action == RuleAction.BLOCK
    @property
    def is_allowed(self)    -> bool: return self.action == RuleAction.ALLOW
    @property
    def is_redirected(self) -> bool: return self.action == RuleAction.REDIRECT

    def __str__(self) -> str:
        rule_str = f" (rule: {self.triggered_rule.rule_id[:8]})" if self.triggered_rule else ""
        return f"RuleResult[{self.action}]{rule_str}: {self.reason}"

# ============================================================================
# FACTORY
# ============================================================================

def create_rule(
    scope:            RuleScope,
    action:           RuleAction,
    match_type:       MatchType,
    pattern:          Optional[str]  = None,
    topic_cluster_id: Optional[int]  = None,
    priority:         int            = 0,
    description:      str            = "",
    use_regex:        bool           = False,
    authority:        RuleAuthority  = RuleAuthority.STANDARD,
) -> RuleData:
    return RuleData(
        scope=scope, action=action, match_type=match_type,
        authority=authority, pattern=pattern,
        topic_cluster_id=topic_cluster_id,
        priority=priority, description=description, use_regex=use_regex,
    )

# Default result เมื่อไม่มี rule trigger
ALLOW_BY_DEFAULT = RuleResult(action=RuleAction.ALLOW, reason="no rule triggered")