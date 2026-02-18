"""
Core/Memory/Structure/KnowletStructure.py

Knowlet — ข้อสรุปที่ได้จาก Atom หลายๆ ตัวที่มี Context เดียวกัน

กฎสำคัญ:
    1. ต้องอ้างอิง parent Atom เสมอ
    2. confidence ต้องสูงกว่า parent
    3. valid เมื่อ Atom ที่ category + primary เดียวกัน > 50% ของ Atom ทั้งหมด
    4. promote ได้ต่อเมื่อมี reviewer_id เท่านั้น

File Structure:
    {tier}/{category}/{primary}/{shard}/{atom_id}.atom
    เช่น short/conversation/python/01A/a1b2c3d4.atom

Shard:
    - ใช้ hex ของ atom_id
    - depth เริ่มต้น 2 ตัว
    - auto-expand เมื่อ folder entries > OS_FOLDER_LIMIT
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================================
# CONSTANTS
# ============================================================================

OS_FOLDER_LIMIT  = 4096   # entries ที่ระบบไฟล์ทำงานได้ดี
SHARD_DEPTH_MIN  = 2      # depth เริ่มต้น เช่น "01"
SHARD_DEPTH_MAX  = 8      # depth สูงสุด เช่น "01A2B3C4"
MAJORITY_RATIO   = 0.5    # Atom ที่ context เดียวกันต้องเกินครึ่ง


# ============================================================================
# KNOWLET DATA
# ============================================================================

@dataclass
class KnowletData:
    """
    Knowlet — ข้อสรุปจาก Atom หลายๆ ตัว

    Attributes:
        knowlet_id:    unique ID (hex)
        parent_ids:    atom_id ของ Atom ที่ใช้สรุป
        category:      TopicCategory ของ parent Atoms
        primary:       primary topic ของ parent Atoms
        summary:       ข้อสรุปที่ได้
        confidence:    ต้องสูงกว่า parent confidence
        parent_confidence: confidence เฉลี่ยของ parent Atoms
        is_promoted:   True เมื่อผ่าน Reviewer แล้ว
        reviewer_id:   ID ของ Reviewer ที่ approve
        created_at:    timestamp (ms)
        promoted_at:   timestamp ที่ promote (ms)
    """
    knowlet_id:        str
    parent_ids:        list[str]
    category:          str
    primary:           str
    summary:           str
    confidence:        float
    parent_confidence: float
    is_promoted:       bool  = False
    reviewer_id:       Optional[str] = None
    created_at:        int   = field(default_factory=lambda: int(time.time() * 1000))
    promoted_at:       Optional[int] = None

    def __post_init__(self):
        """Validate rules"""
        # Rule 1: ต้องมี parent Atom
        if not self.parent_ids:
            raise ValueError(
                f"[KnowletData] must reference at least one parent Atom"
            )

        # Rule 2: confidence ต้องสูงกว่า parent
        if self.confidence <= self.parent_confidence:
            raise ValueError(
                f"[KnowletData] confidence ({self.confidence}) "
                f"must be higher than parent_confidence ({self.parent_confidence})"
            )

        # Clamp confidence
        self.confidence        = max(0.0, min(1.0, self.confidence))
        self.parent_confidence = max(0.0, min(1.0, self.parent_confidence))

    @classmethod
    def create(
        cls,
        parent_ids:        list[str],
        category:          str,
        primary:           str,
        summary:           str,
        confidence:        float,
        parent_confidence: float,
    ) -> 'KnowletData':
        """
        Factory method สร้าง Knowlet พร้อม auto-generate knowlet_id
        """
        raw = f"{category}:{primary}:{summary}:{time.time()}".encode()
        knowlet_id = hashlib.sha256(raw).hexdigest()[:16]

        return cls(
            knowlet_id=knowlet_id,
            parent_ids=parent_ids,
            category=category,
            primary=primary,
            summary=summary,
            confidence=confidence,
            parent_confidence=parent_confidence,
        )

    def promote(self, reviewer_id: str) -> 'KnowletData':
        """
        Promote Knowlet — ต้องมี reviewer_id เท่านั้น
        คืน KnowletData ใหม่ที่ is_promoted = True
        """
        if not reviewer_id:
            raise PermissionError(
                "[KnowletData] promote requires reviewer_id"
            )

        return KnowletData(
            knowlet_id=self.knowlet_id,
            parent_ids=self.parent_ids,
            category=self.category,
            primary=self.primary,
            summary=self.summary,
            confidence=self.confidence,
            parent_confidence=self.parent_confidence,
            is_promoted=True,
            reviewer_id=reviewer_id,
            created_at=self.created_at,
            promoted_at=int(time.time() * 1000),
        )

    def to_dict(self) -> dict:
        return {
            'knowlet_id':        self.knowlet_id,
            'parent_ids':        self.parent_ids,
            'category':          self.category,
            'primary':           self.primary,
            'summary':           self.summary,
            'confidence':        self.confidence,
            'parent_confidence': self.parent_confidence,
            'is_promoted':       self.is_promoted,
            'reviewer_id':       self.reviewer_id,
            'created_at':        self.created_at,
            'promoted_at':       self.promoted_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowletData':
        return cls(
            knowlet_id=data['knowlet_id'],
            parent_ids=data['parent_ids'],
            category=data['category'],
            primary=data['primary'],
            summary=data['summary'],
            confidence=data['confidence'],
            parent_confidence=data['parent_confidence'],
            is_promoted=data.get('is_promoted', False),
            reviewer_id=data.get('reviewer_id'),
            created_at=data['created_at'],
            promoted_at=data.get('promoted_at'),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'KnowletData':
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        status = "promoted" if self.is_promoted else "draft"
        return (
            f"Knowlet({self.knowlet_id[:8]} "
            f"[{self.category}/{self.primary}] "
            f"conf:{self.confidence:.2f} "
            f"parents:{len(self.parent_ids)} "
            f"status:{status})"
        )


# ============================================================================
# SHARD PATH LOGIC
# ============================================================================

class ShardPath:
    """
    จัดการ file path ตาม structure:
        {base}/{tier}/{category}/{primary}/{shard}/{atom_id}.atom

    Shard auto-expand เมื่อ folder entries > OS_FOLDER_LIMIT
    """

    @staticmethod
    def get_shard(atom_id: str, depth: int) -> str:
        """คำนวณ shard folder จาก atom_id + depth"""
        return atom_id[:depth].upper()

    @staticmethod
    def detect_depth(topic_path: Path) -> int:
        """
        อ่าน folder จริงแล้วหา shard depth ที่ใช้อยู่
        ถ้าไม่มี folder ใดเลย → ใช้ SHARD_DEPTH_MIN
        """
        if not topic_path.exists():
            return SHARD_DEPTH_MIN

        shards = [p for p in topic_path.iterdir() if p.is_dir()]
        if not shards:
            return SHARD_DEPTH_MIN

        # ดู depth จาก folder name ที่มีอยู่
        return max(len(s.name) for s in shards)

    @staticmethod
    def should_expand(shard_path: Path) -> bool:
        """ตรวจว่า shard folder เกิน OS_FOLDER_LIMIT ไหม"""
        if not shard_path.exists():
            return False
        entries = sum(1 for _ in shard_path.iterdir())
        return entries > OS_FOLDER_LIMIT

    @staticmethod
    def build_path(
        base: Path,
        tier: str,
        category: str,
        primary: str,
        atom_id: str,
        depth: int,
    ) -> Path:
        """
        Build full path สำหรับ atom file
        {base}/{tier}/{category}/{primary}/{shard}/{atom_id}.atom
        """
        shard = ShardPath.get_shard(atom_id, depth)
        return base / tier / category / primary / shard / f"{atom_id}.atom"

    @staticmethod
    def build_knowlet_path(
        base: Path,
        category: str,
        primary: str,
        knowlet_id: str,
        depth: int,
    ) -> Path:
        """
        Build full path สำหรับ knowlet file
        {base}/knowlet/{category}/{primary}/{shard}/{knowlet_id}.knowlet
        """
        shard = ShardPath.get_shard(knowlet_id, depth)
        return base / "knowlet" / category / primary / shard / f"{knowlet_id}.knowlet"