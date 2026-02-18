"""
Core/Memory/Tiers/Long_term.py

Long-term Memory Tier
- เก็บ Atom ที่เสถียรแล้ว
- ล้างได้เมื่อ expired หรือ MemoryController สั่ง
- MemoryController เป็นคนตัดสินใจว่าจะ promote ขึ้น Immortal หรือล้างทิ้ง

Adapt Layer:
    LongTermMemory  →  serialize (json → bytes)  →  AtomData.payload  →  .atom file
    LongTermMemory  ←  deserialize               ←  AtomData.payload  ←  .atom file
"""

import json
import sys
from pathlib import Path

# ── sys.path setup ───────────────────────────────────────────────
_tiers_dir     = Path(__file__).parent                          # Core/Memory/Tiers/
_memory_dir    = _tiers_dir.parent                              # Core/Memory/
_structure_dir = _memory_dir / "Structure"                      # Core/Memory/Structure/
_project_root  = _memory_dir.parent.parent                      # MindWave#8/

for _p in [str(_structure_dir), str(_tiers_dir),
           str(_memory_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ────────────────────────────────────────────────────────────────

from AtomStructure import AtomData
from .base import BaseTier


class Long_term(BaseTier):

    def __init__(self, base_path: str = "Core/Data/production/long"):
        self._data_path = Path(base_path)
        self.initialize()

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def tier_name(self) -> str:
        return "long"

    @property
    def data_path(self) -> Path:
        return self._data_path

    @property
    def can_delete(self) -> bool:
        return True

    # ─────────────────────────────────────────
    # BaseTier Interface (AtomData layer)
    # ─────────────────────────────────────────

    def write(self, atom_id: str, data: AtomData) -> bool:
        return self._write_file(atom_id, data)

    def read(self, atom_id: str) -> AtomData | None:
        return self._read_file(atom_id)

    def exists(self, atom_id: str) -> bool:
        return self._atom_path(atom_id).exists()

    def list(self) -> list[str]:
        return self._list_files()

    def delete(self, atom_id: str) -> bool:
        self._guard_delete()
        return self._delete_file(atom_id)

    def count(self) -> int:
        return len(self._list_files())

    def clear(self) -> int:
        """
        ล้าง Atom ทั้งหมดใน Long-term
        คืนจำนวน Atom ที่ลบไป
        """
        deleted = 0
        for atom_id in self._list_files():
            if self._delete_file(atom_id):
                deleted += 1

        self._logger.info(f"[{self.tier_name}] CLEAR — {deleted} atoms removed")
        return deleted

    # ─────────────────────────────────────────
    # Adapt Layer (LongTermMemory layer)
    # ─────────────────────────────────────────

    def write_memory(self, memory: LongTermMemory) -> bool:
        """
        รับ LongTermMemory แล้วแปลงเป็น AtomData ก่อนเก็บลง disk

        LongTermMemory → to_dict() → json → bytes → AtomData.payload
        """
        try:
            payload = json.dumps(memory.to_dict()).encode("utf-8")
            data = AtomData(
                payload=payload,
                source=b"long_term",
            )
            return self.write(memory.memory_id, data)
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] write_memory FAILED {memory.memory_id}: {e}")
            return False

    def read_memory(self, atom_id: str) -> LongTermMemory | None:
        """
        อ่าน AtomData จาก disk แล้วแปลงกลับเป็น LongTermMemory

        AtomData.payload → bytes → json → from_dict() → LongTermMemory
        """
        data = self.read(atom_id)
        if data is None:
            return None

        try:
            raw = json.loads(data.payload.decode("utf-8"))
            return LongTermMemory.from_dict(raw)
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] read_memory FAILED {atom_id}: {e}")
            return None

    def list_stale(self) -> list[str]:
        """
        คืน atom_id ที่ is_stale = True
        MemoryController ใช้ตัดสินใจว่าจะลบหรือ promote
        """
        stale = []
        for atom_id in self.list():
            memory = self.read_memory(atom_id)
            if memory and memory.is_stale:
                stale.append(atom_id)
        return stale

    def list_expired(self) -> list[str]:
        """
        คืน atom_id ที่ is_expired = True (เกิน 7 วัน)
        MemoryController ใช้ตัดสินใจว่าจะลบทิ้ง
        """
        expired = []
        for atom_id in self.list():
            memory = self.read_memory(atom_id)
            if memory and memory.is_expired:
                expired.append(atom_id)
        return expired

    def list_promotable(self) -> list[str]:
        """
        คืน atom_id ที่ importance >= LONG_TERM_PROMOTION_THRESHOLD
        MemoryController ใช้ตัดสินใจว่าจะ promote ขึ้น Immortal
        """
        promotable = []
        for atom_id in self.list():
            data = self.read(atom_id)
            if not data or not data.metadata:
                continue
            try:
                meta = json.loads(data.metadata.decode("utf-8"))
                if meta.get("importance", 0) >= LONG_TERM_PROMOTION_THRESHOLD:
                    promotable.append(atom_id)
            except Exception:
                continue
        return promotable

    def is_full(self) -> bool:
        """ตรวจว่า Tier เต็ม capacity ไหม"""
        if LONG_TERM_MAX_CAPACITY is None:
            return False
        return self.count() >= LONG_TERM_MAX_CAPACITY
    
    
# ============================================================================
# CONSTANTS
# ============================================================================

# Long-term memory characteristics
LONG_TERM_RETENTION_SECONDS   = 604800  # 7 days
LONG_TERM_MAX_CAPACITY        = None    # infinity
LONG_TERM_DEFAULT_IMPORTANCE  = 0.7     # minimum importance to store in this tier
LONG_TERM_PROMOTION_THRESHOLD = 0.95    # promote to Immortal when importance >= 0.95