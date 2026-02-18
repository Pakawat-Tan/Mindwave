"""
Core/Memory/Tiers/Short_term.py

Short-term Memory Tier
- เก็บ Atom ของ session ปัจจุบัน
- ล้างได้เมื่อ session จบ
- MemoryController เป็นคนตัดสินใจว่าจะ promote ขึ้น Middle หรือล้างทิ้ง

Adapt Layer:
    ShortTermMemory  →  serialize (json → bytes)  →  AtomData.payload  →  .atom file
    ShortTermMemory  ←  deserialize               ←  AtomData.payload  ←  .atom file
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


class Short_term(BaseTier):

    def __init__(self, base_path: str = "Core/Data/production/short"):
        self._data_path = Path(base_path)
        self.initialize()

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def tier_name(self) -> str:
        return "short"

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
        ล้าง Atom ทั้งหมดใน Short-term
        เรียกเมื่อ session จบ
        คืนจำนวน Atom ที่ลบไป
        """
        deleted = 0
        for atom_id in self._list_files():
            if self._delete_file(atom_id):
                deleted += 1

        self._logger.info(f"[{self.tier_name}] CLEAR — {deleted} atoms removed")
        return deleted

    # ─────────────────────────────────────────
    # Adapt Layer (ShortTermMemory layer)
    # ─────────────────────────────────────────

    def write_memory(self, memory: ShortTermMemory) -> bool:
        """
        รับ ShortTermMemory แล้วแปลงเป็น AtomData ก่อนเก็บลง disk

        ShortTermMemory → to_dict() → json → bytes → AtomData.payload
        """
        try:
            payload = json.dumps(memory.to_dict()).encode("utf-8")
            data = AtomData(
                payload=payload,
                source=b"short_term",
            )
            return self.write(memory.memory_id, data)
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] write_memory FAILED {memory.memory_id}: {e}")
            return False

    def read_memory(self, atom_id: str) -> ShortTermMemory | None:
        """
        อ่าน AtomData จาก disk แล้วแปลงกลับเป็น ShortTermMemory

        AtomData.payload → bytes → json → from_dict() → ShortTermMemory
        """
        data = self.read(atom_id)
        if data is None:
            return None

        try:
            raw = json.loads(data.payload.decode("utf-8"))
            return ShortTermMemory.from_dict(raw)
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

    def list_promotable(self) -> list[str]:
        """
        คืน atom_id ที่ importance >= SHORT_TERM_PROMOTION_THRESHOLD
        MemoryController ใช้ตัดสินใจว่าจะ promote ขึ้น Middle
        """
        promotable = []
        for atom_id in self.list():
            data = self.read(atom_id)
            if not data or not data.metadata:
                continue

            try:
                meta = json.loads(data.metadata.decode("utf-8"))
                if meta.get("importance", 0) >= SHORT_TERM_PROMOTION_THRESHOLD:
                    promotable.append(atom_id)
            except Exception:
                continue

        return promotable

    def is_full(self) -> bool:
        """
        ตรวจว่า Tier เต็ม capacity ไหม
        SHORT_TERM_MAX_CAPACITY = None หมายความว่า infinity — ไม่มีวันเต็ม
        """
        if SHORT_TERM_MAX_CAPACITY is None:
            return False
        return self.count() >= SHORT_TERM_MAX_CAPACITY
    
# ============================================================================
# CONSTANTS
# ============================================================================

# Short-term memory characteristics
SHORT_TERM_RETENTION_SECONDS   = 1800   # 30 minutes
SHORT_TERM_MAX_CAPACITY        = None   # infinity
SHORT_TERM_DEFAULT_IMPORTANCE  = 0.3    # minimum importance to store in this tier
SHORT_TERM_PROMOTION_THRESHOLD = 0.5    # promote to Middle when importance >= 0.5