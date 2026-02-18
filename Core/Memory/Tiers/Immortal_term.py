"""
Core/Memory/Tiers/Immortal_term.py

Immortal Memory Tier — Identity Lock
- ลบไม่ได้เด็ดขาด
- แก้ไขต้องผ่าน Reviewer เท่านั้น
- ไม่มี promote ออก

Adapt Layer:
    ImmortalMemory  →  serialize (json → bytes)  →  AtomData.payload  →  .atom file
    ImmortalMemory  ←  deserialize               ←  AtomData.payload  ←  .atom file
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


class Immortal_term(BaseTier):

    def __init__(self, base_path: str = "Core/Data/production/immortal"):
        self._data_path = Path(base_path)
        self.initialize()

    # ─────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────

    @property
    def tier_name(self) -> str:
        return "immortal"

    @property
    def data_path(self) -> Path:
        return self._data_path

    @property
    def can_delete(self) -> bool:
        return False   # ❌ ลบไม่ได้เด็ดขาด

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
        """❌ Immortal ลบไม่ได้เด็ดขาด"""
        self._guard_delete()  # raises PermissionError เสมอ

    def count(self) -> int:
        return len(self._list_files())

    def clear(self) -> int:
        """❌ Immortal ล้างไม่ได้เด็ดขาด"""
        raise PermissionError(
            "[immortal] clear is not allowed — Immortal tier cannot be wiped"
        )

    # ─────────────────────────────────────────
    # Adapt Layer (ImmortalMemory layer)
    # ─────────────────────────────────────────

    def write_memory(self, memory: ImmortalMemory) -> bool:
        """
        รับ ImmortalMemory แล้วแปลงเป็น AtomData ก่อนเก็บลง disk

        ImmortalMemory → to_dict() → json → bytes → AtomData.payload
        """
        try:
            payload = json.dumps(memory.to_dict()).encode("utf-8")
            data = AtomData(
                payload=payload,
                source=b"immortal_term",
            )
            return self.write(memory.memory_id, data)
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] write_memory FAILED {memory.memory_id}: {e}")
            return False

    def read_memory(self, atom_id: str) -> ImmortalMemory | None:
        """
        อ่าน AtomData จาก disk แล้วแปลงกลับเป็น ImmortalMemory

        AtomData.payload → bytes → json → from_dict() → ImmortalMemory
        """
        data = self.read(atom_id)
        if data is None:
            return None

        try:
            raw = json.loads(data.payload.decode("utf-8"))
            return ImmortalMemory.from_dict(raw)
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] read_memory FAILED {atom_id}: {e}")
            return None

    def list_stale(self) -> list[str]:
        """
        คืน atom_id ที่ is_stale = True
        Immortal ไม่ expire แต่ยังมี stale สำหรับ monitoring
        """
        stale = []
        for atom_id in self.list():
            memory = self.read_memory(atom_id)
            if memory and memory.is_stale:
                stale.append(atom_id)
        return stale

    def is_full(self) -> bool:
        """Immortal ไม่มี capacity limit"""
        return False

# ============================================================================
# CONSTANTS
# ============================================================================

# Immortal memory characteristics
IMMORTAL_RETENTION_SECONDS   = None   # infinity — never expires
IMMORTAL_MAX_CAPACITY        = None   # infinity
IMMORTAL_DEFAULT_IMPORTANCE  = 0.95   # minimum importance to store in this tier
IMMORTAL_PROMOTION_THRESHOLD = 1.0    # no promotion out — 1.0 is unreachable
