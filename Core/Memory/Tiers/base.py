"""
Core/Memory/Tiers/base.py

Base class สำหรับทุก Memory Tier
Tier คือ Storage Handler — รู้แค่ "เขียน/อ่านในพื้นที่ตัวเอง"
การตัดสินใจว่า Atom ควรอยู่ Tier ไหนเป็นหน้าที่ของ MemoryController
"""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

from ..Structure.AtomStructure import AtomData, AtomBinaryFormat
from ..Structure.AtomRepair import quick_check

class BaseTier(ABC):

    # ─────────────────────────────────────────
    # Properties ที่ทุก Tier ต้อง define
    # ─────────────────────────────────────────

    @property
    @abstractmethod
    def tier_name(self) -> str:
        """ชื่อของ Tier เช่น 'short', 'middle', 'long', 'immortal'"""
        ...

    @property
    @abstractmethod
    def data_path(self) -> Path:
        """Path ที่เก็บไฟล์ .atom ของ Tier นี้"""
        ...

    @property
    @abstractmethod
    def can_delete(self) -> bool:
        """
        Immortal = False (ลบไม่ได้เด็ดขาด)
        Short / Middle / Long = True
        """
        ...

    # ─────────────────────────────────────────
    # Interface ที่ทุก Tier ต้อง implement
    # ─────────────────────────────────────────

    @abstractmethod
    def write(self, atom_id: str, data: AtomData) -> bool:
        """บันทึก AtomData ลง Tier นี้"""
        return self._write_file(atom_id, data)

    @abstractmethod
    def read(self, atom_id: str) -> AtomData | None:
        """ดึง AtomData ด้วย atom_id — คืน None ถ้าไม่พบ"""
        return self._read_file(atom_id)

    @abstractmethod
    def exists(self, atom_id: str) -> bool:
        """ตรวจว่า Atom มีอยู่ใน Tier นี้ไหม"""
        return self._atom_path(atom_id).exists()

    @abstractmethod
    def list(self) -> list[str]:
        """คืน list ของ atom_id ทั้งหมดใน Tier นี้"""
        return self._list_files()

    @abstractmethod
    def delete(self, atom_id: str) -> bool:
        """
        ลบ Atom ออกจาก Tier
        Immortal ต้อง override method นี้ให้ raise PermissionError เสมอ
        """
        self._guard_delete()
        return self._delete_file(atom_id)

    @abstractmethod
    def count(self) -> int:
        """
        นับจำนวน Atom ทั้งหมดใน Tier
        MemoryController ใช้ตัดสินใจ promote หรือ cleanup
        """
        return len(self._list_files())

    @abstractmethod
    def clear(self) -> int:
        """
        ล้าง Atom ทั้งหมดใน Tier
        คืนจำนวน Atom ที่ลบไป
        Immortal ต้อง override method นี้ให้ raise PermissionError เสมอ
        """
        deleted = 0
        for atom_id in self._list_files():
            if self._delete_file(atom_id):
                deleted += 1
        self._logger.info(f"[{self.tier_name}] CLEAR — {deleted} atoms removed")
        return deleted

    # ─────────────────────────────────────────
    # Shared logic — ใช้ได้ทุก Tier
    # ─────────────────────────────────────────

    def initialize(self) -> None:
        """สร้าง data_path ถ้ายังไม่มี และเตรียม logger"""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"mindwave.memory.{self.tier_name}")
        self._logger.debug(f"[{self.tier_name}] initialized at {self.data_path}")

    def _atom_path(self, atom_id: str) -> Path:
        """แปลง atom_id เป็น path ของไฟล์ .atom"""
        return self.data_path / f"{atom_id}.atom"

    def _write_file(self, atom_id: str, data: AtomData) -> bool:
        """เขียนไฟล์ .atom ลง disk — ใช้ AtomBinaryFormat.save ตาม convention"""
        try:
            AtomBinaryFormat.save(str(self._atom_path(atom_id)), data)
            self._logger.info(f"[{self.tier_name}] WRITE {atom_id}")
            return True
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] WRITE FAILED {atom_id}: {e}")
            return False

    def _read_file(self, atom_id: str) -> AtomData | None:
        """อ่านไฟล์ .atom จาก disk — ตรวจ checksum ด้วย quick_check ก่อน"""
        path = self._atom_path(atom_id)

        if not path.exists():
            self._logger.debug(f"[{self.tier_name}] NOT FOUND {atom_id}")
            return None

        if not quick_check(str(path)):
            self._logger.warning(f"[{self.tier_name}] CHECKSUM FAIL {atom_id}")
            return None

        try:
            return AtomBinaryFormat.load(str(path))
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] READ FAILED {atom_id}: {e}")
            return None

    def _delete_file(self, atom_id: str) -> bool:
        """ลบไฟล์ .atom จาก disk"""
        path = self._atom_path(atom_id)

        if not path.exists():
            self._logger.debug(f"[{self.tier_name}] DELETE NOT FOUND {atom_id}")
            return False

        try:
            path.unlink()
            self._logger.warning(f"[{self.tier_name}] DELETE {atom_id}")
            return True
        except Exception as e:
            self._logger.error(f"[{self.tier_name}] DELETE FAILED {atom_id}: {e}")
            return False

    def _list_files(self) -> list[str]:
        """คืน atom_id ทั้งหมดจากไฟล์ .atom ที่มีอยู่"""
        return [p.stem for p in self.data_path.glob("*.atom")]

    def _guard_delete(self) -> None:
        """
        เรียกตอนต้นของ delete() ในทุก Tier
        Immortal override ตัวนี้ให้ raise ทันที
        """
        if not self.can_delete:
            raise PermissionError(
                f"[{self.tier_name}] delete is not allowed on this tier"
            )