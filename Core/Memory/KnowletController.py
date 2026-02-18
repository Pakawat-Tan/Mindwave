"""
Core/Memory/KnowletController.py

จัดการ Knowlet lifecycle ทั้งหมด
- สร้าง Knowlet จาก Atom ที่มี context เดียวกัน
- ตรวจ Majority Rule (> 50% ของ Atom ทั้งหมด)
- promote ผ่าน Reviewer เท่านั้น
- จัดการ shard path auto-expand
"""

import json
import logging
from pathlib import Path
import sys

from .Structure.AtomStructure import AtomBinaryFormat
from .Structure.AtomRepair import quick_check
from .Structure.KnowletStructure import (
    KnowletData,
    ShardPath,
    MAJORITY_RATIO,
    OS_FOLDER_LIMIT,
    SHARD_DEPTH_MIN,
)



class KnowletController:

    def __init__(self, base_path: str = "Core/Data"):
        self._base    = Path(base_path)
        self._logger  = logging.getLogger("mindwave.memory.knowlet")
        self._knowlet_base = self._base / "knowlet"
        self._knowlet_base.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # สร้าง Knowlet
    # ─────────────────────────────────────────

    def try_create(
        self,
        tier:              str,
        category:          str,
        primary:           str,
        summary:           str,
        confidence:        float,
    ) -> KnowletData | None:
        """
        ตรวจ Majority Rule แล้วสร้าง Knowlet ถ้าผ่าน

        Majority Rule:
            Atom ที่ category + primary เดียวกัน > 50% ของ Atom ทั้งหมดใน tier

        Returns:
            KnowletData ถ้าผ่าน Majority Rule
            None ถ้ายังไม่ผ่าน
        """
        tier_path = self._base / "production" / tier

        if not tier_path.exists():
            return None

        total_count = 0
        topic_count = 0
        parent_ids  = []

        depth = ShardPath.detect_depth(tier_path)

        for path in tier_path.rglob("*.atom"):
            total_count += 1

            if not quick_check(str(path)):
                continue

            try:
                atom = AtomBinaryFormat.load(str(path))
                meta = json.loads(atom.metadata.decode("utf-8"))

                if meta.get("category") == category and meta.get("primary") == primary:
                    topic_count += 1
                    parent_ids.append(path.stem)

            except Exception:
                continue

        if total_count == 0:
            return None

        ratio = topic_count / total_count
        if ratio < MAJORITY_RATIO:

            self._logger.debug(
                f"[knowlet] majority not reached "
                f"{topic_count}/{total_count} = {ratio:.2%} <= {MAJORITY_RATIO:.0%}"
            )
            return None

        # เก็บ parent_ids ของ Atom ใน topic นี้
        # parent_ids = self._list_atom_ids(topic_path)

        # คำนวณ parent_confidence เฉลี่ยจาก Atom จริง
        parent_confidence = self._avg_confidence(tier, category, primary, parent_ids)

        if confidence <= parent_confidence:
            self._logger.warning(
                f"[knowlet] confidence ({confidence}) must be > "
                f"parent_confidence ({parent_confidence:.3f})"
            )
            return None

        knowlet = KnowletData.create(
            parent_ids=parent_ids,
            category=category,
            primary=primary,
            summary=summary,
            confidence=confidence,
            parent_confidence=parent_confidence,
        )

        if self._write(knowlet):
            self._logger.info(
                f"[knowlet] CREATED {knowlet.knowlet_id[:8]} "
                f"[{category}/{primary}] "
                f"parents:{len(parent_ids)} ratio:{ratio:.2%}"
            )
            return knowlet

        return None

    # ─────────────────────────────────────────
    # Promote
    # ─────────────────────────────────────────

    def promote(self, knowlet_id: str, category: str, primary: str, reviewer_id: str) -> KnowletData | None:
        """
        Promote Knowlet — ต้องมี reviewer_id เท่านั้น
        """
        if not reviewer_id:
            raise PermissionError("[knowlet] promote requires reviewer_id")

        knowlet = self.read(knowlet_id, category, primary)
        if knowlet is None:
            self._logger.warning(f"[knowlet] PROMOTE NOT FOUND {knowlet_id[:8]}")
            return None

        if knowlet.is_promoted:
            self._logger.warning(f"[knowlet] ALREADY PROMOTED {knowlet_id[:8]}")
            return knowlet

        promoted = knowlet.promote(reviewer_id)
        if self._write(promoted):
            self._logger.info(
                f"[knowlet] PROMOTED {knowlet_id[:8]} by reviewer:{reviewer_id}"
            )
            return promoted

        return None

    # ─────────────────────────────────────────
    # Read / Write
    # ─────────────────────────────────────────

    def read(self, knowlet_id: str, category: str, primary: str) -> KnowletData | None:
        """อ่าน Knowlet จาก disk"""
        path = self._knowlet_path(knowlet_id, category, primary)

        if not path.exists():
            self._logger.debug(f"[knowlet] NOT FOUND {knowlet_id[:8]}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return KnowletData.from_json(f.read())
        except Exception as e:
            self._logger.error(f"[knowlet] READ FAILED {knowlet_id[:8]}: {e}")
            return None

    def _write(self, knowlet: KnowletData) -> bool:
        """เขียน Knowlet ลง disk"""
        path = self._knowlet_path(knowlet.knowlet_id, knowlet.category, knowlet.primary)
        path.parent.mkdir(parents=True, exist_ok=True)

        # ตรวจ shard expansion
        self._maybe_expand(path.parent, knowlet.category, knowlet.primary)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(knowlet.to_json())
            return True
        except Exception as e:
            self._logger.error(f"[knowlet] WRITE FAILED {knowlet.knowlet_id[:8]}: {e}")
            return False

    # ─────────────────────────────────────────
    # List
    # ─────────────────────────────────────────

    def list_draft(self, category: str, primary: str) -> list[str]:
        """คืน knowlet_id ที่ยังไม่ promote"""
        return self._list_by_status(category, primary, promoted=False)

    def list_promoted(self, category: str, primary: str) -> list[str]:
        """คืน knowlet_id ที่ promote แล้ว"""
        return self._list_by_status(category, primary, promoted=True)

    def _list_by_status(self, category: str, primary: str, promoted: bool) -> list[str]:
        topic_path = self._knowlet_base / category / primary
        if not topic_path.exists():
            return []

        result = []
        for path in topic_path.rglob("*.knowlet"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    k = KnowletData.from_json(f.read())
                if k.is_promoted == promoted:
                    result.append(k.knowlet_id)
            except Exception:
                continue
        return result

    # ─────────────────────────────────────────
    # Shard Path
    # ─────────────────────────────────────────

    def _knowlet_path(self, knowlet_id: str, category: str, primary: str) -> Path:
        topic_path = self._knowlet_base / category / primary
        depth = ShardPath.detect_depth(topic_path)
        return ShardPath.build_knowlet_path(
            self._base, category, primary, knowlet_id, depth
        )

    def _maybe_expand(self, shard_path: Path, category: str, primary: str) -> None:
        """ขยาย shard depth ถ้า folder เกิน OS_FOLDER_LIMIT"""
        if not ShardPath.should_expand(shard_path):
            return

        topic_path  = self._knowlet_base / category / primary
        old_depth   = ShardPath.detect_depth(topic_path)
        new_depth   = min(old_depth + 1, 8)

        self._logger.info(
            f"[knowlet] SHARD EXPAND [{category}/{primary}] "
            f"depth {old_depth} → {new_depth}"
        )

        # ย้ายไฟล์ไป structure ใหม่
        for old_path in topic_path.rglob("*.knowlet"):
            knowlet_id = old_path.stem
            new_shard  = ShardPath.get_shard(knowlet_id, new_depth)
            new_path   = topic_path / new_shard / old_path.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)

    # ─────────────────────────────────────────
    # Atom Helpers (ใช้ตรวจ Majority Rule)
    # ─────────────────────────────────────────

    def _count_atoms(self, topic_path: Path) -> int:
        """นับ Atom ใน topic path"""
        if not topic_path.exists():
            return 0
        return sum(1 for _ in topic_path.rglob("*.atom"))

    def _count_all_atoms(self, tier_path: Path) -> int:
        """นับ Atom ทั้งหมดใน tier"""
        if not tier_path.exists():
            return 0
        return sum(1 for _ in tier_path.rglob("*.atom"))

    def _list_atom_ids(self, topic_path: Path) -> list[str]:
        """คืน atom_id ทั้งหมดใน topic"""
        if not topic_path.exists():
            return []
        return [p.stem for p in topic_path.rglob("*.atom")]

    def _avg_confidence(
        self,
        tier: str,
        category: str,
        primary: str,
        atom_ids: list[str],
    ) -> float:
        """
        คำนวณ confidence เฉลี่ยจาก metadata ของ Atom
        ถ้าอ่านไม่ได้ → ใช้ 0.5 เป็น fallback
        """
        if not atom_ids:
            return 0.5

        scores  = []
        depth   = ShardPath.detect_depth(
            self._base / "production" / tier / category / primary
        )

        for atom_id in atom_ids:
            path = ShardPath.build_path(
                self._base, tier, category, primary, atom_id, depth
            )
            if not path.exists() or not quick_check(str(path)):
                continue
            try:
                atom = AtomBinaryFormat.load(str(path))
                meta = json.loads(atom.metadata.decode("utf-8"))
                scores.append(float(meta.get("confidence", 0.0)))
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0