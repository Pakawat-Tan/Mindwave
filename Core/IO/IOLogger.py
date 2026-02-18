"""
IOLogger — บันทึกทุก input/output ที่ผ่าน IOController
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from .IOPacket import IOPacket

logger = logging.getLogger("mindwave.io.logger")


class IOLogger:
    def __init__(self, log_dir: str = "Core/Data/io_logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = []
        self._session_start = time.time()

    def log(self, packet: IOPacket) -> None:
        record = packet.to_dict()
        self._records.append(record)
        logger.debug(
            f"[IOLogger] {packet.direction.value.upper()} "
            f"channel={packet.channel.value} "
            f"media={packet.media_type.value} "
            f"len={len(packet.text)}"
        )

    def flush(self) -> str:
        """เขียน log ลง disk"""
        path = self._log_dir / f"session_{int(self._session_start)}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._records.clear()
        return str(path)

    def stats(self) -> Dict[str, Any]:
        by_channel: Dict[str, int] = {}
        by_direction: Dict[str, int] = {"input": 0, "output": 0}
        for r in self._records:
            by_channel[r["channel"]] = by_channel.get(r["channel"], 0) + 1
            by_direction[r["direction"]] = by_direction.get(r["direction"], 0) + 1
        return {
            "total":        len(self._records),
            "by_channel":   by_channel,
            "by_direction": by_direction,
        }

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)