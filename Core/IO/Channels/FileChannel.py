"""Core/IO/Channels/FileChannel.py — txt / json / pdf / docx"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.file")


class FileChannel:
    """อ่าน/เขียนไฟล์ทุกประเภท"""

    # ─── Read ─────────────────────────────────────────────────────

    def read(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """อ่านไฟล์ → IOPacket"""
        p = Path(path)
        if not p.exists():
            logger.warning(f"[FileChannel] NOT FOUND: {path}")
            return None

        ext = p.suffix.lower()
        try:
            if ext in (".txt", ".md"):
                return self._read_text(p, context)
            elif ext == ".json":
                return self._read_json(p, context)
            elif ext == ".pdf":
                return self._read_pdf(p, context)
            elif ext in (".docx", ".doc"):
                return self._read_docx(p, context)
            else:
                # fallback — อ่านเป็น text
                return self._read_text(p, context)
        except Exception as e:
            logger.error(f"[FileChannel] READ FAILED {path}: {e}")
            return None

    def read_all(self, directory: str, pattern: str = "*", context: str = "general") -> List[IOPacket]:
        """อ่านทุกไฟล์ใน directory"""
        packets = []
        for p in Path(directory).glob(pattern):
            if p.is_file():
                pkt = self.read(str(p), context)
                if pkt:
                    packets.append(pkt)
        return packets

    def _read_text(self, path: Path, context: str) -> IOPacket:
        text = path.read_text(encoding="utf-8", errors="replace")
        return IOPacket(
            channel    = ChannelType.FILE,
            direction  = PacketDirection.INPUT,
            media_type = MediaType.TEXT,
            text       = text,
            source     = str(path),
            context    = context,
            meta       = {"filename": path.name, "ext": path.suffix},
        )

    def _read_json(self, path: Path, context: str) -> IOPacket:
        data = json.loads(path.read_text(encoding="utf-8"))
        # แปลง dict/list → text
        if isinstance(data, dict):
            text = "\n".join(f"{k}: {v}" for k, v in data.items())
        elif isinstance(data, list):
            text = "\n".join(str(item) for item in data)
        else:
            text = str(data)
        return IOPacket(
            channel    = ChannelType.FILE,
            direction  = PacketDirection.INPUT,
            media_type = MediaType.JSON,
            text       = text,
            raw        = data,
            source     = str(path),
            context    = context,
            meta       = {"filename": path.name},
        )

    def _read_pdf(self, path: Path, context: str) -> IOPacket:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            text = "\n".join(text_parts)
        except ImportError:
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
            except ImportError:
                logger.warning("[FileChannel] PDF: ต้องติดตั้ง pdfplumber หรือ pypdf")
                text = f"[PDF] {path.name} — ต้องติดตั้ง: pip install pdfplumber"

        return IOPacket(
            channel    = ChannelType.FILE,
            direction  = PacketDirection.INPUT,
            media_type = MediaType.PDF,
            text       = text,
            source     = str(path),
            context    = context,
            meta       = {"filename": path.name, "pages": text.count("\n")},
        )

    def _read_docx(self, path: Path, context: str) -> IOPacket:
        try:
            import docx
            doc  = docx.Document(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            logger.warning("[FileChannel] DOCX: ต้องติดตั้ง python-docx")
            text = f"[DOCX] {path.name} — ต้องติดตั้ง: pip install python-docx"

        return IOPacket(
            channel    = ChannelType.FILE,
            direction  = PacketDirection.INPUT,
            media_type = MediaType.DOCX,
            text       = text,
            source     = str(path),
            context    = context,
            meta       = {"filename": path.name},
        )

    # ─── Write ────────────────────────────────────────────────────

    def write(self, packet: IOPacket, path: str) -> bool:
        """เขียน response ลงไฟล์"""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            ext = p.suffix.lower()

            if ext == ".json":
                p.write_text(
                    json.dumps(packet.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
            else:
                p.write_text(packet.response, encoding="utf-8")
            logger.info(f"[FileChannel] WRITE → {path}")
            return True
        except Exception as e:
            logger.error(f"[FileChannel] WRITE FAILED {path}: {e}")
            return False