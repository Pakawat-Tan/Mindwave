"""
TrainingPipeline — เทรน Brain แบบ Runtime

รับข้อมูลจากทุก source → parse → ส่งเข้า Brain.learn()
ระหว่าง Brain รันอยู่ ไม่ต้อง restart

Sources:
  - txt / md    — plain text หรือ tagged
  - pdf / docx  — เอกสาร (ผ่าน FileChannel)
  - URL         — เว็บไซต์ (ผ่าน InternetChannel)
  - image       — OCR (ผ่าน VideoChannel)

Tags (optional):
  <fact>...</fact>              — ข้อมูล/ความจริง
  <qa>Q: ... A: ...</qa>        — คู่ Q&A
  <context:xxx>...</context>    — ระบุ domain
  <rule>...</rule>              — กฎที่ต้องจำ
  <ignore>...</ignore>          — ข้ามบล็อกนี้

ไม่มี tag → อ่านเป็น plain text chunk ปกติ
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger("mindwave.train")


# ─────────────────────────────────────────────────────────────────────────────
# TrainUnit — หน่วยการเรียนรู้ 1 ชิ้น
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainUnit:
    """ข้อมูล 1 หน่วยที่จะส่งเข้า Brain"""
    text:       str
    context:    str   = "general"
    unit_type:  str   = "fact"    # fact / qa / rule / plain
    source:     str   = ""
    importance: float = 0.7       # high สำหรับ qa/rule

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text":      self.text[:80],
            "context":   self.context,
            "type":      self.unit_type,
            "source":    self.source,
            "importance": self.importance,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TagParser — แยก tagged blocks จาก text
# ─────────────────────────────────────────────────────────────────────────────

class TagParser:
    """
    Parse tagged content → TrainUnit list

    รองรับ:
      <fact>...</fact>
      <qa>Q: ... A: ...</qa>
      <context:xxx>...</context>
      <rule>...</rule>
      <ignore>...</ignore>

    ไม่มี tag → แบ่งเป็น plain chunk
    """

    # regex patterns
    _RE_FACT    = re.compile(r"<fact>(.*?)</fact>",           re.DOTALL | re.IGNORECASE)
    _RE_QA      = re.compile(r"<qa>(.*?)</qa>",               re.DOTALL | re.IGNORECASE)
    _RE_CTX     = re.compile(r"<context:(\w+)>(.*?)</context>", re.DOTALL | re.IGNORECASE)
    _RE_RULE    = re.compile(r"<rule>(.*?)</rule>",           re.DOTALL | re.IGNORECASE)
    _RE_IGNORE  = re.compile(r"<ignore>.*?</ignore>",         re.DOTALL | re.IGNORECASE)

    CHUNK_SIZE  = 400   # chars ต่อ plain chunk

    def parse(self, text: str, default_context: str = "general", source: str = "") -> List[TrainUnit]:
        """แยก text → TrainUnit list"""
        units: List[TrainUnit] = []

        # ลบ <ignore> blocks ก่อน
        clean = self._RE_IGNORE.sub("", text)

        has_tags = any(tag in clean.lower() for tag in ["<fact>", "<qa>", "<context:", "<rule>"])

        if has_tags:
            units += self._parse_tagged(clean, default_context, source)
            # เก็บ text ที่ไม่มี tag ไว้ด้วย
            leftover = self._strip_all_tags(clean).strip()
            if leftover:
                units += self._chunk_plain(leftover, default_context, source)
        else:
            units += self._chunk_plain(clean, default_context, source)

        return [u for u in units if u.text.strip()]

    def _parse_tagged(self, text: str, ctx: str, source: str) -> List[TrainUnit]:
        units = []

        # <fact>
        for m in self._RE_FACT.finditer(text):
            content = m.group(1).strip()
            if content:
                units.append(TrainUnit(
                    text=content, context=ctx,
                    unit_type="fact", source=source, importance=0.75,
                ))

        # <qa>
        for m in self._RE_QA.finditer(text):
            content = m.group(1).strip()
            if content:
                # แยก Q/A ถ้ามี
                q, a = self._split_qa(content)
                if q and a:
                    units.append(TrainUnit(
                        text=f"Q: {q}", context=ctx,
                        unit_type="qa", source=source, importance=0.80,
                    ))
                    units.append(TrainUnit(
                        text=f"A: {a}", context=ctx,
                        unit_type="qa", source=source, importance=0.85,
                    ))
                else:
                    units.append(TrainUnit(
                        text=content, context=ctx,
                        unit_type="qa", source=source, importance=0.80,
                    ))

        # <context:xxx>
        for m in self._RE_CTX.finditer(text):
            domain, content = m.group(1).strip(), m.group(2).strip()
            if content:
                for chunk in self._split_chunks(content):
                    units.append(TrainUnit(
                        text=chunk, context=domain,
                        unit_type="fact", source=source, importance=0.75,
                    ))

        # <rule>
        for m in self._RE_RULE.finditer(text):
            content = m.group(1).strip()
            if content:
                units.append(TrainUnit(
                    text=content, context=ctx,
                    unit_type="rule", source=source, importance=0.90,
                ))

        return units

    def _chunk_plain(self, text: str, ctx: str, source: str) -> List[TrainUnit]:
        """แบ่ง plain text เป็น chunks"""
        chunks = self._split_chunks(text)
        return [
            TrainUnit(
                text=chunk, context=ctx,
                unit_type="plain", source=source, importance=0.65,
            )
            for chunk in chunks if chunk.strip()
        ]

    def _split_qa(self, text: str) -> Tuple[str, str]:
        """แยก 'Q: ... A: ...' """
        m = re.search(r"Q[:\s]+(.+?)\s+A[:\s]+(.+)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return "", ""

    def _split_chunks(self, text: str) -> List[str]:
        if len(text) <= self.CHUNK_SIZE:
            return [text] if text.strip() else []
        # แบ่งตาม sentence boundaries ก่อน
        sentences = re.split(r"(?<=[.!?।\n])\s+", text)
        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) <= self.CHUNK_SIZE:
                cur += (" " if cur else "") + s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks or [text[:self.CHUNK_SIZE]]

    def _strip_all_tags(self, text: str) -> str:
        """ลบ tag ทั้งหมดออก เหลือแต่ text นอก tag"""
        # ลบ blocks ที่มี tag
        for pattern in [self._RE_FACT, self._RE_QA, self._RE_CTX, self._RE_RULE]:
            text = pattern.sub("", text)
        # ลบ tag ที่เหลือ
        text = re.sub(r"<[^>]+>", "", text)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# TrainResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """ผลการ train 1 ครั้ง"""
    source:       str
    total_units:  int   = 0
    learned:      int   = 0
    consolidated: int   = 0
    errors:       int   = 0
    elapsed_s:    float = 0.0
    by_type:      Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        rate = self.learned / max(1, self.total_units) * 100
        return (
            f"✓ {self.source}  "
            f"units={self.total_units}  "
            f"learned={self.learned} ({rate:.0f}%)  "
            f"consolidated={self.consolidated}  "
            f"time={self.elapsed_s:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrainingPipeline
# ─────────────────────────────────────────────────────────────────────────────

class TrainingPipeline:
    """
    เทรน Brain แบบ Runtime จากทุก source

    Usage:
        pipeline = TrainingPipeline(brain, io)
        result = pipeline.train("data.txt")
        result = pipeline.train("https://example.com")
        result = pipeline.train("image.jpg")
    """

    def __init__(self, brain, io=None):
        """
        Args:
            brain: BrainController
            io:    IOController (optional — ใช้สำหรับ file/url/image)
        """
        self._brain  = brain
        self._io     = io
        self._parser = TagParser()
        self._history: List[TrainResult] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main: train จาก source
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        source:   str,
        context:  str = "general",
        on_progress: Optional[Any] = None,
        epochs: int = 3,  # เทรนซ้ำกี่ครั้ง (default: 3)
    ) -> TrainResult:
        """
        เทรนจาก source — detect type อัตโนมัติ

        รองรับ:
          - ไฟล์เดี่ยว:    data.txt, /abs/path/file.pdf
          - directory:     data/  หรือ /abs/path/folder/
          - wildcard:      data/*.txt, docs/**/*.pdf
          - URL:           https://...
          - image:         photo.jpg
          - raw text:      ข้อความโดยตรง
        """
        t0 = time.time()
        result = TrainResult(source=source[:60])

        src_stripped = source.strip()
        src_lower    = src_stripped.lower()

        # ── detect type ───────────────────────────────────────────

        # URL
        if src_lower.startswith(("http://", "https://")):
            return self._train_single(source, context, "url", on_progress, epochs)

        # wildcard pattern
        if "*" in src_stripped or "?" in src_stripped:
            return self._train_glob(src_stripped, context, on_progress, epochs)

        p = Path(src_stripped)

        # directory
        if p.is_dir():
            return self._train_directory(p, context, on_progress, epochs)

        # ไฟล์เดี่ยว (absolute หรือ relative)
        if p.exists() and p.is_file():
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                label = "image"
            elif ext == ".pdf":
                label = "pdf"
            elif ext in (".docx", ".doc"):
                label = "docx"
            else:
                label = "file"
            return self._train_single(str(p), context, label, on_progress, epochs)

        # ไม่เจอไฟล์ — treat as raw text
        return self._train_single(src_stripped, context, "text", on_progress, epochs)

    def _train_glob(self, pattern: str, context: str, on_progress, epochs: int = 3) -> TrainResult:
        """เทรนจาก wildcard pattern เช่น data/*.txt"""
        import glob as _glob
        paths = sorted(_glob.glob(pattern, recursive=True))

        combined = TrainResult(source=pattern[:60])
        if not paths:
            logger.warning(f"[TrainPipeline] GLOB no files: {pattern}")
            combined.errors = 1
            return combined

        for path in paths:
            if Path(path).is_file():
                r = self._train_single(path, context, "file", on_progress, epochs)
                combined.total_units  += r.total_units
                combined.learned      += r.learned
                combined.consolidated += r.consolidated
                combined.errors       += r.errors
                combined.elapsed_s    += r.elapsed_s
                for k, v in r.by_type.items():
                    combined.by_type[k] = combined.by_type.get(k, 0) + v

        self._history.append(combined)
        return combined

    def _train_directory(self, directory: Path, context: str, on_progress, epochs: int = 3) -> TrainResult:
        """เทรนทุกไฟล์ใน directory (recursive)"""
        supported = {".txt", ".md", ".json", ".pdf", ".docx", ".doc",
                     ".jpg", ".jpeg", ".png", ".csv"}
        paths = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        )

        combined = TrainResult(source=str(directory)[:60])
        if not paths:
            logger.warning(f"[TrainPipeline] DIR empty: {directory}")
            combined.errors = 1
            return combined

        logger.info(f"[TrainPipeline] DIR {directory} → {len(paths)} files")
        for path in paths:
            r = self._train_single(str(path), context, "file", on_progress, epochs)
            combined.total_units  += r.total_units
            combined.learned      += r.learned
            combined.consolidated += r.consolidated
            combined.errors       += r.errors
            combined.elapsed_s    += r.elapsed_s
            for k, v in r.by_type.items():
                combined.by_type[k] = combined.by_type.get(k, 0) + v

        self._history.append(combined)
        return combined

    def _train_single(self, source: str, context: str, label: str, on_progress, epochs: int = 3) -> TrainResult:
        """เทรนจาก source เดี่ยว พร้อม multi-epoch"""
        t0     = time.time()
        result = TrainResult(source=source[:60])

        # อ่าน text ตาม label
        if label == "url":
            text = self._fetch_url(source)
        elif label == "image":
            text = self._read_image(source)
        else:
            text = self._read_file(source) if label != "text" else source

        if not text or not text.strip():
            logger.warning(f"[TrainPipeline] EMPTY source={source}")
            result.errors    = 1
            result.elapsed_s = time.time() - t0
            return result

        units = self._parser.parse(text, default_context=context, source=source[:40])
        result.total_units = len(units)
        
        # ── Multi-Epoch Training Loop ──────────────────────────────
        logger.info(
            f"[TrainPipeline] SINGLE source={source[:40]} "
            f"label={label} units={len(units)} epochs={epochs}"
        )
        
        for epoch in range(epochs):
            epoch_learned = 0
            epoch_consolidated = 0
            
            # Log epoch start (ถ้า > 1 epoch)
            if epochs > 1:
                logger.info(f"[TrainPipeline] EPOCH {epoch+1}/{epochs}")
            
            # Train each unit
            self._train_units(
                units, context, source, result, 
                epoch_learned, epoch_consolidated,
                on_progress if epoch == 0 else None  # progress bar แค่ epoch แรก
            )
        
        result.elapsed_s = time.time() - t0
        self._history.append(result)
        logger.info(f"[TrainPipeline] DONE {result.summary()}")
        return result
    
    def _train_units(self, units, context, source, result, epoch_learned, epoch_consolidated, on_progress):
        """Train units (extracted from _train_single for epoch loop)"""

        # ── feed units → Brain ────────────────────────────────────
        # จับคู่ Q/A ก่อน — เก็บ Q เป็น key, A เป็น answer
        qa_pairs: dict = {}
        i = 0
        unit_list = list(units)
        while i < len(unit_list):
            u = unit_list[i]
            if u.unit_type == "qa" and u.text.startswith("Q:"):
                # ดูว่า unit ถัดไปเป็น A: ไหม
                if i + 1 < len(unit_list) and unit_list[i+1].text.startswith("A:"):
                    q_text = u.text[2:].strip()
                    a_text = unit_list[i+1].text[2:].strip()
                    qa_pairs[q_text[:60]] = a_text
                    i += 2
                    continue
            i += 1

        for i, unit in enumerate(unit_list):
            try:
                learn_result = self._brain.learn(unit.text)
                result.learned += 1
                if learn_result.get("consolidated"):
                    result.consolidated += 1
                result.by_type[unit.unit_type] = (
                    result.by_type.get(unit.unit_type, 0) + 1
                )

                # ── Neural Training ────────────────────────────────────
                # Train neural network พร้อมกับ symbolic learning
                if hasattr(self._brain, "train_neural"):
                    # ใช้ unit.text เป็น target response (self-supervised)
                    # สำหรับ fact/rule ให้ Brain จำเป็น pattern
                    target_text = unit.text
                    
                    # สำหรับ QA ให้ใช้ answer เป็น target
                    if unit.unit_type == "qa":
                        # ถ้าเป็น Answer → ใช้เป็น target
                        if unit.text.startswith("A:"):
                            target_text = unit.text[2:].strip()
                        # ถ้าเป็น Question → ใช้ paired answer
                        elif unit.text.startswith("Q:"):
                            q_key = unit.text[2:].strip()[:60]
                            target_text = qa_pairs.get(q_key, unit.text)
                    
                    try:
                        neural_result = self._brain.train_neural(
                            text=unit.text,
                            target_response=target_text,
                            importance=unit.importance,
                        )
                        # log neural training (optional)
                        logger.debug(
                            f"[TrainPipeline] NEURAL loss={neural_result['loss']:.4f} "
                            f"acc={neural_result['accuracy']:.2f}"
                        )
                    except Exception as e:
                        logger.error(f"[TrainPipeline] NEURAL ERROR: {e}")

                # เพิ่ม belief ใน BeliefSystem พร้อม source_text
                if hasattr(self._brain, "_belief_system"):
                    bs = self._brain._belief_system
                    if not hasattr(bs, "_source_texts"):
                        bs._source_texts = {}

                    key = unit.text[:60]
                    bs.update(
                        subject     = key,
                        input_value = unit.importance,
                        context     = unit.context,
                        source      = f"train:{unit.unit_type}",
                    )
                    # เก็บ text จริง
                    bs._source_texts[key] = unit.text

                    # ถ้าเป็น Q: ให้เก็บ A: ด้วย ใช้ Q เป็น key
                    if unit.unit_type == "qa" and unit.text.startswith("Q:"):
                        q_key = unit.text[2:].strip()[:60]
                        a_text = qa_pairs.get(q_key, "")
                        if a_text:
                            bs.update(
                                subject     = q_key,
                                input_value = 0.90,   # high importance สำหรับ QA
                                context     = unit.context,
                                source      = "train:qa_answer",
                            )
                            bs._source_texts[q_key] = a_text

                if on_progress:
                    on_progress(i + 1, len(unit_list), unit)

            except Exception as e:
                logger.error(f"[TrainPipeline] UNIT ERROR: {e}")
                result.errors += 1

    def train_many(
        self,
        sources:  List[str],
        context:  str = "general",
        on_progress: Optional[Any] = None,
    ) -> List[TrainResult]:
        """เทรนจากหลาย source"""
        return [
            self.train(s, context=context, on_progress=on_progress)
            for s in sources
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Source readers
    # ─────────────────────────────────────────────────────────────────────────

    def _read_file(self, path: str) -> str:
        if self._io:
            pkt = self._io.file.read(path)
            return pkt.text if pkt else ""
        # fallback
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"[TrainPipeline] READ FAILED {path}: {e}")
            return ""

    def _fetch_url(self, url: str) -> str:
        if self._io:
            pkt = self._io.internet.fetch(url)
            return pkt.text if pkt else ""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as r:
                raw = r.read().decode("utf-8", errors="replace")
            # strip HTML
            import re as _re
            raw = _re.sub(r"<[^>]+>", " ", raw)
            raw = _re.sub(r"\s+", " ", raw).strip()
            return raw[:10000]
        except Exception as e:
            logger.error(f"[TrainPipeline] FETCH FAILED {url}: {e}")
            return ""

    def _read_image(self, path: str) -> str:
        if self._io:
            pkt = self._io.video.read_image(path)
            return pkt.text if pkt else ""
        return f"[IMAGE] {Path(path).name}"

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total_units   = sum(r.total_units for r in self._history)
        total_learned = sum(r.learned     for r in self._history)
        return {
            "sessions":      len(self._history),
            "total_units":   total_units,
            "total_learned": total_learned,
            "total_consolidated": sum(r.consolidated for r in self._history),
            "total_errors":  sum(r.errors for r in self._history),
        }

    @property
    def history(self) -> List[TrainResult]:
        return list(self._history)