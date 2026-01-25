# ============================================================
# InputAdapter.py
# ------------------------------------------------------------
# Unified Input Adapter
# Converts multi-modal input → normalized brain input
# ============================================================

import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path

from IO.Reader.BaseReader import BaseReader


class InputAdapter:
    def __init__(
        self,
        readers: Optional[List[BaseReader]] = None,
        vocab_size: int = 1000,
        embedding_dim: int = 64,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.readers = readers or []

    # --------------------------------------------------
    # Public entry point
    # --------------------------------------------------
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Normalize ANY input into a standard context dict
        """
        # 1. Command
        if isinstance(input_data, str) and input_data.startswith("/"):
            return {
                "input_type": "command",
                "command": input_data.lstrip("/"),
            }

        # 2. File path
        if isinstance(input_data, str) and Path(input_data).exists():
            return self._process_file(input_data)

        # 3. Pre-parsed content (internet / reader)
        if isinstance(input_data, dict) and "content" in input_data:
            return self._process_content(input_data)

        # 4. Raw text
        return self._process_text(input_data)

    # --------------------------------------------------
    # File handling
    # --------------------------------------------------
    def _process_file(self, file_path: str) -> Dict[str, Any]:
        for reader in self.readers:
            if reader.can_read(file_path):
                content = reader.read(file_path)
                return self._process_content(content)

        raise ValueError(f"No reader supports file: {file_path}")

    # --------------------------------------------------
    # Content handling
    # --------------------------------------------------
    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        vector = self.encode(content.get("content", ""))

        return {
            "input_type": "content",
            "source": content.get("source", "unknown"),
            "file_type": content.get("file_type"),
            "vector": vector,
            "raw_content": content,
            "metadata": content.get("metadata", {}),
        }

    # --------------------------------------------------
    # Text handling
    # --------------------------------------------------
    def _process_text(self, text: Any) -> Dict[str, Any]:
        vector = self.encode(text)

        return {
            "input_type": "text",
            "source": "user",
            "vector": vector,
            "raw_text": str(text),
        }

    # --------------------------------------------------
    # Encoding / Decoding
    # --------------------------------------------------
    def encode(self, input_data: Any) -> np.ndarray:
        text = str(input_data).lower()
        hash_val = hash(text)

        np.random.seed(abs(hash_val) % (2**31))
        vector = np.random.randn(1, self.embedding_dim).astype(np.float32)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        return vector

    def decode(self, prediction: np.ndarray, confidence: float = 0.5) -> Dict[str, Any]:
        if prediction is None:
            return {
                "output": "ไม่สามารถประมวลผลได้",
                "confidence": 0.0
            }

        pred_val = float(np.mean(prediction))

        responses = {
            "greeting": "สวัสดีค่ะ! ยินดีที่ได้พูดคุยกับคุณ",
            "confirm": "ใช่ค่ะ, ผมเข้าใจแล้ว",
            "thinking": "ขอเวลาสักครู่นะค่ะ กำลังคิด...",
            "uncertain": "ขอโทษค่ะ ผมไม่แน่ใจนัก",
            "goodbye": "ลาก่อนค่ะ! ขอบคุณที่พูดคุยกับผม"
        }

        if pred_val < -0.5:
            response = responses["goodbye"]
        elif pred_val < 0:
            response = responses["uncertain"]
        elif pred_val < 0.5:
            response = responses["thinking"]
        elif pred_val < 1.0:
            response = responses["confirm"]
        else:
            response = responses["greeting"]

        return {
            "output": response,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "prediction_value": pred_val
        }
