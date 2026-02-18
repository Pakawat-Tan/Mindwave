"""Core/IO/Channels/VideoChannel.py — video/image → text"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.video")


class VideoChannel:
    """
    Image/Video → text description

    รองรับ:
      - image: OCR (pytesseract) หรือ description
      - video: extract frames → process
      - webcam: live capture

    ต้องการ: pip install opencv-python pytesseract Pillow
    """

    def read_image(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """อ่านรูปภาพ → text (OCR)"""
        try:
            import pytesseract
            from PIL import Image
            img  = Image.open(path)
            text = pytesseract.image_to_string(img, lang="tha+eng")
            text = text.strip()
            if not text:
                text = f"[IMAGE] {Path(path).name} — ไม่พบข้อความในรูป"

            return IOPacket(
                channel    = ChannelType.VIDEO,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.IMAGE,
                text       = text,
                source     = path,
                context    = context,
                meta       = {"file": path, "ocr": True},
            )
        except ImportError:
            logger.warning("[VideoChannel] ต้องติดตั้ง: pip install pytesseract Pillow")
            return IOPacket(
                channel    = ChannelType.VIDEO,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.IMAGE,
                text       = f"[IMAGE] {Path(path).name}",
                source     = path,
                context    = context,
            )
        except Exception as e:
            logger.error(f"[VideoChannel] IMAGE FAILED {path}: {e}")
            return None

    def read_video_frames(self, path: str, context: str = "general", max_frames: int = 5) -> Optional[IOPacket]:
        """แยก frames จาก video → OCR แต่ละ frame"""
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step   = max(1, frame_count // max_frames)
            texts  = []

            for i in range(max_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    import pytesseract
                    from PIL import Image
                    import numpy as np
                    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(img, lang="tha+eng").strip()
                    if text:
                        texts.append(f"[Frame {i+1}] {text}")
                except ImportError:
                    texts.append(f"[Frame {i+1}] — ต้องการ pytesseract")

            cap.release()
            combined = "\n".join(texts) or f"[VIDEO] {Path(path).name}"

            return IOPacket(
                channel    = ChannelType.VIDEO,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.VIDEO,
                text       = combined,
                source     = path,
                context    = context,
                meta       = {"file": path, "frames": len(texts)},
            )
        except ImportError:
            logger.warning("[VideoChannel] ต้องติดตั้ง: pip install opencv-python")
            return None
        except Exception as e:
            logger.error(f"[VideoChannel] VIDEO FAILED {path}: {e}")
            return None

    def capture_webcam(self, context: str = "general") -> Optional[IOPacket]:
        """ถ่ายภาพจาก webcam → OCR"""
        try:
            import cv2
            cap   = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.warning("[VideoChannel] ไม่พบ webcam")
                return None

            # บันทึก temp แล้วอ่าน OCR
            tmp = "/tmp/mindwave_webcam.jpg"
            cv2.imwrite(tmp, frame)
            return self.read_image(tmp, context)

        except ImportError:
            logger.warning("[VideoChannel] ต้องติดตั้ง: pip install opencv-python")
            return None
        except Exception as e:
            logger.error(f"[VideoChannel] WEBCAM FAILED: {e}")
            return None