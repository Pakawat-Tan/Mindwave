"""Core/IO/Channels/InternetChannel.py — web fetch / scrape"""
from __future__ import annotations

import logging
import re
from typing import Optional

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.internet")


class InternetChannel:
    """ดึงข้อมูลจาก URL → text"""

    def __init__(self, timeout: int = 10):
        self._timeout = timeout

    def fetch(self, url: str, context: str = "general") -> Optional[IOPacket]:
        """ดึง URL → แปลงเป็น text"""
        try:
            import urllib.request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MindWave/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                raw = r.read()
                content_type = r.headers.get("Content-Type", "")
                encoding = "utf-8"
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()

            html = raw.decode(encoding, errors="replace")
            text = self._strip_html(html)

            logger.info(f"[InternetChannel] FETCH {url} → {len(text)} chars")
            return IOPacket(
                channel    = ChannelType.INTERNET,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.TEXT,
                text       = text,
                source     = url,
                context    = context,
                meta       = {"url": url, "chars": len(text)},
            )
        except Exception as e:
            logger.error(f"[InternetChannel] FETCH FAILED {url}: {e}")
            return None

    def fetch_json(self, url: str, context: str = "general") -> Optional[IOPacket]:
        """ดึง JSON API"""
        try:
            import urllib.request
            import json
            with urllib.request.urlopen(url, timeout=self._timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            if isinstance(data, dict):
                text = "\n".join(f"{k}: {v}" for k, v in data.items())
            else:
                text = str(data)
            return IOPacket(
                channel    = ChannelType.INTERNET,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.JSON,
                text       = text,
                raw        = data,
                source     = url,
                context    = context,
            )
        except Exception as e:
            logger.error(f"[InternetChannel] JSON FAILED {url}: {e}")
            return None

    def _strip_html(self, html: str) -> str:
        """ลบ HTML tags → plain text"""
        # ลบ script/style
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>",  " ", html, flags=re.DOTALL | re.IGNORECASE)
        # ลบ tags
        html = re.sub(r"<[^>]+>", " ", html)
        # ลบ whitespace ซ้ำ
        html = re.sub(r"\s+", " ", html).strip()
        return html[:10000]  # จำกัด 10k chars