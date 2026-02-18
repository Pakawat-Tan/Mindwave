"""Core/IO/Channels/EventBusChannel.py — event bus ระหว่าง modules"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.eventbus")

Handler = Callable[[IOPacket], None]


class EventBusChannel:
    """ส่ง event ระหว่าง modules แบบ pub/sub"""

    def __init__(self):
        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []

    def subscribe(self, event_type: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[event_type].append(handler)
        logger.debug(f"[EventBus] SUBSCRIBE {event_type}")

    def publish(self, event_type: str, text: str = "", meta: Dict = None, context: str = "general") -> int:
        """ส่ง event → คืนจำนวน handlers ที่รับ"""
        pkt = IOPacket(
            channel    = ChannelType.EVENT_BUS,
            direction  = PacketDirection.INPUT,
            media_type = MediaType.TEXT,
            text       = text or event_type,
            source     = event_type,
            context    = context,
            meta       = {"event_type": event_type, **(meta or {})},
        )
        self._history.append({"event": event_type, "text": text[:80]})

        with self._lock:
            handlers = list(self._handlers.get(event_type, []))

        for h in handlers:
            try:
                h(pkt)
            except Exception as e:
                logger.error(f"[EventBus] HANDLER ERROR {event_type}: {e}")

        logger.debug(f"[EventBus] PUBLISH {event_type} → {len(handlers)} handlers")
        return len(handlers)

    def emit(self, event_type: str, **kwargs) -> int:
        """shorthand สำหรับ publish"""
        return self.publish(event_type, **kwargs)

    @property
    def history(self):
        return list(self._history)