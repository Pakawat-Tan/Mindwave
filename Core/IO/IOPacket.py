"""
IOPacket — โครงสร้างข้อมูลกลางสำหรับทุก channel

ทุก channel แปลง input → IOPacket ก่อนส่งเข้า Brain
Brain ส่ง result กลับ → IOPacket ก่อน output
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ChannelType(Enum):
    CLI       = "cli"
    FILE      = "file"
    SOCKET    = "socket"
    REST      = "rest"
    EVENT_BUS = "event_bus"
    INTERNET  = "internet"
    SOUND     = "sound"
    VIDEO     = "video"


class PacketDirection(Enum):
    INPUT  = "input"   # เข้า Brain
    OUTPUT = "output"  # ออกจาก Brain


class MediaType(Enum):
    TEXT     = "text"
    JSON     = "json"
    BINARY   = "binary"
    AUDIO    = "audio"
    IMAGE    = "image"
    VIDEO    = "video"
    PDF      = "pdf"
    DOCX     = "docx"


@dataclass
class IOPacket:
    """
    normalized packet สำหรับทุก channel

    Brain รับ/ส่งแค่ IOPacket เสมอ — ไม่รู้จัก channel โดยตรง
    """
    packet_id:  str           = field(default_factory=lambda: str(uuid.uuid4())[:8])
    channel:    ChannelType   = ChannelType.CLI
    direction:  PacketDirection = PacketDirection.INPUT
    media_type: MediaType     = MediaType.TEXT

    # content หลัก — Brain ใช้
    text:       str           = ""       # text ที่ normalize แล้ว
    raw:        Any           = None     # raw data ก่อน normalize

    # metadata
    source:     str           = ""       # file path / URL / IP:port
    context:    str           = "general"
    meta:       Dict[str, Any] = field(default_factory=dict)
    timestamp:  float         = field(default_factory=time.time)

    # output fields
    response:   str           = ""
    outcome:    str           = ""
    confidence: float         = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id":  self.packet_id,
            "channel":    self.channel.value,
            "direction":  self.direction.value,
            "media_type": self.media_type.value,
            "text":       self.text[:200],
            "source":     self.source,
            "context":    self.context,
            "response":   self.response[:200],
            "outcome":    self.outcome,
            "confidence": round(self.confidence, 3),
            "timestamp":  self.timestamp,
        }

    def with_response(self, response: str, outcome: str, confidence: float) -> "IOPacket":
        """สร้าง output packet จาก input packet"""
        return IOPacket(
            packet_id  = self.packet_id,
            channel    = self.channel,
            direction  = PacketDirection.OUTPUT,
            media_type = MediaType.TEXT,
            text       = self.text,
            source     = self.source,
            context    = self.context,
            meta       = self.meta,
            response   = response,
            outcome    = outcome,
            confidence = confidence,
        )