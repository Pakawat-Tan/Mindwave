"""Core/IO/Channels/CLIChannel.py — stdin/stdout"""
from __future__ import annotations
import sys
from typing import Optional
from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType


class CLIChannel:
    """stdin/stdout — ใช้ร่วมกับ Main.py"""

    def read(self, prompt: str = "") -> Optional[IOPacket]:
        """อ่านจาก stdin"""
        try:
            if prompt:
                sys.stdout.write(prompt)
                sys.stdout.flush()
            text = sys.stdin.readline()
            if not text:
                return None
            text = text.rstrip("\n")
            return IOPacket(
                channel    = ChannelType.CLI,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.TEXT,
                text       = text,
                source     = "stdin",
            )
        except (KeyboardInterrupt, EOFError):
            return None

    def write(self, packet: IOPacket) -> None:
        """เขียนไป stdout"""
        print(packet.response)

    def write_text(self, text: str) -> None:
        print(text)