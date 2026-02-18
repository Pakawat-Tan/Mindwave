"""
IOController — Gateway เดียวสำหรับทุก IO channel

หน้าที่:
  - รับ input จากทุก channel → normalize → IOPacket → Brain
  - รับ result จาก Brain → format → ส่งออกตาม channel
  - บันทึก IO Log ทุกครั้ง

กฎเหล็ก (ตาม BrainController):
  - Brain ไม่รู้จัก channel โดยตรง
  - ทุก IO ต้องผ่าน IOController เท่านั้น
  - ห้าม relay ตรงจาก channel → Brain

Usage:
    io = IOController(brain)
    io.start_socket(port=9000)
    io.start_rest(port=8000)

    # ส่ง packet โดยตรง
    pkt = io.send_text("hello", context="general")

    # อ่านไฟล์แล้วให้ Brain เรียนรู้
    io.learn_from_file("data.txt")

    # fetch URL แล้วให้ Brain เรียนรู้
    io.learn_from_url("https://example.com/data")
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from .IOPacket import IOPacket, ChannelType, PacketDirection, MediaType
from .IOLogger import IOLogger
from .Channels.CLIChannel import CLIChannel
from .Channels.FileChannel import FileChannel
from .Channels.SocketChannel import SocketChannel
from .Channels.RESTChannel import RESTChannel
from .Channels.EventBusChannel import EventBusChannel
from .Channels.InternetChannel import InternetChannel
from .Channels.SoundChannel import SoundChannel
from .Channels.VideoChannel import VideoChannel

logger = logging.getLogger("mindwave.io")


class IOController:
    """
    Gateway เดียว — รับ/ส่ง IO ทุก channel ผ่านที่นี่

    Brain ไม่รู้ว่า input มาจาก channel ไหน
    ทุกอย่างถูก normalize เป็น IOPacket ก่อน
    """

    def __init__(self, brain=None):
        """
        Args:
            brain: BrainController instance (optional — ใส่ทีหลังได้)
        """
        self._brain = brain
        self._io_logger = IOLogger()

        # ── Channels ──────────────────────────────────────────────
        self.cli       = CLIChannel()
        self.file      = FileChannel()
        self.socket    = SocketChannel()
        self.rest      = RESTChannel()
        self.event_bus = EventBusChannel()
        self.internet  = InternetChannel()
        self.sound     = SoundChannel()
        self.video     = VideoChannel()

        # ── Active servers ────────────────────────────────────────
        self._socket_thread: Optional[threading.Thread] = None
        self._rest_thread:   Optional[threading.Thread] = None

        logger.info("[IOController] INIT — all channels ready")

    def attach_brain(self, brain) -> None:
        """attach Brain หลัง init"""
        self._brain = brain

    # ─────────────────────────────────────────────────────────────
    # Core: process packet → Brain → response
    # ─────────────────────────────────────────────────────────────

    def process(self, packet: IOPacket) -> IOPacket:
        """
        ส่ง IOPacket เข้า Brain → คืน output IOPacket

        ทุก channel ใช้ method นี้
        """
        # log input
        self._io_logger.log(packet)

        if self._brain is None:
            out = packet.with_response("Brain ยังไม่ได้ attach", "error", 0.0)
            self._io_logger.log(out)
            return out

        # เลือก Brain method ตาม meta
        mode = packet.meta.get("mode", "respond")

        try:
            if mode == "learn":
                result = self._brain.learn(packet.text)
                response   = result.get("response", "")
                outcome    = result.get("outcome", "learn")
                confidence = result.get("confidence", 0.0)

            elif mode == "status":
                status = self._brain.status()
                response   = str(status)
                outcome    = "status"
                confidence = 1.0

            else:
                result = self._brain.respond(
                    input_text = packet.text,
                    context    = packet.context,
                )
                response   = result.get("response", "")
                outcome    = result.get("outcome", "")
                confidence = result.get("confidence", 0.0)

        except Exception as e:
            logger.error(f"[IOController] BRAIN ERROR: {e}")
            response, outcome, confidence = f"error: {e}", "error", 0.0

        out = packet.with_response(response, outcome, confidence)
        self._io_logger.log(out)
        return out

    # ─────────────────────────────────────────────────────────────
    # Convenience: send text
    # ─────────────────────────────────────────────────────────────

    def send_text(
        self,
        text:    str,
        context: str = "general",
        channel: ChannelType = ChannelType.CLI,
    ) -> IOPacket:
        """ส่ง text เข้า Brain โดยตรง"""
        pkt = IOPacket(
            channel    = channel,
            direction  = PacketDirection.INPUT,
            media_type = MediaType.TEXT,
            text       = text,
            context    = context,
            source     = channel.value,
        )
        return self.process(pkt)

    # ─────────────────────────────────────────────────────────────
    # File
    # ─────────────────────────────────────────────────────────────

    def learn_from_file(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """อ่านไฟล์ → ให้ Brain เรียนรู้"""
        pkt = self.file.read(path, context)
        if pkt is None:
            return None
        pkt.meta["mode"] = "learn"
        # แบ่ง text เป็น chunks ถ้ายาวเกิน
        chunks = self._split_chunks(pkt.text, chunk_size=500)
        last_out = None
        for chunk in chunks:
            chunk_pkt = IOPacket(
                channel    = ChannelType.FILE,
                direction  = PacketDirection.INPUT,
                media_type = pkt.media_type,
                text       = chunk,
                source     = pkt.source,
                context    = context,
                meta       = {"mode": "learn"},
            )
            last_out = self.process(chunk_pkt)
        logger.info(f"[IOController] LEARN_FILE {path} → {len(chunks)} chunks")
        return last_out

    def respond_from_file(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """อ่านไฟล์ → ส่งเข้า Brain ตอบ"""
        pkt = self.file.read(path, context)
        if pkt is None:
            return None
        return self.process(pkt)

    # ─────────────────────────────────────────────────────────────
    # Internet
    # ─────────────────────────────────────────────────────────────

    def learn_from_url(self, url: str, context: str = "general") -> Optional[IOPacket]:
        """fetch URL → ให้ Brain เรียนรู้"""
        pkt = self.internet.fetch(url, context)
        if pkt is None:
            return None
        chunks = self._split_chunks(pkt.text, chunk_size=500)
        last_out = None
        for chunk in chunks:
            chunk_pkt = IOPacket(
                channel    = ChannelType.INTERNET,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.TEXT,
                text       = chunk,
                source     = url,
                context    = context,
                meta       = {"mode": "learn", "url": url},
            )
            last_out = self.process(chunk_pkt)
        logger.info(f"[IOController] LEARN_URL {url} → {len(chunks)} chunks")
        return last_out

    # ─────────────────────────────────────────────────────────────
    # Socket Server
    # ─────────────────────────────────────────────────────────────

    def start_socket(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        mode: str = "tcp",
    ) -> None:
        """เปิด Socket server"""
        self.socket = SocketChannel(host, port, mode)
        self._socket_thread = self.socket.start_server(
            on_packet=lambda pkt: self.process(pkt).response
        )
        logger.info(f"[IOController] SOCKET {mode.upper()} {host}:{port}")

    def stop_socket(self) -> None:
        self.socket.stop()

    # ─────────────────────────────────────────────────────────────
    # REST Server
    # ─────────────────────────────────────────────────────────────

    def start_rest(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """เปิด REST API server"""
        self.rest = RESTChannel(host, port)
        self._rest_thread = self.rest.start(
            on_packet=lambda pkt: self.process(pkt).response
        )
        logger.info(f"[IOController] REST http://{host}:{port}")

    # ─────────────────────────────────────────────────────────────
    # Sound
    # ─────────────────────────────────────────────────────────────

    def listen_and_respond(self, context: str = "general", speak: bool = True) -> Optional[IOPacket]:
        """ฟัง mic → Brain → พูดตอบ"""
        pkt = self.sound.listen(context=context)
        if pkt is None:
            return None
        out = self.process(pkt)
        if speak and out.response:
            self.sound.speak(out.response)
        return out

    # ─────────────────────────────────────────────────────────────
    # Video / Image
    # ─────────────────────────────────────────────────────────────

    def learn_from_image(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """อ่านรูป → OCR → Brain เรียนรู้"""
        pkt = self.video.read_image(path, context)
        if pkt is None:
            return None
        pkt.meta["mode"] = "learn"
        return self.process(pkt)

    def learn_from_video(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """อ่าน video frames → Brain เรียนรู้"""
        pkt = self.video.read_video_frames(path, context)
        if pkt is None:
            return None
        pkt.meta["mode"] = "learn"
        return self.process(pkt)

    # ─────────────────────────────────────────────────────────────
    # Event Bus
    # ─────────────────────────────────────────────────────────────

    def on_event(self, event_type: str) -> None:
        """subscribe event → Brain"""
        def handler(pkt: IOPacket):
            self.process(pkt)
        self.event_bus.subscribe(event_type, handler)

    def emit(self, event_type: str, text: str = "", **kwargs) -> int:
        return self.event_bus.publish(event_type, text=text, **kwargs)

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _split_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """แบ่ง text ยาว → chunks"""
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "io_log":     self._io_logger.stats(),
            "event_bus":  len(self.event_bus.history),
            "brain":      "attached" if self._brain else "none",
        }

    def flush_log(self) -> str:
        return self._io_logger.flush()