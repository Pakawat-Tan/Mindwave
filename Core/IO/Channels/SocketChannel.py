"""Core/IO/Channels/SocketChannel.py — TCP/UDP"""
from __future__ import annotations

import logging
import socket
import threading
from typing import Callable, Optional

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.socket")

OnPacket = Callable[[IOPacket], Optional[str]]


class SocketChannel:
    """TCP/UDP server/client"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, mode: str = "tcp"):
        self._host   = host
        self._port   = port
        self._mode   = mode.lower()
        self._server: Optional[socket.socket] = None
        self._running = False

    # ─── Server ───────────────────────────────────────────────────

    def start_server(self, on_packet: OnPacket) -> threading.Thread:
        """เปิด server รับ connection — non-blocking"""
        self._running = True
        sock_type = socket.SOCK_STREAM if self._mode == "tcp" else socket.SOCK_DGRAM
        self._server = socket.socket(socket.AF_INET, sock_type)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self._host, self._port))

        if self._mode == "tcp":
            self._server.listen(5)
            t = threading.Thread(
                target=self._tcp_loop, args=(on_packet,), daemon=True
            )
        else:
            t = threading.Thread(
                target=self._udp_loop, args=(on_packet,), daemon=True
            )

        t.start()
        logger.info(f"[SocketChannel] SERVER {self._mode.upper()} {self._host}:{self._port}")
        return t

    def _tcp_loop(self, on_packet: OnPacket) -> None:
        while self._running:
            try:
                conn, addr = self._server.accept()
                threading.Thread(
                    target=self._handle_tcp,
                    args=(conn, addr, on_packet),
                    daemon=True
                ).start()
            except Exception:
                break

    def _handle_tcp(self, conn: socket.socket, addr, on_packet: OnPacket) -> None:
        try:
            data = conn.recv(4096).decode("utf-8", errors="replace").strip()
            if data:
                pkt = IOPacket(
                    channel   = ChannelType.SOCKET,
                    direction = PacketDirection.INPUT,
                    media_type = MediaType.TEXT,
                    text      = data,
                    source    = f"{addr[0]}:{addr[1]}",
                )
                response = on_packet(pkt) or ""
                conn.sendall(response.encode("utf-8"))
        finally:
            conn.close()

    def _udp_loop(self, on_packet: OnPacket) -> None:
        while self._running:
            try:
                data, addr = self._server.recvfrom(4096)
                text = data.decode("utf-8", errors="replace").strip()
                pkt = IOPacket(
                    channel   = ChannelType.SOCKET,
                    direction = PacketDirection.INPUT,
                    media_type = MediaType.TEXT,
                    text      = text,
                    source    = f"{addr[0]}:{addr[1]}",
                )
                response = on_packet(pkt) or ""
                self._server.sendto(response.encode("utf-8"), addr)
            except Exception:
                break

    def stop(self) -> None:
        self._running = False
        if self._server:
            self._server.close()
        logger.info("[SocketChannel] SERVER STOPPED")

    # ─── Client ───────────────────────────────────────────────────

    def send(self, text: str, context: str = "general") -> Optional[str]:
        """ส่ง text ไป server แล้วรอ response"""
        try:
            sock_type = socket.SOCK_STREAM if self._mode == "tcp" else socket.SOCK_DGRAM
            with socket.socket(socket.AF_INET, sock_type) as s:
                s.settimeout(5.0)
                if self._mode == "tcp":
                    s.connect((self._host, self._port))
                    s.sendall(text.encode("utf-8"))
                    return s.recv(4096).decode("utf-8")
                else:
                    s.sendto(text.encode("utf-8"), (self._host, self._port))
                    data, _ = s.recvfrom(4096)
                    return data.decode("utf-8")
        except Exception as e:
            logger.error(f"[SocketChannel] SEND FAILED: {e}")
            return None