"""Core/IO/Channels/RESTChannel.py — HTTP REST (FastAPI)"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.rest")

OnPacket = Callable[[IOPacket], Optional[str]]


class RESTChannel:
    """
    HTTP REST server ด้วย FastAPI

    Endpoints:
      POST /respond  — ส่ง input เข้า Brain
      POST /learn    — ส่งเข้า learn mode
      GET  /status   — Brain status
      GET  /beliefs  — beliefs ทั้งหมด
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self._host = host
        self._port = port
        self._app  = None
        self._on_packet: Optional[OnPacket] = None

    def start(self, on_packet: OnPacket) -> threading.Thread:
        """เปิด REST server — non-blocking"""
        self._on_packet = on_packet
        try:
            from fastapi import FastAPI
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel
            import uvicorn

            app = FastAPI(title="MindWave API")
            self._app = app

            class InputBody(BaseModel):
                text:    str
                context: str = "general"

            @app.post("/respond")
            async def respond(body: InputBody):
                pkt = IOPacket(
                    channel    = ChannelType.REST,
                    direction  = PacketDirection.INPUT,
                    media_type = MediaType.TEXT,
                    text       = body.text,
                    context    = body.context,
                    source     = "rest_api",
                )
                response = on_packet(pkt) or ""
                return {"response": response, "packet_id": pkt.packet_id}

            @app.post("/learn")
            async def learn(body: InputBody):
                pkt = IOPacket(
                    channel    = ChannelType.REST,
                    direction  = PacketDirection.INPUT,
                    media_type = MediaType.TEXT,
                    text       = body.text,
                    context    = body.context,
                    source     = "rest_learn",
                    meta       = {"mode": "learn"},
                )
                response = on_packet(pkt) or ""
                return {"response": response}

            @app.get("/status")
            async def status():
                pkt = IOPacket(
                    channel    = ChannelType.REST,
                    direction  = PacketDirection.INPUT,
                    media_type = MediaType.TEXT,
                    text       = "__status__",
                    source     = "rest_status",
                    meta       = {"mode": "status"},
                )
                response = on_packet(pkt) or "{}"
                return JSONResponse(content={"status": response})

            @app.get("/health")
            async def health():
                return {"ok": True}

            config = uvicorn.Config(
                app, host=self._host, port=self._port,
                log_level="warning"
            )
            server = uvicorn.Server(config)

            t = threading.Thread(target=server.run, daemon=True)
            t.start()
            logger.info(f"[RESTChannel] SERVER http://{self._host}:{self._port}")
            return t

        except ImportError:
            logger.warning(
                "[RESTChannel] ต้องติดตั้ง fastapi + uvicorn: "
                "pip install fastapi uvicorn"
            )
            return threading.Thread(daemon=True)