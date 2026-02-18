"""Core/IO/Channels/SoundChannel.py — audio in/out (speech)"""
from __future__ import annotations

import logging
from typing import Optional

from ..IOPacket import IOPacket, ChannelType, PacketDirection, MediaType

logger = logging.getLogger("mindwave.io.sound")


class SoundChannel:
    """
    Speech-to-Text (mic → text)
    Text-to-Speech (text → audio)

    ต้องการ: pip install SpeechRecognition pyaudio pyttsx3
    """

    def listen(self, context: str = "general", timeout: int = 5) -> Optional[IOPacket]:
        """ฟังจาก microphone → text"""
        try:
            import speech_recognition as sr
            r   = sr.Recognizer()
            mic = sr.Microphone()

            logger.info("[SoundChannel] LISTEN...")
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=timeout)

            text = r.recognize_google(audio, language="th-TH")
            logger.info(f"[SoundChannel] HEARD: {text[:60]}")

            return IOPacket(
                channel    = ChannelType.SOUND,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.AUDIO,
                text       = text,
                source     = "microphone",
                context    = context,
            )
        except ImportError:
            logger.warning(
                "[SoundChannel] ต้องติดตั้ง: "
                "pip install SpeechRecognition pyaudio"
            )
            return None
        except Exception as e:
            logger.error(f"[SoundChannel] LISTEN FAILED: {e}")
            return None

    def speak(self, text: str) -> bool:
        """text → speech"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return True
        except ImportError:
            logger.warning("[SoundChannel] ต้องติดตั้ง: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"[SoundChannel] SPEAK FAILED: {e}")
            return False

    def transcribe_file(self, path: str, context: str = "general") -> Optional[IOPacket]:
        """แปลงไฟล์ audio → text"""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio, language="th-TH")
            return IOPacket(
                channel    = ChannelType.SOUND,
                direction  = PacketDirection.INPUT,
                media_type = MediaType.AUDIO,
                text       = text,
                source     = path,
                context    = context,
                meta       = {"file": path},
            )
        except Exception as e:
            logger.error(f"[SoundChannel] TRANSCRIBE FAILED {path}: {e}")
            return None