"""
จัดการ PersonalityData — init, change, query

Rules:
  - init() สร้างได้ครั้งเดียว — ถ้า init ซ้ำ → PermissionError
  - change() ต้องมี creator_id
  - ทุก mutation ถูก log
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Core.Condition.ConditionController import ConditionController

import logging
from typing import Optional, List

import sys, os
from Core.Personality.PersonalityData import (
    PersonalityData, PersonalityProfile, PersonalityChangeEvent,
    PROFILES, Tone, Friendliness, Firmness,
    ResponseStyle, Humor, Empathy
)


class PersonalityController:

    def __init__(self, condition=None):
        self._personality: Optional[PersonalityData] = None
        self._logger = logging.getLogger("mindwave.personality")
        self._condition = condition

    # ─────────────────────────────────────────────────────────────
    # Init — ครั้งเดียว
    # ─────────────────────────────────────────────────────────────

    def init(self, seed: Optional[int] = None) -> PersonalityData:

        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_personality_allowed()
            if not _allowed:
                self._logger.warning(
                    f"[PersonalityController] init BLOCKED reason={_reason}"
                )
                return None
        """
        สร้าง Personality ครั้งแรก — random from profiles

        ถ้าเรียกซ้ำ → PermissionError (personality is fixed after init)

        Args:
            seed : ใช้สำหรับ test reproducibility เท่านั้น
        """
        if self._personality is not None:
            raise PermissionError(
                "[PersonalityController] personality already initialized — "
                "cannot re-init. Use change() with creator_id."
            )
        self._personality = PersonalityData(seed=seed)
        self._logger.info(
            f"[PersonalityController] INIT "
            f"profile='{self._personality.profile_name}'"
        )
        return self._personality

    def is_initialized(self) -> bool:
        return self._personality is not None

    # ─────────────────────────────────────────────────────────────
    # Change — Creator only
    # ─────────────────────────────────────────────────────────────

    def change(
        self,
        new_profile_name: str,
        creator_id:       str,
        reason:           str = "",
    ) -> PersonalityChangeEvent:
        """
        เปลี่ยน personality — Creator เท่านั้น

        ต้อง init() ก่อน — RuntimeError ถ้ายังไม่ init
        """
        if self._personality is None:
            raise RuntimeError(
                "[PersonalityController] personality not initialized"
            )
        event = self._personality.change(
            new_profile_name = new_profile_name,
            creator_id       = creator_id,
            reason           = reason,
        )
        self._logger.info(
            f"[PersonalityController] CHANGED "
            f"'{event.from_profile}' → '{event.to_profile}' "
            f"by='{creator_id}'"
        )
        return event

    # ─────────────────────────────────────────────────────────────
    # Query — read-only
    # ─────────────────────────────────────────────────────────────

    @property
    def personality(self) -> Optional[PersonalityData]:
        return self._personality

    @property
    def profile(self) -> Optional[PersonalityProfile]:
        return self._personality.profile if self._personality else None

    @property
    def profile_name(self) -> Optional[str]:
        return self._personality.profile_name if self._personality else None

    def get_tone(self)           -> Optional[Tone]:
        return self._personality.tone           if self._personality else None
    def get_friendliness(self)   -> Optional[Friendliness]:
        return self._personality.friendliness   if self._personality else None
    def get_firmness(self)       -> Optional[Firmness]:
        return self._personality.firmness       if self._personality else None
    def get_response_style(self) -> Optional[ResponseStyle]:
        return self._personality.response_style if self._personality else None
    def get_humor(self)          -> Optional[Humor]:
        return self._personality.humor          if self._personality else None
    def get_empathy(self)        -> Optional[Empathy]:
        return self._personality.empathy        if self._personality else None

    def list_available_profiles(self) -> List[str]:
        """รายการ profiles ทั้งหมดที่ใช้ได้"""
        return list(PROFILES.keys())

    def change_history(self) -> List[PersonalityChangeEvent]:
        """ประวัติการเปลี่ยนแปลงทั้งหมด"""
        return self._personality.events if self._personality else []

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if self._personality is None:
            return {"initialized": False}
        return {
            "initialized":    True,
            "profile":        self._personality.profile_name,
            "change_count":   self._personality.change_count,
            "profiles_available": len(PROFILES),
        }