"""
Distributed System — หลาย instances ทำงานร่วมกัน

Features:
  1. instance coordination — ประสานงานระหว่าง instances
  2. consensus            — ตกลงความเห็นร่วมกัน
  3. state sync           — ซิงค์ state ระหว่าง instances
  4. distributed learning — เรียนรู้แบบกระจาย
  5. conflict resolution  — แก้ conflict ระหว่าง instances
  6. leader election      — เลือก leader instance

Algorithm:
  - Simple majority voting for consensus
  - Heartbeat-based leader election
  - Vector clock for conflict resolution
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger("mindwave.distributed")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class InstanceRole(Enum):
    LEADER    = "leader"
    FOLLOWER  = "follower"
    CANDIDATE = "candidate"


class VoteDecision(Enum):
    APPROVE = "approve"
    REJECT  = "reject"
    ABSTAIN = "abstain"


class ConflictStrategy(Enum):
    LAST_WRITE_WINS = "last_write_wins"
    MERGE           = "merge"
    LEADER_DECIDES  = "leader_decides"
    VOTE            = "vote"


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceState:
    """State ของแต่ละ instance"""
    instance_id:   str
    role:          InstanceRole
    last_heartbeat: float
    state_version: int  # vector clock
    data:          Dict[str, Any] = field(default_factory=dict)

    @property
    def is_alive(self, timeout: float = 30.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id":   self.instance_id,
            "role":          self.role.value,
            "last_heartbeat": self.last_heartbeat,
            "state_version": self.state_version,
            "is_alive":      self.is_alive,
        }


@dataclass(frozen=True)
class ConsensusProposal:
    """ข้อเสนอที่ต้องการ consensus"""
    proposal_id: str
    proposer:    str
    action:      str
    payload:     Dict[str, Any]
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposer":    self.proposer,
            "action":      self.action,
            "payload":     self.payload,
            "timestamp":   self.timestamp,
        }


@dataclass
class ConsensusResult:
    """ผลของ consensus"""
    proposal_id: str
    approved:    bool
    votes:       Dict[str, VoteDecision]  # instance_id → vote
    total_votes: int
    approve_count: int
    reject_count:  int

    @property
    def majority(self) -> bool:
        return self.approve_count > (self.total_votes / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id":   self.proposal_id,
            "approved":      self.approved,
            "total_votes":   self.total_votes,
            "approve_count": self.approve_count,
            "reject_count":  self.reject_count,
            "majority":      self.majority,
        }


@dataclass(frozen=True)
class ConflictRecord:
    """บันทึก conflict"""
    conflict_id:   str
    instance_a:    str
    instance_b:    str
    key:           str
    value_a:       Any
    value_b:       Any
    resolved:      bool
    resolution:    Optional[Any] = None
    strategy_used: Optional[ConflictStrategy] = None
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "instance_a":  self.instance_a,
            "instance_b":  self.instance_b,
            "key":         self.key,
            "resolved":    self.resolved,
            "strategy":    self.strategy_used.value if self.strategy_used else None,
        }


@dataclass(frozen=True)
class LearningUpdate:
    """Update สำหรับ distributed learning"""
    update_id:    str
    source:       str
    update_type:  str  # "weight" / "knowledge" / "pattern"
    data:         Dict[str, Any]
    confidence:   float
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id":   self.update_id,
            "source":      self.source,
            "update_type": self.update_type,
            "confidence":  round(self.confidence, 3),
            "timestamp":   self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DistributedSystem
# ─────────────────────────────────────────────────────────────────────────────

class DistributedSystem:
    """
    Distributed System — หลาย instances ทำงานร่วมกัน

    Flow:
      instances register → heartbeat → leader election
      → consensus for decisions → state sync
      → distributed learning → conflict resolution
    """

    def __init__(self, instance_id: str):
        self._instance_id   = instance_id
        self._instances:    Dict[str, InstanceState] = {}
        self._my_role:      InstanceRole = InstanceRole.FOLLOWER
        self._leader_id:    Optional[str] = None

        # register self
        self._instances[instance_id] = InstanceState(
            instance_id   = instance_id,
            role          = self._my_role,
            last_heartbeat = time.time(),
            state_version = 0,
        )

        # consensus
        self._proposals:   Dict[str, ConsensusProposal] = {}
        self._votes:       Dict[str, Dict[str, VoteDecision]] = defaultdict(dict)
        self._results:     Dict[str, ConsensusResult] = {}

        # conflicts
        self._conflicts:   List[ConflictRecord] = []

        # learning
        self._learning_updates: List[LearningUpdate] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Instance Coordination
    # ─────────────────────────────────────────────────────────────────────────

    def register_instance(self, instance_id: str) -> None:
        """ลงทะเบียน instance ใหม่"""
        if instance_id not in self._instances:
            self._instances[instance_id] = InstanceState(
                instance_id   = instance_id,
                role          = InstanceRole.FOLLOWER,
                last_heartbeat = time.time(),
                state_version = 0,
            )
            logger.info(f"[DistributedSystem] REGISTER instance={instance_id}")

    def heartbeat(self, instance_id: Optional[str] = None) -> None:
        """ส่ง heartbeat — บอกว่ายังมีชีวิตอยู่"""
        iid = instance_id or self._instance_id
        if iid in self._instances:
            self._instances[iid].last_heartbeat = time.time()

    def get_alive_instances(self) -> List[str]:
        """คืน instance_ids ที่ยังมีชีวิต"""
        return [
            iid for iid, state in self._instances.items()
            if state.is_alive
        ]

    @property
    def instance_count(self) -> int:
        return len(self.get_alive_instances())

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Consensus
    # ─────────────────────────────────────────────────────────────────────────

    def propose(
        self,
        action:  str,
        payload: Dict[str, Any],
        proposal_id: Optional[str] = None,
    ) -> ConsensusProposal:
        """
        เสนอ proposal ให้ instances อื่นลงคะแนน

        Args:
            action: action ที่ต้องการทำ
            payload: ข้อมูล
            proposal_id: ID (optional)

        Returns:
            ConsensusProposal
        """
        pid = proposal_id or f"{self._instance_id}_{int(time.time())}"

        proposal = ConsensusProposal(
            proposal_id = pid,
            proposer    = self._instance_id,
            action      = action,
            payload     = payload,
        )
        self._proposals[pid] = proposal

        logger.info(
            f"[DistributedSystem] PROPOSE id={pid} action={action}"
        )
        return proposal

    def vote(
        self,
        proposal_id: str,
        decision:    VoteDecision,
        voter_id:    Optional[str] = None,
    ) -> None:
        """
        ลงคะแนนให้ proposal

        Args:
            proposal_id: proposal ที่ต้องการลงคะแนน
            decision: APPROVE / REJECT / ABSTAIN
            voter_id: instance ที่ลงคะแนน (default ตัวเอง)
        """
        voter = voter_id or self._instance_id
        if proposal_id not in self._proposals:
            raise KeyError(f"proposal '{proposal_id}' not found")

        self._votes[proposal_id][voter] = decision
        logger.debug(
            f"[DistributedSystem] VOTE proposal={proposal_id} "
            f"voter={voter} decision={decision.value}"
        )

    def tally_votes(self, proposal_id: str) -> ConsensusResult:
        """
        นับคะแนน

        Args:
            proposal_id: proposal ที่ต้องการนับ

        Returns:
            ConsensusResult
        """
        if proposal_id not in self._proposals:
            raise KeyError(f"proposal '{proposal_id}' not found")

        votes = self._votes.get(proposal_id, {})
        approve = sum(1 for v in votes.values() if v == VoteDecision.APPROVE)
        reject  = sum(1 for v in votes.values() if v == VoteDecision.REJECT)

        result = ConsensusResult(
            proposal_id   = proposal_id,
            approved      = approve > (len(votes) / 2),
            votes         = dict(votes),
            total_votes   = len(votes),
            approve_count = approve,
            reject_count  = reject,
        )
        self._results[proposal_id] = result

        logger.info(
            f"[DistributedSystem] TALLY proposal={proposal_id} "
            f"approve={approve} reject={reject} total={len(votes)}"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 3. State Sync
    # ─────────────────────────────────────────────────────────────────────────

    def sync_state(
        self,
        instance_id: str,
        state_data:  Dict[str, Any],
        version:     int,
    ) -> bool:
        """
        ซิงค์ state จาก instance อื่น

        Args:
            instance_id: instance ที่ส่ง state มา
            state_data: state data
            version: version (vector clock)

        Returns:
            True ถ้า sync สำเร็จ
        """
        if instance_id not in self._instances:
            self.register_instance(instance_id)

        instance = self._instances[instance_id]

        # ถ้า version ใหม่กว่า → sync
        if version > instance.state_version:
            instance.data = state_data.copy()
            instance.state_version = version
            logger.info(
                f"[DistributedSystem] SYNC_STATE from={instance_id} "
                f"version={version}"
            )
            return True
        else:
            logger.debug(
                f"[DistributedSystem] SYNC_STATE_SKIP from={instance_id} "
                f"version={version} (not newer)"
            )
            return False

    def get_state(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """ดู state ของ instance"""
        instance = self._instances.get(instance_id)
        return instance.data.copy() if instance else None

    def broadcast_state(self, state_data: Dict[str, Any]) -> int:
        """
        broadcast state ไปยังทุก instance

        Args:
            state_data: state ที่ต้องการ broadcast

        Returns:
            จำนวน instances ที่ sync สำเร็จ
        """
        my_instance = self._instances[self._instance_id]
        my_instance.state_version += 1
        my_instance.data = state_data.copy()

        synced = 0
        for iid in self.get_alive_instances():
            if iid != self._instance_id:
                success = self.sync_state(
                    iid, state_data, my_instance.state_version
                )
                if success:
                    synced += 1

        logger.info(
            f"[DistributedSystem] BROADCAST_STATE synced={synced}/{self.instance_count-1}"
        )
        return synced

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Distributed Learning
    # ─────────────────────────────────────────────────────────────────────────

    def share_learning(
        self,
        update_type: str,
        data:        Dict[str, Any],
        confidence:  float = 1.0,
        update_id:   Optional[str] = None,
    ) -> LearningUpdate:
        """
        แชร์ learning update ไปยัง instances อื่น

        Args:
            update_type: "weight" / "knowledge" / "pattern"
            data: update data
            confidence: ความมั่นใจใน update
            update_id: ID (optional)

        Returns:
            LearningUpdate
        """
        uid = update_id or f"learn_{self._instance_id}_{int(time.time())}"

        update = LearningUpdate(
            update_id   = uid,
            source      = self._instance_id,
            update_type = update_type,
            data        = data,
            confidence  = confidence,
        )
        self._learning_updates.append(update)

        logger.info(
            f"[DistributedSystem] SHARE_LEARNING id={uid} "
            f"type={update_type} conf={confidence:.2f}"
        )
        return update

    def get_learning_updates(
        self,
        min_confidence: float = 0.0,
        update_type:    Optional[str] = None,
    ) -> List[LearningUpdate]:
        """
        ดึง learning updates จาก instances อื่น

        Args:
            min_confidence: confidence threshold
            update_type: filter by type (optional)

        Returns:
            List[LearningUpdate]
        """
        updates = [
            u for u in self._learning_updates
            if u.confidence >= min_confidence
            and (update_type is None or u.update_type == update_type)
        ]
        return updates

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Conflict Resolution
    # ─────────────────────────────────────────────────────────────────────────

    def detect_conflict(
        self,
        key:        str,
        value_a:    Any,
        value_b:    Any,
        instance_a: str,
        instance_b: str,
    ) -> Optional[ConflictRecord]:
        """
        ตรวจจับ conflict

        Args:
            key: key ที่ conflict
            value_a, value_b: values ที่ต่างกัน
            instance_a, instance_b: instances ที่ conflict

        Returns:
            ConflictRecord ถ้าเจอ conflict
        """
        if value_a == value_b:
            return None  # ไม่ conflict

        conflict = ConflictRecord(
            conflict_id = f"conflict_{int(time.time())}",
            instance_a  = instance_a,
            instance_b  = instance_b,
            key         = key,
            value_a     = value_a,
            value_b     = value_b,
            resolved    = False,
        )
        self._conflicts.append(conflict)

        logger.warning(
            f"[DistributedSystem] CONFLICT_DETECTED key={key} "
            f"A={instance_a} B={instance_b}"
        )
        return conflict

    def resolve_conflict(
        self,
        conflict:  ConflictRecord,
        strategy:  ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
    ) -> Any:
        """
        แก้ conflict

        Args:
            conflict: ConflictRecord
            strategy: วิธีแก้

        Returns:
            resolved value
        """
        if strategy == ConflictStrategy.LAST_WRITE_WINS:
            # ใช้ state_version เป็นตัวตัดสิน
            state_a = self._instances.get(conflict.instance_a)
            state_b = self._instances.get(conflict.instance_b)
            if state_a and state_b:
                resolution = (
                    conflict.value_a if state_a.state_version > state_b.state_version
                    else conflict.value_b
                )
            else:
                resolution = conflict.value_a

        elif strategy == ConflictStrategy.LEADER_DECIDES:
            # leader เลือก
            if self._leader_id == conflict.instance_a:
                resolution = conflict.value_a
            elif self._leader_id == conflict.instance_b:
                resolution = conflict.value_b
            else:
                resolution = conflict.value_a  # fallback

        else:  # MERGE / VOTE
            # ยังไม่ implement ซับซ้อน — fallback ใช้ value_a
            resolution = conflict.value_a

        # update conflict record
        object.__setattr__(conflict, 'resolved', True)
        object.__setattr__(conflict, 'resolution', resolution)
        object.__setattr__(conflict, 'strategy_used', strategy)

        logger.info(
            f"[DistributedSystem] CONFLICT_RESOLVED "
            f"strategy={strategy.value}"
        )
        return resolution

    @property
    def conflicts(self) -> List[ConflictRecord]:
        return list(self._conflicts)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Leader Election
    # ─────────────────────────────────────────────────────────────────────────

    def elect_leader(self) -> str:
        """
        เลือก leader instance

        Algorithm: instance ที่มี ID ต่ำสุด + alive = leader

        Returns:
            leader_id
        """
        alive = self.get_alive_instances()
        if not alive:
            raise RuntimeError("No alive instances for leader election")

        # เลือก ID ต่ำสุด (deterministic)
        leader = min(alive)

        # update roles
        for iid in self._instances:
            if iid == leader:
                self._instances[iid].role = InstanceRole.LEADER
            else:
                self._instances[iid].role = InstanceRole.FOLLOWER

        self._leader_id = leader

        # update my role
        if leader == self._instance_id:
            self._my_role = InstanceRole.LEADER
            logger.info(f"[DistributedSystem] ELECTED_LEADER self")
        else:
            self._my_role = InstanceRole.FOLLOWER
            logger.info(f"[DistributedSystem] ELECTED_LEADER {leader}")

        return leader

    @property
    def leader_id(self) -> Optional[str]:
        return self._leader_id

    @property
    def is_leader(self) -> bool:
        return self._my_role == InstanceRole.LEADER

    @property
    def my_role(self) -> InstanceRole:
        return self._my_role

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        alive = self.get_alive_instances()
        return {
            "instance_id":       self._instance_id,
            "my_role":           self._my_role.value,
            "leader_id":         self._leader_id,
            "is_leader":         self.is_leader,
            "total_instances":   len(self._instances),
            "alive_instances":   len(alive),
            "proposals":         len(self._proposals),
            "consensus_results": len(self._results),
            "conflicts":         len(self._conflicts),
            "conflicts_resolved": sum(1 for c in self._conflicts if c.resolved),
            "learning_updates":  len(self._learning_updates),
        }

    @property
    def instances(self) -> List[InstanceState]:
        return list(self._instances.values())

    def get_instance_state(self, instance_id: str) -> Optional[InstanceState]:
        return self._instances.get(instance_id)