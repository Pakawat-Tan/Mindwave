"""
จัดการ Rule ทั้งหมด — governance, evaluation, proposals, persistence

เหมือน Tier ใน Memory แต่เป็น Rule layer
ConditionController เรียกผ่าน self._rule

API:
  governance_add(rule, creator_id, reviewer_id)
  governance_remove(rule_id, creator_id, reviewer_id)
  check(scope, text, cluster_id)  → RuleResult
  submit_proposal(proposal)
  approve_proposal(proposal_id, reviewer_id, creator_id)
  reject_proposal(proposal_id, reviewer_id)
  load_defaults(creator_id, reviewer_id, defaults_dir)
  save() / load()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
from Core.Condition.Rule.RuleData import (
    RuleData, RuleScope, RuleAction, RuleAuthority,
    RuleResult, ALLOW_BY_DEFAULT, create_rule, MatchType
)
from ..Proposal import (
    ProposalData, ProposalAction, ProposalTarget, ProposalStatus
)


class RuleController:

    def __init__(self, data_path: str):
        self._path    = Path(data_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._logger  = logging.getLogger("mindwave.condition.rule")
        self._rules:     Dict[str, RuleData]     = {}
        self._proposals: Dict[str, ProposalData] = {}
        self.load()

    # ─────────────────────────────────────────────────────────────
    # Governance — direct (startup / admin)
    # ─────────────────────────────────────────────────────────────

    def governance_add(
        self,
        rule:        RuleData,
        reviewer_id: Optional[str] = None,
        creator_id:  Optional[str] = None,
    ) -> None:
        actor = self._authorize(rule.authority, reviewer_id, creator_id, "add")
        self._rules[rule.rule_id] = rule
        self._logger.warning(
            f"[RuleController] ADD {rule.rule_id[:8]} "
            f"[{rule.authority}] by='{actor}' | {rule}"
        )

    def governance_remove(
        self,
        rule_id:     str,
        reviewer_id: Optional[str] = None,
        creator_id:  Optional[str] = None,
    ) -> bool:
        rule = self._rules.get(rule_id)
        if rule is None:
            return False
        actor = self._authorize(rule.authority, reviewer_id, creator_id, "remove")
        del self._rules[rule_id]
        self._logger.warning(
            f"[RuleController] REMOVE {rule_id[:8]} "
            f"[{rule.authority}] by='{actor}'"
        )
        return True

    def get(self, rule_id: str) -> Optional[RuleData]:
        return self._rules.get(rule_id)

    def list(self, scope: Optional[RuleScope] = None) -> List[RuleData]:
        rules = list(self._rules.values())
        if scope is not None:
            rules = [r for r in rules if r.scope == scope]
        return sorted(rules, key=lambda r: r.priority, reverse=True)

    # ─────────────────────────────────────────────────────────────
    # Check (Rule Layer evaluation)
    # ─────────────────────────────────────────────────────────────

    def check(
        self,
        scope:      RuleScope,
        text:       str           = "",
        cluster_id: Optional[int] = None,
    ) -> RuleResult:
        """First-match wins — priority สูง → ต่ำ"""
        for rule in self.list(scope=scope):
            if rule.matches(text=text, cluster_id=cluster_id):
                self._logger.debug(
                    f"[RuleController] MATCH [{scope}] → {rule.action} "
                    f"({rule.rule_id[:8]})"
                )
                return RuleResult(
                    action         = rule.action,
                    triggered_rule = rule,
                    reason         = rule.description or f"matched {rule.rule_id[:8]}",
                )
        return ALLOW_BY_DEFAULT

    # ─────────────────────────────────────────────────────────────
    # Proposal system (for rules)
    # ─────────────────────────────────────────────────────────────

    def submit_proposal(self, proposal: ProposalData) -> str:
        if proposal.target_type != ProposalTarget.RULE:
            raise ValueError(
                f"[RuleController] proposal target must be RULE, "
                f"got {proposal.target_type}"
            )
        if not proposal.is_pending:
            raise ValueError(
                f"[RuleController] proposal must be PENDING, "
                f"got {proposal.status}"
            )
        self._proposals[proposal.proposal_id] = proposal
        self._logger.info(
            f"[RuleController] PROPOSAL_SUBMITTED {proposal.proposal_id[:8]} "
            f"{proposal.action} by='{proposal.proposed_by}'"
        )
        return proposal.proposal_id

    def approve_proposal(
        self,
        proposal_id: str,
        reviewer_id: Optional[str] = None,
        creator_id:  Optional[str] = None,
        note:        str           = "",
    ) -> bool:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        if not proposal.is_pending:
            raise ValueError(
                f"[RuleController] proposal {proposal_id[:8]} "
                f"is already {proposal.status}"
            )
        # ตรวจสิทธิ์ตาม authority ของ Rule ที่จะเพิ่ม/แก้ไข
        from Core.Review.Proposal import RuleAuthority as PA
        auth = RuleAuthority(proposal.authority.value)
        actor = self._authorize(auth, reviewer_id, creator_id, "approve proposal")
        self._apply_proposal(proposal)
        proposal._approve(actor, note)
        self._logger.warning(
            f"[RuleController] PROPOSAL_APPROVED {proposal_id[:8]} "
            f"by='{actor}'"
        )
        return True

    def reject_proposal(
        self,
        proposal_id: str,
        reviewer_id: str,
        note:        str = "",
    ) -> bool:
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[RuleController] reject_proposal requires reviewer_id"
            )
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False
        if not proposal.is_pending:
            raise ValueError(
                f"[RuleController] proposal {proposal_id[:8]} "
                f"is already {proposal.status}"
            )
        proposal._reject(reviewer_id, note)
        self._logger.info(
            f"[RuleController] PROPOSAL_REJECTED {proposal_id[:8]} "
            f"by='{reviewer_id}'"
        )
        return True

    def get_proposal(self, proposal_id: str) -> Optional[ProposalData]:
        return self._proposals.get(proposal_id)

    def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
    ) -> List[ProposalData]:
        proposals = list(self._proposals.values())
        if status is not None:
            proposals = [p for p in proposals if p.status == status]
        return sorted(proposals, key=lambda p: p.created_ts, reverse=True)

    # ─────────────────────────────────────────────────────────────
    # Default JSON loader
    # ─────────────────────────────────────────────────────────────

    def load_defaults(
        self,
        creator_id:   str,
        reviewer_id:  str           = "default_system",
        defaults_dir: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        โหลด Default Rules จากทุก JSON ใน defaults_dir

        Returns:
            {category: count}
        """
        d = Path(defaults_dir) if defaults_dir else \
            Path(__file__).parent / "Defaults"

        summary = {}
        for json_path in sorted(d.glob("*.json")):
            category = json_path.stem
            try:
                count = self._load_json_file(
                    json_path, creator_id, reviewer_id
                )
                summary[category] = count
                self._logger.info(
                    f"[RuleController] loaded {count} rules from {category}"
                )
            except Exception as e:
                self._logger.error(
                    f"[RuleController] failed to load {category}: {e}"
                )
        return summary

    # ─────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────

    def save(self) -> None:
        (self._path / "rules.json").write_text(
            json.dumps([r.to_dict() for r in self._rules.values()], indent=2),
            encoding="utf-8"
        )
        (self._path / "rule_proposals.json").write_text(
            json.dumps([p.to_dict() for p in self._proposals.values()], indent=2),
            encoding="utf-8"
        )

    def load(self) -> None:
        rules_path    = self._path / "rules.json"
        proposals_path = self._path / "rule_proposals.json"
        if rules_path.exists():
            for d in json.loads(rules_path.read_text(encoding="utf-8")):
                r = RuleData.from_dict(d)
                self._rules[r.rule_id] = r
        if proposals_path.exists():
            for d in json.loads(proposals_path.read_text(encoding="utf-8")):
                p = ProposalData.from_dict(d)
                self._proposals[p.proposal_id] = p

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "rules_total":    len(self._rules),
            "rules_system":   sum(1 for r in self._rules.values() if r.authority == RuleAuthority.SYSTEM),
            "rules_standard": sum(1 for r in self._rules.values() if r.authority == RuleAuthority.STANDARD),
            "proposals_pending": sum(1 for p in self._proposals.values() if p.is_pending),
        }

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _authorize(
        self,
        authority:   RuleAuthority,
        reviewer_id: Optional[str],
        creator_id:  Optional[str],
        action:      str,
    ) -> str:
        if authority == RuleAuthority.SYSTEM:
            if not creator_id or not creator_id.strip():
                raise PermissionError(
                    f"[RuleController] SYSTEM rule {action} requires creator_id"
                )
            return creator_id
        else:
            if not reviewer_id or not reviewer_id.strip():
                raise PermissionError(
                    f"[RuleController] STANDARD rule {action} requires reviewer_id"
                )
            return reviewer_id

    def _apply_proposal(self, proposal: ProposalData) -> None:
        if proposal.action == ProposalAction.ADD:
            rule = RuleData.from_dict(proposal.payload)
            self._rules[rule.rule_id] = rule
        elif proposal.action == ProposalAction.REMOVE:
            self._rules.pop(proposal.payload.get("rule_id", ""), None)
        elif proposal.action == ProposalAction.MODIFY:
            rule = RuleData.from_dict(proposal.payload)
            self._rules[rule.rule_id] = rule

    def _load_json_file(
        self,
        path:        Path,
        creator_id:  str,
        reviewer_id: str,
    ) -> int:
        data      = json.loads(path.read_text(encoding="utf-8"))
        authority = RuleAuthority(data.get("_authority", "standard"))
        count     = 0
        for rd in data.get("rules", []):
            rule = create_rule(
                scope            = RuleScope(rd["scope"]),
                action           = RuleAction(rd["action"]),
                match_type       = MatchType(rd["match_type"]),
                pattern          = rd.get("pattern"),
                topic_cluster_id = rd.get("topic_cluster_id"),
                priority         = rd.get("priority", 0),
                description      = rd.get("description", ""),
                use_regex        = rd.get("use_regex", False),
                authority        = authority,
            )
            self.governance_add(
                rule,
                creator_id  = creator_id  if authority == RuleAuthority.SYSTEM else None,
                reviewer_id = reviewer_id if authority == RuleAuthority.STANDARD else None,
            )
            count += 1
        return count