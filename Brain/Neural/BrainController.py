"""
BrainController.py
Main brain controller integrating all rules, learning, and review components
"""

from typing import Dict, Any
import sys
import time
from pathlib import Path

# Add Rules directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Rules"))

from .Brain import BrainStructure as brain
from .BrainState import BrainState

from RuleEngine import RuleEngine
from System.SystemRule import SystemRule
from System.RuntimePolicy import RuntimePolicy
from Safety.SafetyRule import SafetyRule
from Routing.RoutingRule import RoutingEngine
from Memory.MemoryRule import MemoryRule
from Memory.AcquisitionRule import AcquisitionRule
from Memory.EmotionRule import EmotionRule
from Memory.TopicRule import TopicRule
from Learning.LearnRule import LearnRule
from Learning.GradientRule import GradientRule
from Adaption.AdaptionRule import AdaptionRule
from Adaption.StructureRule import StructureRule
from Adaption.EmotionDrivenRule import EmotionDrivenRule

from .Learning.LearningEngine import LearningEngine
from .Learning.GradientLearner import GradientLearner
from .Learning.ReplayLearner import ReplayLearner
from .Learning.SelfLearner import SelfLearner
from .Learning.EvolutionLearner import EvolutionLearner

from Review.ApproveEngine import ApproveEngine
from Review.ConfidenceScorer import ConfidenceScorer
from Review.PerformanceMonitor import PerformanceMonitor


# ==========================================
# Brain summary ASCII formatter (function)
# ==========================================

WIDTH = 67  # ปรับได้ แต่เลขนี้เผื่อ emoji แล้ว

def box_line(text: str = "") -> str:
    return f"│ {text:<{WIDTH - 4}} │"


def _format_brain_summary_ascii(brain) -> str:
    nodes = getattr(brain, "nodes", {})
    conns = getattr(brain, "connections", {})
    biases = getattr(brain, "biases", {})

    total_nodes = len(nodes)
    total_connections = sum(1 for c in conns.values() if c.get("enabled"))
    total_weights = total_connections
    total_biases = len(biases)
    total_params = total_weights + total_biases

    role_count = {"input": 0, "hidden": 0, "output": 0}
    layers = set()
    total_usage = 0.0

    for n in nodes.values():
        role = n.get("role", "hidden")
        role_count[role] = role_count.get(role, 0) + 1
        layers.add(n.get("layer", 0))
        total_usage += n.get("usage", 0.0)

    avg_usage = total_usage / total_nodes if total_nodes > 0 else 0.0

    lines = []

    # ================= HEADER =================
    lines.append("┌" + "─" * (WIDTH - 2) + "┐")
    lines.append(box_line("🧠 Brain Summary"))
    lines.append("├" + "─" * (WIDTH - 2) + "┤")
    lines.append(box_line(f"Model type         : {getattr(brain, 'model_type', 'Unknown')}"))
    lines.append(box_line(f"Layers             : {len(layers)}"))
    lines.append(box_line(f"Nodes              : {total_nodes}"))
    lines.append(box_line(f"  ├─ Input          : {role_count['input']}"))
    lines.append(box_line(f"  ├─ Hidden         : {role_count['hidden']}"))
    lines.append(box_line(f"  └─ Output         : {role_count['output']}"))
    lines.append(box_line(f"Active connections : {total_connections}"))
    lines.append(box_line(f"Parameters         : {total_params}"))
    lines.append(box_line(f"  ├─ Weights        : {total_weights}"))
    lines.append(box_line(f"  └─ Biases         : {total_biases}"))
    lines.append(box_line(f"Avg usage / node   : {avg_usage:.2f}"))
    lines.append("└" + "─" * (WIDTH - 2) + "┘")

    # ================= TABLE =================
    lines.append("")
    lines.append("┌──────┬──────────────────────┬──────────┬──────────┬────────┬────────┐")
    lines.append("│Layer │ Node ID              │ Role     │ Head     │ Usage% │ Params │")
    lines.append("├──────┼──────────────────────┼──────────┼──────────┼────────┼────────┤")

    if total_nodes == 0:
        lines.append("│  --  │      (no nodes)      │    --    │    --    │   --   │   --   │")
    else:
        for nid, n in sorted(nodes.items(), key=lambda x: (x[1].get("layer", 0), x[0])):
            usage = n.get("usage", 0.0)
            usage_pct = (usage / total_usage * 100.0) if total_usage > 0 else 0.0

            param_count = 1  # bias
            for c in conns.values():
                if c.get("enabled") and c.get("destination") == nid:
                    param_count += 1

            lines.append(
                f"│ {n.get('layer', 0):<4} "
                f"│ {nid:<20} "
                f"│ {n.get('role', 'hidden'):<8} "
                f"│ {str(n.get('head', '-')):<8} "
                f"│ {usage_pct:>6.2f} "
                f"│ {param_count:>6} │"
            )

    lines.append("└──────┴──────────────────────┴──────────┴──────────┴────────┴────────┘")

    return "\n".join(lines)


class BrainController:
    """Central brain controller orchestrating all systems."""

    def __init__(self) -> None:
        
        # Braiin structure reference
        self.brain = brain()
        
        # Core rule engine
        self.rule_engine = RuleEngine()

        # System rules
        self.system_rule = SystemRule()
        self.runtime_policy = RuntimePolicy()

        # Safety
        self.safety_rule = SafetyRule()

        # Routing
        self.routing_engine = RoutingEngine()

        # Memory
        self.memory_rule = MemoryRule()
        self.acquisition_rule = AcquisitionRule()
        self.emotion_rule = EmotionRule()
        self.topic_rule = TopicRule()

        # Learning rules (symbolic)
        self.learn_rule = LearnRule()
        self.gradient_rule = GradientRule()

        # Adaptation
        self.adaption_rule = AdaptionRule()
        self.structure_rule = StructureRule()
        self.emotion_driven_rule = EmotionDrivenRule()

        # Brain State
        self.brain_state = BrainState(storage_path="./weights")

        # Learning engines
        self.learning_engine = LearningEngine()
        self.gradient_learner = GradientLearner()
        self.replay_learner = ReplayLearner()
        self.self_learner = SelfLearner()
        self.evolution_learner = EvolutionLearner()

        self.learning_engine.register_learner("gradient", self.gradient_learner)
        self.learning_engine.register_learner("replay", self.replay_learner)
        self.learning_engine.register_learner("self", self.self_learner)
        self.learning_engine.register_learner("evolution", self.evolution_learner)

        # Review layer
        self.approve_engine = ApproveEngine()
        self.confidence_scorer = ConfidenceScorer()
        self.performance_monitor = PerformanceMonitor()

        # Runtime state
        self.running = False
        self.cycle_count = 0
        self.system_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            self.rule_engine.reload_configs()
            self.runtime_policy.load_from_json()
            self.safety_rule.load_from_json()
            self.memory_rule.load_from_json()
            self.acquisition_rule.load_from_json()
            self.learn_rule.load_from_json()
            self.adaption_rule.load_from_json()
            return True
        except Exception as e:
            print(f"[BrainController] Init error: {e}")
            return False

    def start(self) -> bool:
        if not self.initialize():
            return False
        self.running = True
        return True

    def stop(self) -> bool:
        self.running = False
        return True

    # ------------------------------------------------------------------
    def handle_command(self, context: Dict[str, Any]) -> Dict[str, Any]:
        command = context.get("command")
        payload = context.get("payload", {})

        if command == "summary":
            return {
                "type": "system_response",
                "command": "summary",
                "structure": self.get_brain_structure_summary()
            }

        if command == "start_learning":
            self.learning_engine.enable_learning()
            return {"status": "learning_enabled"}

        if command == "stop_learning":
            self.learning_engine.disable_learning()
            return {"status": "learning_disabled"}

        if command == "reset_brain":
            self.brain_state.reset()
            return {"status": "brain_reset"}

        if command == "emergency_stop":
            return {"status": "stopped"} if self.handle_emergency_stop() else {"error": "failed"}

        return {
            "type": "system_response",
            "error": f"Unknown command: {command}"
        }
        
    def get_brain_structure_summary(self) -> str:
        """
        Return formatted brain structure summary (ASCII table)
        """
        try:
            return _format_brain_summary_ascii(self.brain)
        except Exception as e:
            return f"[BrainController] Summary failed: {e}"


    def process_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.running:
            return {"error": "Brain controller not running"}
        
        # ===============================
        # Command handling (SYSTEM LEVEL)
        # ===============================
        if context.get("input_type") == "command":
            return self.handle_command(context)

        self.cycle_count += 1

        cycle_result = {
            "cycle_number": self.cycle_count,
            "timestamp": time.time(),
            "decisions": [],
            "actions": [],            # internal actions
            "approved_actions": [],   # externally allowed actions
            "insights": [],
            "review": {}
        }

        # 1. Resource check
        resource_status = self.runtime_policy.check_resource_usage()
        if resource_status.get("memory_critical") or resource_status.get("cpu_critical"):
            cycle_result["recovery_triggered"] = self.runtime_policy.trigger_recovery()
            return cycle_result

        # 2. Safety (pre-check)
        if context.get("proposed_action"):
            safety = self.safety_rule.check_action_safety(
                action_type="motion",
                action=context["proposed_action"]
            )
            cycle_result["safety_check"] = safety
            if not safety.get("safe"):
                return cycle_result

        # 3. Routing & memory
        if context.get("incoming_data"):
            routed = self.routing_engine.route_data(
                context.get("data_type"),
                context["incoming_data"]
            )
            cycle_result["routed_data"] = routed

            acquire, _ = self.acquisition_rule.should_acquire_memory(
                context["incoming_data"]
            )
            if acquire:
                cycle_result["memory_acquisition"] = \
                    self.acquisition_rule.acquire_memory(context["incoming_data"])

        # 4. Rule execution → decisions/actions
        rule_output = self.rule_engine.execute_rules(context)
        cycle_result["rule_results"] = rule_output

        # normalize rule outputs
        cycle_result["decisions"] = rule_output.get("decisions", [])
        cycle_result["actions"] = rule_output.get("actions", [])

        # 5. Gradient learning
        if context.get("learning_data"):
            gradients = context["learning_data"].get("gradients")
            if gradients:
                deltas = self.gradient_learner.update_weights(
                    self.brain_state.get_weights().weights,
                    gradients
                )
                self.brain_state.apply_weight_updates(deltas, source="gradient")
                cycle_result["weight_update"] = list(deltas.keys())

        # 6. Meta learning engine
        if context.get("learning_context"):
            engine_result = self.learning_engine.coordinate_learning(
                context["learning_context"]
            )
            if engine_result.get("weight_deltas"):
                self.brain_state.apply_weight_updates(
                    engine_result["weight_deltas"],
                    source="learning_engine"
                )
            cycle_result["learning_engine"] = engine_result

        # 7. Replay memory
        if context.get("experience"):
            self.replay_learner.store_experience(context["experience"])

        # 8. Self reflection
        for decision in cycle_result["decisions"]:
            reflection = self.self_learner.reflect_on_decision(decision.get("id"))
            insight = self.self_learner.extract_insight(reflection)
            if insight:
                cycle_result["insights"].append(insight)

        # 9. Adaptation
        if cycle_result["insights"]:
            improvement = self.self_learner.identify_improvement_areas()
            if improvement:
                self.adaption_rule.register_internal_feedback(improvement)

        # --------------------------------------------------
        # Review Phase (judgment before external output)
        # --------------------------------------------------
        review_results = []

        for action in cycle_result["actions"]:
            confidence = self.confidence_scorer.compute_confidence(
                output=action,
                evidence={
                    "brain_state": self.brain_state.get_state_summary(),
                    "rules": cycle_result.get("rule_results"),
                    "learning": self.learning_engine.get_learning_progress()
                }
            )

            approved = self.approve_engine.evaluate_output(
                output=action,
                context={"confidence": confidence}
            )

            self.performance_monitor.record_performance("confidence", confidence)
            self.performance_monitor.record_performance("approved", int(approved))

            review_results.append({
                "action_id": action.get("id"),
                "confidence": confidence,
                "approved": approved
            })

            if approved:
                cycle_result["approved_actions"].append(action)
            else:
                self.approve_engine.reject_output(
                    output_id=action.get("id", "unknown"),
                    reason="Low confidence or review rejection"
                )
                self.self_learner.learn_from_mistakes({
                    "action": action,
                    "confidence": confidence,
                    "reason": "review_rejection"
                })

        cycle_result["review"] = {
            "results": review_results,
            "approval_rate": self.approve_engine.get_approval_rate()
        }

        # Long-term degradation feedback
        degradation = self.performance_monitor.detect_performance_degradation()
        if degradation:
            self.adaption_rule.register_internal_feedback({
                "performance_degradation": degradation
            })

        # 10. Metrics
        self.system_metrics = {
            "cycle": self.cycle_count,
            "brain_state": self.brain_state.get_state_summary(),
            "learning": self.learning_engine.get_learning_progress(),
            "rules": self.rule_engine.get_rule_status(),
            "review": cycle_result["review"]
        }

        return cycle_result

    # ------------------------------------------------------------------

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "brain_state": self.brain_state.get_state_summary(),
            "resource_status": self.runtime_policy.get_resource_status(),
            "learning_status": self.learning_engine.get_learning_progress()
        }

    def handle_emergency_stop(self) -> bool:
        self.running = False
        self.safety_rule.trigger_emergency_stop()
        return True