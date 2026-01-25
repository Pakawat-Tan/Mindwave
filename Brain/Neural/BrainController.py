

"""
BrainController.py
Main brain controller integrating all rules and neural components
"""

from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add Rules directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Rules"))

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


class BrainController:
    """Central brain controller orchestrating all systems."""
    
    def __init__(self) -> None:
        """Initialize brain controller with all rule engines."""
        self.rule_engine = RuleEngine()
        
        # System management rules
        self.system_rule = SystemRule()
        self.runtime_policy = RuntimePolicy()
        
        # Safety rules
        self.safety_rule = SafetyRule()
        
        # Information routing
        self.routing_engine = RoutingEngine()
        
        # Memory management rules
        self.memory_rule = MemoryRule()
        self.acquisition_rule = AcquisitionRule()
        self.emotion_rule = EmotionRule()
        self.topic_rule = TopicRule()
        
        # Learning rules
        self.learn_rule = LearnRule()
        self.gradient_rule = GradientRule()
        
        # Adaptation rules
        self.adaption_rule = AdaptionRule()
        self.structure_rule = StructureRule()
        self.emotion_driven_rule = EmotionDrivenRule()
        
        # Controller state
        self.running = False
        self.cycle_count = 0
        self.system_metrics: Dict[str, Any] = {}
    
    def initialize(self) -> bool:
        """Initialize all brain systems."""
        try:
            # Load all rule configurations
            self.rule_engine.reload_configs()
            
            # Initialize runtime policy
            self.runtime_policy.load_from_json()
            
            # Initialize safety rules
            self.safety_rule.load_from_json()
            
            # Initialize other rules
            self.memory_rule.load_from_json()
            self.acquisition_rule.load_from_json()
            self.learn_rule.load_from_json()
            self.adaption_rule.load_from_json()
            
            return True
        except Exception as e:
            print(f"Error initializing BrainController: {e}")
            return False
    
    def start(self) -> bool:
        """Start the brain controller."""
        if not self.initialize():
            return False
        
        self.running = True
        return True
    
    def stop(self) -> bool:
        """Stop the brain controller."""
        self.running = False
        return True
    
    def process_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process one brain cycle.
        
        Parameters
        ----------
        context : Dict[str, Any]
            Current system context and inputs
            
        Returns
        -------
        Dict[str, Any]
            Cycle results and actions to take
        """
        if not self.running:
            return {"error": "Brain controller not running"}
        
        self.cycle_count += 1
        cycle_result = {
            "cycle_number": self.cycle_count,
            "timestamp": __import__("time").time(),
            "actions": [],
            "decisions": []
        }
        
        # 1. Check resource constraints
        resource_status = self.runtime_policy.check_resource_usage()
        if not resource_status.get("memory_critical") and not resource_status.get("cpu_critical"):
            # 2. Execute safety checks on proposed actions
            if context.get("proposed_action"):
                safety_check = self.safety_rule.check_action_safety(
                    action_type="motion",
                    action=context["proposed_action"]
                )
                cycle_result["safety_check"] = safety_check
                
                if not safety_check.get("safe"):
                    return cycle_result  # Stop processing if unsafe
            
            # 3. Route incoming information
            if context.get("incoming_data"):
                routed_data = self.routing_engine.route_data(
                    data_type=context.get("data_type"),
                    data=context["incoming_data"]
                )
                cycle_result["routed_data"] = routed_data
                
                # 4. Determine if data should be acquired to memory
                should_acquire = self.acquisition_rule.should_acquire_memory(context["incoming_data"])
                if should_acquire[0]:
                    acquisition_result = self.acquisition_rule.acquire_memory(context["incoming_data"])
                    cycle_result["memory_acquisition"] = acquisition_result
            
            # 5. Check learning opportunities
            if context.get("learning_data"):
                if self.learn_rule.is_mode_enabled(self.learn_rule.active_modes[0]):
                    gradient_step = self.gradient_rule.compute_gradient_step(
                        context["learning_data"].get("gradients", {})
                    )
                    cycle_result["learning_updates"] = gradient_step
            
            # 6. Check for adaptation needs
            if context.get("system_metrics"):
                adaptation_needs = self.adaption_rule.evaluate_adaptation_need(context["system_metrics"])
                if any(adaptation_needs.values()):
                    cycle_result["adaptation_needed"] = adaptation_needs
            
            # 7. Execute all applicable rules
            rule_results = self.rule_engine.execute_rules(context)
            cycle_result["rule_results"] = rule_results
        else:
            # Trigger recovery if resources are critical
            recovery = self.runtime_policy.trigger_recovery()
            cycle_result["recovery_triggered"] = recovery
        
        # Update system metrics
        self.system_metrics = {
            "cycle_count": self.cycle_count,
            "resource_status": resource_status,
            "rule_engine_status": self.rule_engine.get_rule_status()
        }
        
        return cycle_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns
        -------
        Dict[str, Any]
            Current system status
        """
        return {
            "running": self.running,
            "cycle_count": self.cycle_count,
            "resource_status": self.runtime_policy.get_resource_status(),
            "rule_engine_status": self.rule_engine.get_rule_status(),
            "memory_status": self.acquisition_rule.get_storage_status(),
            "routing_status": self.routing_engine.get_buffer_status(),
            "learning_status": self.learn_rule.get_learning_statistics(),
            "adaptation_status": self.adaption_rule.get_adaptation_status()
        }
    
    def handle_emergency_stop(self) -> bool:
        """Handle emergency stop request."""
        self.running = False
        self.safety_rule.trigger_emergency_stop()
        return True
    
    