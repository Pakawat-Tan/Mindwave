"""
RuleEngine.py (CoreRule)
ตัว dispatch กฎทั้งหมด
Central rule engine for managing all rules - connects to BrainController.py
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
from ConfigLoader import ConfigLoader


@dataclass
class Rule:
    """Represents a single rule"""
    name: str
    rule_type: str  # e.g., "safety", "learning", "memory", "adaption"
    enabled: bool = True
    priority: int = 0
    conditions: Dict[str, Any] | None = None
    actions: Dict[str, Any] | None = None
    handler: Optional[Callable[..., Any]] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.actions is None:
            self.actions = {}


class RuleEngine:
    """Central engine for dispatching and managing all rules."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        """Initialize rule engine.
        
        Parameters
        ----------
        config_loader : ConfigLoader, optional
            Config loader instance for loading rule configurations
        """
        self.rules: Dict[str, Rule] = {}
        self.rule_priorities: Dict[str, int] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.config_loader = config_loader or ConfigLoader()
        
        # Load all rule configurations
        self.configs = self.config_loader.load_all_system_configs()
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize all rules from configurations"""
        # System Rules
        if self.configs.get("SystemRule"):
            self._init_system_rules(self.configs["SystemRule"])
        
        # Runtime Policy
        if self.configs.get("RuntimePolicy"):
            self._init_runtime_rules(self.configs["RuntimePolicy"])
        
        # Safety Rules
        if self.configs.get("SafetyPolicy"):
            self._init_safety_rules(self.configs["SafetyPolicy"])
        
        # Learning Rules
        if self.configs.get("LearningRule"):
            self._init_learning_rules(self.configs["LearningRule"])
        
        # Memory Rules
        if self.configs.get("MemoryRule"):
            self._init_memory_rules(self.configs["MemoryRule"])
        if self.configs.get("AcquisitionRule"):
            self._init_acquisition_rules(self.configs["AcquisitionRule"])
        
        # Routing Rules
        if self.configs.get("RoutingRule"):
            self._init_routing_rules(self.configs["RoutingRule"])
        
        # Adaption Rules
        if self.configs.get("AdaptionRule"):
            self._init_adaption_rules(self.configs["AdaptionRule"])
    
    def _init_system_rules(self, config: Dict[str, Any]):
        """Initialize system-level rules"""
        capabilities = config.get("capability_flags", {})
        governance = config.get("governance", {})
        
        self.register_rule(
            name="allow_learning",
            rule_type="system",
            enabled=capabilities.get("allow_learning", True),
            priority=4
        )
        self.register_rule(
            name="allow_external_learning",
            rule_type="system",
            enabled=capabilities.get("allow_external_learning", False),
            priority=4
        )
        self.register_rule(
            name="allow_adaptation",
            rule_type="system",
            enabled=capabilities.get("allow_adaptation", True),
            priority=4
        )
        self.register_rule(
            name="require_review_for_new_rules",
            rule_type="system",
            enabled=governance.get("require_review_for_new_rules", True),
            priority=3
        )
    
    def _init_runtime_rules(self, config: Dict[str, Any]):
        """Initialize runtime policy rules"""
        rm = config.get("resource_management", {})
        timeouts = config.get("timeout_policies", {})
        
        self.register_rule(
            name="memory_limit",
            rule_type="runtime",
            enabled=True,
            priority=3,
            conditions={
                "max_usage_percent": rm.get("memory", {}).get("max_usage_percent", 80),
                "warning_threshold": rm.get("memory", {}).get("warning_threshold", 70)
            }
        )
        
        self.register_rule(
            name="cpu_limit",
            rule_type="runtime",
            enabled=True,
            priority=3,
            conditions={
                "max_usage_percent": rm.get("cpu", {}).get("max_usage_percent", 90),
                "warning_threshold": rm.get("cpu", {}).get("warning_threshold", 75)
            }
        )
        
        self.register_rule(
            name="operation_timeout",
            rule_type="runtime",
            enabled=True,
            priority=2,
            conditions={
                "operation_timeout_ms": timeouts.get("operation_timeout_ms", 5000),
                "learning_timeout_ms": timeouts.get("learning_timeout_ms", 30000),
                "inference_timeout_ms": timeouts.get("inference_timeout_ms", 1000)
            }
        )
    
    def _init_safety_rules(self, config: Dict[str, Any]):
        """Initialize safety rules"""
        motion = config.get("motion_safety", {})
        output = config.get("output_safety", {})
        
        self.register_rule(
            name="motion_safety",
            rule_type="safety",
            enabled=motion.get("enabled", True),
            priority=4,
            conditions={
                "max_velocity": motion.get("max_movement_velocity", 0.5),
                "max_force": motion.get("max_force_newtons", 100)
            }
        )
        
        self.register_rule(
            name="output_safety",
            rule_type="safety",
            enabled=output.get("filter_harmful_content", True),
            priority=4,
            conditions={
                "max_risk_level": output.get("max_response_risk_level", 0.3),
                "require_review_threshold": output.get("require_review_for_risk_level", 0.7)
            }
        )
    
    def _init_learning_rules(self, config: Dict[str, Any]):
        """Initialize learning rules"""
        gradient = config.get("gradient_learning", {})
        advisor = config.get("advisor_learning", {})
        
        self.register_rule(
            name="gradient_learning",
            rule_type="learning",
            enabled=gradient.get("enabled", True),
            priority=3,
            conditions={
                "learning_rate": gradient.get("learning_rate", 0.01),
                "momentum": gradient.get("momentum", 0.9),
                "gradient_clipping": gradient.get("gradient_clipping", 1.0)
            }
        )
        
        self.register_rule(
            name="advisor_learning",
            rule_type="learning",
            enabled=advisor.get("enabled", True),
            priority=3,
            conditions={
                "respects_external_flag": advisor.get("respects_external_learning_flag", True),
                "feedback_weight": advisor.get("advisor_feedback_weight", 0.5)
            }
        )
    
    def _init_memory_rules(self, config: Dict[str, Any]):
        """Initialize memory rules"""
        consolidation = config.get("memory_consolidation", {})
        decay = config.get("memory_decay", {})
        
        self.register_rule(
            name="consolidation",
            rule_type="memory",
            enabled=consolidation.get("enable_consolidation", True),
            priority=2,
            conditions={
                "interval_minutes": consolidation.get("consolidation_interval_minutes", 30),
                "threshold": consolidation.get("consolidation_threshold", 0.5)
            }
        )
        
        self.register_rule(
            name="memory_decay",
            rule_type="memory",
            enabled=decay.get("enable_forgetting", True),
            priority=2,
            conditions={
                "base_decay_rate": decay.get("base_decay_rate", 0.05),
                "emotional_protection": decay.get("emotional_protection", True)
            }
        )
    
    def _init_acquisition_rules(self, config: Dict[str, Any]):
        """Initialize acquisition rules"""
        filters = config.get("filtering", {})
        triggers = config.get("acquisition_triggers", {})
        
        self.register_rule(
            name="memory_acquisition",
            rule_type="memory",
            enabled=True,
            priority=2,
            conditions={
                "novelty_threshold": triggers.get("novelty_threshold", 0.6),
                "emotion_threshold": triggers.get("emotion_strength_threshold", 0.5),
                "filter_duplicates": filters.get("filter_duplicate_content", True)
            }
        )
    
    def _init_routing_rules(self, config: Dict[str, Any]):
        """Initialize routing rules"""
        routing = config.get("routing_rules", {})
        
        for data_type, route_config in routing.items():
            self.register_rule(
                name=f"route_{data_type}",
                rule_type="routing",
                enabled=True,
                priority=route_config.get("priority", 1),
                conditions={
                    "route_to": route_config.get("route_to", []),
                    "buffer_size": route_config.get("buffer_size", 100)
                }
            )
    
    def _init_adaption_rules(self, config: Dict[str, Any]):
        """Initialize adaption rules"""
        triggers = config.get("adaptation_triggers", {})
        structural = config.get("structural_adaptation", {})
        
        self.register_rule(
            name="structural_adaptation",
            rule_type="adaption",
            enabled=structural.get("enabled", True),
            priority=3,
            conditions={
                "allow_neuron_addition": structural.get("allow_neuron_addition", True),
                "max_change_rate": structural.get("max_structural_change_rate", 0.05),
                "require_approval": structural.get("require_advisor_approval", True)
            }
        )
    
    def register_rule(self, name: str, rule_type: str, enabled: bool = True, 
                     priority: int = 0, conditions: Dict[str, Any] | None = None,
                     actions: Dict[str, Any] | None = None, handler: Optional[Callable[..., Any]] = None) -> bool:
        """Register a rule."""
        rule = Rule(
            name=name,
            rule_type=rule_type,
            enabled=enabled,
            priority=priority,
            conditions=conditions or {},
            actions=actions or {},
            handler=handler
        )
        
        self.rules[name] = rule
        self.rule_priorities[name] = priority
        return True
    
    def execute_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute applicable rules based on context."""
        applicable_rules = self.get_applicable_rules(context)
        results = []
        
        # Sort by priority (highest first)
        sorted_rules = sorted(applicable_rules, key=lambda r: self.rule_priorities.get(r.name, 0), reverse=True)
        
        for rule in sorted_rules:
            if rule.enabled:
                result = self.apply_rule_actions(rule, context)
                results.append(result)
                self._log_execution(rule, result)
        
        return {
            "executed_count": len(results),
            "results": results
        }
    
    def check_rule_conditions(self, rule_name: str, context: Dict[str, Any]) -> bool:
        """Check if rule conditions are met."""
        if rule_name not in self.rules:
            return False
        
        rule = self.rules[rule_name]
        
        # Custom condition checking logic
        for condition_key, condition_value in rule.conditions.items():
            if condition_key in context:
                if context[condition_key] != condition_value:
                    return False
        
        return True
    
    def apply_rule_actions(self, rule: Rule, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply actions of a rule."""
        result = {
            "rule_name": rule.name,
            "rule_type": rule.rule_type,
            "executed": False,
            "actions": []
        }
        
        # Check conditions first
        if not self.check_rule_conditions(rule.name, context):
            return result
        
        # Execute custom handler if provided
        if rule.handler:
            try:
                action_result = rule.handler(context)
                result["actions"].append(action_result)
                result["executed"] = True
            except Exception as e:
                result["error"] = str(e)
                return result
        
        # Execute standard actions
        for action_name, action_value in rule.actions.items():
            result["actions"].append({
                "action": action_name,
                "value": action_value
            })
        
        result["executed"] = True
        return result
    
    def get_applicable_rules(self, context: Dict[str, Any]) -> List[Rule]:
        """Get rules applicable to current context."""
        context_type = context.get("type", "general")
        applicable = []
        
        for rule in self.rules.values():
            if rule.enabled:
                # Match by type
                if context_type == "any" or rule.rule_type == context_type:
                    applicable.append(rule)
        
        return applicable
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            return True
        return False
    
    def _log_execution(self, rule: Rule, result: Dict[str, Any]):
        """Log rule execution"""
        log_entry = {
            "rule_name": rule.name,
            "rule_type": rule.rule_type,
            "executed": result.get("executed", False),
            "timestamp": __import__("time").time()
        }
        self.execution_log.append(log_entry)
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_log[-limit:]
    
    def get_rule_status(self) -> Dict[str, Any]:
        """Get comprehensive rule engine status"""
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "disabled_rules": sum(1 for r in self.rules.values() if not r.enabled),
            "execution_log_size": len(self.execution_log),
            "rule_types": list(set(r.rule_type for r in self.rules.values())),
            "rules_by_type": {
                rule_type: [r.name for r in self.rules.values() if r.rule_type == rule_type]
                for rule_type in set(r.rule_type for r in self.rules.values())
            }
        }
    
    def reload_configs(self) -> bool:
        """Reload all rule configurations from JSON"""
        try:
            self.config_loader.clear_cache()
            self.configs = self.config_loader.load_all_system_configs()
            self.rules.clear()
            self.rule_priorities.clear()
            self._initialize_rules()
            return True
        except Exception as e:
            print(f"Error reloading configs: {e}")
            return False
