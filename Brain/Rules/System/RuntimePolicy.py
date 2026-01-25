"""
RuntimePolicy.py
Runtime operational policies and resource management
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import psutil
import sys
from pathlib import Path
from Brain.enum import RuntimeMode

# Add parent directory to path for imports
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


@dataclass
class ResourceLimits:
    """Resource usage limits"""
    memory_max_percent: float = 80.0
    memory_warning_percent: float = 70.0
    cpu_max_percent: float = 90.0
    cpu_warning_percent: float = 75.0
    cycle_time_ms: int = 1000


@dataclass
class TimeoutConfig:
    """Timeout configurations"""
    operation_timeout_ms: int = 5000
    learning_timeout_ms: int = 30000
    inference_timeout_ms: int = 1000
    consolidation_timeout_ms: int = 60000


class RuntimePolicy:
    """Manages runtime operational policies and resource constraints."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize runtime policy.
        
        Parameters
        ----------
        config_path : str, optional
            Path to RuntimePolicy.json configuration file
        """
        self.resource_limits = ResourceLimits()
        self.timeout_config = TimeoutConfig()
        self.runtime_mode = RuntimeMode.NORMAL
        self.config_loader = ConfigLoader()
        self.operation_timers: Dict[str, float] = {}
        self.performance_tuning: Dict[str, Any] = {}
        self.monitoring: Dict[str, Any] = {}
        self.recovery_policies: Dict[str, Any] = {}
        
        if config_path:
            self.load_from_json(config_path)
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load runtime policy from JSON configuration.
        
        Parameters
        ----------
        config_path : str, optional
            Path to JSON file. If not provided, loads from default location
            
        Returns
        -------
        bool
            True if successfully loaded
        """
        try:
            config = self.config_loader.load_config("RuntimePolicy", "System")
            
            if not config:
                print("Warning: RuntimePolicy configuration not found")
                return False
            
            # Load resource management settings
            rm = config.get("resource_management", {})
            self.resource_limits.memory_max_percent = rm.get("memory", {}).get("max_usage_percent", 80.0)
            self.resource_limits.memory_warning_percent = rm.get("memory", {}).get("warning_threshold", 70.0)
            self.resource_limits.cpu_max_percent = rm.get("cpu", {}).get("max_usage_percent", 90.0)
            self.resource_limits.cpu_warning_percent = rm.get("cpu", {}).get("warning_threshold", 75.0)
            self.resource_limits.cycle_time_ms = rm.get("cycle_time_ms", 1000)
            
            # Load timeout policies
            timeouts = config.get("timeout_policies", {})
            self.timeout_config.operation_timeout_ms = timeouts.get("operation_timeout_ms", 5000)
            self.timeout_config.learning_timeout_ms = timeouts.get("learning_timeout_ms", 30000)
            self.timeout_config.inference_timeout_ms = timeouts.get("inference_timeout_ms", 1000)
            self.timeout_config.consolidation_timeout_ms = timeouts.get("consolidation_timeout_ms", 60000)
            
            # Load performance tuning
            self.performance_tuning = config.get("performance_tuning", {})
            
            # Load monitoring settings
            self.monitoring = config.get("monitoring", {})
            
            # Load recovery policies
            self.recovery_policies = config.get("recovery", {})
            
            return True
        except Exception as e:
            print(f"Error loading RuntimePolicy from JSON: {e}")
            return False
    
    def set_runtime_mode(self, mode: RuntimeMode) -> bool:
        """Set the current runtime mode.
        
        Parameters
        ----------
        mode : RuntimeMode
            The runtime mode to set
            
        Returns
        -------
        bool
            True if mode was set successfully
        """
        if mode not in RuntimeMode:
            return False
        
        self.runtime_mode = mode
        
        # Adjust limits based on mode
        if mode == RuntimeMode.FAST:
            # Reduce timeouts for faster operation
            self.timeout_config.operation_timeout_ms = int(self.timeout_config.operation_timeout_ms * 0.5)
        elif mode == RuntimeMode.EFFICIENT:
            # Reduce resource usage
            self.resource_limits.memory_max_percent = min(70.0, self.resource_limits.memory_max_percent)
            self.resource_limits.cpu_max_percent = min(80.0, self.resource_limits.cpu_max_percent)
        elif mode == RuntimeMode.DEBUG:
            # Increase limits for debugging
            self.timeout_config.operation_timeout_ms = int(self.timeout_config.operation_timeout_ms * 2)
        
        return True
    
    def check_resource_usage(self) -> Dict[str, Any]:
        """Check current system resource usage.
        
        Returns
        -------
        Dict[str, Any]
            Resource usage metrics
        """
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            "memory_percent": memory_percent,
            "cpu_percent": cpu_percent,
            "memory_warning": memory_percent >= self.resource_limits.memory_warning_percent,
            "memory_critical": memory_percent >= self.resource_limits.memory_max_percent,
            "cpu_warning": cpu_percent >= self.resource_limits.cpu_warning_percent,
            "cpu_critical": cpu_percent >= self.resource_limits.cpu_max_percent
        }
    
    def enforce_resource_limits(self) -> bool:
        """Enforce resource limits.
        
        Returns
        -------
        bool
            True if all resources are within limits
        """
        usage = self.check_resource_usage()
        
        if usage["memory_critical"] or usage["cpu_critical"]:
            # Trigger recovery if critical
            self.trigger_recovery()
            return False
        
        return not (usage["memory_warning"] or usage["cpu_warning"])
    
    def start_operation(self, operation_name: str, timeout_ms: Optional[int] = None) -> str:
        """Start tracking an operation timeout.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation
        timeout_ms : int, optional
            Custom timeout in milliseconds
            
        Returns
        -------
        str
            Operation ID for tracking
        """
        import time
        import uuid
        
        op_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        self.operation_timers[op_id] = time.time()
        
        return op_id
    
    def check_operation_timeout(self, operation_id: str, timeout_ms: int) -> bool:
        """Check if an operation has exceeded timeout.
        
        Parameters
        ----------
        operation_id : str
            Operation ID to check
        timeout_ms : int
            Timeout in milliseconds
            
        Returns
        -------
        bool
            True if operation has timed out
        """
        import time
        
        if operation_id not in self.operation_timers:
            return False
        
        elapsed_ms = (time.time() - self.operation_timers[operation_id]) * 1000
        return elapsed_ms >= timeout_ms
    
    def handle_timeout(self, operation_id: str, operation_type: str = "general") -> Dict[str, Any]:
        """Handle a timeout event.
        
        Parameters
        ----------
        operation_id : str
            Operation ID that timed out
        operation_type : str
            Type of operation (general, learning, inference, consolidation)
            
        Returns
        -------
        Dict[str, Any]
            Recovery action to take
        """
        if operation_id in self.operation_timers:
            del self.operation_timers[operation_id]
        
        recovery_action = self.recovery_policies.get(operation_type, {})
        
        return {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "action": recovery_action.get("action", "retry"),
            "max_retries": recovery_action.get("max_retries", 3),
            "backoff_ms": recovery_action.get("backoff_ms", 100)
        }
    
    def trigger_recovery(self) -> Dict[str, Any]:
        """Trigger system recovery procedures."""
        recovery = self.recovery_policies.get("resource_exhaustion", {})
        
        return {
            "recovery_type": "resource_exhaustion",
            "actions": recovery.get("actions", ["gc", "clear_cache", "reduce_buffers"]),
            "priority": recovery.get("priority", "high")
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status.
        
        Returns
        -------
        Dict[str, Any]
            Current resource status and limits
        """
        usage = self.check_resource_usage()
        
        return {
            "current_usage": usage,
            "limits": {
                "memory_max_percent": self.resource_limits.memory_max_percent,
                "cpu_max_percent": self.resource_limits.cpu_max_percent,
                "cycle_time_ms": self.resource_limits.cycle_time_ms
            },
            "runtime_mode": self.runtime_mode.name,
            "active_operations": len(self.operation_timers),
            "status": "healthy" if self.enforce_resource_limits() else "degraded"
        }
