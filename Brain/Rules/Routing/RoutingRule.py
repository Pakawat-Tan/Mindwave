"""
RoutingRule.py
Implements topic-based and priority-based routing logic
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class DataType(Enum):
    """Types of data that can be routed"""
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    LEARNING = "learning"
    MEMORY = "memory"
    INTERNAL = "internal"


@dataclass
class RoutingRule:
    """Represents a routing rule for data flow"""
    name: str
    data_type: DataType
    route_to: List[str] = field(default_factory=list)
    priority: int = 1
    buffer_size: int = 100
    enabled: bool = True


class RoutingEngine:
    """Manages routing of information through the system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize routing engine.
        
        Parameters
        ----------
        config_path : str, optional
            Path to RoutingRule.json configuration file
        """
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.route_buffers: Dict[str, List[Any]] = {}
        self.config_loader = ConfigLoader()
        self.routing_stats: Dict[str, int] = {}
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load routing rules from JSON configuration.
        
        Returns
        -------
        bool
            True if successfully loaded
        """
        try:
            config = self.config_loader.load_config("RoutingRule", "Routing")
            
            if not config:
                print("Warning: RoutingRule configuration not found")
                return False
            
            routing_rules = config.get("routing_rules", {})
            
            for data_type_name, route_config in routing_rules.items():
                try:
                    data_type = DataType(data_type_name)
                    
                    rule = RoutingRule(
                        name=f"route_{data_type_name}",
                        data_type=data_type,
                        route_to=route_config.get("route_to", []),
                        priority=route_config.get("priority", 1),
                        buffer_size=route_config.get("buffer_size", 100),
                        enabled=route_config.get("enabled", True)
                    )
                    
                    self.routing_rules[rule.name] = rule
                    self.route_buffers[rule.name] = []
                    self.routing_stats[rule.name] = 0
                except ValueError:
                    print(f"Warning: Unknown data type '{data_type_name}'")
            
            return True
        except Exception as e:
            print(f"Error loading RoutingRule from JSON: {e}")
            return False
    
    def route_data(self, data_type: DataType, data: Any) -> Dict[str, List[Any]]:
        """Route data based on rules.
        
        Parameters
        ----------
        data_type : DataType
            Type of data being routed
        data : Any
            The data to route
            
        Returns
        -------
        Dict[str, List[Any]]
            Mapping of destinations to routed data
        """
        rule_name = f"route_{data_type.value}"
        
        if rule_name not in self.routing_rules:
            return {}
        
        rule = self.routing_rules[rule_name]
        
        if not rule.enabled:
            return {}
        
        # Add to buffer
        if len(self.route_buffers[rule_name]) < rule.buffer_size:
            self.route_buffers[rule_name].append(data)
            self.routing_stats[rule_name] += 1
        
        # Route to destinations
        routed_data = {}
        for destination in rule.route_to:
            routed_data[destination] = data
        
        return routed_data
    
    def get_route_priority(self, data_type: DataType) -> int:
        """Get priority for a data type's routing.
        
        Parameters
        ----------
        data_type : DataType
            Type of data
            
        Returns
        -------
        int
            Priority level (higher = more important)
        """
        rule_name = f"route_{data_type.value}"
        
        if rule_name in self.routing_rules:
            return self.routing_rules[rule_name].priority
        
        return 0
    
    def get_buffer_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all route buffers.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Buffer status for each route
        """
        status = {}
        
        for rule_name, buffer in self.route_buffers.items():
            rule = self.routing_rules[rule_name]
            status[rule_name] = {
                "current_size": len(buffer),
                "max_size": rule.buffer_size,
                "utilization_percent": (len(buffer) / rule.buffer_size * 100) if rule.buffer_size > 0 else 0,
                "items_routed": self.routing_stats[rule_name],
                "enabled": rule.enabled
            }
        
        return status
    
    def clear_buffer(self, rule_name: str) -> bool:
        """Clear a routing buffer.
        
        Parameters
        ----------
        rule_name : str
            Name of the routing rule
            
        Returns
        -------
        bool
            True if buffer was cleared
        """
        if rule_name in self.route_buffers:
            self.route_buffers[rule_name].clear()
            return True
        return False
    
    def get_buffered_data(self, rule_name: str) -> List[Any]:
        """Get all data in a buffer.
        
        Parameters
        ----------
        rule_name : str
            Name of the routing rule
            
        Returns
        -------
        List[Any]
            Buffered data
        """
        return self.route_buffers.get(rule_name, [])
    
    def drain_buffer(self, rule_name: str) -> List[Any]:
        """Get and clear a buffer.
        
        Parameters
        ----------
        rule_name : str
            Name of the routing rule
            
        Returns
        -------
        List[Any]
            Buffered data, buffer is cleared
        """
        data = self.get_buffered_data(rule_name)
        self.clear_buffer(rule_name)
        return data
