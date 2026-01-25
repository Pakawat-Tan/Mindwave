"""
TopicRouter.py
Routes information to appropriate processing modules based on topic.
"""

from typing import Dict, Callable, Optional, List, Any, Tuple
from dataclasses import dataclass
import time


@dataclass
class RoutingRule:
    """A routing rule"""
    rule_id: str
    condition: str  # Condition to match
    destination: str  # Where to route
    priority: int = 1


class TopicRouter:
    """Routes information to appropriate modules based on topic."""
    
    def __init__(self):
        """Initialize topic router."""
        self.topic_handlers: Dict[str, Callable] = {}
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def register_handler(self, topic_id: str, handler: Callable) -> bool:
        """Register a handler for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
        handler : Callable
            Handler function
            
        Returns
        -------
        bool
            Success
        """
        self.topic_handlers[topic_id] = handler
        return True
    
    def unregister_handler(self, topic_id: str) -> bool:
        """Unregister a handler.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
            
        Returns
        -------
        bool
            Success
        """
        if topic_id in self.topic_handlers:
            del self.topic_handlers[topic_id]
            return True
        return False
    
    def route_information(self, data: Any, topic_id: str) -> Optional[Any]:
        """Route information to appropriate handler.
        
        Parameters
        ----------
        data : Any
            Data to route
        topic_id : str
            Topic ID
            
        Returns
        -------
        Optional[Any]
            Handler result or None
        """
        if topic_id in self.topic_handlers:
            try:
                handler = self.topic_handlers[topic_id]
                result = handler(data)
                
                # Record in history
                self.routing_history.append({
                    "timestamp": time.time(),
                    "topic_id": topic_id,
                    "data_type": type(data).__name__,
                    "success": True
                })
                
                if len(self.routing_history) > self.max_history:
                    self.routing_history.pop(0)
                
                return result
            except Exception as e:
                self.routing_history.append({
                    "timestamp": time.time(),
                    "topic_id": topic_id,
                    "data_type": type(data).__name__,
                    "success": False,
                    "error": str(e)
                })
                return None
        
        return None
    
    def add_routing_rule(self, rule_id: str, condition: str, 
                        destination: str, priority: int = 1) -> bool:
        """Add a routing rule.
        
        Parameters
        ----------
        rule_id : str
            Rule ID
        condition : str
            Condition to match
        destination : str
            Destination/handler
        priority : int
            Rule priority
            
        Returns
        -------
        bool
            Success
        """
        self.routing_rules[rule_id] = RoutingRule(
            rule_id=rule_id,
            condition=condition,
            destination=destination,
            priority=priority
        )
        return True
    
    def get_routing_rules(self) -> List[RoutingRule]:
        """Get all routing rules.
        
        Returns
        -------
        List[RoutingRule]
            Routing rules sorted by priority
        """
        rules = sorted(self.routing_rules.values(), key=lambda x: x.priority, reverse=True)
        return rules
    
    def get_handler(self, topic_id: str) -> Optional[Callable]:
        """Get handler for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
            
        Returns
        -------
        Optional[Callable]
            Handler or None
        """
        return self.topic_handlers.get(topic_id)
    
    def get_registered_topics(self) -> List[str]:
        """Get all registered topics.
        
        Returns
        -------
        List[str]
            Topic IDs
        """
        return list(self.topic_handlers.keys())
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics
        """
        successful = sum(1 for r in self.routing_history if r.get("success", False))
        failed = sum(1 for r in self.routing_history if not r.get("success", False))
        
        # Count by topic
        by_topic = {}
        for entry in self.routing_history:
            topic = entry.get("topic_id")
            if topic:
                by_topic[topic] = by_topic.get(topic, 0) + 1
        
        return {
            "total_routings": len(self.routing_history),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self.routing_history) if self.routing_history else 0.0,
            "by_topic": by_topic,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "registered_topics": len(self.topic_handlers),
            "routing_rules": len(self.routing_rules),
            "history_length": len(self.routing_history),
            "statistics": self.get_routing_statistics(),
        }
    
    def get_handler(self, topic_id):
        """Get the handler for a topic."""
        pass
