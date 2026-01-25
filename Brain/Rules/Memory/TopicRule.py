"""
TopicRule.py
Implements topic-based routing and management rules
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


@dataclass
class TopicConfig:
    """Configuration for topics"""
    name: str
    priority: int = 1
    max_items: int = 100
    enabled: bool = True


class TopicRule:
    """Manages topic-based routing and management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize topic rule."""
        self.topics: Dict[str, TopicConfig] = {}
        self.config_loader = ConfigLoader()
        self.topic_buffers: Dict[str, List[Any]] = {}
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load topic rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("MemoryRule", "Memory")
            
            if not config:
                return False
            
            # Initialize with default topics
            default_topics = ["vision", "audio", "learning", "memory", "internal"]
            
            for topic_name in default_topics:
                self.topics[topic_name] = TopicConfig(
                    name=topic_name,
                    priority=1,
                    max_items=100,
                    enabled=True
                )
                self.topic_buffers[topic_name] = []
            
            return True
        except Exception as e:
            print(f"Error loading TopicRule from JSON: {e}")
            return False
    
    def route_to_topic(self, topic: str, data: Any) -> bool:
        """Route data to a topic.
        
        Parameters
        ----------
        topic : str
            Topic name
        data : Any
            Data to route
            
        Returns
        -------
        bool
            True if successfully routed
        """
        if topic not in self.topic_buffers:
            return False
        
        buffer = self.topic_buffers[topic]
        topic_config = self.topics[topic]
        
        if len(buffer) < topic_config.max_items:
            buffer.append(data)
            return True
        
        return False
    
    def get_topic_data(self, topic: str) -> List[Any]:
        """Get data from a topic."""
        return self.topic_buffers.get(topic, [])
