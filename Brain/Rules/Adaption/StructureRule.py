"""
StructureRule.py
Implements structural adaptation rules
"""

from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class StructureRule:
    """Manages structural adaptation rules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize structure rule."""
        self.config_loader = ConfigLoader()
        self.structural_changes: List[Dict[str, Any]] = []
        self.max_change_rate = 0.05
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load structure rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("AdaptionRule", "Adaption")
            
            if not config:
                return False
            
            structural = config.get("structural_adaptation", {})
            self.max_change_rate = structural.get("max_structural_change_rate", 0.05)
            
            return True
        except Exception as e:
            print(f"Error loading StructureRule from JSON: {e}")
            return False
    
    def propose_structural_change(self, change_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a structural change.
        
        Parameters
        ----------
        change_type : str
            Type of structural change
        params : Dict[str, Any]
            Parameters for the change
            
        Returns
        -------
        Dict[str, Any]
            Proposed change
        """
        proposal = {
            "change_type": change_type,
            "parameters": params,
            "status": "proposed"
        }
        
        return proposal
    
    def validate_structural_change(self, change: Dict[str, Any]) -> bool:
        """Validate a structural change.
        
        Parameters
        ----------
        change : Dict[str, Any]
            The change to validate
            
        Returns
        -------
        bool
            True if change is valid
        """
        # Check if change rate is within limits
        if len(self.structural_changes) > 0:
            recent_changes = len([c for c in self.structural_changes[-10:]])
            if recent_changes / 10 > self.max_change_rate:
                return False
        
        return True
