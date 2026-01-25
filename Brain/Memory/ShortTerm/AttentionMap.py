"""
AttentionMap.py
Tracks attention and focus across different inputs
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class AttentionType(Enum):
    """Types of attention"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    PROPRIOCEPTIVE = "proprioceptive"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"


@dataclass
class AttentionFocus:
    """Represents a focus of attention"""
    attention_type: AttentionType
    target: str
    intensity: float = 0.5  # 0.0-1.0
    duration_ms: int = 0
    timestamp: float = field(default_factory=time.time)


class AttentionMap:
    """Maps and manages attention across different modalities."""
    
    def __init__(self):
        """Initialize attention map."""
        self.foci: Dict[AttentionType, AttentionFocus] = {}
        self.history: List[AttentionFocus] = []
        self.primary_focus: Optional[AttentionType] = None
        self.attention_weights: Dict[AttentionType, float] = {
            at: 0.2 for at in AttentionType
        }
    
    def set_focus(self, attention_type: AttentionType, target: str, intensity: float = 0.7) -> bool:
        """Set focus for an attention type.
        
        Parameters
        ----------
        attention_type : AttentionType
            Type of attention
        target : str
            Target of attention
        intensity : float
            Intensity of attention (0.0-1.0)
            
        Returns
        -------
        bool
            True if focus was set
        """
        focus = AttentionFocus(
            attention_type=attention_type,
            target=target,
            intensity=max(0.0, min(1.0, intensity))
        )
        
        self.foci[attention_type] = focus
        self.history.append(focus)
        
        # Update primary focus if high intensity
        if intensity > 0.7:
            self.primary_focus = attention_type
        
        return True
    
    def get_focus(self, attention_type: AttentionType) -> Optional[AttentionFocus]:
        """Get current focus for attention type.
        
        Parameters
        ----------
        attention_type : AttentionType
            Type of attention
            
        Returns
        -------
        Optional[AttentionFocus]
            Current focus, or None
        """
        return self.foci.get(attention_type)
    
    def get_primary_focus(self) -> Optional[AttentionFocus]:
        """Get the primary focus of attention.
        
        Returns
        -------
        Optional[AttentionFocus]
            Primary focus, or None
        """
        if self.primary_focus and self.primary_focus in self.foci:
            return self.foci[self.primary_focus]
        return None
    
    def update_intensity(self, attention_type: AttentionType, intensity: float) -> bool:
        """Update intensity of attention.
        
        Parameters
        ----------
        attention_type : AttentionType
            Type of attention
        intensity : float
            New intensity (0.0-1.0)
            
        Returns
        -------
        bool
            True if updated
        """
        if attention_type in self.foci:
            self.foci[attention_type].intensity = max(0.0, min(1.0, intensity))
            return True
        return False
    
    def shift_focus(self, from_type: AttentionType, to_type: AttentionType) -> bool:
        """Shift focus from one type to another.
        
        Parameters
        ----------
        from_type : AttentionType
            Source attention type
        to_type : AttentionType
            Target attention type
            
        Returns
        -------
        bool
            True if shift was successful
        """
        if from_type not in self.foci:
            return False
        
        focus = self.foci.pop(from_type)
        focus.attention_type = to_type
        self.foci[to_type] = focus
        
        return True
    
    def get_active_attention_types(self) -> List[AttentionType]:
        """Get all active attention types.
        
        Returns
        -------
        List[AttentionType]
            Active attention types sorted by intensity
        """
        active = list(self.foci.keys())
        return sorted(active, key=lambda x: self.foci[x].intensity, reverse=True)
    
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get distribution of attention across types.
        
        Returns
        -------
        Dict[str, float]
            Normalized attention distribution
        """
        if not self.foci:
            return {}
        
        total_intensity = sum(f.intensity for f in self.foci.values())
        
        if total_intensity == 0:
            return {}
        
        return {
            at.value: focus.intensity / total_intensity
            for at, focus in self.foci.items()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get attention map status.
        
        Returns
        -------
        Dict[str, Any]
            Current status
        """
        return {
            "active_foci": len(self.foci),
            "primary_focus": self.primary_focus.value if self.primary_focus else None,
            "attention_types": self.get_active_attention_types(),
            "distribution": self.get_attention_distribution(),
            "total_history": len(self.history)
        }
    
    def get_high_attention_items(self, threshold=0.7):
        """Get items with high attention weights."""
        pass
    
    def update_saliency(self, item_id, importance):
        """Update saliency for an item."""
        pass
    
    def get_attention_distribution(self):
        """Get overall attention distribution."""
        pass