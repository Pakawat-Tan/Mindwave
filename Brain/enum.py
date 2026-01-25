from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, TypedDict, Union 


# ==================== System Enums ====================

class SystemState(Enum):
    """System operational states"""
    IDLE = "idle"
    RUNNING = "running"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class SystemPriority(Enum):
    """Priority levels for system operations"""
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1


class RuntimeMode(Enum):
    """Runtime operational modes"""
    NORMAL = "normal"
    FAST = "fast"
    EFFICIENT = "efficient"
    DEBUG = "debug"


# ==================== Safety Enums ====================

class SafetyLevel(Enum):
    """Safety operational levels"""
    PERMISSIVE = 1
    NORMAL = 2
    CAUTIOUS = 3
    STRICT = 4
    LOCKDOWN = 5


# ==================== TypedDicts ====================

class NodeSchema(TypedDict):
    layer: int
    role: Literal["input", "hidden", "output"]
    head: Optional[str]
    activation: Optional[str]
    value: float | None
    gradient: float | None
    usage: float


class EvolutionContext(TypedDict):
    loss: float
    loss_trend: float
    num_nodes: int
    num_connections: int
    usage: Dict[str, float]
    model_type: str

