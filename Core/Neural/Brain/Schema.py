"""
TypedDict schemas สำหรับ BrainStructure — strict typing
"""

from __future__ import annotations
from typing import Dict, List, Literal, Optional, TypedDict


NodeRole = Literal["input", "hidden", "output"]
MDNHead  = Literal["mdn_pi", "mdn_mu", "mdn_sigma"]

class NodeSchema(TypedDict):
    layer:      int
    role:       NodeRole
    head:       Optional[MDNHead]
    activation: Optional[str]
    value:      Optional[float]
    gradient:   Optional[float]
    usage:      float


class ConnectionSchema(TypedDict):
    source:      str
    destination: str
    enabled:     bool


class EvolutionContext(TypedDict):
    loss:             float
    loss_trend:       float       # loss - prev_loss (ลบ = ดีขึ้น)
    num_nodes:        int
    num_connections:  int
    usage:            Dict[str, float]
    model_type:       str


class StructureSnapshot(TypedDict):
    """snapshot ก่อน evolve — ใช้สำหรับ rollback"""
    nodes:       Dict[str, NodeSchema]
    connections: Dict[str, ConnectionSchema]
    weights:     Dict[str, float]
    biases:      Dict[str, float]