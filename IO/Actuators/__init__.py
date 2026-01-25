"""
__init__.py
Actuators module initialization
"""

from .OutputRouter import OutputRouter
from . import Motion

__all__ = ['OutputRouter', 'Motion']
