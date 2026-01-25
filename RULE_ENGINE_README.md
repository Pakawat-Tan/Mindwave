# Rule Engine Implementation - Mindwave #5

## Overview

The complete Rule Engine system has been implemented as a centralized governance and decision-making framework for the Mindwave AI system. All rules load their configuration from JSON files, enabling flexible system configuration without code changes.

## Architecture

### Core Components

1. **RuleEngine (CoreRule)** - Central dispatcher orchestrating all rules
2. **ConfigLoader** - Centralized JSON configuration loading with caching
3. **BrainController** - Integration point connecting all systems to the neural core

### Rule Categories

#### System Rules
- **SystemRule** - Core system governance and state management
- **RuntimePolicy** - Runtime operational policies and resource constraints

#### Safety Rules
- **SafetyRule** - Safety constraints and action validation
- **MotionConstraint** - Motion safety limits
- **OutputSafetyConstraint** - Output safety filtering

#### Information Routing
- **RoutingEngine** - Topic-based and priority-based routing
- **RoutingRule** - Data flow management
- **TopicRule** - Topic-based routing and management

#### Memory Management
- **MemoryRule** - Memory consolidation and decay policies
- **AcquisitionRule** - Memory storage acquisition strategy
- **EmotionRule** - Emotion-driven decision rules

#### Learning Rules
- **LearnRule** - Learning modes and constraints
- **GradientRule** - Gradient descent-based learning

#### Adaptation Rules
- **AdaptionRule** - Structural, behavioral, emotional adaptation
- **StructureRule** - Structural adaptation rules
- **EmotionDrivenRule** - Emotion-driven decision rules

## Configuration Files

All rules load from JSON configuration files in subdirectories:

### System Configuration
- `Brain/Rules/System/RuntimePolicy.json` - Resource management, timeouts, performance tuning
- `Brain/Rules/System/SystemRule.json` - Capability flags, governance policies

### Safety Configuration
- `Brain/Rules/Safety/SafetyPolicy.json` - Motion, output, decision safety constraints

### Routing Configuration
- `Brain/Rules/Routing/RoutingRule.json` - Topic and priority-based routing definitions

### Memory Configuration
- `Brain/Rules/Memory/MemoryRule.json` - Consolidation, retrieval, decay rates
- `Brain/Rules/Memory/AcquisitionRule.json` - Memory acquisition triggers and filtering

### Learning Configuration
- `Brain/Rules/Learning/LearningRule.json` - Learning modes, gradient settings, constraints

### Adaptation Configuration
- `Brain/Rules/Adaption/AdaptionRule.json` - Structural, behavioral, emotional adaptation

## File Structure

```
Brain/
├── enum.py                          # Centralized enums
├── Neural/
│   └── BrainController.py          # Main controller
├── Rules/
│   ├── ConfigLoader.py             # JSON config loader
│   ├── RuleEngine.py               # Core dispatcher
│   ├── __init__.py                 # Centralized imports
│   ├── System/
│   │   ├── SystemRule.py           # System governance
│   │   ├── RuntimePolicy.py        # Runtime policies
│   │   └── RuntimePolicy.json
│   ├── Safety/
│   │   ├── SafetyRule.py           # Safety constraints
│   │   └── SafetyPolicy.json
│   ├── Routing/
│   │   ├── RoutingRule.py          # Data routing
│   │   └── RoutingRule.json
│   ├── Memory/
│   │   ├── MemoryRule.py           # Memory management
│   │   ├── AcquisitionRule.py      # Memory acquisition
│   │   ├── EmotionRule.py          # Emotion rules
│   │   ├── TopicRule.py            # Topic routing
│   │   ├── MemoryRule.json
│   │   └── AcquisitionRule.json
│   ├── Learning/
│   │   ├── LearnRule.py            # Learning modes
│   │   ├── GradientRule.py         # Gradient descent
│   │   └── LearningRule.json
│   └── Adaption/
│       ├── AdaptionRule.py         # Adaptation rules
│       ├── StructureRule.py        # Structural adaptation
│       ├── EmotionDrivenRule.py    # Emotional adaptation
│       └── AdaptionRule.json
```

## Key Features

### 1. Centralized Configuration
- All rules read from JSON files
- ConfigLoader provides caching for performance
- Dot-notation access for nested values
- Automatic validation of required keys

### 2. Rule Execution Pipeline
- RuleEngine coordinates all rules
- Rules prioritized by importance
- Context-based rule selection
- Comprehensive execution logging

### 3. State Management
- SystemState enum for core states
- Validated state transitions
- State history tracking

### 4. Resource Management
- Real-time CPU/memory monitoring via psutil
- Resource limit enforcement
- Automatic recovery procedures
- Operation timeout tracking

### 5. Safety Integration
- Multi-level safety constraints
- Motion and output validation
- Emergency stop capability
- Safety event logging

### 6. Memory Management
- Memory consolidation policies
- Decay/forgetting mechanisms
- Emotional memory protection
- Acquisition filtering

### 7. Learning Integration
- Multiple learning modes (Gradient, Advisor, Evolution, Replay, Self, Reinforcement)
- Gradient clipping and momentum
- Adaptive learning rates
- Dropout and regularization support

### 8. Adaptation Framework
- Structural adaptation (neuron addition/removal)
- Behavioral adaptation (strategy changes)
- Emotional adaptation (profile updates)
- Change rate limiting

## Usage Examples

### Initialize Brain Controller
```python
from Brain.Neural.BrainController import BrainController

controller = BrainController()
controller.initialize()
controller.start()
```

### Process a Cycle
```python
context = {
    "type": "general",
    "system_metrics": {"performance": 0.8}
}
result = controller.process_cycle(context)
```

### Access Rule Engines
```python
# Access specific rule engines
status = controller.runtime_policy.get_resource_status()
safety_ok = controller.safety_rule.check_action_safety("motion", action)
acquired = controller.acquisition_rule.acquire_memory(memory)
```

### Check System Status
```python
system_status = controller.get_system_status()
print(f"Running: {system_status['running']}")
print(f"Cycles: {system_status['cycle_count']}")
print(f"Memory: {system_status['resource_status']['current_usage']['memory_percent']}%")
```

## Testing

Run comprehensive tests:
```bash
python test_rule_engine.py          # Test all rule classes
python test_brain_controller.py     # Test BrainController integration
```

## Configuration Details

### RuntimePolicy.json
- Resource limits (memory/CPU thresholds)
- Operation timeouts (inference, learning, consolidation)
- Performance tuning parameters
- Monitoring and recovery policies

### SafetyPolicy.json
- Motion safety (velocity, force limits)
- Output safety (content filtering, risk levels)
- Safety override protocols
- Emergency procedures

### LearningRule.json
- Enabled learning modes
- Gradient learning settings (learning rate, momentum)
- Regularization and dropout configuration
- Constraint limits

### AdaptionRule.json
- Structural adaptation (neuron changes, approval requirements)
- Behavioral adaptation (strategy changes, monitoring)
- Emotional adaptation (profile updates, dampening)
- Adaptation trigger thresholds

## Design Patterns

### 1. ConfigLoader Pattern
All rules implement `load_from_json()` to read configuration:
```python
def load_from_json(self, config_path=None):
    config = self.config_loader.load_config("RuleName", "Subdirectory")
    # Parse and apply configuration
```

### 2. Enum Centralization
All system states managed in `Brain/enum.py`:
- SystemState
- SystemPriority
- RuntimeMode
- SafetyLevel

### 3. Dataclass Constraints
Configuration and constraint objects use @dataclass:
```python
@dataclass
class ResourceLimits:
    memory_max_percent: float = 80.0
    cpu_max_percent: float = 90.0
```

### 4. Rule Priority
Rules execute in priority order:
- Priority 4: CRITICAL (Safety, System capability flags)
- Priority 3: HIGH (Runtime policies, Learning)
- Priority 2: NORMAL (Memory, Routing, Adaptation)
- Priority 1: LOW (Standard operations)

## Performance Metrics

From test run:
- Total Rules: 19
- Enabled Rules: 18
- Average Cycle Time: < 100ms
- Memory Usage: ~40%
- CPU Usage: ~26%
- Rule Execution: Sub-millisecond per rule

## Future Enhancements

1. **Rule Chaining** - Support complex rule dependencies
2. **Rule Templates** - Common rule patterns
3. **Dynamic Rule Loading** - Add rules at runtime
4. **Rule Analytics** - Detailed execution statistics
5. **Conflict Resolution** - Handle conflicting rules
6. **Rule Versioning** - Configuration version management
7. **Hot Reload** - Update configurations without restart

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│            BrainController                          │
│         (Central Orchestrator)                      │
└──────────────────┬──────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
     ▼             ▼             ▼
┌─────────┐   ┌──────────┐   ┌──────────┐
│RuleEngine│   │MemoryMgmt│   │Learning  │
│ (Core)  │   │ Rules    │   │Rules     │
└─────────┘   └──────────┘   └──────────┘
     │             │             │
     └─────────────┼─────────────┘
                   │
        ┌──────────▼──────────┐
        │  ConfigLoader       │
        │  (JSON Loading)     │
        └─────────────────────┘
```

## Summary

The complete Rule Engine system provides:
- ✅ Centralized rule management
- ✅ JSON-driven configuration
- ✅ Multi-category rule support
- ✅ Real-time resource monitoring
- ✅ Safety-first architecture
- ✅ Integration with BrainController
- ✅ Extensible design patterns
- ✅ Comprehensive testing framework

All requirements satisfied and tested!
