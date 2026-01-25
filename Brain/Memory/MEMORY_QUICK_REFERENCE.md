# Memory System Quick Reference

## Quick Start

```python
from Brain.Memory import MemoryEngine

# Create memory engine
memory = MemoryEngine()

# Store memory
mem_id = memory.store_memory("Important fact", "HIGH")

# Retrieve memory
content = memory.retrieve_memory(mem_id)

# Set attention
memory.set_attention("VISUAL", "red_button", intensity=0.9)

# Add topics
memory.add_topic("learning")
memory.relate_topics("learning", "memory")

# Consolidate
result = memory.consolidate_memory()
print(f"Consolidated {result['promoted_items']} items")

# Get status
status = memory.get_engine_status()
```

## Memory Tiers

### ShortTerm (Immediate)
```python
# WorkingMemory
wm = WorkingMemory(max_items=50)
mem_id = wm.store("content", MemoryPriority.HIGH)
item = wm.retrieve(mem_id)
candidates = wm.get_consolidation_candidates()

# AttentionMap
am = AttentionMap()
am.set_focus("VISUAL", "target", intensity=0.8)
primary = am.get_primary_focus()
```

### MiddleTerm (Context)
```python
# ContextBuffer
cb = ContextBuffer(max_frames=100, retention_time_seconds=300)
cb.start_new_frame()
cb.add_event({"type": "user_input"})
recent = cb.get_recent_context(5)

# TopicContext
tc = TopicContext(max_topics=50)
tc.create_topic("topic_name")
tc.relate_topics("topic1", "topic2")
tc.update_relevance("topic1", 0.9)
```

### LongTerm (Persistent)
```python
# ExperienceStore
es = ExperienceStore(max_experiences=1000)
es.store_experience("exp_001", context, actions)
es.record_outcome("exp_001", outcome, reward=1.0, success=True)
learning = es.learn_from_experiences()

# KnowledgeStore
ks = KnowledgeStore()
ks.add_fact("fact_001", "water", "is_liquid", "true")
ks.add_concept("water", "H2O", "clear liquid")
facts = ks.query_facts(subject="water")

# WeightArchive
wa = WeightArchive(max_snapshots=50)
wa.create_snapshot("snap_001", weights)
wa.rollback_to_snapshot("snap_001")
```

### Emotions (State & Personality)
```python
# EmotionProfile - personality traits
profile = EmotionProfile()
profile.set_trait("joy", 0.8)
profile.set_trait("fear", 0.2)

# EmotionState - current emotions
state = EmotionState()
state.update_emotion("joy", 0.9, source="success")
dominant = state.get_dominant_emotion()
valence, arousal = state.get_valence_arousal()
state.apply_emotion_decay(0.1)

# EmotionWeight - decision biasing
weight = EmotionWeight()
emotional_state = {"joy": 0.8, "fear": 0.1}
biased = weight.apply_emotional_bias(0.5, emotional_state)
confidence = weight.modulate_confidence(0.7, emotional_state)

# EmotionHistory - trend analysis
history = EmotionHistory(max_history=1000)
history.record_emotion_state({"joy": 0.5, "fear": 0.2})
trend = history.get_emotional_trend("joy")
stability = history.get_emotional_stability()

# EmotionEncoder - embeddings
encoder = EmotionEncoder(vector_dim=32)
vector = encoder.encode_emotion({"joy": 0.8, "fear": 0.1})
decoded = encoder.decode_vector(vector)
similarity = encoder.compute_similarity(state1, state2)
```

### Topics (Tracking & Routing)
```python
# TopicProfile - expertise
profile = TopicProfile()
profile.set_proficiency("python", 0.8)
profile.set_interest("python", 0.9)
best = profile.get_best_topics(5)

# TopicState - current topic
state = TopicState()
state.set_topic("programming", confidence=0.95)

# TopicWeight - importance
weight = TopicWeight()
weight.set_weight("machine_learning", 0.9, priority=8)

# TopicHistory - sequence tracking
history = TopicHistory(max_history=500)
history.record_topic_change("new_topic", "old_topic")
sequence = history.get_topic_sequence(10)
frequent = history.get_most_frequent_topics(5)
transitions = history.get_topic_transitions()

# TopicEmbedding - semantic similarity
embedding = TopicEmbedding(embedding_dim=64)
vec = embedding.embed_topic("topic_id")
similarity = embedding.compute_similarity("topic1", "topic2")
similar = embedding.find_similar_topics("topic_id", top_k=5)

# TopicRouter - handling
router = TopicRouter()
router.register_handler("math", lambda x: process_math(x))
result = router.route_information(data, "math")
router.add_routing_rule("rule_1", "condition", "destination")
```

## Key Constants

| Component | Capacity | Retention |
|-----------|----------|-----------|
| WorkingMemory | 50 items | Decay 0.01/min |
| AttentionMap | 5 types | Real-time |
| ContextBuffer | 100 frames | 5 minutes |
| TopicContext | 50 topics | LRU evict |
| ExperienceStore | 1000 items | LRU evict |
| KnowledgeStore | 5000 facts | LRU evict |
| EmotionHistory | 1000 entries | Deque |
| TopicHistory | 500 entries | Deque |
| WeightArchive | 50 snapshots | Oldest first |

## Enums

```python
# Memory Priority
from Brain.Memory import MemoryPriority
- CRITICAL (4)
- HIGH (3)
- NORMAL (2)
- LOW (1)

# Attention Types
from Brain.Memory import AttentionType
- VISUAL
- AUDITORY
- PROPRIOCEPTIVE
- SEMANTIC
- EMOTIONAL
```

## Status Methods

Every component has a `get_status()` method:

```python
status = working_memory.get_status()
# Returns: {
#   "current_items": int,
#   "max_items": int,
#   "utilization_percent": float,
#   ...
# }
```

## Consolidation Workflow

```
WorkingMemory (Immediate)
      ↓
  (decay > 0.6)
      ↓
ContextBuffer (Context)
      ↓
  (prune old frames)
      ↓
LongTerm Memory
  - ExperienceStore
  - KnowledgeStore
  - WeightArchive
```

## Integration Example

```python
from Brain.Memory import MemoryEngine
from Brain.Neural.BrainController import BrainController

# Create components
memory = MemoryEngine()
brain = BrainController()

# Processing cycle
def process_cycle():
    # Receive input
    input_data = receive_input()
    
    # Store in memory
    mem_id = memory.store_memory(input_data["content"], "NORMAL")
    
    # Set attention
    memory.set_attention("VISUAL", input_data.get("target", ""))
    
    # Get context
    context = memory.context_buffer.get_recent_context(5)
    
    # Process with brain
    output = brain.process_context(context)
    
    # Consolidate
    memory.consolidate_memory()
    
    return output
```

## Vector Spaces

### Emotion Space (32-dim)
- Each emotion has a basis vector
- Linear combination creates state vectors
- Normalized by intensity weights

### Topic Space (64-dim)
- Random initialization (seed=42)
- Feature-influenced embeddings
- Cosine similarity for comparison

## Performance Considerations

1. **Decay**: Applied during retrieval, O(1)
2. **Consolidation**: O(n) where n = candidates
3. **Similarity**: O(m) where m = stored items
4. **Eviction**: O(log n) priority-based
5. **Vector ops**: O(d) where d = dimension

## Common Patterns

### Storing Episodic Memory
```python
mem_id = memory.store_memory(
    content=experience_data,
    priority="HIGH",
    context={"topic": "learning", "emotion": "joy"}
)
```

### Tracking Emotional State
```python
emotions = {"joy": 0.8, "interest": 0.7}
memory.emotion_state.update_emotion("joy", 0.8, source="success")
valence, arousal = memory.emotion_state.get_valence_arousal()
```

### Topic-Specific Processing
```python
# Register handler
memory.topic_router.register_handler("math", math_processor)

# Route information
memory.topic_router.route_information(equation, "math")
```

### Learning from Experience
```python
exp_store = memory.experience_store
learning = exp_store.learn_from_experiences()
if learning["success_rate"] > 0.8:
    apply_reinforcement()
```
