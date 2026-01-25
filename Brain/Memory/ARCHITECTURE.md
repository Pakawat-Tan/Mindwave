# Memory System Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY ENGINE (Master)                      │
│                    Consolidation Coordinator                    │
└──────┬──────────────┬──────────────┬──────────────┬──────────────┘
       │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼
   ┌────────────┐  ┌──────────┐  ┌─────────┐  ┌────────────┐  ┌────────────┐
   │ SHORTTERM  │  │ MIDDLETERM│ │ LONGTERM│  │  EMOTIONS  │  │   TOPICS   │
   └────────────┘  └──────────┘  └─────────┘  └────────────┘  └────────────┘
       │              │              │              │              │
       ├─ Working     ├─ Context   ├─ Episodic  ├─ Profile   ├─ Profile
       │  Memory      │  Buffer    │  Exp.Store ├─ State     ├─ State
       └─ Attention  └─ Topic     ├─ Semantic  ├─ Weight    ├─ Weight
          Map        Context      │  Know.Store├─ History   ├─ History
                                  └─ Weights   ├─ Encoder   ├─ Embedding
                                     Archive   └─ History   └─ Router
```

## Memory Tier Flow

```
INPUT DATA
    │
    ▼
┌─────────────────────┐
│  Working Memory     │  ← Immediate storage (50 items)
│  - Decay: 0.01/min  │
│  - Priority eviction│
└──────────┬──────────┘
           │ (relevance > 0.6)
           ▼
┌─────────────────────┐
│ Context Buffer      │  ← Context maintenance (100 frames, 5 min)
│ - Temporal frames   │
│ - Event logging     │
└──────────┬──────────┘
           │ (relevance > 0.7)
           ▼
┌──────────────────────────────────┐
│    Long-Term Storage             │
├──────────────────────────────────┤
│ Episodic: Experiences (1000)     │
│ Semantic: Facts+Concepts (5500)  │
│ Archive: Weight Snapshots (50)   │
└──────────────────────────────────┘
```

## Emotion Processing Pipeline

```
STIMULUS
    │
    ▼
┌─────────────────────────────────┐
│ EmotionProfile                  │  ← Personality traits
│ Predicts response               │
└──────────────┬──────────────────┘
               │
               ▼
        ┌──────────────────┐
        │ EmotionState     │  ← Current state
        │ Updates & Decays │
        └────────┬─────────┘
                 │
                 ├─ Valence (-1 to 1)
                 └─ Arousal (0 to 1)
                 │
                 ▼
        ┌──────────────────┐
        │ EmotionWeight    │  ← Decision bias
        │ Modulates        │
        │ Confidence       │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ EmotionHistory   │  ← Trend analysis
        │ Records & Trends │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ EmotionEncoder   │  ← 32-dim vectors
        │ Embeddings       │
        └────────┬─────────┘
                 │
                 ▼
            DECISION BIAS
```

## Topic Routing System

```
INFORMATION
    │
    ▼
┌──────────────────────┐
│ Topic Detection      │
└──────────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │ TopicProfile │  ← Expertise tracking
    │ Proficiency  │
    │ Interest     │
    └──────┬───────┘
           │
           ├─────────────────────┐
           │                     │
           ▼                     ▼
    ┌──────────────┐     ┌──────────────┐
    │ TopicState   │     │ TopicWeight  │
    │ Current      │     │ Importance   │
    │ History      │     │ Priority     │
    └──────┬───────┘     └──────┬───────┘
           │                    │
           └────────┬───────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ TopicEmbedding  │  ← 64-dim vectors
           │ Similarity      │
           │ Clustering      │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ TopicRouter     │  ← Handler dispatch
           │ Route to handler│
           └────────┬────────┘
                    │
                    ▼
              PROCESSED OUTPUT
```

## Vector Space Representations

### Emotion Space (32-dim)
```
Each emotion (joy, fear, etc.) has a basis vector
Encoded state is weighted combination of basis vectors

joy (0.8) + fear (0.1) + ... = 32-dim vector
      ↓        ↓
    intense  minimal
```

### Topic Space (64-dim)
```
Each topic has unique semantic position
Similar topics cluster together
Cosine similarity measures semantic distance

topic1     topic2
  ●           ●
   \         /
    \       /  ← high similarity
     \     /
      \   /
       \ /
```

## Data Flow Example

```
PROCESSING CYCLE
│
├─ INPUT: "I made a mistake"
│  │
│  ├─ Store in WorkingMemory (id: mem_001)
│  │
│  ├─ Emotions: sadness↑ (0.8), fear↑ (0.6)
│  │  └─ Update EmotionState
│  │     └─ Calculate valence (-0.7), arousal (0.6)
│  │
│  ├─ Set Attention: SEMANTIC focused on "mistake"
│  │
│  ├─ Add topic "error_recovery"
│  │  └─ Route to error_recovery_handler
│  │
│  ├─ Retrieve similar past experiences
│  │  └─ How did we handle this before?
│  │
│  ├─ Generate response (biased by emotions)
│  │  └─ Apply emotional modulation
│  │
│  ├─ Record decision in ContextBuffer
│  │  └─ What we did and felt
│  │
│  └─ Every 10 cycles: Consolidate
│     ├─ ContextBuffer → LongTerm
│     ├─ Update experience store
│     └─ Record emotional pattern
│
└─ OUTPUT: Response with emotional coloring
```

## Component Relationships

```
MemoryEngine
├─ Owns WorkingMemory
│  └─ Feeds to ContextBuffer
├─ Manages AttentionMap
│  └─ Affects processing priority
├─ Coordinates ContextBuffer
│  └─ Consolidates to LongTerm
├─ Controls ExperienceStore
│  └─ Learns from outcomes
├─ Manages KnowledgeStore
│  └─ Reasons with facts
├─ Maintains WeightArchive
│  └─ Tracks evolution
├─ Monitors EmotionState
│  └─ Modulates decisions via EmotionWeight
├─ Reads EmotionHistory
│  └─ Detects emotional trends
├─ Uses EmotionEncoder
│  └─ Computes emotion vectors
├─ Tracks TopicState
│  └─ Routes via TopicRouter
├─ Manages TopicContext
│  └─ Maintains relationships
└─ Records TopicHistory
   └─ Analyzes sequences
```

## Consolidation Workflow

```
CONSOLIDATION CYCLE
     │
     ▼
┌──────────────────────────┐
│ Get candidates from      │  ← Items reaching 0.6
│ WorkingMemory            │     relevance threshold
└────────────┬─────────────┘
             │
             ▼
      ┌──────────────┐
      │ Check if     │
      │ > 0.6        │  ← Consolidation threshold
      │ relevance    │
      └────────┬─────┘
               │ (YES)
               ▼
      ┌──────────────────┐
      │ Add to           │
      │ ContextBuffer    │
      │ Prune old items  │
      └────────┬─────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Ready for          │
    │ LongTerm if time > │  ← 5 minutes retention
    │ threshold          │
    └─────────┬──────────┘
              │
              ▼
    ┌──────────────────────┐
    │ Move to              │
    │ ExperienceStore or   │
    │ KnowledgeStore       │
    └─────────────────────┘
```

## Capacity and Eviction

```
WorkingMemory (50 items)
├─ Eviction: Lowest priority first
├─ Decay: Exponential (0.01/min)
└─ Full? → Remove LOW priority items

AttentionMap (5 types)
├─ Real-time management
└─ Update: Set new focus (auto-primary if >0.7)

ContextBuffer (100 frames)
├─ Eviction: Oldest first (FIFO)
├─ Time-based: Prune if >5 min
└─ Full? → Remove oldest frame

ExperienceStore (1000 items)
├─ Eviction: Lowest (importance, retrievals, recency)
└─ Full? → Remove least important

KnowledgeStore (5500 total)
├─ Facts (5000): Lowest (verified, confidence, usage)
├─ Concepts (500): Lowest (confidence, usage)
└─ Full? → Remove by type then priority

TopicContext (50 topics)
├─ Eviction: Lowest (active, relevance)
└─ Full? → Remove inactive low-relevance

WeightArchive (50 snapshots)
├─ Eviction: Oldest first (FIFO)
└─ Full? → Remove first snapshot created
```

## Status Monitoring

All components provide `get_status()`:

```python
status = {
    "working_memory": {
        "current_items": 45,
        "max_items": 50,
        "utilization_percent": 90.0,
        "decay_rate": 0.01
    },
    "attention": {
        "active_types": 3,
        "primary_focus": "VISUAL",
        "intensity": 0.85
    },
    "context": {
        "current_frames": 87,
        "max_frames": 100,
        "retention_seconds": 300,
        "average_importance": 0.65
    },
    ...
}
```

## Integration Points

```
┌─────────────────────────────────────────────────┐
│            BrainController / Neural             │
│                  (Caller)                       │
└────────────────┬────────────────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ▼           ▼           ▼
┌────────┐  ┌─────────┐  ┌─────────┐
│ Rules  │  │ Memory  │  │ Learning│
│ Engine │  │ Engine  │  │ Engines │
└────────┘  └─────────┘  └─────────┘
     │           │           │
     └───────────┼───────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ▼           ▼           ▼
┌────────┐  ┌─────────┐  ┌─────────┐
│Output  │  │Decisions│  │Learning │
└────────┘  └─────────┘  └─────────┘
```

---

This architecture provides a complete, scalable memory system suitable for artificial cognitive systems.
