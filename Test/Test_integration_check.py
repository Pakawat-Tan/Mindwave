#!/usr/bin/env python3
"""
Test/Integration/test_full_integration.py

Integration test ทุก feature ของ Mindwave
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Core.BrainController import BrainController
from Core.Train.TrainingPipeline import TrainingPipeline

def test_full_integration():
    """Test ทุก feature รวมกัน"""
    
    print("=" * 70)
    print("🧪 Mindwave Full Integration Test")
    print("=" * 70)
    
    # ═══════════════════════════════════════════════════════════════
    # 1. Brain Creation
    # ═══════════════════════════════════════════════════════════════
    print("\n1️⃣  Testing Brain Creation...")
    brain = BrainController()
    assert brain is not None
    assert brain._instance_id is not None
    print(f"   ✓ Brain created: {brain._instance_id}")
    
    # ═══════════════════════════════════════════════════════════════
    # 2. Basic Response
    # ═══════════════════════════════════════════════════════════════
    print("\n2️⃣  Testing Basic Response...")
    result = brain.respond("สวัสดี", context="general")
    assert result["response"] is not None
    assert result["outcome"] in ["commit", "conditional", "ask", "silence", "reject"]
    assert 0.0 <= result["confidence"] <= 1.0
    print(f"   ✓ Response: {result['response'][:50]}")
    print(f"   ✓ Outcome: {result['outcome']}, Confidence: {result['confidence']:.2f}")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. Training Pipeline
    # ═══════════════════════════════════════════════════════════════
    print("\n3️⃣  Testing Training Pipeline...")
    
    # สร้าง sample training file
    sample_data = """
<qa>
Q: AI คืออะไร?
A: AI คือระบบที่เรียนรู้และแก้ปัญหาได้
</qa>

<qa>
Q: Neural network คืออะไร?
A: Neural network คือโครงข่ายประสาทเทียม
</qa>

<fact>Deep learning ใช้ neural network หลายชั้น</fact>

<rule>ตอบเป็นภาษาไทยเป็นหลัก</rule>
"""
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_data)
        temp_file = f.name
    
    pipeline = TrainingPipeline(brain)
    result = pipeline.train(temp_file, context="general", epochs=3)
    
    assert result.total_units > 0
    assert result.learned > 0
    assert result.errors == 0
    print(f"   ✓ Trained {result.learned} units")
    print(f"   ✓ By type: {result.by_type}")
    
    os.unlink(temp_file)
    
    # ═══════════════════════════════════════════════════════════════
    # 4. Trained Knowledge Recall
    # ═══════════════════════════════════════════════════════════════
    print("\n4️⃣  Testing Trained Knowledge Recall...")
    result = brain.respond("AI คืออะไร", context="general")
    print(f"   ✓ Response: {result['response']}")
    print(f"   ✓ Confidence: {result['confidence']:.2f}")
    
    # ═══════════════════════════════════════════════════════════════
    # 5. BeliefSystem
    # ═══════════════════════════════════════════════════════════════
    print("\n5️⃣  Testing BeliefSystem...")
    beliefs = list(brain._belief_system._beliefs.values())
    assert len(beliefs) > 0
    print(f"   ✓ Total beliefs: {len(beliefs)}")
    
    stable = [b for b in beliefs if b.belief_variance <= 0.10]
    print(f"   ✓ Stable beliefs: {len(stable)}")
    
    strong = [b for b in beliefs if b.confidence_score >= 0.75]
    print(f"   ✓ Strong beliefs: {len(strong)}")
    
    # ═══════════════════════════════════════════════════════════════
    # 6. Neural Network
    # ═══════════════════════════════════════════════════════════════
    print("\n6️⃣  Testing Neural Network...")
    
    # Check structure
    nodes = len(brain._brain_struct.nodes)
    connections = len(brain._brain_struct.connections)
    print(f"   ✓ Nodes: {nodes}")
    print(f"   ✓ Connections: {connections}")
    
    # Train neural network
    result = brain.train_neural(
        text="Test input",
        target_response="Test output",
        importance=0.8
    )
    assert result["loss"] >= 0
    assert 0 <= result["accuracy"] <= 1
    print(f"   ✓ Neural training: loss={result['loss']:.4f}, acc={result['accuracy']:.2f}")
    
    # ═══════════════════════════════════════════════════════════════
    # 7. Neural Evolution
    # ═══════════════════════════════════════════════════════════════
    print("\n7️⃣  Testing Neural Evolution...")
    stats = brain._neural_trainer.stats()
    print(f"   ✓ Evolution enabled: {stats['evolution_enabled']}")
    print(f"   ✓ Evolve every: {stats['evolve_every']} samples")
    print(f"   ✓ Current nodes: {stats['current_nodes']}")
    print(f"   ✓ Evolution count: {stats['evolution_count']}")
    
    # ═══════════════════════════════════════════════════════════════
    # 8. Memory System
    # ═══════════════════════════════════════════════════════════════
    print("\n8️⃣  Testing Memory System...")
    memory_stats = brain._memory.stats()
    total_atoms = sum(memory_stats.values())
    print(f"   ✓ Total atoms: {total_atoms}")
    print(f"   ✓ By tier: {memory_stats}")
    
    # ═══════════════════════════════════════════════════════════════
    # 9. Emotional Processing
    # ═══════════════════════════════════════════════════════════════
    print("\n9️⃣  Testing Emotional Processing...")
    emotion_state = brain._emotion.get_emotional_state()
    print(f"   ✓ Primary emotion: {emotion_state.primary_emotion.value}")
    print(f"   ✓ Intensity: {emotion_state.intensity:.2f}")
    
    # ═══════════════════════════════════════════════════════════════
    # 10. MetaCognition
    # ═══════════════════════════════════════════════════════════════
    print("\n🔟 Testing MetaCognition...")
    logs = brain._logs
    if logs:
        reflection = brain._metacognition.reflect(logs)
        print(f"   ✓ Reflection completed")
    else:
        print(f"   ✓ MetaCognition ready (no logs yet)")
    
    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("✅ All Integration Tests Passed!")
    print("=" * 70)
    print(f"""
Summary:
  • Brain: {brain._instance_id}
  • Beliefs: {len(beliefs)} ({len(stable)} stable, {len(strong)} strong)
  • Memory: {total_atoms} atoms
  • Neural: {stats['current_nodes']} nodes, {stats['current_connections']} connections
  • Evolution: {stats['evolution_count']} times
  • Emotion: {emotion_state.primary_emotion.value} (intensity={emotion_state.intensity:.2f})
""")

if __name__ == "__main__":
    test_full_integration()