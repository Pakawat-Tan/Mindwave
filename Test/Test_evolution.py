#!/usr/bin/env python3
"""
test_evolution.py — Test Neural Evolution

ทดสอบว่า network evolve อัตโนมัติหรือไม่
"""

import sys
sys.path.insert(0, '.')

from Core.BrainController import BrainController

print("=" * 60)
print("🧬 Testing Neural Evolution")
print("=" * 60)

# Create brain
brain = BrainController()
print(f"✓ Brain created: {brain._instance_id}")

# Check initial structure
initial_nodes = len(brain._brain_struct.nodes)
initial_connections = len(brain._brain_struct.connections)
print(f"✓ Initial structure:")
print(f"  Nodes: {initial_nodes}")
print(f"  Connections: {initial_connections}")

# Train many samples to trigger evolution
print(f"\n{'─'*60}")
print("Training 100 samples to trigger evolution...")
print(f"{'─'*60}")

text = "AI คืออะไร?"
response = "AI คือระบบที่เรียนรู้ได้"

for i in range(100):
    result = brain.train_neural(text, response, 0.8)
    
    # แสดงทุก 10 samples
    if (i + 1) % 10 == 0:
        stats = brain._neural_trainer.stats()
        print(
            f"Sample {i+1:3d}: "
            f"loss={result['loss']:.4f} "
            f"nodes={stats['current_nodes']} "
            f"evolutions={stats['evolution_count']}"
        )

# Check final structure
final_nodes = len(brain._brain_struct.nodes)
final_connections = len(brain._brain_struct.connections)

print(f"\n{'─'*60}")
print("Final Results:")
print(f"{'─'*60}")

stats = brain._neural_trainer.stats()
print(f"✓ Structure changes:")
print(f"  Nodes: {initial_nodes} → {final_nodes} ({final_nodes - initial_nodes:+d})")
print(f"  Connections: {initial_connections} → {final_connections} ({final_connections - initial_connections:+d})")
print(f"\n✓ Evolution stats:")
print(f"  Total evolutions: {stats['evolution_count']}")
print(f"  Evolve every: {stats['evolve_every']} samples")
print(f"  Average loss: {stats['avg_loss']:.4f}")
print(f"  Recent loss: {stats['recent_loss']:.4f}")

# Show evolution log
if brain._neural_trainer.evolution_log:
    print(f"\n✓ Evolution history:")
    for i, evo in enumerate(brain._neural_trainer.evolution_log, 1):
        print(
            f"  {i}. Sample {evo['sample']}: {evo['intent']} "
            f"(nodes {evo['nodes_before']}→{evo['nodes_after']}, "
            f"loss={evo['loss']:.4f})"
        )

print(f"\n{'='*60}")
if stats['evolution_count'] > 0:
    print("✅ Network evolved successfully!")
else:
    print("⚠️  No evolution occurred (may need more samples or different loss)")
print(f"{'='*60}")