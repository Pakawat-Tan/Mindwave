"""
Main.py — MindWave Entry Point

รัน BrainController แบบ Realtime
รับ input จาก stdin → ส่งเข้า Brain → แสดง response

Usage:
    python3 Main.py
    python3 Main.py --context math
    python3 Main.py --verbose
    python3 Main.py --instance my_brain
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

from Core.BrainController import BrainController
from Core.IO.IOController import IOController
from Core.IO.IOPacket import ChannelType
from Core.Train.TrainingPipeline import TrainingPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level  = level,
        format = "[%(levelname)s] %(name)s: %(message)s",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║            🧠  M I N D W A V E  v1.0                 ║
║       Synthetic Intelligence — Realtime Mode         ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  /context <name>  — เปลี่ยน context (เช่น math, science, general)
  /status          — ดูสถานะ Brain
  /meta            — ดู MetaCognition stats
  /emotion         — ดูสถานะอารมณ์ปัจจุบัน
  /patterns        — ดู patterns ที่พบ
  /topics          — ดู topic clusters
  /strategy        — แนะนำ strategy ปัจจุบัน
  /feedback        — ดู implicit feedback signals
  /beliefs         — ดูสิ่งที่ Brain เรียนรู้ไว้ (auto-updated ทุก interaction)
  /summary         — สรุปภาพรวมโมเดลทั้งหมด
  /io              — ดูสถานะ IO channels
  /loadfile <path> — อ่านไฟล์แล้วให้ Brain เรียนรู้ (txt/json/pdf/docx)
  /loadurl <url>   — fetch URL แล้วให้ Brain เรียนรู้
  /train <source>  — เทรน Brain จาก file/URL/image (รองรับ tags)
  /trainstats      — ดูสถิติการเทรน
  /reset           — reset session (ไม่ล้าง learning)
  /help            — แสดง help นี้
  /quit or /exit   — ออกจากโปรแกรม
"""


def print_response(result: dict, verbose: bool = False) -> None:
    """แสดง response จาก Brain"""
    outcome    = result.get("outcome", "?")
    confidence = result.get("confidence", 0.0)
    response   = result.get("response", "")
    learned    = result.get("learned", False)

    # color by outcome
    outcome_colors = {
        "commit":      "\033[92m",   # green
        "conditional": "\033[93m",   # yellow
        "ask":         "\033[96m",   # cyan
        "silence":     "\033[90m",   # gray
        "reject":      "\033[91m",   # red
    }
    reset  = "\033[0m"
    color  = outcome_colors.get(outcome, "\033[97m")

    print(f"\n{color}[{outcome.upper()}]{reset} {response}")

    if verbose:
        learned_str = "✓ learned" if learned else ""
        print(f"  conf={confidence:.2f}  {learned_str}")


def print_status(brain: BrainController) -> None:
    """แสดงสถานะ Brain"""
    s = brain.status()
    print(f"""
┌─ Brain Status ────────────────────────────────
│  instance   : {s['instance_id']}
│  mode       : {s['mode']}
│  personality: {s['personality']}
│  skills     : {s['skill_count']}
│  logs       : {s['logs_total']}
├─ Modules ─────────────────────────────────────""")
    for name, active in s["modules"].items():
        mark = "✓" if active else "✗"
        print(f"│  {mark} {name}")
    print("└───────────────────────────────────────────────")


def print_meta(brain: BrainController) -> None:
    """แสดง MetaCognition stats"""
    s = brain.metacognition.stats()
    print(f"""
┌─ MetaCognition ───────────────────────────────
│  reflections   : {s['reflections']}
│  calibrations  : {s['calibrations']}
│  errors found  : {s['errors_detected']}
│  tracks        : {s['learning_tracks']}
│  confidence bias: {s['confidence_bias']:+.3f}""")

    if s["last_strategy"]:
        st = s["last_strategy"]
        print(f"│  strategy      : {st['recommended']} (conf={st['confidence']:.2f})")
        print(f"│  reason        : {st['reason']}")
    print("└───────────────────────────────────────────────")


def print_emotion(brain: BrainController) -> None:
    """แสดงสถานะอารมณ์"""
    state = brain.emotion.get_emotional_state()
    scores = {
        e: round(s, 2)
        for e, s in state.emotion_scores.items()
        if s > 0.01
    }
    print(f"""
┌─ Emotional State ─────────────────────────────
│  primary   : {state.primary_emotion.value}
│  intensity : {state.intensity:.2f}
│  sentiment : {state.sentiment.value}
│  scores    : {scores}
└───────────────────────────────────────────────""")


def print_patterns(brain: BrainController) -> None:
    """แสดง patterns"""
    pr = brain.pattern
    print(f"""
┌─ Pattern Recognition ─────────────────────────
│  sequences   : {len(pr.sequences)}
│  behaviors   : {len(pr.behaviors)}
│  transitions : {len(pr.transitions)}
│  errors      : {len(pr.errors)}
│  successes   : {len(pr.successes)}""")

    if pr.behaviors:
        b = pr.behaviors[-1]
        print(f"│  prefer ctx  : {b.preferred_contexts}")
        print(f"│  style       : {b.interaction_style}")
    print("└───────────────────────────────────────────────")


def print_topics(brain: BrainController) -> None:
    """แสดง topic clusters"""
    s = brain.topic.stats()
    print(f"""
┌─ Topic Clusters ──────────────────────────────
│  clusters    : {s.get('total_clusters', 0)}
│  topics      : {s.get('total_topics', 0)}
│  avg size    : {s.get('avg_cluster_size', 0.0):.2f}""")

    for cluster in brain.topic.clusters[:5]:
        print(f"│  [{cluster.cluster_id}] {cluster.centroid} ({cluster.size} topics)")
    print("└───────────────────────────────────────────────")


def print_strategy(brain: BrainController, context: str) -> None:
    """แนะนำ strategy"""
    logs = brain.logs
    if len(logs) < 3:
        print("  ℹ️  ต้องมี interactions อย่างน้อย 3 ครั้งก่อน suggest strategy")
        return

    rec = brain.metacognition.suggest_strategy(logs, context)
    print(f"""
┌─ Strategy Recommendation ─────────────────────
│  strategy  : {rec.recommended.value}
│  confidence: {rec.confidence:.2f}
│  reason    : {rec.reason}
│  alt       : {[s.value for s in rec.alternatives]}
└───────────────────────────────────────────────""")


WIDTH = 56  # box width

def print_feedback(brain: BrainController) -> None:
    """แสดง implicit feedback stats"""
    s = brain.feedback.stats()
    print(f"""
┌─ Implicit Feedback ───────────────────────────
│  total signals : {s['total_signals']}
│  sealed atoms  : {s['sealed_atoms']}
│  current session: {s['current_session']} signals
│  by type       : {s['by_type']}
│  positive      : {s['by_polarity']['positive']}
│  negative      : {s['by_polarity']['negative']}
│  conf delta    : {s['cumulative_conf']:+.3f}
│  skill delta   : {s['cumulative_skill']:+.3f}
└───────────────────────────────────────────────""")

def box_line(text: str) -> str:
    """จัดข้อความให้อยู่ใน box"""
    inner = WIDTH - 4
    return f"│ {text:<{inner}} │"


def _format_brain_summary_ascii(brain) -> str:
    """Brain node/connection summary จาก MainController"""
    nodes  = getattr(brain, "nodes",       {})
    conns  = getattr(brain, "connections", {})
    biases = getattr(brain, "biases",      {})

    total_nodes       = len(nodes)
    total_connections = sum(1 for c in conns.values() if c.get("enabled"))
    total_weights     = total_connections
    total_biases      = len(biases)
    total_params      = total_weights + total_biases

    role_count  = {"input": 0, "hidden": 0, "output": 0}
    layers      = set()
    total_usage = 0.0
    for n in nodes.values():
        role = n.get("role", "hidden")
        role_count[role] = role_count.get(role, 0) + 1
        layers.add(n.get("layer", 0))
        total_usage += n.get("usage", 0.0)
    avg_usage = total_usage / total_nodes if total_nodes > 0 else 0.0

    lines = []

    # ── Header ────────────────────────────────────────────────
    lines.append("  ┌" + "─" * (WIDTH - 2) + "┐")
    lines.append("  " + box_line("🧠 Brain Structure"))
    lines.append("  ├" + "─" * (WIDTH - 2) + "┤")
    lines.append("  " + box_line(f"Model type         : {getattr(brain, 'model_type', 'NeuralBrain')}"))
    lines.append("  " + box_line(f"Layers             : {len(layers)}"))
    lines.append("  " + box_line(f"Nodes              : {total_nodes}"))
    lines.append("  " + box_line(f"  ├─ Input          : {role_count['input']}"))
    lines.append("  " + box_line(f"  ├─ Hidden         : {role_count['hidden']}"))
    lines.append("  " + box_line(f"  └─ Output         : {role_count['output']}"))
    lines.append("  " + box_line(f"Active connections : {total_connections}"))
    lines.append("  " + box_line(f"Parameters         : {total_params}"))
    lines.append("  " + box_line(f"  ├─ Weights        : {total_weights}"))
    lines.append("  " + box_line(f"  └─ Biases         : {total_biases}"))
    lines.append("  " + box_line(f"Avg usage / node   : {avg_usage:.2f}"))
    lines.append("  └" + "─" * (WIDTH - 2) + "┘")

    if total_nodes == 0:
        return "\n".join(lines)

    # ── Node Table ────────────────────────────────────────────
    lines.append("")
    lines.append("  ┌──────┬──────────────────────┬──────────┬──────────┬────────┬────────┐")
    lines.append("  │Layer │ Node ID              │ Role     │ Head     │ Usage% │ Params │")
    lines.append("  ├──────┼──────────────────────┼──────────┼──────────┼────────┼────────┤")

    for nid, n in sorted(nodes.items(), key=lambda x: (x[1].get("layer", 0), x[0])):
        usage     = n.get("usage", 0.0)
        usage_pct = (usage / total_usage * 100.0) if total_usage > 0 else 0.0
        param_count = 1  # bias
        for c in conns.values():
            if c.get("enabled") and c.get("destination") == nid:
                param_count += 1
        lines.append(
            f"  │ {n.get('layer', 0):<4} "
            f"│ {nid:<20} "
            f"│ {n.get('role', 'hidden'):<8} "
            f"│ {str(n.get('head', '-')):<8} "
            f"│ {usage_pct:>6.2f} "
            f"│ {param_count:>6} │"
        )

    lines.append("  └──────┴──────────────────────┴──────────┴──────────┴────────┴────────┘")
    return "\n".join(lines)
    """แสดง implicit feedback stats"""
    s = brain.feedback.stats()
    print(f"""
┌─ Implicit Feedback ───────────────────────────
│  total signals : {s['total_signals']}
│  sealed atoms  : {s['sealed_atoms']}
│  current session: {s['current_session']} signals
│  by type       : {s['by_type']}
│  positive      : {s['by_polarity']['positive']}
│  negative      : {s['by_polarity']['negative']}
│  conf delta    : {s['cumulative_conf']:+.3f}
│  skill delta   : {s['cumulative_skill']:+.3f}
└───────────────────────────────────────────────""")


def print_summary(brain: BrainController, context: str, start_time: float, interaction_count: int) -> None:
    """สรุปภาพรวม Mindwave ทั้งหมด"""
    import time as _time

    uptime_s  = int(_time.time() - start_time)
    uptime    = f"{uptime_s // 3600}h {(uptime_s % 3600) // 60}m {uptime_s % 60}s"

    s         = brain.status()
    bs        = brain.belief_system.stats()
    lm        = brain.learn_mode.stats()
    pat       = brain.pattern.stats() if hasattr(brain.pattern, "stats") else {}
    fb        = brain.feedback.stats()
    meta_s    = brain.metacognition.stats()
    emo_state = brain.emotion.get_emotional_state()
    personality = s.get("personality", "—")
    skill_count = s.get("skill_count", 0)
    logs_total  = s.get("logs_total", 0)

    # active modules
    modules     = s.get("modules", {})
    active_mods = [name for name, on in modules.items() if on]

    # beliefs top 3
    top_beliefs = brain.belief_system.strongest(n=3)

    print(f"""
╔══════════════════════════════════════════════════════╗
║         🧠  M I N D W A V E  —  S U M M A R Y      ║
╚══════════════════════════════════════════════════════╝

  ┌─ Identity ────────────────────────────────────
  │  name        : Mindwave v1.0
  │  instance    : {s['instance_id']}
  │  uptime      : {uptime}
  │  context     : {context}
  │  personality : {personality}
  └───────────────────────────────────────────────

  ┌─ Session ─────────────────────────────────────
  │  interactions : {interaction_count}
  │  logs total   : {logs_total}
  │  skills       : {skill_count}
  │  conf bias    : {meta_s['confidence_bias']:+.3f}
  │  emotion      : {emo_state.primary_emotion.value} (intensity={emo_state.intensity:.2f})
  └───────────────────────────────────────────────

  ┌─ Learning ────────────────────────────────────
  │  beliefs total    : {bs['total']}
  │    stable         : {bs['stable']}
  │    strong         : {bs['strong']}
  │    conflicted     : {bs['conflicted']}
  │    avg confidence : {bs['avg_confidence']:.2f}
  │  learn sessions   : {lm['sessions']}
  │    consolidated   : {lm['consolidated']}
  │  feedback signals : {fb['total_signals']}
  │    positive       : {fb['by_polarity']['positive']}
  │    negative       : {fb['by_polarity']['negative']}
  └───────────────────────────────────────────────""")

    if top_beliefs:
        print(f"  ┌─ Top Beliefs ──────────────────────────────────")
        for b in top_beliefs:
            status = "✓" if b.is_stable else "~"
            print(f"  │  {status} {b.subject[:36]:<36} conf={b.confidence_score:.2f}")
        print(f"  └───────────────────────────────────────────────")

    print(f"""
  ┌─ Active Modules ({len(active_mods)}/{len(modules)}) ───────────────────────""")
    for i in range(0, len(active_mods), 3):
        row = active_mods[i:i+3]
        print(f"  │  " + "   ".join(f"✓ {m}" for m in row))
    print(f"  └───────────────────────────────────────────────")

    # โครงสร้าง Brain — nodes/connections table
    try:
        brain_obj = getattr(brain, "_brain_struct", brain)
        print(_format_brain_summary_ascii(brain_obj))
    except Exception:
        pass

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def run(
    context:      str  = "general",
    verbose:      bool = False,
    instance_id:  str  = "",
    socket_port:  int  = 0,    # 0 = ไม่เปิด
    rest_port:    int  = 0,    # 0 = ไม่เปิด
    epochs:       int  = 3,    # จำนวน epochs สำหรับ /train
) -> None:
    """Main realtime loop"""

    print(BANNER)
    print(f"  Context: {context}")
    print(f"  Type /help for commands\n")

    # init Brain
    brain = BrainController()
    print(f"  Brain initialized: instance={brain._instance_id}")

    # init IOController — เชื่อม Brain เข้า IO gateway
    io = IOController(brain)

    # init TrainingPipeline — เทรน Runtime
    pipeline = TrainingPipeline(brain, io)

    # เปิด servers ถ้า port ระบุไว้
    if socket_port:
        io.start_socket(port=socket_port)
        print(f"  Socket TCP  : port {socket_port}")
    if rest_port:
        io.start_rest(port=rest_port)
        print(f"  REST API    : http://0.0.0.0:{rest_port}")

    print(f"  IO channels : {', '.join(['cli','file','socket','rest','event','internet','sound','video'])}\n")

    interaction_count = 0
    start_time = time.time()

    while True:
        try:
            # prompt
            prompt = f"\033[96m[{context}]\033[0m > "
            user_input = input(prompt).strip()

        except (KeyboardInterrupt, EOFError):
            brain.seal_session(silence=True)
            io.flush_log()
            print("\n\n  👋 MindWave ปิดตัวแล้ว\n")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            args = user_input.split()[1:] if len(user_input.split()) > 1 else []

            if cmd in ("/quit", "/exit"):
                print("\n  👋 MindWave ปิดตัวแล้ว\n")
                break

            elif cmd == "/help":
                print(HELP_TEXT)

            elif cmd == "/context":
                if args:
                    context = args[0]
                    print(f"  ✓ context เปลี่ยนเป็น '{context}'")
                else:
                    print(f"  context ปัจจุบัน: {context}")

            elif cmd == "/status":
                print_status(brain)

            elif cmd == "/meta":
                print_meta(brain)

            elif cmd == "/emotion":
                print_emotion(brain)

            elif cmd == "/patterns":
                print_patterns(brain)

            elif cmd == "/topics":
                print_topics(brain)

            elif cmd == "/strategy":
                print_strategy(brain, context)

            elif cmd == "/feedback":
                print_feedback(brain)

            elif cmd == "/learn":
                learn_text = user_input[6:].strip()
                if not learn_text:
                    print("  ใช้: /learn <ข้อความที่ต้องการเรียนรู้>")
                else:
                    result = brain.learn(learn_text)
                    print(f"\n{result['response']}")

            elif cmd == "/beliefs":
                # LearnMode beliefs
                # LearnMode beliefs
                lm_summary = brain.learn_mode.summary()
                lm_stats   = brain.learn_mode.stats()
                # BeliefSystem
                bs_summary = brain.belief_system.summary(n=8)
                bs_stats   = brain.belief_system.stats()

                print(f"\n┌─ Beliefs ─────────────────────────────────────")
                print(f"│  [LearnMode]")
                for line in lm_summary.split("\n"):
                    print(f"│    {line}")
                print(f"│")
                print(f"│  [BeliefSystem] {bs_stats['total']} beliefs "
                      f"stable={bs_stats['stable']} "
                      f"strong={bs_stats['strong']} "
                      f"conflict={bs_stats['conflicted']}")
                for line in bs_summary.split("\n")[1:6]:
                    print(f"│    {line}")
                print(f"│")
                print(f"│  sessions={lm_stats['sessions']} "
                      f"consolidated={lm_stats['consolidated']}")
                print(f"└───────────────────────────────────────────────")

            elif cmd == "/summary":
                print_summary(brain, context, start_time, interaction_count)

            elif cmd == "/io":
                s = io.stats()
                log = s["io_log"]
                print(f"""
┌─ IO Status ───────────────────────────────────
│  brain      : {s['brain']}
│  total logs : {log['total']}
│  input      : {log['by_direction'].get('input', 0)}
│  output     : {log['by_direction'].get('output', 0)}
│  by channel : {log.get('by_channel', {})}
│  event bus  : {s['event_bus']} events
├─ Channels ────────────────────────────────────
│  ✓ CLI       stdin/stdout
│  ✓ File      txt / json / pdf / docx
│  ✓ Internet  web fetch
│  ✓ Sound     speech in/out
│  ✓ Video     image / video OCR
│  ✓ Socket    TCP/UDP {"(active)" if socket_port else "(standby)"}
│  ✓ REST      HTTP API {"(active)" if rest_port else "(standby)"}
│  ✓ EventBus  pub/sub
└───────────────────────────────────────────────""")

            elif cmd == "/loadfile":
                if not args:
                    print("  ใช้: /loadfile <path>")
                else:
                    path = " ".join(args)
                    print(f"  กำลังอ่าน '{path}'...")
                    out = io.learn_from_file(path, context=context)
                    if out:
                        print(f"  ✓ เรียนรู้จากไฟล์แล้ว → {out.response[:80]}")
                    else:
                        print(f"  ✗ ไม่พบไฟล์หรืออ่านไม่ได้")

            elif cmd == "/loadurl":
                if not args:
                    print("  ใช้: /loadurl <url>")
                else:
                    url = args[0]
                    print(f"  กำลัง fetch '{url}'...")
                    out = io.learn_from_url(url, context=context)
                    if out:
                        print(f"  ✓ เรียนรู้จาก URL แล้ว → {out.response[:80]}")
                    else:
                        print(f"  ✗ fetch ไม่ได้ (network หรือ URL ไม่ถูกต้อง)")

            elif cmd == "/train":
                if not args:
                    print("  ใช้: /train <path หรือ URL หรือ image>")
                    print("  เช่น: /train data.txt")
                    print("        /train https://example.com")
                    print("        /train photo.jpg")
                else:
                    source = " ".join(args)
                    print(f"\n  🎓 กำลังเทรนจาก '{source}'...")

                    # progress bar แบบ inline
                    _progress = [0]
                    def on_progress(done, total, unit):
                        pct = int(done / total * 30)
                        bar = "█" * pct + "░" * (30 - pct)
                        print(f"\r  [{bar}] {done}/{total} ({unit.unit_type})", end="", flush=True)
                        _progress[0] = done

                    result = pipeline.train(source, context=context, on_progress=on_progress, epochs=epochs)
                    print()  # newline หลัง progress bar

                    # แสดงผล
                    rate = result.learned / max(1, result.total_units) * 100
                    print(f"""
  ┌─ Train Result ─────────────────────────────────
  │  source       : {result.source}
  │  units        : {result.total_units}
  │  learned      : {result.learned} ({rate:.0f}%)
  │  consolidated : {result.consolidated}
  │  errors       : {result.errors}
  │  time         : {result.elapsed_s:.1f}s
  │  by type      : {result.by_type}
  └───────────────────────────────────────────────""")

            elif cmd == "/trainstats":
                s = pipeline.stats()
                print(f"""
  ┌─ Training Stats ───────────────────────────────
  │  sessions     : {s['sessions']}
  │  total units  : {s['total_units']}
  │  total learned: {s['total_learned']}
  │  consolidated : {s['total_consolidated']}
  │  errors       : {s['total_errors']}
  └───────────────────────────────────────────────""")
                brain.seal_session(silence=False)
                brain = BrainController()
                interaction_count = 0
                print(f"  ✓ Session reset (Brain ใหม่, learning เริ่มใหม่)")

            else:
                print(f"  ❓ ไม่รู้จัก command '{cmd}' — พิมพ์ /help")

            continue

        # ── Normal input → IOController → Brain ──────────────────
        try:
            out = io.send_text(user_input, context=context, channel=ChannelType.CLI)
            result = {
                "response":   out.response,
                "outcome":    out.outcome,
                "confidence": out.confidence,
                "learned":    False,
            }
            interaction_count += 1
            print_response(result, verbose)

        except Exception as e:
            print(f"\n  ⚠️  Error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MindWave — Cognitive AI Realtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--context", "-c",
        default = "general",
        help    = "Initial context (default: general)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action  = "store_true",
        help    = "Show detailed output (confidence, learned)",
    )
    parser.add_argument(
        "--instance", "-i",
        default = "",
        help    = "Instance ID (optional)",
    )
    parser.add_argument(
        "--socket-port", "-s",
        type    = int,
        default = 0,
        help    = "เปิด TCP socket server (เช่น 9000)",
    )
    parser.add_argument(
        "--rest-port", "-r",
        type    = int,
        default = 0,
        help    = "เปิด REST API server (เช่น 8000)",
    )
    parser.add_argument(
        "--log-level",
        default = "WARNING",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Logging level",
    )
    parser.add_argument(
        "--epochs", "-e",
        type    = int,
        default = 3,
        help    = "จำนวน epochs สำหรับ /train (default: 3)",
    )

    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level  = getattr(logging, args.log_level),
        format = "[%(levelname)s] %(name)s: %(message)s",
    )

    run(
        context     = args.context,
        verbose     = args.verbose,
        instance_id = args.instance,
        socket_port = args.socket_port,
        rest_port   = args.rest_port,
        epochs      = args.epochs,
    )


if __name__ == "__main__":
    main()