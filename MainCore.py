# ============================================================
# MainCore.py
# ------------------------------------------------------------
# Central orchestrator between IO, Brain, and API
# Can be run directly
# ============================================================

from typing import Any, Dict
import time

# --- Import dependencies ---
from Brain.Neural.BrainController import BrainController
from IO.InputAdapter import InputAdapter


class MainCore:
    def __init__(self, brain_controller, input_adapter):
        self.brain = brain_controller
        self.adapter = input_adapter

        # Auto-start brain
        if not self.brain.running:
            self.brain.start()

    # ===============================
    # External API entry point
    # ===============================
    def handle_input(self, raw_input: Any, input_type: str = "text") -> Dict[str, Any]:
        """
        Entry point for API / UI / File input
        """

        encoded = self.adapter.encode(raw_input)

        context = {
            "input_type": input_type,
            "incoming_data": encoded,
            "data_type": "vector",
            "timestamp": time.time()
        }

        cycle_result = self.brain.process_cycle(context)

        actions = cycle_result.get("approved_actions", [])

        if not actions:
            return self.adapter.decode(None, confidence=0.0)

        action = actions[0]
        prediction = (
            action.get("value")
            or action.get("output")
            or action.get("result")
        )

        return self.adapter.decode(
            prediction=prediction,
            confidence=cycle_result.get("review", {}).get("approval_rate", 0.5)
        )

    # ===============================
    # System / Brain commands
    # ===============================
    def send_command(self, command: str, payload: Dict[str, Any] = None) -> Dict[str, Any]:
        context = {
            "input_type": "command",
            "command": command,
            "payload": payload or {},
            "timestamp": time.time()
        }
        return self.brain.process_cycle(context)


# ============================================================
# Run directly
# ============================================================
if __name__ == "__main__":
    print("🧠 Mindwave MainCore starting...")

    brain = BrainController()
    adapter = InputAdapter()
    core = MainCore(brain, adapter)

    print("✅ System ready")
    print("Commands:")
    print("  /summary   → show brain structure")
    print("  /exit      → shutdown")
    print("-" * 40)

    while True:
        try:
            user_input = input("🧑 > ").strip()

            if not user_input:
                continue

            # ----- system commands -----
            if user_input.startswith("/"):
                cmd = user_input[1:].strip()

                if cmd == "exit":
                    print("🛑 Shutting down...")
                    brain.stop()
                    break

                response = core.send_command(cmd)
                print("🧠 >", response)
                continue

            # ----- normal text input -----
            response = core.handle_input(user_input)
            print("🤖 >", response.get("output"))
            print("    confidence:", response.get("confidence"))

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            brain.stop()
            break

        except Exception as e:
            print("❌ Error:", e)
