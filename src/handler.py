import os
import sys
import traceback

# ----------------------------
# Environment (must be first)
# ----------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf/hub")

print("🚀 Handler starting...", flush=True)

# ----------------------------
# Safe imports (no GPU work)
# ----------------------------
from routes.classify import classify_video
from routes.process import process_video
from helper import load_models


def handler(event):
    """RunPod Serverless handler."""
    try:
        payload = (event or {}).get("input") or {}
        action = payload.get("action", "process")

        if action == "process":
            return process_video(payload)
        elif action == "classify":
            return classify_video(payload)
        else:
            return {
                "error": "invalid_action",
                "detail": f"Unknown action: {action}",
            }
    except Exception as e:
        traceback.print_exc()
        return {"error": "internal_server_error", "detail": str(e)}


# ----------------------------
# Entry point (CRITICAL)
# ----------------------------
if __name__ == "__main__":
    try:
        print("🔄 Loading models...", flush=True)
        load_models()
        print("✅ Models loaded successfully", flush=True)
    except Exception:
        print("❌ MODEL LOAD FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)  # hard fail so logs appear

    import runpod

    print("🟢 Starting RunPod serverless worker", flush=True)
    runpod.serverless.start({"handler": handler})
