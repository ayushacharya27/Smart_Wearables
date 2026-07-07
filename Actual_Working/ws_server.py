# ws_server.py
# Drop this file next to run.py.
# Run:  python ws_server.py  (starts pipeline + WebSocket server on ws://localhost:8765)
# The UI connects to this and receives live JSON frames.

import asyncio
import json
import threading
import argparse
import numpy as np
import websockets

from serial_rec import collect_window
from prerna_preprocessor import HARPreprocessor
from model_predict import ModelPredictor
from prerna_belief import BeliefNode
from temporal import TemporalNode
from alert_engine import AlertEngine

ACTIVITY_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
MODEL_PATH = "C:/ai/Smart_Wearables/Model_Files/har_model1.keras"

# Shared state between pipeline thread and WebSocket server
_clients = set()
_latest_frame = {}
_lock = threading.Lock()


# ── WebSocket server ──────────────────────────────────────────────────────────

async def handler(websocket):
    _clients.add(websocket)
    try:
        # Send last known frame immediately on connect
        with _lock:
            if _latest_frame:
                await websocket.send(json.dumps(_latest_frame))
        await websocket.wait_closed()
    finally:
        _clients.discard(websocket)


async def broadcast(frame: dict):
    if not _clients:
        return
    msg = json.dumps(frame)
    await asyncio.gather(*[c.send(msg) for c in list(_clients)], return_exceptions=True)


# ── Pipeline (runs in its own thread) ────────────────────────────────────────

def run_pipeline(args, loop):
    import os
    model_dir = os.path.dirname(args.model)

    preprocessor = HARPreprocessor(model_dir)
    predictor    = ModelPredictor(args.model)
    belief       = BeliefNode(num_classes=len(ACTIVITY_NAMES), alpha=args.alpha,
                              activity_names=ACTIVITY_NAMES)
    temporal     = TemporalNode(num_classes=len(ACTIVITY_NAMES))
    alert_engine = AlertEngine()

    windows = 0
    while True:
        try:
            raw_data  = collect_window(port=args.port, baudrate=args.baudrate,
                                       window_size=args.window_size)
            X, C      = preprocessor.preprocess(raw_data)
            raw_probs = predictor.predict(X, C)
            smoothed  = belief.update(raw_probs)
            idx       = temporal.update(smoothed)
            activity  = ACTIVITY_NAMES[idx]
            confidence = float(smoothed[idx])
            alerts    = alert_engine.update(activity, confidence)
            windows  += 1

            frame = {
                "activity":   activity,
                "confidence": round(confidence, 4),
                "probs":      [round(float(p), 4) for p in smoothed],
                "alerts":     alerts,          # list of {key, message, level, ts}
                "windows":    windows,
                "raw_mean":   raw_data.mean(axis=0).round(4).tolist(),
            }

            with _lock:
                _latest_frame.update(frame)

            # Schedule broadcast on the asyncio event loop
            asyncio.run_coroutine_threadsafe(broadcast(frame), loop)

            print(f"[{windows}] {activity} ({confidence:.0%})")
            if alerts:
                for a in alerts:
                    print(f"  ALERT [{a['level']}] {a['message']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Pipeline error] {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args):
    loop = asyncio.get_event_loop()

    # Start pipeline in background thread
    t = threading.Thread(target=run_pipeline, args=(args, loop), daemon=True)
    t.start()

    print(f"WebSocket server running on ws://localhost:{args.ws_port}")
    print("Open the UI in your browser — it will connect automatically.\n")

    async with websockets.serve(handler, "localhost", args.ws_port):
        await asyncio.Future()  # run forever


def parse_args():
    p = argparse.ArgumentParser(description="HAR Pipeline + WebSocket Server")
    p.add_argument("--port",        type=str,   default="COM6")
    p.add_argument("--baudrate",    type=int,   default=115200)
    p.add_argument("--window_size", type=int,   default=128)
    p.add_argument("--model",       type=str,   default=MODEL_PATH)
    p.add_argument("--alpha",       type=float, default=0.6)
    p.add_argument("--ws_port",     type=int,   default=8765)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))