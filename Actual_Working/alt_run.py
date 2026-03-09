# run.py
# Integrates: serial_rec, prerna_preprocessor, model_predict, prerna_belief, temporal_node

import argparse
import numpy as np

from serial_rec import collect_window
from prerna_preprocessor import HARPreprocessor
from model_predict import ModelPredictor
from prerna_belief import BeliefNode
from prerna_temporal import TemporalNode


# ── Config ────────────────────────────────────────────────────────────────────

ACTIVITY_NAMES = [
    "Walking",
    "Jogging",
    "Upstairs",
    "Downstairs",
    "Sitting",
    "Standing",
]

MODEL_PATH = "C:/ai/Smart_Wearables/Model_Files/har_model1.keras"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_pipeline(model_path: str, alpha: float = 0.6):
    """Instantiate and return all pipeline components."""

    import os
    model_dir = os.path.dirname(model_path)

    preprocessor = HARPreprocessor(model_dir)

    predictor = ModelPredictor(model_path)

    belief = BeliefNode(
        num_classes=len(ACTIVITY_NAMES),
        alpha=alpha,
        activity_names=ACTIVITY_NAMES,
    )

    temporal = TemporalNode(num_classes=len(ACTIVITY_NAMES))

    return preprocessor, predictor, belief, temporal


def run_once(preprocessor, predictor, belief, temporal, port, baudrate, window_size):
    """Collect one window, run inference, update belief, apply temporal reasoning."""

    # 1. Collect raw sensor data from Arduino over serial
    raw_data = collect_window(port=port, baudrate=baudrate, window_size=window_size)
    print(f"\n[Serial]  Collected window shape: {raw_data.shape}")

    # ---- DEBUG: CHECK SENSOR STATISTICS ----
    print("Raw mean:", raw_data.mean(axis=0))
    print("Raw std :", raw_data.std(axis=0))
    # ----------------------------------------

    # 2. Preprocess → (X, C)
    X, C = preprocessor.preprocess(raw_data)
    print(f"[Preproc] X: {X.shape}  C: {C.shape}")

    # 3. Neural inference → raw softmax probs
    raw_probs = predictor.predict(X, C)
    print(f"[Model]   Raw probs: {np.round(raw_probs, 3)}")

    # 4. Belief smoothing
    smoothed = belief.update(raw_probs)

    # 5. Temporal reasoning
    idx = temporal.update(smoothed)
    confidence = smoothed[idx]
    activity = ACTIVITY_NAMES[idx]

    # 6. Report
    print(f"\n{'─'*40}")
    print(f"  Predicted Activity : {activity}")
    print(f"  Confidence         : {confidence:.1%}")
    print(f"{'─'*40}")

    for i, (name, p) in enumerate(zip(ACTIVITY_NAMES, smoothed)):
        bar = "█" * int(p * 30)
        marker = " ◄" if i == idx else ""
        print(f"  {name:<22} {p:.3f}  {bar}{marker}")

    return idx, confidence, activity


def run_loop(preprocessor, predictor, belief, temporal, port, baudrate, window_size, n_windows):
    """Continuously collect windows and infer."""

    print(f"\nStarting HAR pipeline — {'continuous' if n_windows == 0 else str(n_windows) + ' window(s)'}\n")

    count = 0
    try:
        while True:
            run_once(preprocessor, predictor, belief, temporal, port, baudrate, window_size)
            count += 1

            if n_windows and count >= n_windows:
                break

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    print(f"\n[Done] Processed {count} window(s).")


# ── Entry Point ───────────────────────────────────────────────────────────────

def parse_args():

    parser = argparse.ArgumentParser(description="HAR Real-Time Inference Pipeline")

    parser.add_argument("--port", type=str, default="COM6")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--n_windows", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    preprocessor, predictor, belief, temporal = build_pipeline(
        args.model,
        alpha=args.alpha
    )

    run_loop(
        preprocessor=preprocessor,
        predictor=predictor,
        belief=belief,
        temporal=temporal,
        port=args.port,
        baudrate=args.baudrate,
        window_size=args.window_size,
        n_windows=args.n_windows,
    )