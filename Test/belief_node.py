"""
agents/belief_node.py
Maintains a running belief (smoothed probability) over the 6 activity classes.

Why this exists:
  The raw neural output is noisy — a single bad window might say "Jogging"
  even when the person has been walking for 10 seconds.
  Exponential smoothing gives the system memory so short blips are absorbed.

Formula:
  B_t = alpha * P_t  +  (1 - alpha) * B_{t-1}
  where P_t  = new raw prediction from the model
        B_t  = updated belief
        alpha = how much to weight the new observation (0.6 = fairly reactive)
"""

import numpy as np
from collections import deque


class BeliefNode:

    def __init__(self, num_classes=6, alpha=0.6, history_len=10, activity_names=None):
        # alpha: higher = trusts new predictions more, lower = more smoothing
        assert 0 < alpha <= 1.0, "alpha must be in (0, 1]"
        self.num_classes    = num_classes
        self.alpha          = alpha
        self.activity_names = activity_names or [str(i) for i in range(num_classes)]

        # start with uniform distribution — no preference before first observation
        self._belief = np.ones(num_classes, dtype=np.float32) / num_classes

        # keep the last N raw probability vectors for diagnostics
        self._history = deque(maxlen=history_len)

        self._step = 0   # counts how many updates have happened

    # ── main call ─────────────────────────────────────────────────────────────

    def update(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Feed in the neural model's softmax output and get back a smoothed belief.
        raw_probs : np.ndarray shape (num_classes,)
        returns   : np.ndarray shape (num_classes,)  — the updated belief
        """
        raw_probs = np.asarray(raw_probs, dtype=np.float32)

        # normalise just in case floating-point drift made it not sum to 1
        raw_probs = raw_probs / (raw_probs.sum() + 1e-12)

        # store raw prediction for diagnostics before smoothing
        self._history.append(raw_probs.copy())

        # exponential smoothing: blend new observation with existing belief
        self._belief = self.alpha * raw_probs + (1.0 - self.alpha) * self._belief

        # keep it as a valid probability distribution
        self._belief = self._belief / (self._belief.sum() + 1e-12)

        self._step += 1
        return self._belief.copy()

    # ── helpers ───────────────────────────────────────────────────────────────

    def reset(self):
        """Call this between subjects so old belief doesn't bleed into new session."""
        self._belief = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self._history.clear()
        self._step = 0

    @property
    def belief(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def top(self):
        """Returns (class_index, probability) for the highest-belief activity."""
        idx = int(np.argmax(self._belief))
        return idx, float(self._belief[idx])

    def entropy(self) -> float:
        """
        Shannon entropy in nats.
        High → uncertain (belief spread across many classes)
        Low  → confident (belief concentrated on one class)
        """
        eps = 1e-12
        return float(-np.sum(self._belief * np.log(self._belief + eps)))

    def summary(self) -> str:
        lines = [f"[BeliefNode]  step={self._step}  alpha={self.alpha}  entropy={self.entropy():.3f}"]
        for i, (name, p) in enumerate(zip(self.activity_names, self._belief)):
            bar  = "█" * int(p * 25)
            mark = " ←" if i == int(np.argmax(self._belief)) else ""
            lines.append(f"  {name:<12} {p:.4f}  {bar}{mark}")
        return "\n".join(lines)
