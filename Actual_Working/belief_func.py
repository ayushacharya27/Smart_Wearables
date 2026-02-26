# agents/belief_node.py

import numpy as np


class BeliefNode:
    """
    Maintains an exponentially smoothed belief over activity classes.

    B_t = alpha * P_t + (1 - alpha) * B_{t-1}
    """

    def __init__(self,
                 num_classes=6,
                 alpha=0.6,
                 activity_names=None):

        if not (0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.num_classes = num_classes
        self.alpha = alpha
        self.activity_names = activity_names or [str(i) for i in range(num_classes)]

        # Initialize belief memory (uniform distribution)
        self._belief = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self._step = 0

    

    def update(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Update belief using model softmax output.

        raw_probs: shape (num_classes,)
        returns: updated belief (shape (num_classes,))
        """

        raw_probs = np.asarray(raw_probs, dtype=np.float32)

        if raw_probs.shape[0] != self.num_classes:
            raise ValueError("Wrong number of classes")

        # Normalize just in case
        raw_probs = raw_probs / (raw_probs.sum() + 1e-12)

        # Exponential smoothing
        self._belief = (
            self.alpha * raw_probs +
            (1.0 - self.alpha) * self._belief
        )

        # Renormalize
        self._belief = self._belief / (self._belief.sum() + 1e-12)

        self._step += 1
        return self._belief.copy()

    

    def reset(self):
        """Reset belief to uniform distribution."""
        self._belief = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self._step = 0

    

    @property
    def belief(self):
        return self._belief.copy()

    @property
    def top(self):
        idx = int(np.argmax(self._belief))
        return idx, float(self._belief[idx])