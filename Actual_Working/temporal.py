import numpy as np


class TemporalNode:
    """
    Temporal reasoning over activity states using transition probabilities.
    """

    def __init__(self, num_classes=6):

        self.num_classes = num_classes
        self.previous_state = None

        # Transition probability matrix
        # rows = current state
        # cols = next state

        # Basically all this bs is just for how can a action progress from one to another
        self.transition_matrix = np.array([
            # W    J    U    D    S    ST
            [0.60,0.20,0.10,0.10,0.00,0.00],  # Walking
            [0.30,0.60,0.00,0.00,0.00,0.10],  # Jogging
            [0.60,0.00,0.30,0.00,0.00,0.10],  # Upstairs
            [0.60,0.00,0.00,0.30,0.00,0.10],  # Downstairs
            [0.00,0.00,0.00,0.00,0.70,0.30],  # Sitting
            [0.40,0.00,0.00,0.00,0.30,0.30]   # Standing
        ])

        self.history = []

    def update(self, smoothed_probs: np.ndarray) -> int:
        """
        Input:
            smoothed_probs shape (6,)
        Output:
            selected_state (int)
        """

        probs = np.asarray(smoothed_probs)

        # First step
        if self.previous_state is None:
            state = int(np.argmax(probs))
            self.previous_state = state
            self.history.append(state)
            return state

        scores = []

        for s in range(self.num_classes):

            transition_prob = self.transition_matrix[self.previous_state][s]

            # combine belief + transition
            # score = probs[s] * transition_prob
            score = 0.6 * probs[s] + 0.4 * transition_prob

            scores.append(score)

        best_state = int(np.argmax(scores))

        self.previous_state = best_state
        self.history.append(best_state)

        # keep last 10 states
        if len(self.history) > 10:
            self.history.pop(0)

        return best_state



# import numpy as np
# from prerna_belief import BeliefNode


# ACTIVITY_NAMES = [
#     "Walking",
#     "Jogging",
#     "Upstairs",
#     "Downstairs",
#     "Sitting",
#     "Standing",
# ]


# class TemporalFSM:
#     """
#     Temporal reasoning layer with:
#         1. Soft transition matrix        (probability-weighted, applied BEFORE smoothing)
#         2. Exponential smoothing         (via BeliefNode)
#         3. Stability enforcement         (pending state must hold for N steps)
#         4. Confidence thresholding       (suppress low-confidence outputs)

#     Transition matrix interpretation:
#         Row  = current confirmed state
#         Col  = next candidate state
#         Value = how likely that transition is (0.0 = impossible, 1.0 = free)

#     Example: Sitting → Jogging = 0.0  (physically blocked)
#              Standing is gateway: can reach any state but with dampened probs
#     """

#     def __init__(self,
#                  num_classes: int = 6,
#                  alpha: float = 0.6,
#                  activity_names: list = None,
#                  decision_threshold: float = 0.5,
#                  stable_required: int = 2):
#         """
#         Args:
#             num_classes        : Number of activity classes.
#             alpha              : BeliefNode smoothing factor in (0, 1].
#             activity_names     : List of label strings.
#             decision_threshold : Minimum confidence to commit a prediction.
#             stable_required    : Consecutive steps candidate must hold to commit.
#         """

#         self.activity_names     = activity_names or ACTIVITY_NAMES
#         self.num_classes        = num_classes
#         self.decision_threshold = decision_threshold
#         self.stable_required    = stable_required

#         # ── BeliefNode (exponential smoothing) ────────────────────────────────
#         self.belief_node = BeliefNode(
#             num_classes=num_classes,
#             alpha=alpha,
#             activity_names=self.activity_names,
#         )

#         # ── Soft Transition Matrix ────────────────────────────────────────────
#         # Row  = current confirmed state index
#         # Col  = next candidate state index
#         # Value = transition weight (0.0 blocked → 1.0 free)
#         #
#         #                   Walk  Jog   Up   Down  Sit  Stand
#         self.transition = np.array([
#             # [1, 1, 1, 1, 0, 1],   # Walking
#             # [1, 1, 1, 1, 0, 1],   # Jogging
#             # [1, 1, 1, 1, 0, 1],   # Upstairs
#             # [1, 1, 1, 1, 0, 1],   # Downstairs
#             # [0, 0, 0, 0, 1, 1],   # Sitting  (can only stay or go to Standing)
#             # [1, 0, 0, 0, 1, 1],   # Standing (can go to Walking or Sitting)
        
#             [0.6,  0.2,  0.1,  0.1,  0.0,  0.0],   # Walking
#             [0.3,  0.6,  0.0,  0.0,  0.0,  0.1],   # Jogging
#             [0.2,  0.0,  0.6,  0.0,  0.1,  0.1],   # Upstairs
#             [0.2,  0.0,  0.0,  0.6,  0.1,  0.1],   # Downstairs
#             [0.0,  0.0,  0.0,  0.0,  0.7,  0.3],   # Sitting   → only Stay/Stand
#             [0.1,  0.1,  0.1,  0.1,  0.3,  0.3],   # Standing  → gateway state
#         ], dtype=np.float32)

#         # ── State tracking ────────────────────────────────────────────────────
#         self.current_state   = None   # last committed state index
#         self._pending_state  = None   # candidate being evaluated
#         self._stable_counter = 0      # consecutive steps pending has held

#     # ──────────────────────────────────────────────────────────────────────────

#     def update(self, raw_probs: np.ndarray):
#         """
#         Run one step of the temporal FSM.

#         Args:
#             raw_probs : shape (num_classes,)
#                         Raw softmax output from ModelPredictor.predict()

#         Returns:
#             final_activity (str)        : Committed label, "Stabilizing..." or "Uncertain"
#             confidence     (float)      : Confidence of top candidate.
#             smoothed_probs (np.ndarray) : shape (num_classes,) smoothed belief.
#         """

#         raw_probs = np.asarray(raw_probs, dtype=np.float32)

#         # ── Step 1: Weight raw probs by transition row BEFORE smoothing ───────
#         # Soft weighting: impossible transitions → zeroed out
#         #                 unlikely transitions  → dampened
#         #                 likely transitions    → boosted
#         # Applying BEFORE BeliefNode keeps its internal memory clean.
#         if self.current_state is not None:
#             weights   = self.transition[self.current_state]   # shape (num_classes,)
#             raw_probs = raw_probs * weights
#             total     = raw_probs.sum()
#             if total > 0:
#                 raw_probs = raw_probs / total
#             else:
#                 # Fallback: all transitions blocked (matrix misconfiguration)
#                 raw_probs = np.ones(self.num_classes, dtype=np.float32) / self.num_classes

#         # ── Step 2: Exponential smoothing via BeliefNode ──────────────────────
#         smoothed = self.belief_node.update(raw_probs)

#         # ── Step 3: Pick candidate ────────────────────────────────────────────
#         idx        = int(np.argmax(smoothed))
#         confidence = float(smoothed[idx])

#         # ── Step 4: Stability tracking ────────────────────────────────────────
#         if idx != self._pending_state:
#             self._pending_state  = idx
#             self._stable_counter = 1
#         else:
#             self._stable_counter += 1

#         # ── Step 5: Decision rule ─────────────────────────────────────────────
#         if confidence < self.decision_threshold:
#             final_activity = "Uncertain"

#         elif self._stable_counter < self.stable_required:
#             final_activity = "Stabilizing..."

#         else:
#             self.current_state = self._pending_state
#             final_activity     = self.activity_names[idx]

#         return final_activity, confidence, smoothed.copy()

#     # ──────────────────────────────────────────────────────────────────────────

#     def reset(self):
#         """Reset all state (use between sessions or on device reconnect)."""
#         self.belief_node.reset()
#         self.current_state   = None
#         self._pending_state  = None
#         self._stable_counter = 0

#     # ──────────────────────────────────────────────────────────────────────────

#     @property
#     def smoothed_probs(self) -> np.ndarray:
#         """Current smoothed belief distribution (all classes)."""
#         return self.belief_node.belief

#     @property
#     def state_label(self) -> str:
#         """Human-readable label of the current committed state."""
#         if self.current_state is None:
#             return "Unknown"
#         return self.activity_names[self.current_state]


# # ── Quick self-test ────────────────────────────────────────────────────────────

# if __name__ == "__main__":

#     fsm = TemporalFSM(
#         num_classes=6,
#         alpha=0.6,
#         activity_names=ACTIVITY_NAMES,
#         decision_threshold=0.5,
#         stable_required=1,
#     )

#     print("=" * 60)
#     print("  TemporalFSM Self-Test  (Soft Transition Matrix)")
#     print("=" * 60)

#     # Simulate physically realistic and unrealistic transitions
#     test_sequence = [
#         ("Sitting  (step 1)",        [0.00, 0.00, 0.00, 0.00, 0.90, 0.10]),
#         ("Sitting  (step 2)",        [0.00, 0.00, 0.00, 0.00, 0.90, 0.10]),
#         ("Jog attempt (blocked)",    [0.00, 0.90, 0.00, 0.00, 0.05, 0.05]),  # 0.0 weight → zeroed
#         ("Jog attempt (blocked)",    [0.00, 0.90, 0.00, 0.00, 0.05, 0.05]),  # still zeroed
#         ("Standing (step 1)",        [0.00, 0.00, 0.00, 0.00, 0.10, 0.90]),
#         ("Standing (step 2)",        [0.00, 0.00, 0.00, 0.00, 0.10, 0.90]),
#         ("Jogging via gateway (1)",  [0.00, 0.90, 0.00, 0.00, 0.00, 0.10]),  # now allowed (0.1 weight)
#         ("Jogging via gateway (2)",  [0.00, 0.90, 0.00, 0.00, 0.00, 0.10]),
#     ]

#     for label, probs in test_sequence:
#         raw = np.array(probs, dtype=np.float32)
#         activity, confidence, smoothed = fsm.update(raw)

#         print(f"\n  Input          : {label}")
#         print(f"  Raw Probs      : {np.round(raw, 2)}")
#         print(f"  Smoothed       : {np.round(smoothed, 3)}")
#         print(f"  Output         : {activity}  ({confidence:.1%})")
#         print(f"  Stable Count   : {fsm._stable_counter}  |  "
#               f"Committed State : {fsm.state_label}")

#     print("\n" + "=" * 60)
#     fsm.reset()
#     print(f"  After reset → State: {fsm.state_label}")
