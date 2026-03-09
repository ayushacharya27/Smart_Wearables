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
            score = probs[s] * transition_prob

            scores.append(score)

        best_state = int(np.argmax(scores))

        self.previous_state = best_state
        self.history.append(best_state)

        # keep last 10 states
        if len(self.history) > 10:
            self.history.pop(0)

        return best_state