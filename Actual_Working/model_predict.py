# agents/model_predict.py

import numpy as np
from tensorflow import keras


class ModelPredictor:
    """
    Wraps trained HAR model.
    Responsible ONLY for neural inference.
    """

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    

    def predict(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Inputs:
            X → shape (1, 128, 6)
            C → shape (1, 10)

        Returns:
            raw_probs → shape (6,)
        """

        preds = self.model.predict([X, C], verbose=0)

        # Return flattened softmax vector
        return preds[0]