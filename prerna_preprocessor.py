import numpy as np
import joblib
from scipy import signal


class HARPreprocessor:
    def __init__(self):
        # Load saved objects
        self.sensor_scaler = joblib.load("sensor_scaler1.pkl")
        self.context_scaler = joblib.load("context_scaler1.pkl")
        self.config = joblib.load("config1.pkl")

        self.window_size = self.config['window']
        self.sampling_rate = self.config['sr']

    # Low-pass filter 
    def _lowpass(self, data, cutoff=20):
        nyq = 0.5 * self.sampling_rate
        b, a = signal.butter(4, cutoff / nyq, btype='low')
        return signal.filtfilt(b, a, data, axis=0)

    # Gravity
    def _separate_gravity(self, acc):
        nyq = 0.5 * self.sampling_rate
        b, a = signal.butter(3, 0.3 / nyq, btype='low')
        gravity = signal.filtfilt(b, a, acc, axis=0)
        body = acc - gravity
        return gravity, body

    # Extracting 10 Feautures
    def _extract_context(self, window, gravity):
        acc = window[:, :3]
        mag = np.linalg.norm(acc, axis=1)

        jerk = np.diff(acc, axis=0)
        jerk_mag = np.linalg.norm(jerk, axis=1)

        z = acc[:, 2] - np.mean(acc[:, 2])
        zero_cross = np.sum(np.diff(np.sign(z)) != 0)
        step_freq = zero_cross / (len(acc) / self.sampling_rate)

        features = [
            np.mean(mag),
            np.std(mag),
            np.mean(jerk_mag),
            np.sum(acc**2),
            np.var(acc),
            np.sum(np.abs(acc)) / len(acc),
            np.mean(np.abs(gravity[:, 2])),
            np.mean(acc[:, 2]),
            step_freq,
            np.sum(mag < 0.1) / len(mag)
        ]

        return np.array(features)

    
    def preprocess(self, sensor_data):
        """
        sensor_data: numpy array of shape (128, 6)
        returns:
            X -> shape (1, 128, 6)
            C -> shape (1, 10)
        """

        if sensor_data.shape != (self.window_size, 6):
            raise ValueError(f"Input must be shape ({self.window_size}, 6)")

        # Filter
        filtered = self._lowpass(sensor_data)

        acc = filtered[:, :3]
        gyro = filtered[:, 3:]

        # Separate gravity
        gravity, body = self._separate_gravity(acc)

        processed = np.hstack([body, gyro])

        # Context features
        context = self._extract_context(processed, gravity)

        # Scale
        X = self.sensor_scaler.transform(
            processed.reshape(-1, 6)
        ).reshape(1, self.window_size, 6)

        C = self.context_scaler.transform(
            context.reshape(1, -1)
        )

        return X, C