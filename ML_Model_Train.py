import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy import signal
import joblib
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings("ignore")

# ==================== ACTIVITY LABELS ====================
ACTIVITIES = {
    0: 'Walking',
    1: 'Jogging',
    2: 'Upstairs',
    3: 'Downstairs',
    4: 'Sitting',
    5: 'Standing'
}

# ==================== IMU DATA PROCESSOR ====================
class IMUDataProcessor:
    def __init__(self, window_size=128, overlap=0.5, sampling_rate=50):
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate

    def apply_filters(self, data, cutoff_freq=20):
        nyq = 0.5 * self.sampling_rate
        cutoff = cutoff_freq / nyq
        b, a = signal.butter(4, cutoff, btype='low')
        return signal.filtfilt(b, a, data, axis=0)

    def separate_gravity(self, acc):
        nyq = 0.5 * self.sampling_rate
        cutoff = 0.3 / nyq
        b, a = signal.butter(3, cutoff, btype='low')
        gravity = signal.filtfilt(b, a, acc, axis=0)
        body = acc - gravity
        return gravity, body

    def create_windows(self, data, labels):
        step = int(self.window_size * (1 - self.overlap))
        X, y = [], []

        for i in range(0, len(data) - self.window_size, step):
            win = data[i:i+self.window_size]
            lab = labels[i:i+self.window_size]
            X.append(win)
            y.append(np.bincount(lab).argmax())

        return np.array(X), np.array(y)

# ==================== CONTEXT FEATURE EXTRACTOR ====================
class ContextExtractor:
    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate

    def extract_features(self, window, gravity):
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

# ==================== MODEL ====================
class ContextAwareHAR:
    def __init__(self, input_shape, context_dim, num_classes):
        self.model = self.build(input_shape, context_dim, num_classes)

    def build(self, input_shape, context_dim, num_classes):
        sensor_in = layers.Input(shape=input_shape)
        context_in = layers.Input(shape=(context_dim,))

        x = layers.Conv1D(64, 5, activation='relu', padding='same')(sensor_in)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)

        c = layers.Dense(32, activation='relu')(context_in)

        merged = layers.Concatenate()([x, c])

        x = layers.Dense(128, activation='relu')(merged)
        x = layers.Dropout(0.4)(x)
        out = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model([sensor_in, context_in], out)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

# ==================== MOTIONSENSE LOADER (FIXED) ====================
class MotionSenseLoader:
    def __init__(self, path):
        self.path = path
        self.map = {'wlk':0,'jog':1,'ups':2,'dws':3,'sit':4,'std':5}

    def load(self):
        # Find the correct data folder
        data_folder = self.path
        
        # Check if path exists
        if not os.path.exists(data_folder):
            # Try nested folder structure
            nested = os.path.join(self.path, "A_DeviceMotion_data", "A_DeviceMotion_data")
            if os.path.exists(nested):
                data_folder = nested
            else:
                raise FileNotFoundError(f"Cannot find data at {self.path}")
        
        # If the provided path already contains A_DeviceMotion_data folder, navigate to it
        if os.path.exists(os.path.join(data_folder, "A_DeviceMotion_data", "A_DeviceMotion_data")):
            data_folder = os.path.join(data_folder, "A_DeviceMotion_data", "A_DeviceMotion_data")
        elif os.path.exists(os.path.join(data_folder, "A_DeviceMotion_data")):
            # Check if this is the outer folder
            inner = os.path.join(data_folder, "A_DeviceMotion_data", "A_DeviceMotion_data")
            if os.path.exists(inner):
                data_folder = inner
            else:
                data_folder = os.path.join(data_folder, "A_DeviceMotion_data")

        print(f"📂 Loading data from: {data_folder}")
        
        data, labels = [], []
        found_files = 0

        for act in self.map:
            pattern = os.path.join(data_folder, f"{act}_*")
            folders = glob.glob(pattern)
            
            if not folders:
                print(f"⚠️  Warning: No folders found for activity '{act}'")
            
            for folder in folders:
                csv_pattern = os.path.join(folder, "sub_*.csv")
                csv_files = glob.glob(csv_pattern)
                
                for csv in csv_files:
                    try:
                        df = pd.read_csv(csv)
                        if 'userAcceleration.x' not in df.columns:
                            continue

                        d = df[['userAcceleration.x','userAcceleration.y','userAcceleration.z',
                                'rotationRate.x','rotationRate.y','rotationRate.z']].values
                        d = d[~np.isnan(d).any(axis=1)]

                        if len(d) > 0:
                            data.append(d)
                            labels.append(np.full(len(d), self.map[act]))
                            found_files += 1
                    except Exception as e:
                        print(f"❌ Error reading {csv}: {e}")
                        continue

        if not data:
            raise ValueError(f"No valid data files found in {data_folder}. Please check the path.")
        
        total_samples = len(np.vstack(data))
        print(f"✅ Successfully loaded {found_files} files with {total_samples:,} total samples")
        
        return np.vstack(data), np.hstack(labels), 50

# ==================== TRAIN PIPELINE ====================
def train_model(path):
    print("=" * 60)
    print("Context-Aware HAR System - Training")
    print("=" * 60)
    
    # Load data
    loader = MotionSenseLoader(path)
    raw, labels, sr = loader.load()

    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(raw):,}")
    print(f"   Activities: {[ACTIVITIES[i] for i in sorted(np.unique(labels))]}")
    print(f"   Sampling rate: {sr} Hz")

    # Process data
    print("\n🔄 Processing data...")
    proc = IMUDataProcessor(128, 0.5, sr)
    filt = proc.apply_filters(raw)

    acc, gyro = filt[:,:3], filt[:,3:]
    grav, body = proc.separate_gravity(acc)
    final = np.hstack([body, gyro])

    # Create windows
    print("📦 Creating windows...")
    X, y = proc.create_windows(final, labels)
    gX, _ = proc.create_windows(grav, labels)

    print(f"   Window shape: {X.shape}")
    print(f"   Number of windows: {len(X):,}")

    # Split data
    print("\n✂️  Splitting data...")
    Xtr, Xte, gtr, gte, ytr, yte = train_test_split(
        X, gX, y, test_size=0.2, stratify=y, random_state=42
    )

    # Extract features
    print("🔍 Extracting context features...")
    ext = ContextExtractor(sr)
    Ctr = np.array([ext.extract_features(w,g) for w,g in zip(Xtr,gtr)])
    Cte = np.array([ext.extract_features(w,g) for w,g in zip(Xte,gte)])

    # Scale data
    print("⚖️  Scaling data...")
    ss = StandardScaler()
    cs = StandardScaler()

    Xtr = ss.fit_transform(Xtr.reshape(-1,6)).reshape(Xtr.shape)
    Xte = ss.transform(Xte.reshape(-1,6)).reshape(Xte.shape)

    Ctr = cs.fit_transform(Ctr)
    Cte = cs.transform(Cte)

    # Build and train model
    print("\n🏗️  Building model...")
    model = ContextAwareHAR(Xtr.shape[1:], Ctr.shape[1], len(ACTIVITIES))
    
    print("\n🚀 Training model...")
    print("-" * 60)
    history = model.model.fit([Xtr, Ctr], ytr, epochs=30, batch_size=64,
                    validation_split=0.2, verbose=1)

    # Evaluate
    print("\n📈 Evaluating model...")
    yp = np.argmax(model.model.predict([Xte, Cte]), axis=1)
    
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(yte, yp, 
                                target_names=[ACTIVITIES[i] for i in range(len(ACTIVITIES))]))
    
    print("\n" + "=" * 60)
    print("Confusion Matrix:")
    print("=" * 60)
    print(confusion_matrix(yte, yp))

    # Save model
    print("\n💾 Saving model and scalers...")
    model.model.save("har_model1.h5")
    joblib.dump(ss, "sensor_scaler1.pkl")
    joblib.dump(cs, "context_scaler1.pkl")
    joblib.dump({'window':128,'sr':sr}, "config1.pkl")
    
    print("✅ Model saved successfully!")
    print("   - har_model1.h5")
    print("   - sensor_scaler1.pkl")
    print("   - context_scaler1.pkl")
    print("   - config1.pkl")

    # Plot training history
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("📊 Training plot saved as 'training_history.png'")
    except Exception as e:
        print(f"⚠️  Could not save plot: {e}")

# ==================== REAL-TIME PREDICTION ====================
def predict_activity(sensor_data):
    """
    Predict activity from real-time sensor data
    
    Args:
        sensor_data: numpy array of shape (128, 6) 
                    [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    
    Returns:
        activity_name: predicted activity
        confidence: prediction confidence
    """
    model = keras.models.load_model("har_model1.h5")
    ss = joblib.load("sensor_scaler1.pkl")
    cs = joblib.load("context_scaler1.pkl")
    cfg = joblib.load("config1.pkl")

    proc = IMUDataProcessor(cfg['window'], sampling_rate=cfg['sr'])
    ext = ContextExtractor(cfg['sr'])

    f = proc.apply_filters(sensor_data)
    acc, gyro = f[:,:3], f[:,3:]
    grav, body = proc.separate_gravity(acc)
    proc_data = np.hstack([body, gyro])

    context = ext.extract_features(proc_data, grav)

    X = ss.transform(proc_data.reshape(-1,6)).reshape(1,128,6)
    C = cs.transform(context.reshape(1,-1))

    p = model.predict([X,C], verbose=0)[0]
    idx = np.argmax(p)

    return ACTIVITIES[idx], float(p[idx])

# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Context-Aware Human Activity Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py --path motion-sense-data/
  python chat.py --path A_DeviceMotion_data/A_DeviceMotion_data
  python chat.py --real --data motionsense --path A_DeviceMotion_data/A_DeviceMotion_data

Download MotionSense Dataset:
  https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
        """
    )
    
    parser.add_argument(
        '--path', 
        type=str, 
        default='motion-sense-data/',
        help='Path to the MotionSense dataset folder (default: motion-sense-data/)'
    )
    
    parser.add_argument(
        '--real', 
        action='store_true',
        help='Enable real-time prediction mode (for future use)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='motionsense',
        help='Dataset name (default: motionsense)'
    )
    
    args = parser.parse_args()
    
    print(f"\n🔍 Configuration:")
    print(f"   Data path: {args.path}")
    print(f"   Dataset: {args.data}")
    print(f"   Real-time mode: {args.real}")
    print()
    
    # Train model
    try:
        train_model(args.path)
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
