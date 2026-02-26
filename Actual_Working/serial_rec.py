import serial
import numpy as np


def collect_window(port="/dev/ttyUSB0",
                   baudrate=115200,
                   window_size=128):
    """
    Reads MPU6050 data from serial until window_size samples collected.

    Expected Arduino format per line:
        ax,ay,az,gx,gy,gz

    Returns:
        numpy array of shape (window_size, 6)
    """

    ser = serial.Serial(port, baudrate, timeout=1)

    buffer = []

    print(f"Serial connected on {port}")
    print(f"Collecting {window_size} samples...")

    while len(buffer) < window_size:

        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        try:
            values = list(map(float, line.split(',')))

            if len(values) == 6:
                buffer.append(values)

        except:
            continue

    ser.close()

    return np.array(buffer)