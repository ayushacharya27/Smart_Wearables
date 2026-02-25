## Model Requirement

### Input:
It Takes IMU Data as well as context features


#### IMU Data
```bash
(1, 128, 6) 6 -> Feautures per Sample
128 time samples
6 features per sample: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
```

#### Context Data
```bash
(1, 10) 10 -> Feautures per Sample
[
 mean(|acc|),
 std(|acc|),
 mean(jerk),
 energy,
 variance,
 mean absolute acc,
 mean gravity_z,
 mean acc_z,
 step_freq,
 low_motion_ratio
]
```

##### How this is Calculated:

First Window is what's the IMU is Sending at each time interval:
```bash
t0  ax ay az gx gy gz
t1  ax ay az gx gy gz
t2  ax ay az gx gy gz
t3  ax ay az gx gy gz
...
```

Total Accelaration:
```bash
acc = window[:, :3] # First 3 Values i.e. ax, ay, az
mag = np.linalg.norm(acc, axis=1) # mag = mag = sqrt(ax² + ay² + az²)
```

Now we can calculate the 6 Context Feautures:
```python
# mean(|acc|)
np.std(mag)

# std(|acc|)
np.std(mag) # It gives Deviation, means standing -> Low Deviation, Jogging -> High Deviation

# mean(jerk) 
jerk = np.diff(acc, axis=0) # Difference b/w the Accelaration (jerk = a(t+1) - a(t))
jerk_mag = np.linalg.norm(jerk, axis=1) # Magnitude
np.mean(jerk_mag) # High Jerk -> Running........

# energy
np.sum(acc**2) # Energy = Σ(ax² + ay² + az²), Jogging(Energy)>Sitting(Energy)

# variance
np.var(acc) # More spread = more dynamic movement

# mean absolute acc
np.sum(np.abs(acc)) / len(acc) # Finds Average Accelaration
```

#### 7. mean gravity_z
```bash
np.mean(np.abs(gravity[:, 2]))
```

#### Now what's Gravity??? Fucking Hell Bruh why do we need this shit
```bash
a_total(t) = a_body(t) + g(t) # Always
```
#### If Standing Still
```bash
a_total(t) = 9.8m/s^2 # Since this part changes very slowly so we seperate it using a low pass filter
a_total = motion + 9.81 # When Moving
```
#### Low pass filter
Very Slow Frequency to catch very slowly changing components, thus extracting gravity in a array

```bash
gravity.shape = (128, 3)

gravity[:,0] → gravity_x
gravity[:,1] → gravity_y
gravity[:,2] → gravity_z
```
```python
# Extracting and taking mean
gravity_z = gravity[:, 2]
```

#### 8. Other Context Features
```python
# mean acc_z, Its Easy
```

#### 9. step_freq
Calculates How Frequently Steps are Taken

```python
z = acc[:, 2] - np.mean(acc[:, 2])
zero_cross = np.sum(np.diff(np.sign(z)) != 0)
step_freq = zero_cross / (len(acc) / sampling_rate)
```
How many times the z value has crossed 0, thus calculating steps


#### 10. low_motion_ratio
Counts how many samples have very low acceleration.
```python
np.sum(mag < 0.1) / len(mag)
```
If sitting -> Nearly 0, if Jogging none are near 0.
















