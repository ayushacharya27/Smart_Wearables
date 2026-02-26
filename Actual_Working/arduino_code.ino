#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

const int SAMPLE_RATE_HZ = 50;
unsigned long lastTime = 0;
const unsigned long interval = 1000 / SAMPLE_RATE_HZ;

void setup() {
  Wire.begin();
  Serial.begin(115200);

  mpu.initialize();

  if (!mpu.testConnection()) {
    while (1); 
  }
}

void loop() {

  if (millis() - lastTime >= interval) {
    lastTime = millis();

    int16_t ax, ay, az;
    int16_t gx, gy, gz;

    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

    // Convert to proper units
    float accX = ax / 16384.0;  
    float accY = ay / 16384.0;
    float accZ = az / 16384.0;

    float gyroX = gx / 131.0;   
    float gyroY = gy / 131.0;
    float gyroZ = gz / 131.0;

    //CSV format
    Serial.print(accX, 6);
    Serial.print(",");
    Serial.print(accY, 6);
    Serial.print(",");
    Serial.print(accZ, 6);
    Serial.print(",");
    Serial.print(gyroX, 6);
    Serial.print(",");
    Serial.print(gyroY, 6);
    Serial.print(",");
    Serial.println(gyroZ, 6);
  }
}