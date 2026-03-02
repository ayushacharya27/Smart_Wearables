#include <Wire.h>
#include <MPU6050.h>

#define SDA_PIN 8
#define SCL_PIN 9

MPU6050 mpu;

void setup() {

  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  mpu.initialize();

  // DO NOT USE testConnection() for MPU9250
  Serial.println("MPU9250 Ready");
}

void loop() {

  int16_t ax, ay, az;
  int16_t gx, gy, gz;

  mpu.getMotion6(&ax,&ay,&az,&gx,&gy,&gz);

  Serial.print(ax/16384.0); Serial.print(",");
  Serial.print(ay/16384.0); Serial.print(",");
  Serial.print(az/16384.0); Serial.print(",");
  Serial.print(gx/131.0);   Serial.print(",");
  Serial.print(gy/131.0);   Serial.print(",");
  Serial.println(gz/131.0);

  delay(20); // ~50Hz
}