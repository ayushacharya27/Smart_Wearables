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




// #include <Wire.h>
// #include <MPU6050.h>

// MPU6050 mpu;

// const int SAMPLE_RATE_HZ = 50;
// unsigned long lastTime = 0;
// const unsigned long interval = 1000 / SAMPLE_RATE_HZ;

// // void setup() {
// //   Wire.begin();
// //   Serial.begin(115200);

// //   mpu.initialize();

// //   if (!mpu.testConnection()) {
// //     while (1); 
// //   }
// // }
// void setup() {
//   Wire.begin();
//   Serial.begin(115200);
//   delay(1000); // Give Serial time to connect
  
//   Serial.println("System Check: Start");

//   mpu.initialize();
//   Serial.println("Sensor Initialized...");

//   if (!mpu.testConnection()) {
//     Serial.println("ERROR: MPU6050 connection failed! Check wiring or I2C address.");
//     // while (1); // Comment this out for now so you can see if the loop starts anyway
//   } else {
//     Serial.println("SUCCESS: MPU6050 connected.");
//   }
// }

// void loop() {

//   if (millis() - lastTime >= interval) {
//     lastTime = millis();

//     int16_t ax, ay, az;
//     int16_t gx, gy, gz;

//     mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

//     // Convert to proper units
//     float accX = ax / 16384.0;  
//     float accY = ay / 16384.0;
//     float accZ = az / 16384.0;

//     float gyroX = gx / 131.0;   
//     float gyroY = gy / 131.0;
//     float gyroZ = gz / 131.0;

//     //CSV format
//     Serial.print(accX, 6);
//     Serial.print(",");
//     Serial.print(accY, 6);
//     Serial.print(",");
//     Serial.print(accZ, 6);
//     Serial.print(",");
//     Serial.print(gyroX, 6);
//     Serial.print(",");
//     Serial.print(gyroY, 6);
//     Serial.print(",");
//     Serial.println(gyroZ, 6);
//   }
// }