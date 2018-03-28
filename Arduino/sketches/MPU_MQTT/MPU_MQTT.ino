#include <WiFi.h>
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"

#define AIO_SERVER      "192.168.4.2"
#define AIO_SERVERPORT  1883                   // use 8883 for SSL
//#define AIO_USERNAME    "...your AIO username (see https://accounts.adafruit.com)..."
//#define AIO_KEY         "...your AIO key..."

MPU6050 mpu;
int gyro_sen = MPU6050_GYRO_FS_2000;
int acc_sen = MPU6050_ACCEL_FS_16;
int offsets[6] = {-2719,2506,4818,186,45,8};

const char *ssid = "ESP32ap";
const char *password = "qweqweqwe";

WiFiClient client;
Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, "test", "", "");
Adafruit_MQTT_Publish publisher = Adafruit_MQTT_Publish(&mqtt, "espdata");

struct data_t {
  unsigned long time;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
};

void acquireData(data_t* data) {
    mpu.getMotion6(&(data->ax), &(data->ay), &(data->az), &(data->gx), &(data->gy), &(data->gz));
    data->time = micros();
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.print("Configuring access point...");
  /* You can remove the password parameter if you want the AP to be open. */
  WiFi.softAP(ssid, password);

  IPAddress myIP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(myIP);


  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties

  // initialize serial communication
//  Serial.begin(115200);
//  while (!Serial); // wait for Leonardo enumeration, others continue immediately
  
  Serial.println(F("Initializing I2C device..."));
 
    // initialize device
    mpu.initialize();
    
    // verify connection
    Serial.print("Testing MPU"); Serial.print(" connection... ");
    bool connection = mpu.testConnection();
    Serial.println(connection ? F("connection successful!") : F("connection FAILED!"));

    mpu.setFullScaleGyroRange(gyro_sen);
    mpu.setFullScaleAccelRange(acc_sen);
    mpu.setXAccelOffset(offsets[0]);
    mpu.setYAccelOffset(offsets[1]);
    mpu.setZAccelOffset(offsets[2]);
    mpu.setXGyroOffset(offsets[3]);
    mpu.setYGyroOffset(offsets[4]);
    mpu.setZGyroOffset(offsets[5]);
}

void loop() {
  MQTT_connect();
  
  data_t data;
  acquireData(&data);
  /*Serial.print("DATA: ");
  Serial.print(data.time);
  Serial.print(" ax: ");
  Serial.print(data.ax);
  Serial.print(" ay: ");
  Serial.print(data.ay);
  Serial.print(" az: ");
  Serial.print(data.az);
  Serial.print(" gx: ");
  Serial.print(data.gx);
  Serial.print(" gy: ");
  Serial.print(data.gy);
  Serial.print(" gz: ");
  Serial.println(data.gz);*/
  publisher.publish((uint8_t*)&data, sizeof(struct data_t));
  
}

void MQTT_connect() {
  int8_t ret;

  // Stop if already connected.
  if (mqtt.connected()) {
    return;
  }
  Serial.print("Connecting to MQTT... ");

  uint8_t retries = 200;
  while ((ret = mqtt.connect()) != 0) { // connect will return 0 for connected
       Serial.println(mqtt.connectErrorString(ret));
       Serial.println("Retrying MQTT connection in 5 seconds...");
       mqtt.disconnect();
       delay(5000);  // wait 5 seconds
       retries--;
       if (retries == 0) {
         // basically die and wait for WDT to reset me
         while (1);
       }
  }
  Serial.println("MQTT Connected!");
}
