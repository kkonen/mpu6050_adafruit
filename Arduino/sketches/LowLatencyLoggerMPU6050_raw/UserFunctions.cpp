// User data functions.  Modify these functions for your data items.
#include "UserTypes.h"
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"
//------------------------------------------------------------------------------
MPU6050 mpu;
int addPins[4] = {A0,A1,A2,A3};
static unsigned long startMicros;
// Acquire a data record.

void selectDevice(int dev){
  for(int i=0; i<devices; i++){
    if(i == dev){
      digitalWrite(addPins[i],LOW);
    }else{
      digitalWrite(addPins[i],HIGH);
    }
  }
}

void acquireData(data_t* data) {
  for(int i = 0; i < devices; i++){
    selectDevice(i);
    mpu.getMotion6(&(data->ax[i]), &(data->ay[i]), &(data->az[i]), &(data->gx[i]), &(data->gy[i]), &(data->gz[i]));
    data->time[i] = micros();
  }
}

// setup AVR I2C
void userSetup() {

  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties


  //SET ALL DEVICES TO ADDRESS 0x69
  for(int i=0; i<devices; i++){
    pinMode(addPins[i], OUTPUT);
    digitalWrite(addPins[i], HIGH);
  }
  
  // initialize serial communication
  Serial.begin(115200);
  while (!Serial); // wait for Leonardo enumeration, others continue immediately
  
  Serial.println(F("Initializing I2C devices..."));
  for(int i=0; i<devices; i++){
    selectDevice(i);
    // initialize device
    mpu.initialize();
    
    // verify connection
    Serial.print("Testing MPU"); Serial.print(i); Serial.print(" connection... ");
    bool connection = mpu.testConnection();
    Serial.println(connection ? F("connection successful!") : F("connection FAILED!"));
  }
}

// Print a data record.
void printData(Print* pr, data_t* data) {
  
  if (startMicros == 0) {
    startMicros = data->time[0];
  }
  for(int i = 0; i < devices; i++){
    pr->print(i);
    pr->write(",");
    pr->print(data->time[i] - startMicros);
    pr->write(',');
    pr->print(data->ax[i]);
    pr->write(',');
    pr->print(data->ay[i]);
    pr->write(',');
    pr->print(data->az[i]);
    pr->write(',');
    pr->print(data->gx[i]);
    pr->write(',');
    pr->print(data->gy[i]);
    pr->write(',');
    pr->println(data->gz[i]);
  }
}

// Print data header.
void printHeader(Print* pr) {
    startMicros = 0;
  pr->println(F("device,micros,ax,ay,az,gx,gy,gz"));
}
