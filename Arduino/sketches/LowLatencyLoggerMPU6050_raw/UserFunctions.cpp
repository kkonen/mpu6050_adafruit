// User data functions.  Modify these functions for your data items.
#include "UserTypes.h"
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050.h"
//------------------------------------------------------------------------------
MPU6050 mpu;
int addPins[4] = {A0,A1,A2,A3};



int gyro_sen[4] = {MPU6050_GYRO_FS_2000,
                   MPU6050_GYRO_FS_2000,
                   MPU6050_GYRO_FS_2000,
                   MPU6050_GYRO_FS_2000};
int acc_sen[4] = {MPU6050_ACCEL_FS_16,
                   MPU6050_ACCEL_FS_16,
                   MPU6050_ACCEL_FS_16,
                   MPU6050_ACCEL_FS_16};

int offsets[4][6] = {{-2719,2506,4818,186,45,8},
                     {1129,-1971,4966,-28,-1,2},
                     {-2627,-1065,5137,110,-19,0},
                     {-2627,-1065,5137,110,-19,0}};


/*
int offsets[4][6] = {{-2737,2478,4808,170,41,9},
                     {-1142,-2002,5025,-34,7,9},
                     {-2672,-1114,5193,119,-30,-4},
                     {-2672,-1114,5193,119,-30,-4}};
*/
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
//  Serial.begin(115200);
//  while (!Serial); // wait for Leonardo enumeration, others continue immediately
  
  Serial.println(F("Initializing I2C devices..."));
  for(int i=0; i<devices; i++){
    selectDevice(i);
    // initialize device
    mpu.initialize();
    
    // verify connection
    Serial.print("Testing MPU"); Serial.print(i); Serial.print(" connection... ");
    bool connection = mpu.testConnection();
    Serial.println(connection ? F("connection successful!") : F("connection FAILED!"));

    mpu.setFullScaleGyroRange(gyro_sen[i]);
    mpu.setFullScaleAccelRange(acc_sen[i]);
    mpu.setXAccelOffset(offsets[i][0]);
    mpu.setYAccelOffset(offsets[i][1]);
    mpu.setZAccelOffset(offsets[i][2]);
    mpu.setXGyroOffset(offsets[i][3]);
    mpu.setYGyroOffset(offsets[i][4]);
    mpu.setZGyroOffset(offsets[i][5]);

    
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
