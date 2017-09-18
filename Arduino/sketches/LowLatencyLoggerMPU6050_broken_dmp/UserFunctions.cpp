// User data functions.  Modify these functions for your data items.
#include "UserTypes.h"
#include "Wire.h"
#include "I2Cdev.h"
#include "MPU6050_6Axis_MotionApps20.h"
//------------------------------------------------------------------------------
MPU6050 mpu;
static uint32_t startMicros[devices];
static uint32_t e;
// MPU control/status vars

uint8_t devStatus[devices];      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount[devices];     // count of all bytes currently in FIFO
uint8_t fifoBuffer[devices][64]; // FIFO storage buffer
int addPins[4] = {A0,A1,A2,A3};


Quaternion q;


void selectDevice(int dev){
  for(int i=0; i<devices; i++){
    if(i == dev){
      digitalWrite(addPins[i],LOW);
    }else{
      digitalWrite(addPins[i],HIGH);
    }
  }
}


// Acquire a data record.
void acquireData(data_t* data) {

  for(int i = 0; i < devices; i++){
    //selectDevice(i);
    fifoCount[i] = mpu.getFIFOCount();
    if(fifoCount[i] >= packetSize){
        mpu.getFIFOBytes(data->fifoBuffer[i], packetSize);
        data->fifoCount[i] = fifoCount[i];
        data->fifoCountAfter[i] = mpu.getFIFOCount();
        data->time[i] = micros();
      } else {        
        data->time[i] = micros();
        data->fifoCount[i] = fifoCount[i];        
        data->fifoCountAfter[i] = mpu.getFIFOCount();
    }
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

    if(connection){
      // load and configure the DMP
      Serial.println(F("   Initializing DMP..."));
      devStatus[i] = mpu.dmpInitialize();
  
      // supply your own gyro offsets here, scaled for min sensitivity
      mpu.setXGyroOffset(220);
      mpu.setYGyroOffset(76);
      mpu.setZGyroOffset(-85);
      mpu.setZAccelOffset(1788); // 1688 factory default for my test chip
  
      // make sure it worked (returns 0 if so)
      if (devStatus[i] == 0) {
          // turn on the DMP, now that it's ready
          Serial.println(F("   Enabling DMP..."));
          mpu.setDMPEnabled(true);  
          // get expected DMP packet size for later comparison
          packetSize = mpu.dmpGetFIFOPacketSize();
      } else {
          // ERROR!
          // 1 = initial memory load failed
          // 2 = DMP configuration updates failed
          // (if it's going to break, usually the code will be 1)
          Serial.print(F("   DMP Initialization failed (code "));
          Serial.print(devStatus[i]);
          Serial.println(F(")"));
      }
    }

    

  
    //next dev
  }

  // wait for ready
  Serial.println(packetSize);
  Serial.println(F("\nSend any character to begin data collection: "));
  while (Serial.available() && Serial.read()); // empty buffer
  while (!Serial.available());                 // wait for data
  while (Serial.available() && Serial.read()); // empty buffer again

  for(int i=0; i<devices; i++){
    selectDevice(i);
    mpu.resetFIFO();
  }
  selectDevice(0);
}

// Print a data record.
void printData(Print* pr, data_t* data) {
  for(int i = 0; i < devices; i++){
    pr->print(i);
    pr->write(" ");
    if (startMicros[i] == 0) {
      startMicros[i] = data->time[i];
    }
    mpu.dmpGetQuaternion(&q, data->fifoBuffer[i]);
    pr->print(data->time[i] - startMicros[i]);
    pr->write(',');
    pr->print(q.w);
    pr->write(',');
    pr->print(q.x);
    pr->write(',');
    pr->print(q.y);
    pr->write(',');
    pr->print(q.z);
    pr->write('|');
    pr->print(data->fifoCount[i]);
    pr->write(',');
    pr->print(data->fifoCountAfter[i]);
    pr->write(' : ');
  }
  pr->println();
}

// Print data header.
void printHeader(Print* pr) {
  startMicros[0] = 0;
  pr->println(F("time, qw, qx, qy, qz, fifoCount"));
}
