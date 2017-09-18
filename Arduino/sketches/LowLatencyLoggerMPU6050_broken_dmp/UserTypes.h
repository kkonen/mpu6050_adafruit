#ifndef UserTypes_h
#define UserTypes_h
#include "Arduino.h"

const int devices = 1;

#define FILE_BASE_NAME "mpuraw"
struct data_t {
  uint32_t time[devices];
  uint8_t fifoBuffer[devices][64];
  uint16_t fifoCount[devices];
  uint16_t fifoCountAfter[devices];
  
};
void acquireData(data_t* data);
void printData(Print* pr, data_t* data);
void printHeader(Print* pr);
void userSetup();
void dmpDataReady();
#endif  // UserTypes_h
