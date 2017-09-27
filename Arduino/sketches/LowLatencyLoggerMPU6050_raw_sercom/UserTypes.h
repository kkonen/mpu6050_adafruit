#ifndef UserTypes_h
#define UserTypes_h
#include "Arduino.h"
#define FILE_BASE_NAME "realdata"

const int devices = 1;

struct data_t {
  unsigned long time[devices];
  int16_t ax[devices];
  int16_t ay[devices];
  int16_t az[devices];
  int16_t gx[devices];
  int16_t gy[devices];
  int16_t gz[devices];
};
void acquireData(data_t* data);
void printData(Print* pr, data_t* data);
void printHeader(Print* pr);
void userSetup();
#endif  // UserTypes_h
