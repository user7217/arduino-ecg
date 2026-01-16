#include <SPI.h>
#include "protocentral_max30003.h"

#define MAX30003_CS_PIN 10
#define MEGA_SS_PIN 53 

MAX30003 max30003(MAX30003_CS_PIN);

void forceRegisterWrite(uint8_t address, uint32_t data) {
    digitalWrite(MAX30003_CS_PIN, LOW); 
    SPI.transfer((address << 1) | 0); 
    SPI.transfer((data >> 16) & 0xFF);
    SPI.transfer((data >> 8) & 0xFF);
    SPI.transfer(data & 0xFF);
    digitalWrite(MAX30003_CS_PIN, HIGH);
    delay(10);
}

void setup() {
    Serial.begin(57600); // Fast communication
    
    pinMode(MEGA_SS_PIN, OUTPUT);
    digitalWrite(MEGA_SS_PIN, HIGH);
    pinMode(MAX30003_CS_PIN, OUTPUT);
    digitalWrite(MAX30003_CS_PIN, HIGH); 

    SPI.begin();
    SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
    
    max30003.begin();
    
    // --- REAL MODE SETUP ---
    forceRegisterWrite(0x00, 0x000000); // Reset
    delay(100);
    forceRegisterWrite(0x10, 0x081007); // Enable ECG + RBias
    delay(50);
    forceRegisterWrite(0x15, 0x805000); // Rate 128, Gain 20x
    delay(50);
    forceRegisterWrite(0x14, 0x000000); // Connect Pads
    delay(50);
    forceRegisterWrite(0x00, 0x000000); // Sync
}

void loop() {
    int32_t sample = 0;
    if (max30003.readEcgSample(sample)) {
        Serial.println(sample); // Just send raw numbers
    }
    delay(4); // ~200 samples per second
}