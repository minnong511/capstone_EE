#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define DEVICE_NAME         "Sensor-01"  // 보드별로 "Sensor-02", "Sensor-03" 등으로 변경
#define SERVICE_UUID        "12345678-1234-5678-1234-56789abcdef0"
#define CHAR_HEARTBEAT_UUID "12345678-1234-5678-1234-56789abcdef1"

BLEServer* pServer = nullptr;
BLECharacteristic* pHeartbeatChar = nullptr;
volatile bool deviceConnected = false;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("[BLE] Central connected");
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("[BLE] Central disconnected");
    BLEDevice::startAdvertising(); // 재광고
  }
};

void setup() {
  Serial.begin(115200);
  delay(200);

  BLEDevice::init(DEVICE_NAME);

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pHeartbeatChar = pService->createCharacteristic(
      CHAR_HEARTBEAT_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pHeartbeatChar->addDescriptor(new BLE2902()); // CCCD
  pHeartbeatChar->setValue("boot");

  pService->start();

  BLEAdvertising* pAdv = BLEDevice::getAdvertising();
  pAdv->addServiceUUID(SERVICE_UUID);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising started");
}

void loop() {
  static uint32_t last = 0;
  uint32_t now = millis();
  if (now - last >= 1000) {
    last = now;
    char msg[32];
    snprintf(msg, sizeof(msg), "beat:%lu", now);
    pHeartbeatChar->setValue((uint8_t*)msg, strlen(msg));
    if (deviceConnected) pHeartbeatChar->notify();
    Serial.println(msg);
  }
}
