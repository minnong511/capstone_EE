#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <vector>
#include <algorithm>

#define DEVICE_NAME          "Sensor-01"

#define SERVICE_UUID         "12345678-1234-5678-1234-56789abcdef0"
#define CHAR_HEARTBEAT_UUID  "12345678-1234-5678-1234-56789abcdef1"
#define CHAR_FILE_META_UUID  "12345678-1234-5678-1234-56789abcdef2"
#define CHAR_FILE_DATA_UUID  "12345678-1234-5678-1234-56789abcdef3"

BLEServer* pServer = nullptr;
BLECharacteristic* pHeartbeatChar = nullptr;
BLECharacteristic* pMetaChar = nullptr;
BLECharacteristic* pDataChar = nullptr;

volatile bool deviceConnected = false;
static const size_t mtu_payload = 120; // 청크 크기 줄임
static constexpr float PI_F = 3.14159265358979323846f;

uint32_t lastSendTime = 0;  // 마지막 전송 시각
static uint32_t file_id = 0; // 파일 ID

// ===== WAV 생성 =====
std::vector<uint8_t> make_wav_1s_8k16_mono_tone() {
  const uint32_t sampleRate = 8000;
  const uint16_t bitsPerSample = 16;
  const uint16_t numChannels = 1;
  const uint32_t numSamples = sampleRate;
  const uint32_t byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const uint16_t blockAlign = numChannels * (bitsPerSample / 8);
  const uint32_t dataSize = numSamples * blockAlign;
  const uint32_t chunkSize = 36 + dataSize;

  std::vector<uint8_t> wav;
  wav.reserve(44 + dataSize);

  auto push_u32 = [&](uint32_t v) {
    wav.push_back(v & 0xFF);
    wav.push_back((v >> 8) & 0xFF);
    wav.push_back((v >> 16) & 0xFF);
    wav.push_back((v >> 24) & 0xFF);
  };
  auto push_u16 = [&](uint16_t v) {
    wav.push_back(v & 0xFF);
    wav.push_back((v >> 8) & 0xFF);
  };

  // RIFF header
  wav.insert(wav.end(), {'R','I','F','F'});
  push_u32(chunkSize);
  wav.insert(wav.end(), {'W','A','V','E'});
  // fmt chunk
  wav.insert(wav.end(), {'f','m','t',' '});
  push_u32(16);
  push_u16(1);
  push_u16(numChannels);
  push_u32(sampleRate);
  push_u32(byteRate);
  push_u16(blockAlign);
  push_u16(bitsPerSample);
  // data chunk
  wav.insert(wav.end(), {'d','a','t','a'});
  push_u32(dataSize);

  // 440Hz sine wave
  const float freq = 440.0f;
  for (uint32_t n = 0; n < numSamples; ++n) {
    float t = (float)n / (float)sampleRate;
    float s = sinf(2.0f * PI_F * freq * t);
    int16_t sample = (int16_t)(s * 30000);
    wav.push_back(sample & 0xFF);
    wav.push_back((sample >> 8) & 0xFF);
  }
  return wav;
}

// ===== WAV 전송 =====
void sendWav() {
  std::vector<uint8_t> wav = make_wav_1s_8k16_mono_tone();
  const size_t totalSize = wav.size();
  file_id++;

  // META 전송
  char meta[128];
  snprintf(meta, sizeof(meta),
           "{\"sensor\":\"%s\",\"mime\":\"audio/wav\",\"size\":%lu,\"id\":%lu}",
           DEVICE_NAME, (unsigned long)totalSize, (unsigned long)file_id);
  pMetaChar->setValue((uint8_t*)meta, strlen(meta));
  pMetaChar->notify();
  Serial.printf("[BLE] META sent: %s\n", meta);

  // 첫 청크 전송 전에 대기 (첫 패킷 유실 방지)
  delay(30);

  // DATA 청크 전송
  uint16_t seq = 0;
  size_t offset = 0;
  while (offset < totalSize) {
    size_t remain = totalSize - offset;
    size_t chunk = std::min<size_t>(mtu_payload, remain);
    std::vector<uint8_t> pkt;
    pkt.reserve(3 + chunk);
    pkt.push_back(seq & 0xFF);
    pkt.push_back((seq >> 8) & 0xFF);
    uint8_t flags = (chunk == remain) ? 0x01 : 0x00; // EOF
    pkt.push_back(flags);
    pkt.insert(pkt.end(), wav.begin() + offset, wav.begin() + offset + chunk);

    pDataChar->setValue(pkt.data(), pkt.size());
    pDataChar->notify();

    offset += chunk;
    seq++;
    delay(15); // 청크 간 간격 늘림
  }

  Serial.printf("[BLE] WAV sent: %lu bytes in %u chunks\n",
                (unsigned long)totalSize, (unsigned int)seq);
}

// ===== BLE 콜백 =====
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("[BLE] Central connected");
    lastSendTime = millis();
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("[BLE] Central disconnected");
    BLEDevice::startAdvertising();
  }
};

void setup() {
  Serial.begin(115200);
  delay(200);

  BLEDevice::init(DEVICE_NAME);
  BLEDevice::setMTU(247);

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  pHeartbeatChar = pService->createCharacteristic(
    CHAR_HEARTBEAT_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pHeartbeatChar->addDescriptor(new BLE2902());
  pHeartbeatChar->setValue("boot");

  pMetaChar = pService->createCharacteristic(
    CHAR_FILE_META_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  pMetaChar->addDescriptor(new BLE2902());

  pDataChar = pService->createCharacteristic(
    CHAR_FILE_DATA_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pDataChar->addDescriptor(new BLE2902());

  pService->start();

  BLEAdvertising* pAdv = BLEDevice::getAdvertising();
  pAdv->addServiceUUID(SERVICE_UUID);
  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising started");
}

void loop() {
  // Heartbeat
  static uint32_t lastBeat = 0;
  uint32_t now = millis();
  if (now - lastBeat >= 1000) {
    lastBeat = now;
    char msg[32];
    snprintf(msg, sizeof(msg), "beat:%lu", (unsigned long)now);
    pHeartbeatChar->setValue((uint8_t*)msg, strlen(msg));
    if (deviceConnected) pHeartbeatChar->notify();
  }

  // 5초마다 전송
  if (deviceConnected && (millis() - lastSendTime >= 5000)) {
    Serial.println("[BLE] Sending WAV file...");
    sendWav();
    Serial.println("[BLE] Send complete");
    lastSendTime = millis();
  }
}
