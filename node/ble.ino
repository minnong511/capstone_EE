#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <memory>
#include <cstring>

// ====== ESP-IDF I2S driver (Arduino core exposes it) ======
extern "C" {
  #include "driver/i2s.h"
}

// ---------------- BLE identity / UUIDs (기존 유지) ----------------
#define DEVICE_NAME          "Sensor-01"
#define SERVICE_UUID         "12345678-1234-5678-1234-56789abcdef0"
#define CHAR_HEARTBEAT_UUID  "12345678-1234-5678-1234-56789abcdef1"
#define CHAR_FILE_META_UUID  "12345678-1234-5678-1234-56789abcdef2"
#define CHAR_FILE_DATA_UUID  "12345678-1234-5678-1234-56789abcdef3"

// ---------------- BLE globals (기존 유지) ----------------
BLEServer* pServer = nullptr;
BLECharacteristic* pHeartbeatChar = nullptr;
BLECharacteristic* pMetaChar = nullptr;
BLECharacteristic* pDataChar = nullptr;

volatile bool deviceConnected = false;
static const size_t mtu_payload = 120; // 필요시 160~200 등으로 조정
uint32_t lastSendTime = 0;             // (사용 안 하지만 기존 변수 유지)
static uint32_t file_id = 0;           // 파일 ID (증가)

// 전송 안정화용 상태
static volatile bool g_sending = false;     // 전송 중이면 true → 녹음/트리거 정지
static uint32_t g_connectedAtMs = 0;        // 연결 직후 쿨다운 타이밍

// ===========================================================
//                Audio / Trigger Configuration
// ===========================================================
// ESP32-WROOM 권장 핀 (INMP441)
#define I2S_PORT            I2S_NUM_0
#define I2S_WS_PIN          25   // LRCLK / WS
#define I2S_SD_PIN          32   // SD (mic DOUT) -> ESP32 input
#define I2S_SCK_PIN         26   // BCLK

static const int   SAMPLE_RATE       = 16000;   // Hz
static const int   NUM_CHANNELS      = 1;       // mono
static const int   BITS_PER_SAMPLE   = 16;      // 저장은 16-bit PCM

// dBFS 트리거 파라미터 (민감도 ↑)
static const float CALIB_SEC         = 10.0f;   // 주변 소음 캘리브레이션 시간
static const float HOP_SEC           = 0.032f;  // RMS 측정 블록 (32ms)
static const float RECORD_SEC        = 2.0f;    // 트리거 시 녹음 길이
static const float THRESH_DB_ABOVE   = 3.0f;    // ambient + 3 dBFS
static const uint16_t MIN_BLOCKS_OVER= 1;       // 연속 1블록 넘으면 트리거
static const uint32_t REFRACT_MS     = 1500;    // 전송 후 불응기

// 파생 크기/버퍼
static const size_t HOP_SAMPLES = (size_t)(SAMPLE_RATE * HOP_SEC);
static const size_t REC_SAMPLES = (size_t)(SAMPLE_RATE * RECORD_SEC);
static std::vector<int16_t> g_block(HOP_SAMPLES);
static std::vector<int16_t> g_pcm(REC_SAMPLES);

// 상태
static float    g_ambient_db = -80.0f;
static uint16_t g_overCount  = 0;
static uint32_t g_lastSendMs = 0;

// ===========================================================
//                        WAV Helpers
// ===========================================================
#pragma pack(push, 1)
struct WavHeader {
  char     riff[4];
  uint32_t chunkSize;
  char     wave[4];
  char     fmt[4];
  uint32_t subchunk1Size;
  uint16_t audioFormat;
  uint16_t numChannels;
  uint32_t sampleRate;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
  char     data[4];
  uint32_t dataSize;
};
#pragma pack(pop)

static inline void makeWavHeader(WavHeader& h, uint32_t pcmBytes) {
  memcpy(h.riff, "RIFF", 4);
  memcpy(h.wave, "WAVE", 4);
  memcpy(h.fmt , "fmt ", 4);
  memcpy(h.data, "data", 4);
  h.subchunk1Size = 16;
  h.audioFormat   = 1; // PCM
  h.numChannels   = NUM_CHANNELS;
  h.sampleRate    = SAMPLE_RATE;
  h.bitsPerSample = BITS_PER_SAMPLE;
  h.byteRate      = SAMPLE_RATE * NUM_CHANNELS * (BITS_PER_SAMPLE/8);
  h.blockAlign    = NUM_CHANNELS * (BITS_PER_SAMPLE/8);
  h.dataSize      = pcmBytes;
  h.chunkSize     = 36 + h.dataSize;
}

static void buildWavFromPCM(const int16_t* pcm, size_t samples, std::vector<uint8_t>& out) {
  const uint32_t pcmBytes = (uint32_t)(samples * sizeof(int16_t));
  WavHeader hdr; makeWavHeader(hdr, pcmBytes);
  out.resize(sizeof(WavHeader) + pcmBytes);
  memcpy(out.data(), &hdr, sizeof(WavHeader));
  memcpy(out.data() + sizeof(WavHeader), pcm, pcmBytes);
}

// ===========================================================
//                     I2S Helpers (INMP441)
// ===========================================================
static bool i2sInit() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT; // 24-bit in 32-bit container
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;  // Left 채널 사용
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256; // frames per DMA buf
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;

  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;

  i2s_pin_config_t pins = {};
  pins.mck_io_num = I2S_PIN_NO_CHANGE;
  pins.bck_io_num = I2S_SCK_PIN;   // 26
  pins.ws_io_num  = I2S_WS_PIN;    // 25
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num  = I2S_SD_PIN;  // 32
  if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) return false;

  if (i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, (i2s_channel_t)NUM_CHANNELS) != ESP_OK) return false;
  return true;
}

// 32-bit 샘플(상위 24 유효) → 16-bit PCM 변환하여 dst에 채움
static size_t i2sReadToPCM16(int16_t* dst, size_t samplesWanted) {
  const size_t CHUNK = 256;
  int32_t tmp32[CHUNK];
  size_t samplesFilled = 0, bytesRead = 0;

  while (samplesFilled < samplesWanted) {
    size_t need = samplesWanted - samplesFilled;
    size_t toRead = (need > CHUNK) ? CHUNK : need;
    if (i2s_read(I2S_PORT, (void*)tmp32, toRead * sizeof(int32_t), &bytesRead, portMAX_DELAY) != ESP_OK) break;
    size_t got = bytesRead / sizeof(int32_t);
    for (size_t i = 0; i < got; ++i) {
      int32_t s = tmp32[i] >> 8;          // 32->24 비트 정렬
      dst[samplesFilled++] = (int16_t)s;  // 24->16 (truncate)
    }
    if (got == 0) break;
  }
  return samplesFilled;
}

static float rmsDBFS(const int16_t* x, size_t n) {
  if (n == 0) return -120.0f;
  double sumsq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double v = (double)x[i];
    sumsq += v * v;
  }
  double mean = sumsq / (double)n;
  double rms = sqrt(mean);
  double d = rms / 32768.0 + 1e-12; // log(0) 방지
  return (float)(20.0 * log10(d));
}

// ===========================================================
//                 BLE 안정화: CCCD 구독 확인 (수정판)
// ===========================================================
static bool notificationsEnabled(BLECharacteristic* ch) {
  BLEDescriptor* d = ch->getDescriptorByUUID(BLEUUID((uint16_t)0x2902));
  if (!d) return false;

  // NimBLE 사용 시: BLE2902가 getNotifications()/getIndications() 제공
  #if defined(CONFIG_BT_NIMBLE_ENABLED)
    BLE2902* cccd = (BLE2902*)d;
    if (!cccd) return false;
    return cccd->getNotifications();   // notify bit
  #else
    // 블루드로이드 계열 등: 원시 2바이트 값에서 notify bit(LSB) 확인
    uint8_t* raw = d->getValue();      // 보통 2 bytes: [0]=notify, [1]=indicate
    if (raw == nullptr) return false;
    // 길이 API가 없는 경우가 있어 보수적으로 1바이트만 체크
    return (raw[0] & 0x01) != 0;
  #endif
}
// ===========================================================
//                 BLE META/DATA send (호환 유지)
// ===========================================================
static void sendWavBuffer(const uint8_t* wav, size_t totalSize) {
  if (!deviceConnected) return;
  if (!notificationsEnabled(pMetaChar) || !notificationsEnabled(pDataChar)) {
    Serial.println("[BLE] notify not enabled; skip send");
    return;
  }

  g_sending = true;                     // ---- 전송 시작: 녹음/트리거 정지
  file_id++;

  // META 전송
  char meta[128];
  snprintf(meta, sizeof(meta),
           "{\"sensor\":\"%s\",\"mime\":\"audio/wav\",\"size\":%lu,\"id\":%lu}",
           DEVICE_NAME, (unsigned long)totalSize, (unsigned long)file_id);
  pMetaChar->setValue((uint8_t*)meta, strlen(meta));
  pMetaChar->notify();
  Serial.printf("[BLE] META sent: %s\n", meta);

  delay(80); // 수신측 준비시간 약간 부여

  // DATA 청크: seq_lo | seq_hi | flags | payload
  uint16_t seq = 0;
  size_t offset = 0;
  while (offset < totalSize && deviceConnected) {
    size_t remain = totalSize - offset;
    size_t chunk  = std::min<size_t>(mtu_payload, remain);

    std::vector<uint8_t> pkt;
    pkt.reserve(3 + chunk);
    pkt.push_back(seq & 0xFF);
    pkt.push_back((seq >> 8) & 0xFF);
    uint8_t flags = (chunk == remain) ? 0x01 : 0x00; // EOF
    pkt.push_back(flags);

    pkt.insert(pkt.end(), wav + offset, wav + offset + chunk);

    pDataChar->setValue(pkt.data(), pkt.size());
    pDataChar->notify();

    offset += chunk;
    seq++;
    delay(20);  // BLE 스택 여유
    yield();    // 워치독 양보
  }

  Serial.printf("[BLE] WAV sent: %lu bytes in %u chunks\n",
                (unsigned long)totalSize, (unsigned int)seq);
  g_sending = false;                    // ---- 전송 종료
}

// ===========================================================
//                       BLE Callbacks
// ===========================================================
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    g_connectedAtMs = millis();           // 연결 직후 시각 기록 (쿨다운)
    Serial.println("[BLE] Central connected");
    lastSendTime = millis();
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    g_sending = false;                    // 안전하게 해제
    Serial.println("[BLE] Central disconnected");
    BLEDevice::startAdvertising();
  }
};

// ===========================================================
//                        Setup / Loop
// ===========================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  // ------ BLE (기존 코드 유지) ------
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

  // ------ I2S + dBFS 캘리브레이션 ------
  if (!i2sInit()) {
    Serial.println("[I2S] init failed");
    while (1) { delay(1000); }
  }

  // FIFO 프라임
  (void)i2sReadToPCM16(g_block.data(), HOP_SAMPLES);

  // 주변 소음 평균 dBFS 측정
  const size_t blocks = (size_t)(CALIB_SEC / HOP_SEC);
  double sumdb = 0.0;
  for (size_t b = 0; b < blocks; ++b) {
    size_t got = i2sReadToPCM16(g_block.data(), HOP_SAMPLES);
    sumdb += rmsDBFS(g_block.data(), got);
  }
  g_ambient_db = (float)(sumdb / (double)blocks);
  Serial.printf("[I2S] Ambient: %.1f dBFS, threshold: %.1f (+%.1f)\n",
                g_ambient_db, g_ambient_db + THRESH_DB_ABOVE, THRESH_DB_ABOVE);
}

void loop() {
  // Heartbeat 유지 (기존)
  static uint32_t lastBeat = 0;
  static uint32_t lastDbPrint = 0;     // 5초 간격 출력용
  uint32_t now = millis();
  if (now - lastBeat >= 1000) {
    lastBeat = now;
    char msg[32];
    snprintf(msg, sizeof(msg), "beat:%lu", (unsigned long)now);
    pHeartbeatChar->setValue((uint8_t*)msg, strlen(msg));
    if (deviceConnected) pHeartbeatChar->notify();
  }

  // ===== 전송 중이면 감지/녹음 중단 (반이중) =====
  if (g_sending) {
    // (선택) DMA 오버런 방지용 드레인 + 5초마다 상태 출력
    (void)i2sReadToPCM16(g_block.data(), HOP_SAMPLES);
    if (now - lastDbPrint >= 5000) {
      lastDbPrint = now;
      float thrDbg = g_ambient_db + THRESH_DB_ABOVE;
      float dbDbg  = rmsDBFS(g_block.data(), HOP_SAMPLES);
      Serial.printf("[DB] (sending) rms=%.1f dBFS, thr=%.1f dBFS\n", dbDbg, thrDbg);
    }
    delay(10);
    return;
  }

  // 연결 안 됐으면 I2S만 살짝 비워주고 대기
  if (!deviceConnected) {
    (void)i2sReadToPCM16(g_block.data(), HOP_SAMPLES);
    if (now - lastDbPrint >= 5000) {
      lastDbPrint = now;
      float thrDbg = g_ambient_db + THRESH_DB_ABOVE;
      float dbDbg  = rmsDBFS(g_block.data(), HOP_SAMPLES);
      Serial.printf("[DB] rms=%.1f dBFS, thr=%.1f dBFS (BLE not connected)\n", dbDbg, thrDbg);
    }
    delay(10);
    return;
  }

  // 전송 후 불응기
  if (g_lastSendMs && (millis() - g_lastSendMs < REFRACT_MS)) {
    (void)i2sReadToPCM16(g_block.data(), HOP_SAMPLES);
    if (now - lastDbPrint >= 5000) {
      lastDbPrint = now;
      float dbDbg = rmsDBFS(g_block.data(), HOP_SAMPLES);
      float thrDbg = g_ambient_db + THRESH_DB_ABOVE;
      Serial.printf("[DB] rms=%.1f dBFS, thr=%.1f dBFS\n", dbDbg, thrDbg);
    }
    return;
  }

  // 32ms 블록 RMS(dBFS) 측정 및 트리거 판정
  size_t got = i2sReadToPCM16(g_block.data(), HOP_SAMPLES);
  float db = rmsDBFS(g_block.data(), got);
  float thr = g_ambient_db + THRESH_DB_ABOVE;

  // 5초마다 현재 dB/임계치 출력
  if (now - lastDbPrint >= 5000) {
    lastDbPrint = now;
    Serial.printf("[DB] rms=%.1f dBFS, thr=%.1f dBFS\n", db, thr);
  }

  if (db >= thr) {
    if (++g_overCount >= MIN_BLOCKS_OVER) {
      g_overCount = 0;

      // 트리거: 2초 녹음
      Serial.printf("[TRIGGER] dBFS=%.1f >= %.1f, recording %.2fs...\n", db, thr, RECORD_SEC);
      size_t filled = 0;
      while (filled < REC_SAMPLES) {
        filled += i2sReadToPCM16(g_pcm.data() + filled, REC_SAMPLES - filled);
      }
      Serial.println("[TRIGGER] Recording done.");

      // ---- 전송 가능 여부 판단 (연결 직후 쿨다운 + CCCD 구독 확인)
      bool canSend = deviceConnected
        && (millis() - g_connectedAtMs >= 300)     // 연결 직후 300ms 대기
        && notificationsEnabled(pMetaChar)
        && notificationsEnabled(pDataChar);

      // WAV 만들기
      std::vector<uint8_t> wav;
      buildWavFromPCM(g_pcm.data(), g_pcm.size(), wav);

      if (canSend) {
        Serial.println("[TRIGGER] Sending...");
        sendWavBuffer(wav.data(), wav.size());
        Serial.println("[TRIGGER] Send complete.");
        g_lastSendMs = millis();
      } else {
        Serial.printf("[TRIGGER] Captured %u samples (BLE not ready for notify)\n",
                      (unsigned)g_pcm.size());
        // 전송은 스킵, 다음 트리거를 기다림
      }
    }
  } else {
    g_overCount = 0;
  }
}
