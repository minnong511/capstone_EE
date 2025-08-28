#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <vector>
#include <algorithm>
#include <math.h>
#include <memory>
#include <cstring>
#include <time.h>

// ====== ESP-IDF I2S driver (Arduino core exposes it) ======
extern "C" {
  #include "driver/i2s.h"
}

// ---------------- Device / Network Config ----------------
#define DEVICE_NAME "Sensor-01"

// ★★★ TODO: 네트워크 환경 맞게 수정 ★★★
static const char* WIFI_SSID = "YOUR-SSID";
static const char* WIFI_PASS = "YOUR-PASS";

// 서버 업로드 엔드포인트 (Flask 예시: http://<server-ip>:5050/upload)
static const char* UPLOAD_URL = "http://192.168.0.10:5050/upload";

// NTP로 시간 동기화(파일명 타임스탬프 안정)
static const char* NTP_POOL = "pool.ntp.org";
static const long  GMT_OFFSET_SEC = 9 * 3600; // Asia/Seoul (+9h)
static const int   DAYLIGHT_OFFSET_SEC = 0;

// 전송 상태
static volatile bool g_sending = false;     // 전송 중이면 true → 녹음/트리거 정지
static uint32_t g_connectedAtMs = 0;        // (Wi-Fi) 연결 직후 쿨다운
static uint32_t g_lastSendMs = 0;

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

// dBFS 트리거 파라미터
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
//                    Wi-Fi helpers / HTTP upload
// ===========================================================
static void wifiConnectBlocking() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (millis() - t0 > 20000) { // 20s 타임아웃 후 재시작 시도
      Serial.println("\n[WiFi] connect timeout, retry...");
      t0 = millis();
      WiFi.disconnect(true, true);
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }
  Serial.printf("\n[WiFi] Connected. IP=%s\n", WiFi.localIP().toString().c_str());
  g_connectedAtMs = millis();
  // 시간 동기화(선택)
  configTime(GMT_OFFSET_SEC, DAYLIGHT_OFFSET_SEC, NTP_POOL);
}

static uint32_t nowEpoch() {
  time_t now = time(nullptr);
  if (now < 100000) { // 아직 동기화 전이면 millis 기반 임시치
    return (uint32_t)(millis() / 1000);
  }
  return (uint32_t)now;
}

static bool httpUploadWav(const uint8_t* wav, size_t totalSize) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] Wi-Fi not connected; abort upload");
    return false;
  }

  HTTPClient http;
  http.setTimeout(15000); // 15s
  http.begin(UPLOAD_URL);
  http.addHeader("Content-Type", "application/octet-stream");
  http.addHeader("X-Room-ID", "Room102");
  http.addHeader("X-Mic-ID", DEVICE_NAME);
  http.addHeader("X-Timestamp", String(nowEpoch()));

  Serial.printf("[HTTP] POST %s (%u bytes)\n", UPLOAD_URL, (unsigned)totalSize);
  int code = http.POST(wav, totalSize);

  if (code > 0) {
    Serial.printf("[HTTP] Response code=%d\n", code);
    if (code == 200) {
      String resp = http.getString();
      Serial.printf("[HTTP] Body: %s\n", resp.c_str());
      http.end();
      return true;
    }
  } else {
    Serial.printf("[HTTP] Failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
  return false;
}

// ===========================================================
//                        Setup / Loop
// ===========================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  // ------ Wi-Fi ------
  wifiConnectBlocking();

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
  // Heartbeat 비슷한 주기 로그
  static uint32_t lastBeat = 0;
  static uint32_t lastDbPrint = 0;     // 5초 간격 출력용
  uint32_t now = millis();

  // Wi-Fi 연결 유지
  if (WiFi.status() != WL_CONNECTED) {
    wifiConnectBlocking();
  }

  if (now - lastBeat >= 1000) {
    lastBeat = now;
    Serial.printf("[HB] t=%lu, IP=%s\n", (unsigned long)now, WiFi.localIP().toString().c_str());
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

      // WAV 만들기
      std::vector<uint8_t> wav;
      buildWavFromPCM(g_pcm.data(), g_pcm.size(), wav);

      // ---- 업로드 (최대 2회 재시도) ----
      g_sending = true;
      bool ok = false;
      for (int attempt = 1; attempt <= 2; ++attempt) {
        if (millis() - g_connectedAtMs < 300) delay(300 - (millis() - g_connectedAtMs));
        ok = httpUploadWav(wav.data(), wav.size());
        if (ok) break;
        Serial.printf("[HTTP] retry %d...\n", attempt);
        delay(500);
        if (WiFi.status() != WL_CONNECTED) wifiConnectBlocking();
      }
      g_sending = false;

      if (ok) {
        Serial.println("[TRIGGER] Upload complete.");
        g_lastSendMs = millis();
      } else {
        Serial.println("[TRIGGER] Upload failed; will wait for next trigger.");
      }
    }
  } else {
    g_overCount = 0;
  }
}