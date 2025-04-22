#include <driver/i2s.h>
#include <algorithm>
#include <math.h>

#define SAMPLE_RATE          32000
#define SAMPLE_BUFFER_SIZE   1024
#define I2S_NUM              I2S_NUM_0

//--- Calibration parameters ---
#define CALIBRATION_SEC      30
#define DROP_RATIO           0.10f    // 상위 10%
#define SENSITIVITY_COEF     1.0f     // 0.5~2.0 조정 가능

//--- Recording parameters ---
#define DEFAULT_SEC          7        // 기본 녹음 길이
#define MAX_SEC             15        // 최대 연장 길이
#define SILENCE_LIMIT_MS   1500       // 1.5초 이내 정적이면 조기 종료
#define MIN_WRAP_SEC         5        // 짧게 멈추면 5초만 녹음

// 전역 변수
float threshold_dB = 0.0f;
bool  isRecording  = false;
bool  continuousAbove = true;
bool  earlyStopEligible = false;
unsigned long recordStartTime = 0;
unsigned long silenceStartTime = 0;

// I2S 설정
i2s_config_t i2s_config = {
  .mode              = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
  .sample_rate       = SAMPLE_RATE,
  .bits_per_sample   = I2S_BITS_PER_SAMPLE_32BIT,
  .channel_format    = I2S_CHANNEL_FMT_ONLY_LEFT,
  .communication_format = I2S_COMM_FORMAT_I2S,
  .intr_alloc_flags  = ESP_INTR_FLAG_LEVEL1,
  .dma_buf_count     = 2,
  .dma_buf_len       = SAMPLE_BUFFER_SIZE,
  .use_apll          = false,
  .tx_desc_auto_clear= false,
  .fixed_mclk        = 0
};

i2s_pin_config_t i2s_pins = {
  .bck_io_num   = 32,
  .ws_io_num    = 33,
  .data_out_num = I2S_PIN_NO_CHANGE,
  .data_in_num  = 35
};

int32_t rawBuf[SAMPLE_BUFFER_SIZE];

// RMS 계산 함수
float computeRMS(int32_t *buf, int len) {
  double sumSq = 0;
  for(int i=0;i<len;i++){
    double v = buf[i];
    sumSq += v * v;
  }
  return sqrt(sumSq / len);
}

// 1) 캘리브레이션
void calibrateBaseline() {
  const int blocks = (CALIBRATION_SEC * SAMPLE_RATE) / SAMPLE_BUFFER_SIZE;
  float dbArr[blocks];

  size_t bytes;
  for(int i=0; i<blocks; i++){
    // I2S에서 버퍼 읽기
    i2s_read(I2S_NUM, rawBuf, sizeof(rawBuf), &bytes, portMAX_DELAY);
    int samples = bytes / sizeof(int32_t);
    float rms = computeRMS(rawBuf, samples);
    float db  = 20.0f * log10(rms + 1e-6f);
    dbArr[i] = db;
  }

  // 오름차순 정렬 후 상위 DROP_RATIO 비율 제거
  std::sort(dbArr, dbArr + blocks);
  int keep = blocks - int(blocks * DROP_RATIO);

  // 평균·표준편차 계산
  double sum=0, sumSq=0;
  for(int i=0;i<keep;i++){
    sum   += dbArr[i];
    sumSq += dbArr[i] * dbArr[i];
  }
  float mean   = sum / keep;
  float var    = sumSq / keep - mean * mean;
  float stddev = sqrt(var);

  threshold_dB = mean + stddev * SENSITIVITY_COEF;
  Serial.printf("[Calib] mean=%.2f dB, σ=%.2f dB → thresh=%.2f dB\n",
                mean, stddev, threshold_dB);
}

// 녹음 시작·종료 훅 (사용자 코드로 구현)
void onRecordingStart() {
  Serial.println(">>> Recording START");
  // TODO: Wi‑Fi 전송 또는 SD 카드 기록 등
}
void onRecordingStop() {
  Serial.println(">>> Recording STOP");
  // TODO: 버퍼 전송 마무리
}

void setup() {
  Serial.begin(115200);
  i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM, &i2s_pins);

  // 1) 캘리브레이션
  Serial.println(">> Calibrating ambient noise...");
  calibrateBaseline();
}

void loop() {
  size_t bytes;
  i2s_read(I2S_NUM, rawBuf, sizeof(rawBuf), &bytes, portMAX_DELAY);
  int samples = bytes / sizeof(int32_t);
  float rms = computeRMS(rawBuf, samples);
  float db  = 20.0f * log10(rms + 1e-6f);

  // 2) 트리거 로직
  if (!isRecording) {
    if (db > threshold_dB) {
      // 녹음 시작
      isRecording       = true;
      continuousAbove   = true;
      earlyStopEligible = false;
      recordStartTime   = millis();
      silenceStartTime  = 0;
      onRecordingStart();
    }
  }
  else {
    // 녹음 중: 샘플 버퍼 처리 또는 전송
    // 예) sendAudioChunk(rawBuf, bytes);

    // 소리 레벨 검사
    if (db < threshold_dB && silenceStartTime == 0) {
      silenceStartTime = millis();
      unsigned long sinceStart = silenceStartTime - recordStartTime;
      if (sinceStart <= SILENCE_LIMIT_MS) {
        earlyStopEligible = true;
      }
      continuousAbove = false;
    }
    else if (db >= threshold_dB) {
      // 소리 다시 올라오면
      continuousAbove = false; // 한번 떨어지면 연속성 끝
    }

    unsigned long now = millis();
    unsigned long elapsed = now - recordStartTime;

    // 종료 조건
    if (earlyStopEligible && elapsed >= MIN_WRAP_SEC * 1000UL) {
      isRecording = false;
      onRecordingStop();
    }
    else if (!continuousAbove && !earlyStopEligible
             && elapsed >= DEFAULT_SEC * 1000UL) {
      isRecording = false;
      onRecordingStop();
    }
    else if (continuousAbove && elapsed >= MAX_SEC * 1000UL) {
      isRecording = false;
      onRecordingStop();
    }
  }

  delay(10);  // 약간의 여유
}

// wifi로 전송하는 부분 만들어야 한다. 