import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import os

file_path = "flitered_signal.wav"
print("파일 존재 여부:", os.path.exists(file_path))  # True여야 함

# 모델 로딩
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 오디오 불러오기
waveform, sample_rate = torchaudio.load("flitered_signal.wav")

# 리샘플링 (16kHz 필수)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# 모노 처리: [1, time]
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# 배치 차원 없이 벡터로 변환: [time]
waveform = waveform.squeeze()

# 입력 전처리
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

# 추론
with torch.no_grad():
    logits = model(**inputs).logits

# 디코딩
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("🗣️ 인식된 문장:", transcription)
