import torch
import torch.nn as nn
from models import Cnn10  # models.py에서 정의됨

# -------------------------------- 4월 13일 -------------------------------- # 

# 모델 찾음 : PANN // 

class PANNsCNN10(nn.Module):
    def __init__(self, checkpoint_path='./Model/pretrained/Cnn10.pth'):
        super().__init__()
        self.model = Cnn10(sample_rate=32000, window_size=1024, hop_size=320,
                           mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()  # freeze

    def forward(self, x):
        with torch.no_grad():
            output = self.model(x)
            return output['embedding']  # (batch, 1024)

# 커스텀 분류기 (전이학습용)
class TransferClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# 테스트 실행
# if __name__ == '__main__':
#     model = PANNsCNN10('./Model/pretrained/Cnn10.pth')

#     # (batch_size, waveform_length) → float32 중요!
#     dummy_input = torch.randn(2, 32000).float()

#     # PANNs 내부에서 자동으로 (B, 1, T)로 reshape 하기 때문에 
#     # unsqueeze() 쓰지 말고 바로 float32 2차원 텐서를 넘겨야 함.

#     emb = model(dummy_input)
#     print("임베딩 shape:", emb.shape)

# -------------------------------- 4월 14일 -------------------------------- # 

# 1. .wav 파일 로드 (torchaudio 또는 librosa)

# 2. 32Khz 리샘플링 + 모노 변환 

# 3. 모델에 넣기 위한 Waveform -> embedding 추출 

# 4. (embedding, label) 쌍을 모아서 학습 