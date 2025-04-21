import torch
import torch.nn as nn
import torch.optim as optim 
import torchaudio
import platform
import os 

from models import Cnn10  # models.py에서 정의됨
from torch.utils.data import Dataset

#-------------------------------------------------------------------------# 

def get_device():
    system = platform.system()

    if system == 'Darwin':  # macOS
        if torch.backends.mps.is_available():
            print("macOS + MPS 사용")
            return torch.device('mps')
        else:
            print("macOS지만 MPS 사용 불가 → CPU로 대체")
            return torch.device('cpu')

    elif torch.cuda.is_available():  # Windows/Linux with GPU
        print("CUDA GPU 사용")
        return torch.device('cuda')

    else:
        print("GPU 사용 불가 → CPU 사용")
        return torch.device('cpu')
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
    def __init__(self, input_dim=1024, num_classes=3): # 내가 출력하고 싶은 클래스는 여기서 수정하면 된다
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
# 여러 개를 전처리하고 싶으니 그에 맞게 클래스를 구현하면 된다. 

# 1. .wav 파일 여러개 로드 (torchaudio 또는 librosa)

# 2. 32Khz 리샘플링 + 모노 변환 

# 3. 모델에 넣기 위한 Waveform -> embedding 추출 

# 4. (embedding, label) 쌍을 모아서 학습 

class AudioEmbeddingDataset(Dataset): 
    def __init__(self, root_dir, model, sample_rate = 32000): 
        self.samples = [] 
        self.model = model 
        self.model.eval() 
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)  # default wav: 44.1kHz
        self.label_dict = {}

        for idx, label_name in enumerate(sorted(os.listdir(root_dir))): 
            self.label_dict[label_name] = idx 
            label_dir = os.path.join(root_dir,label_name)
            for frame in os.listdir(label_dir): 
                if frame.endswith(".wav"): 
                    fpath = os.path.join(label_dir, name)
                    self.samples.append((fpath,idx)) 
    
    def __len__(self):
        return len(self.samples) 
    
    def __getitem__(self,idx): 
        fpath, label = self.samples[idx] 
        waveform, sr = torchaudio.load(fpath) 

        # 모노 변환 
        if waveform.shape[0] > 1 : 
            waveform = waveform.mean(dim = 0 , keepdim= True) 
        
        # 리샘플링 -> 32000으로 만들어줘야 한다. 
        if sr != 32000:
            waveform = self.resampler(dim = 0, keepdim = True) 
        
        waveform = waveform.squeeze(0).unsqueeze(0)  # → (1, length)

        # 모델에 넣어서 임베딩만 추출하되, 학습을 하지 않기 위해 no_grad를 사용한다. 
        with torch.no_grad():
            emb = self.model(waveform)[0]  # → (512,) or (1024,) depending on model

        return emb, label

# 데이터셋 자동 처리 모델 

# embedding  -> classifer -> label 
# 일단은 최대한 간단하게 모델 구성 
# 학습 루프 구성 
# 클래스는 몇 개로 분류?? 

class TransferClassifier(nn.Module):
    # input_dim은 CNN10 = 1024, CNN6 = 512 
    def __init__(self,input_dim = 1024,num_classes =3): 
        super().__init__() 
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,num_classes)
        )
    
    def forward(self,x): 
        return self.classifier(x) 


# ----------- 전이학습 후에 임베딩 추출하고, 추출된 임베딩과 라벨로 Classifier를 구현하는 부분임
def train_classifier(classifier, dataloader, num_classes, epochs=10):
    device = get_device() # 맥에서는 쿠다 안되니 윈도우 컴에서 구현 ㄱㄱ 
    classifier = classifier.to(device) 
    optimizer = optim.Adam(classifier.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss() 

    for epoch in range(epoch): 
        classifier.train()
        total_loss = 0 
        correct = 0 
        total = 0 

        for x,y in dataloader: 
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = classifier(x) 
            loss = criterion(logits,y) 
            loss.backward() 
            optimizer.step()

            total_loss += loss.item() 
            pred = torch.argmax(logits, dim = 1) 
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total 
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")


#------------------------ 4월 15일 개발 ---------------------# 
# --- 오디오 추론 모델 개발 --- #  

def infer_audio(file_path, room_id, panns_model, classifier_model, label_dict, device = "cpu"):
    import torchaudio
    import torch 

    # 1. 오디오 로드 
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:  
        waveform = waveform.mean(dim = 0, keepdim= True)
    
    # 2. 리샘플링
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq = 32000)
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0).unsqueeze(0).to(device)
    
    # 3. 임베딩 추출
    with torch.no_grad():
        embedding = panns_model(waveform) 

    # 4. 분류기 추출 
    with torch.no_grad():
        logits = classifier_model(embedding.to(device))
        pred_idx = torch.argmax(logits, dim = 1).item()

    # 5. 출력 
    idx_to_label = {v: k for k, v in label_dict.items()}
    pred_label = idx_to_label[pred_idx]

    print(f"🧠 Predicted: {pred_label} | 📍 Room {room_id}")
    return {"room_id": room_id, "predicted_class": pred_label}