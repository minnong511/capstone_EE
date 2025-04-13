from base_model_panns import AudioEmbeddingDataset
from base_model_panns import PANNsCNN10
import torch 
import torch.nn
from torch.utils.data import DataLoader
import os 


# ------------------------- 데이터 전처리 ------------------------- # 


# 1. 모델 준비
model = PANNsCNN10('./Model/pretrained/Cnn10.pth')

# 2. Dataset & Dataloader
dataset = AudioEmbeddingDataset(root_dir='./your_dataset', model=model)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. 학습 루프에서 사용
for x, y in loader:
    print(x.shape)  # torch.Size([8, 512]) or [8, 1024]
    print(y.shape)  # torch.Size([8])
    break

print(dataset.label_dict)

# ------------------------- 전이 학습 -----------------------------