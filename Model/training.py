# 필요할 때마다 base 모델에서 불러오는 것을 추천함 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.base_model_panns import (
    AudioEmbeddingDataset,
    PANNsCNN10,
    TransferClassifier,
    train_classifier,
    infer_audio,
    get_device
)

import torch 
import torch.nn as nn
import torch.optim  as optim 
from torch.utils.data import DataLoader

# 이건 학습이라서 데이터셋에 방에 관련된 정보는 들어오지 않을 것임. ----------
# 학습용 데이터에는 음성 및 라벨링만 들어있을 예정 

# - 데이터셋은 폴더별로 관리해서 라벨링하면 된다. 

device = get_device()# device를 먼저 밝히는 게 먼저이다~

# ------------------------- 데이터 전처리 및 임베딩 준비------------------------- # 

# 1. 모델 준비
model = PANNsCNN10('./Model/pretrained/Cnn10.pth')

# 2. Dataset & Dataloader
dataset = AudioEmbeddingDataset(root_dir='./Dataset/Dataset', model=model)
loader = DataLoader(dataset, batch_size=16, shuffle=True)  # 배치 사이즈는 16으로 통일

# (선택) 데이터 확인용: 한 번만 출력
for x, y in loader:
    print(x.shape)  # torch.Size([16, 1024]) → 열은 임베딩 사이즈
    print(y.shape)  # torch.Size([16])
    break

print(dataset.label_dict)

# 3. 분류기 정의
classifier = TransferClassifier(input_dim=512, num_classes=len(dataset.label_dict))

# 4. 전이 학습 수행
train_classifier(classifier, loader, num_classes=len(dataset.label_dict), epochs=2)
