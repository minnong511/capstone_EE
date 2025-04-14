# 필요할 때마다 base 모델에서 불러오는 것을 추천함 
from base_model_panns import (
    AudioEmbeddingDataset,
    PANNsCNN10,
    TransferClassifier,
    train_classifier,
    infer_audio
)
import torch 
import torch.nn as nn
import torch.optim  as optim 
from torch.utils.data import DataLoader
import os 

# ------ multimodal로 구현하고 싶다.. ㅋㅋㅋ ---- 방학의 정보를 내가 학습하지 않으면 좋을 거 같은데 
# 이건 학습이라서 데이터셋에 방에 관련된 정보는 들어오지 않을 것임. ----------
# 학습용 데이터에는 음성 및 라벨링만 들어있을 예정 

# - 데이터셋은 폴더별로 관리해서 라벨링하면 된다. 

device = get_device()# device를 먼저 밝히는 게 먼저이다~

# ------------------------- 데이터 전처리 및 임베딩 준비------------------------- # 


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

# 전처리된 embedding dataset & dataloader
dataset = AudioEmbeddingDataset(root_dir='./your_dataset', model=model)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# classifier 정의
classifier = TransferClassifier(input_dim=1024, num_classes=len(dataset.label_dict))

# 학습 시작
train_classifier(classifier, loader, num_classes=len(dataset.label_dict), epochs=20)

# ----------------------- 모델 추론 -------------------------------

# Validation Set으로 데이터셋을 평가해야 함. -> 범용성 있는 확인해야 한다. 
# 여기서는 음성 데이터, 방 번호가 들어가게 되는데, 음성 데이터만 추론 모델에 들어가게 되고, 방 번호는 그대로 통과해서 출력 레이블에 그대로 붙게 될 것임.

# result = infer_audio(
#     filepath="./inputs/dog_bark.wav",
#     room_id=1,
#     panns_model=panns,
#     classifier_model=classifier,
#     label_dict=label_dict,
#     device=device
# )

# 👇 출력 결과 딕셔너리 활용
# {'room_id': 1, 'predicted_class': 'dog_bark'}

#------------------------- 알림 시스템 개발 --------------------------- # 















#--------- 최종적으로는 알림 시스템과, 모델 학습된 거 추출하기 ----------------------- #