# 필요할 때마다 base 모델에서 불러오는 것을 추천함 
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
import os 

# 이건 학습이라서 데이터셋에 방에 관련된 정보는 들어오지 않을 것임. ----------
# 학습용 데이터에는 음성 및 라벨링만 들어있을 예정 

# - 데이터셋은 폴더별로 관리해서 라벨링하면 된다. 

device = get_device()# device를 먼저 밝히는 게 먼저이다~

# ----------------------- 모델 추론 -------------------------------

## 추론 파일 저장하기 
test_folder = "./Test_Dataset"

test_files = [f for f in os.listdir(test_folder) if f.endswith(".wav")]

# 추론 반복 
for fname in test_files: 
    file_path = os.path.join(test_folder, fname)

    # 예시 : 방 번호를 파일명에서 추출하거나 고정값 사용 
    room_id = ""