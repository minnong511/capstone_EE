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
    get_device, 
    get_label_dict
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

# 테스트 폴더 경로
test_folder = "./Infer_Test_Dataset"


# 모델, 레이블, 디바이스 정의
panns_model = PANNsCNN10('./Model/pretrained/Cnn10.pth')
panns_model.to(device)
panns_model.eval()

# 2. Transfer Classifier (사전 학습된 분류기)
classifier_model_path = "Model/classifier_model.pth"
label_dict = get_label_dict(root_dir='./Dataset/Dataset')  # 예: {"speech": 0, "dog_bark": 1, ...}

classifier_model = TransferClassifier(input_dim=512, num_classes=len(label_dict))
classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=device))
classifier_model.to(device)
classifier_model.eval()

device = get_device()
# 파일 리스트 가져오기
test_files = [f for f in os.listdir(test_folder) if f.endswith(".wav")]

# 파일별 추론 수행
for filename in test_files:
    try:
        room_id = filename.split("_")[0]
        file_path = os.path.join(test_folder, filename)

        infer_audio(
            file_path=file_path,
            room_id=room_id,
            panns_model=panns_model,
            classifier_model=classifier_model,  # ← 여기 바뀜!
            label_dict=label_dict,
            device=device
        )

    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")