from watchdog.observers import Observer 
from watchdog.events import FileSystemEventHandler
import time 
import os  

from Model.base_model_panns import (
    AudioEmbeddingDataset,
    PANNsCNN10,
    TransferClassifier,
    LabelDict,
    train_classifier,
    infer_audio,
    get_device
)

class AudioFileHandler(FileSystemEventHandler):
    
    def __init__(self, model, classifier, label_dict, device):
        self.model = model 
        self.classifier = classifier
        self.label_dict = label_dict 
        self.device = device

    def on_created(self, event):
        if event.src_path.lower().endswith(".wav"):
            print(f"New file detected : {event.src_path}")
            time.sleep(0.5)  # 파일 저장이 끝나도록 살짝 대기 (옵션)
            try:
                result = infer_audio(
                    file_path = event.src_path,
                    room_id = "RealtimeRoom",
                    panns_model = self.model, 
                    classifier_model = self.classifier,
                    label_dict = self.label_dict, 
                    device = self.device  
                )
                print("🎧 Inference Result:", result)
            except Exception as e:
                print(f"Error during inference: {e}")


# 설정 

device = get_device()
label_manager = LabelDict('./Dataset/Dataset')
label_dict = label_manager.get()
model = PANNsCNN10('./Model/pretrained/Cnn10.pth').to(device)
classifier = TransferClassifier(input_dim=1024, num_classes=len(label_dict)).to(device)
dataset = AudioEmbeddingDataset(root_dir="./Dataset", model=model)

watch_path = "./RealtimeInput"
handler = AudioFileHandler(model, classifier, dataset.label_dict, device)
observer = Observer()
observer.schedule(handler, path = watch_path, recursive=False)
observer.start()

print(f"👀 Watching {watch_path} for new .wav files...")

try:
    while True: 
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
