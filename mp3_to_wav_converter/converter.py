from pydub import AudioSegment
import os

# 입력 폴더와 출력 폴더 경로
input_dir = './original_file'
output_dir = './wav_file'


# 출력 폴더 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 폴더 내 모든 파일 확인
for filename in os.listdir(input_dir):
    if filename.endswith('.mp3'):
        mp3_path = os.path.join(input_dir, filename)
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(output_dir, wav_filename)

        # mp3 파일 불러오기
        sound = AudioSegment.from_mp3(mp3_path)
        
        # 필요하면 리샘플링 (모노 + 32kHz로)
        sound = sound.set_channels(1).set_frame_rate(32000)
        
        # wav로 저장
        sound.export(wav_path, format='wav')
        print(f"변환 완료: {wav_filename}")