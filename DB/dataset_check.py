import os
import torchaudio
import subprocess

def find_broken_wav_files(root_dir):
    broken_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".wav"):
                fpath = os.path.join(dirpath, fname)
                try:
                    torchaudio.load(fpath)
                except Exception as e:
                    print(f"[ERROR] {fpath} - {e}")
                    broken_files.append(fpath)

    print(f"\n총 {len(broken_files)}개의 손상된/비표준 파일 발견")
    return broken_files

def reencode_wav_files(file_list):
    for fpath in file_list:
        print(f"🔁 Re-encoding {fpath}")
        output_path = fpath.replace(".wav", "_fixed.wav")
        try:
            # ffmpeg로 재인코딩: 16bit PCM, 32kHz
            cmd = [
                "ffmpeg",
                "-y",  # 자동 덮어쓰기
                "-i", fpath,
                "-acodec", "pcm_s16le",
                "-ar", "32000",
                output_path
            ]
            subprocess.run(cmd, check=True)

            # 원본 파일 삭제 후 교체
            os.remove(fpath)
            os.rename(output_path, fpath)
            print(f"복구 완료: {fpath}")
        except subprocess.CalledProcessError:
            print(f"복구 실패: {fpath}")
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    ROOT_DIR = "./Dataset/Dataset"
    broken_files = find_broken_wav_files(ROOT_DIR)
    reencode_wav_files(broken_files)