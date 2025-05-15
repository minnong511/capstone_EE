import os

dataset_path = "./Dataset/Dataset"  # 데이터셋 루트 폴더 경로
category_counts = {}

for category in sorted(os.listdir(dataset_path)):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        category_counts[category] = len(files)

total_files = sum(category_counts.values())

print("Category Counts:")
for category, count in category_counts.items():
    print(f"- {category}: {count} files")
print(f"\nTotal number of audio files: {total_files}")
