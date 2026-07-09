import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Source folders
normal_dir = r"Desktop\Lungh phenumonia\Lung X-Ray Image\Normal"
pneumonia_dir = r"\Desktop\Lungh phenumonia\Lung X-Ray Image\Viral Pneumonia"

# Destination folder
output_dir = r"C:\Desktop\Lungh phenumonia\Dataset_Split"

# Create folder structure
for split in ['train', 'val', 'test']:
    for cls in ['Normal', 'Viral_Pneumonia']:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


def split_and_copy(source_dir, class_name):
    # Get all image files
    images = [f for f in os.listdir(source_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)

    # 70% Train, 30% Temp
    train_files, temp_files = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # Split Temp into 15% Validation and 15% Test
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=42
    )

    # Copy files
    for file in train_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(output_dir, 'train', class_name, file)
        )

    for file in val_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(output_dir, 'val', class_name, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(output_dir, 'test', class_name, file)
        )

    print(f"\n{class_name}")
    print("Train:", len(train_files))
    print("Validation:", len(val_files))
    print("Test:", len(test_files))


# Split both classes
split_and_copy(normal_dir, "Normal")
split_and_copy(pneumonia_dir, "Viral_Pneumonia")

print("\nDataset split completed successfully!")