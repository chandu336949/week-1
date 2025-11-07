import os
import shutil

original_train_normal = "chest_xray/train/NORMAL/"
original_train_pneumonia = "chest_xray/train/PNEUMONIA/"
original_test_normal = "chest_xray/test/NORMAL/"
original_test_pneumonia = "chest_xray/test/PNEUMONIA/"

project_train_normal = "train/normal/"
project_train_pneumonia = "train/pneumonia/"
project_test_normal = "test/normal/"
project_test_pneumonia = "test/pneumonia/"


def copy_some_images(src_folder, dst_folder, n):
    if not os.path.exists(src_folder):
        print(f"ERROR: Source folder not found: {src_folder}")
        return
    
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images in {src_folder}, copying first {n}...")
    
    for img in images[:n]:
        shutil.copy2(os.path.join(src_folder, img), os.path.join(dst_folder, img))
    
    print(f"✓ Copied {min(n, len(images))} images to {dst_folder}")


print("Starting image copy process...\n")
copy_some_images(original_train_normal, project_train_normal, 120)
copy_some_images(original_train_pneumonia, project_train_pneumonia, 120)
copy_some_images(original_test_normal, project_test_normal, 30)
copy_some_images(original_test_pneumonia, project_test_pneumonia, 30)
print("\nDone! Images copied successfully.")
