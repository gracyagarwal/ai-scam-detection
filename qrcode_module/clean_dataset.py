import os
import cv2
import shutil
import random

# Source directories (update paths as needed)
benign_path = "C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\qrcode_module\\archive\\QR codes\\benign\\benign"
malicious_path = "C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\qrcode_module\\archive\\QR codes\\malicious\\malicious"

# Destination directories
clean_base = "C:\\Users\\Gracy\\OneDrive\\Desktop\\vit\\24-25winter\\project-2\\qrcode_module\\archive\\QR codes cleaned"
os.makedirs(os.path.join(clean_base, "benign"), exist_ok=True)
os.makedirs(os.path.join(clean_base, "malicious"), exist_ok=True)

# Max number of samples to balance
MAX_SAMPLES = 20000  # adjust as needed

def clean_and_copy_images(src_dir, dst_dir, max_samples):
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(all_files)

    count = 0
    for file in all_files:
        if count >= max_samples:
            break
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)

        try:
            img = cv2.imread(src_file)
            if img is None:
                continue  # unreadable image
            cv2.imwrite(dst_file, img)  # save clean copy
            count += 1
        except Exception as e:
            print(f"Error with {file}: {e}")
    
    print(f"✅ Cleaned {count} images for {dst_dir}")

clean_and_copy_images(benign_path, os.path.join(clean_base, "benign"), MAX_SAMPLES)
clean_and_copy_images(malicious_path, os.path.join(clean_base, "malicious"), MAX_SAMPLES)
