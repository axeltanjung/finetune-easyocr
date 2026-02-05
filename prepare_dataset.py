import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def preprocess_image(image_path, output_path):
    """
    Preprocessing gambar untuk training OCR
    """
    # Baca gambar
    img = cv2.imread(str(image_path))

    if img is None:
        print(f"Error reading {image_path}")
        return False

    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # Adaptive thresholding (opsional, tergantung dataset)
    # thresh = cv2.adaptiveThreshold(denoised, 255,
    #                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 11, 2)

    # Resize jika terlalu besar (max height 128)
    height, width = denoised.shape
    if height > 128:
        scale = 128 / height
        new_width = int(width * scale)
        denoised = cv2.resize(denoised, (new_width, 128))

    # Save
    cv2.imwrite(str(output_path), denoised)
    return True

def validate_dataset(label_file, image_dir):
    """
    Validasi dataset - cek semua gambar ada dan readable
    """
    issues = []

    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Validating {len(lines)} samples...")

    for idx, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) != 2:
            issues.append(f"Line {idx+1}: Invalid format (expected 2 columns)")
            continue

        image_name, text = parts
        image_path = os.path.join(image_dir, image_name)

        # Check file exists
        if not os.path.exists(image_path):
            issues.append(f"Line {idx+1}: Image not found - {image_name}")
            continue

        # Check image readable
        img = cv2.imread(image_path)
        if img is None:
            issues.append(f"Line {idx+1}: Cannot read image - {image_name}")
            continue

        # Check text not empty
        if len(text.strip()) == 0:
            issues.append(f"Line {idx+1}: Empty text label - {image_name}")

    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues)-10} more")
        return False
    else:
        print(f"✅ All {len(lines)} samples validated successfully!")
        return True

# Jalankan validasi
if __name__ == "__main__":
    # Validasi training set
    validate_dataset(
        "dataset/labels/train_labels.txt",
        "dataset/raw/train"
    )

    # Validasi validation set
    validate_dataset(
        "dataset/labels/val_labels.txt",
        "dataset/raw/validation"
    )
