import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm

def create_lmdb_dataset(image_dir, label_file, output_path):
    """
    Membuat LMDB dataset untuk EasyOCR training

    Args:
        image_dir: Directory berisi gambar
        label_file: Path ke file label (TSV format)
        output_path: Path output LMDB database
    """

    # Baca label file
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    # Calculate map size (1TB)
    map_size = 1099511627776

    # Create LMDB environment
    env = lmdb.open(output_path, map_size=map_size)

    cache = {}
    cnt = 1

    print(f"Processing {len(lines)} samples...")

    for idx, line in enumerate(tqdm(lines)):
        # Parse line
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"Skipping line {idx+1}: Invalid format")
            continue

        image_name, label = parts
        image_path = os.path.join(image_dir, image_name)

        # Check file exists
        if not os.path.exists(image_path):
            print(f"Skipping line {idx+1}: Image not found - {image_name}")
            continue

        # Read image as binary
        with open(image_path, 'rb') as f:
            image_bin = f.read()

        if not image_bin:
            print(f"Skipping line {idx+1}: Empty image - {image_name}")
            continue

        # Create keys (format yang dibutuhkan EasyOCR)
        image_key = f'image-{cnt:09d}'.encode()
        label_key = f'label-{cnt:09d}'.encode()

        # Store in cache
        cache[image_key] = image_bin
        cache[label_key] = label.encode('utf-8')

        cnt += 1

        # Write to LMDB in batches (setiap 1000 samples)
        if cnt % 1000 == 0:
            with env.begin(write=True) as txn:
                for key, value in cache.items():
                    txn.put(key, value)
            cache = {}
            print(f"Written {cnt-1} samples to LMDB")

    # Write remaining cache
    if cache:
        with env.begin(write=True) as txn:
            for key, value in cache.items():
                txn.put(key, value)

    # Write total number of samples
    with env.begin(write=True) as txn:
        txn.put('num-samples'.encode(), str(cnt-1).encode())

    env.close()
    print(f"\n✅ Created LMDB dataset with {cnt-1} samples at {output_path}")
    return cnt-1

def verify_lmdb(lmdb_path):
    """
    Verifikasi LMDB dataset
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        num_samples = txn.get('num-samples'.encode())
        if num_samples:
            num_samples = int(num_samples.decode())
            print(f"Total samples in LMDB: {num_samples}")

            # Check beberapa sample
            print("\nChecking first 3 samples:")
            for i in range(1, min(4, num_samples+1)):
                image_key = f'image-{i:09d}'.encode()
                label_key = f'label-{i:09d}'.encode()

                image_bin = txn.get(image_key)
                label = txn.get(label_key)

                if image_bin and label:
                    print(f"  Sample {i}: Label = '{label.decode('utf-8')}', Image size = {len(image_bin)} bytes")
                else:
                    print(f"  Sample {i}: Missing data!")

    env.close()

if __name__ == "__main__":
    # Create training LMDB
    print("Creating training LMDB...")
    train_count = create_lmdb_dataset(
        image_dir="dataset/raw/train",
        label_file="dataset/labels/train_labels.txt",
        output_path="dataset/lmdb/train"
    )

    print("\nVerifying training LMDB...")
    verify_lmdb("dataset/lmdb/train")

    print("\n" + "="*50 + "\n")

    # Create validation LMDB
    print("Creating validation LMDB...")
    val_count = create_lmdb_dataset(
        image_dir="dataset/raw/validation",
        label_file="dataset/labels/val_labels.txt",
        output_path="dataset/lmdb/validation"
    )

    print("\nVerifying validation LMDB...")
    verify_lmdb("dataset/lmdb/validation")

    print("\n" + "="*50)
    print(f"✅ Dataset ready for training!")
    print(f"   Training samples: {train_count}")
    print(f"   Validation samples: {val_count}")
    print(f"   Ratio: {val_count/train_count*100:.1f}% validation")
