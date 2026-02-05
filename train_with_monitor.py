import subprocess
import os
import sys
from datetime import datetime

def train_with_monitoring():
    """
    Training dengan automatic monitoring dan checkpoint management
    """

    # Training configuration
    config = {
        'train_data': 'dataset/lmdb/train',
        'valid_data': 'dataset/lmdb/validation',
        'batch_size': 192,
        'num_iter': 300000,
        'valInterval': 2000,
        'saved_model': './saved_models/',
        'Transformation': 'TPS',
        'FeatureExtraction': 'ResNet',
        'SequenceModeling': 'BiLSTM',
        'Prediction': 'Attn',
        'lr': 1.0,
        'workers': 4,
    }

    # Create save directory dengan timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"experiments/exp_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    config['saved_model'] = save_dir

    # Build command
    cmd = ['python', 'train.py']
    for key, value in config.items():
        cmd.extend([f'--{key}', str(value)])

    # Log configuration
    with open(f'{save_dir}/config.txt', 'w') as f:
        f.write(f"Training started: {timestamp}\n")
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

    print(f"Starting training...")
    print(f"Logs will be saved to: {save_dir}")
    print(f"Monitor with: tensorboard --logdir={save_dir}")

    # Run training
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Stream output
        log_file = f'{save_dir}/training.log'
        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(line, end='')  # Print to console
                f.write(line)  # Write to log file
                f.flush()

        process.wait()

        if process.returncode == 0:
            print(f"\n✅ Training completed successfully!")
        else:
            print(f"\n❌ Training failed with code {process.returncode}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        process.terminate()
        process.wait()

    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    train_with_monitoring()
