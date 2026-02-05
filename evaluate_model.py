import easyocr
import cv2
import os
from tqdm import tqdm
import Levenshtein
import numpy as np
import json

def calculate_metrics(predicted, ground_truth):
    """
    Calculate CER, WER, dan exact match accuracy
    """
    # Character Error Rate
    cer = Levenshtein.distance(predicted, ground_truth) / max(len(ground_truth), 1)

    # Word Error Rate
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    wer = Levenshtein.distance(' '.join(pred_words), ' '.join(gt_words)) / max(len(gt_words), 1)

    # Exact match
    exact_match = 1.0 if predicted == ground_truth else 0.0

    return {
        'cer': cer,
        'wer': wer,
        'exact_match': exact_match
    }

def evaluate_on_dataset(model_path, test_data_dir, label_file):
    """
    Evaluasi model pada test dataset
    """
    # Initialize reader
    print("Loading model...")
    reader = easyocr.Reader(
        ['en'],
        gpu=True,
        model_storage_directory='./models/',
        user_network_directory=model_path,
        recog_network='custom'
    )

    # Load test labels
    with open(label_file, 'r', encoding='utf-8') as f:
        test_samples = [line.strip().split('\t') for line in f.readlines()]

    results = []
    total_cer = 0
    total_wer = 0
    total_exact = 0
    errors = []

    print(f"Evaluating on {len(test_samples)} samples...")

    for img_name, ground_truth in tqdm(test_samples):
        img_path = os.path.join(test_data_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found - {img_path}")
            continue

        # Predict
        try:
            result = reader.readtext(img_path, detail=0)
            predicted = ' '.join(result) if result else ""
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            predicted = ""

        # Calculate metrics
        metrics = calculate_metrics(predicted, ground_truth)

        # Store results
        results.append({
            'image': img_name,
            'ground_truth': ground_truth,
            'predicted': predicted,
            **metrics
        })

        total_cer += metrics['cer']
        total_wer += metrics['wer']
        total_exact += metrics['exact_match']

        # Track significant errors
        if metrics['cer'] > 0.3:  # CER > 30%
            errors.append({
                'image': img_name,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'cer': metrics['cer']
            })

    # Calculate averages
    n_samples = len(results)
    avg_metrics = {
        'avg_cer': total_cer / n_samples,
        'avg_wer': total_wer / n_samples,
        'exact_match_accuracy': total_exact / n_samples,
        'total_samples': n_samples
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {n_samples}")
    print(f"Average CER: {avg_metrics['avg_cer']:.4f} ({avg_metrics['avg_cer']*100:.2f}%)")
    print(f"Average WER: {avg_metrics['avg_wer']:.4f} ({avg_metrics['avg_wer']*100:.2f}%)")
    print(f"Exact Match Accuracy: {avg_metrics['exact_match_accuracy']:.4f} ({avg_metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"Significant errors (CER > 30%): {len(errors)}")

    # CER Distribution
    cer_values = [r['cer'] for r in results]
    print("\nCER Distribution:")
    print(f"  Min: {min(cer_values):.4f}")
    print(f"  Max: {max(cer_values):.4f}")
    print(f"  Median: {np.median(cer_values):.4f}")
    print(f"  Std Dev: {np.std(cer_values):.4f}")

    # Show worst errors
    if errors:
        print("\nTop 5 worst predictions:")
        errors.sort(key=lambda x: x['cer'], reverse=True)
        for i, err in enumerate(errors[:5], 1):
            print(f"\n{i}. {err['image']} (CER: {err['cer']:.4f})")
            print(f"   GT: {err['ground_truth']}")
            print(f"   Pred: {err['predicted']}")

    # Save detailed results
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': avg_metrics,
            'detailed_results': results,
            'errors': errors
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detailed results saved to: evaluation_results.json")

    return avg_metrics, results

if __name__ == "__main__":
    model_path = "./saved_models/best_accuracy.pth"
    test_data_dir = "dataset/raw/test"
    label_file = "dataset/labels/test_labels.txt"

    evaluate_on_dataset(model_path, test_data_dir, label_file)
