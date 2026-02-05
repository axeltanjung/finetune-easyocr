# Finetuning EasyOCR with Dataset Custom

##  Character Set Configuration
PENTING: Sesuaikan character set dengan kebutuhan dataset Anda.

Contoh untuk berbagai kasus:

Angka saja (digit recognition):

- character = "0123456789"

Alphanumeric + basic punctuation:

- character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,- "

Indonesian text (dengan huruf khusus):

- character = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ àáèéìíòóùú"

Full ASCII:

```
import string
character = string.printable[:-5]  # exclude \t\n\r\v\f
```

Tips Character Set:

- Include SEMUA karakter yang muncul di dataset
- Jangan include karakter yang tidak ada di dataset
- Case sensitive: 'A' != 'a'
- Space character penting untuk text dengan spasi


## Training dari Scratch
Training model baru dari nol (tanpa pretrained weights):

```python train.py \
  --train_data dataset/lmdb/train \
  --valid_data dataset/lmdb/validation \
  --select_data train \
  --batch_ratio 1.0 \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --batch_size 192 \
  --num_iter 300000 \
  --valInterval 2000 \
  --saved_model ./saved_models/ \
  --lr 1.0 \
  --workers 4 \
  --manualSeed 1111 \
  --rgb False
```

Penjelasan parameter:

- --train_data: Path ke LMDB training dataset
- --valid_data: Path ke LMDB validation dataset
- --Transformation TPS: Gunakan Thin Plate Spline untuk handle distortion
- --FeatureExtraction ResNet: Gunakan ResNet untuk feature extraction
- --SequenceModeling BiLSTM: Bidirectional LSTM untuk sequence modeling
- --Prediction Attn: Attention mechanism untuk prediction
- --batch_size 192: Batch size (reduce jika OOM)
- --num_iter 300000: Total training iterations
- --valInterval 2000: Validasi setiap 2000 iterations
- --lr 1.0: Learning rate (untuk Adadelta)
- --workers 4: Jumlah data loader workers
- --rgb False: Grayscale images


## Finetuning dari Pretrained Model
Finetuning dari model pretrained EasyOCR untuk accelerate training:

Download pretrained model:

- Latin alphabet model (English)
```
wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip
unzip latin_g2.zip
```

- Atau model bahasa lain
Indonesian
```
wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/indonesian_g2.zip
```

Finetune command:

```python train.py \
  --train_data dataset/lmdb/train \
  --valid_data dataset/lmdb/validation \
  --select_data train \
  --batch_ratio 1.0 \
  --Transformation TPS \
  --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM \
  --Prediction Attn \
  --batch_size 192 \
  --num_iter 50000 \
  --valInterval 1000 \
  --saved_model ./finetuned_models/ \
  --lr 0.0001 \
  --workers 4 \
  --manualSeed 1111 \
  --saved_model latin_g2.pth \
  --FT
```

Perbedaan penting saat finetuning:

- --saved_model latin_g2.pth: Path ke pretrained model
- --FT: Flag untuk enable finetuning mode
- --lr 0.0001: Learning rate LEBIH KECIL (1000x lebih kecil)
- --num_iter 50000: Iterasi LEBIH SEDIKIT (6x lebih sedikit)

## Monitoring Training dengan TensorBoard
EasyOCR otomatis log metrics ke TensorBoard. Buka di terminal baru:

```tensorboard --logdir=./saved_models/```
Buka browser: http://localhost:6006

Metrics yang perlu dimonitor:

- Training Loss: Harus menurun secara konsisten

- Good: Smooth decrease
- Bad: Plateau atau naik
- Solution jika bad: Reduce learning rate
- Validation Accuracy: Target > 95% untuk dataset berkualitas

- Monitor overfitting: jika train acc naik tapi val acc turun
- Solution: Early stopping, regularization, atau augmentation
- Character Error Rate (CER): Semakin rendah semakin baik

- Good: < 0.05 (5% error)
- Acceptable: 0.05 - 0.15
- Bad: > 0.15
- Learning Rate Schedule: Track apakah LR sudah optimal

## Testing dan Evaluasi Model
### Evaluasi Comprehensive pada Test Set

Output expectation

```
Loading model...
Evaluating on 1000 samples...
100%|████████████████████| 1000/1000 [02:15<00:00,  7.38it/s]

============================================================
EVALUATION RESULTS
============================================================
Total samples: 1000
Average CER: 0.0342 (3.42%)
Average WER: 0.0891 (8.91%)
Exact Match Accuracy: 0.8520 (85.20%)
Significant errors (CER > 30%): 15

CER Distribution:
  Min: 0.0000
  Max: 0.8571
  Median: 0.0200
  Std Dev: 0.0876

Top 5 worst predictions:
1. img_789.jpg (CER: 0.8571)
   GT: Invoice #INV-2024-001
   Pred: Jnvoice #1NV-Z0Z4-00l

✅ Detailed results saved to: evaluation_results.json
```

# Best Practices dan Rekomendasi
## 1. Dataset Preparation
Rekomendasi ukuran dataset:

- Minimum: 1,000 samples
- Good: 5,000 - 10,000 samples
- Excellent: 50,000+ samples

Split ratio:

- Training: 80%
- Validation: 10%
- Test: 10%

Data quality checklist:

- Semua images readable dan clear
- Labels 100% accurate (audit manual)
- Diverse fonts, sizes, styles
- Various backgrounds dan lighting
- Balanced character distribution
- No duplicate images

## 2. Training Strategy
For small dataset (< 5k samples):

- Gunakan pretrained model + finetuning
- Heavy augmentation
- Small learning rate (1e-4 to 1e-5)
- Monitor overfitting closely
- Early stopping dengan patience=10

For large dataset (> 50k samples):

- Train from scratch OK
- Moderate augmentation
- Standard learning rate (1.0 untuk Adadelta)
- Longer training (300k iterations)
- Less prone to overfitting

### 3. Monitoring dan Checkpointing

Save checkpoint strategy
```if iteration % save_interval == 0:
    # Save checkpoint
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_accuracy': val_accuracy,
    }
    torch.save(checkpoint, f'checkpoint_iter_{iteration}.pth')
```

Keep best model

```
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy
    torch.save(model.state_dict(), 'best_accuracy.pth')

if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_loss.pth')
```

### 4. Hyperparameter Tuning Priority
Order of importance untuk tuning:

- Learning Rate (paling penting)

- Start: 1.0 (Adadelta) atau 1e-3 (Adam)
- Tune dengan LR finder
- Batch Size

- Larger = more stable tapi butuh memory
- Typical: 64-256
- Architecture

- TPS + ResNet + BiLSTM + Attn (recommended)
- Experiment dengan VGG atau RCNN
- Image Size

- Default: 64x256
- Larger untuk text detail lebih baik
- Augmentation Strength

- Start conservative, increase jika overfit
