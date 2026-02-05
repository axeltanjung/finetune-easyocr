from collections import Counter
import matplotlib.pyplot as plt

def analyze_character_distribution(label_file):
    """
    Analisis distribusi karakter dalam dataset
    """
    all_chars = []

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                text = parts[1]
                all_chars.extend(list(text))

    # Count characters
    char_counts = Counter(all_chars)

    print(f"Total characters: {len(all_chars)}")
    print(f"Unique characters: {len(char_counts)}")
    print("\nTop 20 most common characters:")
    for char, count in char_counts.most_common(20):
        if char == ' ':
            char = '<SPACE>'
        elif char == '\t':
            char = '<TAB>'
        elif char == '\n':
            char = '<NEWLINE>'
        print(f"  '{char}': {count} ({count/len(all_chars)*100:.2f}%)")

    # All unique characters
    print("\nAll unique characters:")
    unique_chars = sorted(char_counts.keys())
    print(''.join(unique_chars))

    # Generate character string for config
    print("\nCharacter string untuk config:")
    print(f'character = "{' '.join(unique_chars)}"')

    return char_counts

if __name__ == "__main__":
    print("Analyzing training set...")
    train_chars = analyze_character_distribution("dataset/labels/train_labels.txt")

    print("\n" + "="*60 + "\n")

    print("Analyzing validation set...")
    val_chars = analyze_character_distribution("dataset/labels/val_labels.txt")

    # Check if val set has chars not in train set
    train_set = set(train_chars.keys())
    val_set = set(val_chars.keys())

    extra_in_val = val_set - train_set
    if extra_in_val:
        print(f"\n⚠️  Warning: Validation set has {len(extra_in_val)} characters not in training set:")
        print(f"  {extra_in_val}")
