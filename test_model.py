import easyocr
import cv2
import os
from pathlib import Path

def test_custom_model(model_path, test_image_path):
    """
    Test custom trained model
    """
    # Initialize reader dengan custom model
    reader = easyocr.Reader(
        ['en'],  # Language (sesuaikan)
        gpu=True,
        model_storage_directory='./models/',
        user_network_directory=model_path,
        recog_network='custom'
    )

    # Read image
    result = reader.readtext(test_image_path, detail=1)

    # Print results
    print(f"\nResults for: {test_image_path}")
    print("-" * 60)
    for bbox, text, confidence in result:
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.4f}")
        print(f"BBox: {bbox}")
        print()

    return result

def visualize_results(image_path, result, output_path):
    """
    Visualisasi hasil OCR pada gambar
    """
    img = cv2.imread(image_path)

    for bbox, text, conf in result:
        # Extract coordinates
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Draw rectangle
        color = (0, 255, 0) if conf > 0.8 else (0, 165, 255)  # Green if confident, Orange if not
        cv2.rectangle(img, top_left, bottom_right, color, 2)

        # Draw text
        cv2.putText(
            img,
            f"{text} ({conf:.2f})",
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Save result
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")

# Test pada single image
if __name__ == "__main__":
    model_path = "./saved_models/best_accuracy.pth"
    test_image = "dataset/test/sample_001.jpg"

    result = test_custom_model(model_path, test_image)
    visualize_results(test_image, result, "output_visualized.jpg")
