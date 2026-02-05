import albumentations as A
import cv2

def get_training_augmentation():
    """
    Augmentation pipeline untuk training
    """
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.5
        ),

        # Noise
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.3),

        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Brightness & Contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),

        # Additional
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    ])

# Apply augmentation
transform = get_training_augmentation()
image = cv2.imread('sample.jpg')
augmented = transform(image=image)['image']
