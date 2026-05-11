import cv2
import albumentations as A
from pathlib import Path
import zipfile

# Make paths absolute based on where the Python script is located
script_dir = Path(__file__).resolve().parent
input_dir = script_dir / "personnel"
output_dir = script_dir / "augmented_personnel"
output_dir.mkdir(parents=True, exist_ok=True)

zip_path = script_dir / "personnel.zip"
if zip_path.exists():
    print(f"Unzipping {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(script_dir)


# Dynamically iterate over all PNG images in the directory
image_paths = list(input_dir.glob("*.png"))
if not image_paths:
    print(f"No PNG images found in: {input_dir}")

for img_path in image_paths:
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Warning: Failed to load {img_path}")
        continue

    for i in range(50):
        # Progressively increase the intensity of transformations based on `i`
        transform = A.Compose([
            A.Perspective(scale=(0.05, 0.10 + i * 0.0018), p=0.7),
            A.Affine(scale=(0.9 - i * 0.0036, 1.1 + i * 0.0036), rotate=(-10 - i * 0.36, 10 + i * 0.36), shear=(-5 - i * 0.2, 5 + i * 0.2), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1 + i * 0.0018, contrast_limit=0.1 + i * 0.0018, p=0.7),
            A.MotionBlur(blur_limit=3 + (i // 8) * 2, p=0.3),
            A.GaussNoise(std_range=(0.03, 0.06 + i * 0.001), p=0.3),
            A.ImageCompression(quality_range=(max(10, int(80 - i * 0.96)), max(20, int(100 - i * 0.36))), p=0.4),
            A.CoarseDropout(num_holes_range=(1, max(2, 2 + i // 15)), hole_height_range=(10, 20 + i // 5), hole_width_range=(10, 20 + i // 5), p=0.3),
        ])

        augmented = transform(image=image)["image"]
    
        train_path = output_dir / "train" / img_path.stem / f"{img_path.stem}_augmented_{i}{img_path.suffix}"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(train_path), augmented)
        # print(f"Exported: {train_path}")

    for j in range(10):
        transform_test = A.Compose([
            A.Perspective(scale=(0.05, 0.10 + j * 0.01), p=0.7),
            A.Affine(scale=(0.9 - j * 0.02, 1.1 + j * 0.02), rotate=(-10 - j * 2.0, 10 + j * 2.0), shear=(-5 - j * 1.0, 5 + j * 1.0), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.1 + j * 0.01, contrast_limit=0.1 + j * 0.01, p=0.7),
            A.MotionBlur(blur_limit=3 + (j // 2) * 2, p=0.3),
            A.GaussNoise(std_range=(0.03, 0.06 + j * 0.006), p=0.3),
            A.ImageCompression(quality_range=(max(10, int(80 - j * 5)), max(20, int(100 - j * 2))), p=0.4),
            A.CoarseDropout(num_holes_range=(1, max(2, 2 + j // 3)), hole_height_range=(10, 20 + j), hole_width_range=(10, 20 + j), p=0.3),
        ])
        augmented_test = transform_test(image=image)["image"]
        test_aug_path = output_dir / "test" / img_path.stem / f"{img_path.stem}_augmented_{j}{img_path.suffix}"
        test_aug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(test_aug_path), augmented_test)
        # print(f"Exported: {test_aug_path}")

        
    test_path = output_dir / "test" / img_path.stem / f"{img_path.stem}_original{img_path.suffix}"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), image)