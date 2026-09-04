from pathlib import Path
from PIL import Image

# Path to the raw dataset
DATASET_PATH = Path("data/raw/Plant_leave_diseases_dataset_without_augmentation")


def inspect_dataset(dataset_path: Path):
    classes = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

    print(f"Number of classes: {len(classes)}")
    print("\nClasses and image counts:\n")

    total_images = 0

    for class_folder in sorted(classes):
        image_count = 0

        for image_path in class_folder.iterdir():
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_count += 1

        total_images += image_count
        print(f"{class_folder.name}: {image_count} images")

    print(f"\nTotal images: {total_images}")


if __name__ == "__main__":
    inspect_dataset(DATASET_PATH)