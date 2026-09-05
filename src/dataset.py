from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =========================
# Configuration
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Plant_leave_diseases_dataset_without_augmentation"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =========================
# Get All Image Paths
# =========================

class_dirs = sorted(
    [folder for folder in DATASET_PATH.iterdir() if folder.is_dir()]
)

class_names = [folder.name for folder in class_dirs]
num_classes = len(class_names)

image_paths = []
labels = []

for class_index, class_dir in enumerate(class_dirs):
    for image_path in class_dir.iterdir():
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(str(image_path))
            labels.append(class_index)


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total images: {len(image_paths)}")
print(f"Number of classes: {num_classes}")


# =========================
# Stratified Split
# =========================

# 80% Train, 20% Temporary
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=SEED,
    stratify=labels,
)

# Split temporary 50/50
# => 10% Validation, 10% Test
validation_paths, test_paths, validation_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_labels,
)


# =========================
# Dataset Creation
# =========================

def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    return image, label


def create_dataset(paths, labels, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =========================
# Create Train / Validation / Test
# =========================

train_dataset = create_dataset(
    train_paths,
    train_labels,
    shuffle=True
)

validation_dataset = create_dataset(
    validation_paths,
    validation_labels,
    shuffle=False
)

test_dataset = create_dataset(
    test_paths,
    test_labels,
    shuffle=False
)


# =========================
# Print Split Information
# =========================

print("\nSplit sizes:")
print(f"Train:      {len(train_paths)}")
print(f"Validation: {len(validation_paths)}")
print(f"Test:       {len(test_paths)}")

print("\nNumber of batches:")
print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")


# =========================
# Verify Every Class Exists
# =========================

print("\nClass distribution check:")

for class_index, class_name in enumerate(class_names):
    train_count = train_labels.count(class_index)
    validation_count = validation_labels.count(class_index)
    test_count = test_labels.count(class_index)

    print(
        f"{class_name}: "
        f"Train={train_count}, "
        f"Validation={validation_count}, "
        f"Test={test_count}"
    )