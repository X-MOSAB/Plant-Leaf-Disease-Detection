from pathlib import Path
import tensorflow as tf

# =========================
# Configuration
# =========================

DATASET_PATH = Path(
    "data/raw/Plant_leave_diseases_dataset_without_augmentation"
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# =========================
# Load Training Dataset
# =========================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# =========================
# Load Validation + Test
# =========================

validation_test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

total_batches = len(validation_test_dataset)
test_batches = total_batches // 2

validation_dataset = validation_test_dataset.take(test_batches)
test_dataset = validation_test_dataset.skip(test_batches)


# =========================
# Class Names
# =========================

class_names = train_dataset.class_names
num_classes = len(class_names)


# =========================
# Data Augmentation
# =========================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")


# =========================
# Normalize Pixel Values
# =========================

normalization = tf.keras.layers.Rescaling(
    1.0 / 255
)


# =========================
# Prepare Train Dataset
# =========================

train_dataset = train_dataset.map(
    lambda images, labels: (
        data_augmentation(images, training=True),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# Prepare Validation Dataset
# =========================

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# Prepare Test Dataset
# =========================

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# =========================
# Improve Performance
# =========================

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# =========================
# Dataset Information
# =========================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print(f"Number of classes: {num_classes}")

print("\nNumber of batches:")
print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")