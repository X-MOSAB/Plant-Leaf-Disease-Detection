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
# 80% of the data
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
# Load Validation + Test Pool
# 20% of the data
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


# =========================
# Split the 20% pool equally
# 10% Validation + 10% Test
# =========================

total_batches = len(validation_test_dataset)
test_batches = total_batches // 2

validation_dataset = validation_test_dataset.take(test_batches)
test_dataset = validation_test_dataset.skip(test_batches)


# =========================
# Class Names
# =========================

class_names = train_dataset.class_names


# =========================
# Dataset Information
# =========================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print(f"Number of classes: {len(class_names)}")

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print("\nNumber of batches:")
print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(validation_dataset)}")
print(f"Test:       {len(test_dataset)}")