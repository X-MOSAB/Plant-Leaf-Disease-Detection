# 🌱 Plant Disease Detection using Deep Learning

A deep learning project for detecting plant diseases from leaf images using Transfer Learning with EfficientNetB0.

## 📌 Project Overview

The goal of this project is to build an image classification system capable of identifying plant diseases from leaf images.

The model classifies images into **39 different plant disease and healthy-plant categories**.

## 🧠 Model

We use **EfficientNetB0** pretrained on ImageNet as the backbone.

### Architecture

```text
Input Image
     ↓
Data Augmentation
     ↓
EfficientNetB0
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense Layer
     ↓
39 Classes

The pretrained EfficientNetB0 backbone was initially frozen, and a classification head was trained for the plant disease classes.

📊 Dataset
Total Images: 55,448
Number of Classes: 39
Training Set: approximately 80%
Validation Set: approximately 10%
Test Set: approximately 10%

A stratified split was used to ensure that all classes are represented across training, validation, and test sets.

🔧 Data Preprocessing

The image pipeline includes:

Image resizing to 224 × 224
Batch processing
Data augmentation on the training set
Horizontal flipping
Random rotation
Random zoom
Random contrast
📈 Results
Validation Performance
Validation Accuracy: 97.24%
Test Performance
Test Accuracy: 97.58%
Test Loss: 0.0853
Macro F1-Score: 96.84%
Weighted F1-Score: 97.59%
📌 Per-Class Performance

The model performs strongly across most of the 39 classes.

Some of the more challenging classes include:

Tomato___Target_Spot
Potato___healthy
Tomato___Early_blight
Corn___Cercospora_leaf_spot Gray_leaf_spot
📊 Confusion Matrix

The confusion matrix is available here:

results/confusion_matrix.png
🔮 Prediction

The project includes a prediction script that can classify a single image.

Example:

python src/predict.py "path/to/image.jpg"

Example output:

==================================================
TOP 3 PREDICTIONS
==================================================
1. Tomato___healthy: 100.00%
2. Tomato___Septoria_leaf_spot: 0.00%
3. Tomato___Target_Spot: 0.00%
📁 Project Structure
plant_disease/
│
├── app/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── results/
│   └── confusion_matrix.png
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
🚀 Future Improvements

Planned improvements include:

Fine-tuning EfficientNetB0
Improving performance on difficult classes
Building a web interface for image upload
Adding prediction confidence visualization
Deploying the application
👨‍💻 Technologies
Python
TensorFlow / Keras
NumPy
Scikit-learn
Pillow
Matplotlib
Git & GitHub
📜 License
This project is developed for educational and academic purposes.