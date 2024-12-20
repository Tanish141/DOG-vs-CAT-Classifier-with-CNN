### 🐶🐱 Dog vs Cat Classifier Using CNN

This project implements a Convolutional Neural Network (CNN) in Python using TensorFlow and Keras to classify images as either dogs or cats. The model is trained on an image dataset with data augmentation and evaluated for accuracy and loss.

---

### 🚀 Features

Image Augmentation: Uses ImageDataGenerator to augment training data for better generalization.

Binary Classification: Classifies input images into two categories: Dogs or Cats.

Custom Prediction: Allows users to test the model with custom images.

Visualization: Plots training and validation accuracy/loss over epochs.

---

### 🛠️ Technologies Used

Python 🐍

TensorFlow & Keras: For building and training the CNN.

Matplotlib: For visualizing training performance.

NumPy: For numerical computations.

---

### 📂 Dataset Structure

The dataset should be organized as follows:

Dataset/
    training_set/
        dogs/
        cats/
    test_set/
        dogs/
        cats/

Training Set: Contains images of dogs and cats in respective subdirectories.

Test Set: Contains images of dogs and cats for validation.

---

### ⚙️ How It Works

Data Augmentation: Applies transformations such as rotation, width/height shift, shear, zoom, and horizontal flips to enhance training data.

CNN Architecture:

4 convolutional layers with ReLU activation.

MaxPooling layers to reduce spatial dimensions.

Fully connected (dense) layers, culminating in a single sigmoid output for binary classification.

Model Training: Uses binary_crossentropy loss and the Adam optimizer to minimize the error.

Visualization: Training and validation performance metrics are plotted.

Custom Prediction: Allows testing of the model with custom images for classification.

---

### 🛠️ Setup Instructions

Clone the Repository:

git clone https://github.com/Tanish141/dog-vs-cat-classifier.git
cd dog-vs-cat-classifier

Install Dependencies:

pip install tensorflow numpy matplotlib

Prepare the Dataset:
Ensure the dataset is structured as described above.

Run the Training Script:

python dog_vs_cat_classifier.py

Test the Model:
Provide the path to a custom image in the script and run it:

predict_image(model, 'path/to/your-image.jpeg')

---

### 📊 Results

The model provides the following insights during training:

Training and Validation Accuracy: Shows how well the model is learning over epochs.

Training and Validation Loss: Indicates overfitting or underfitting trends.

---

### 🏅 Key Learning Outcomes

Hands-on experience with CNN architecture for image classification.

Using data augmentation to improve model generalization.

Visualizing training metrics for model evaluation.

Testing a trained model with unseen data.

---

### 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request. Let’s make this project better together. 🌟

---

### 📧 Contact

For any queries or suggestions, reach out via:

Email: mrtanish14@gmail.com

GitHub: https://github.com/Tanish141

---

### 🎉 Happy Coding!
