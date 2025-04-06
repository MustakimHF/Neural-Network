# 🧠 TensorFlow Neural Network Classifier

A simple neural network built with TensorFlow and Keras to classify synthetic data into multiple classes. This project demonstrates key steps like data preprocessing, model training, early stopping, and visualizing training progress.

---

### 🚀 What This Project Does

- Creates synthetic classification data
- Preprocesses input features using standardization
- Defines a multi-layer neural network using Keras Sequential API
- Trains with early stopping to prevent overfitting
- Evaluates the model and visualizes training history

---

### 🧰 Tech Stack

- **TensorFlow / Keras** – for building and training the neural network
- **NumPy** – for data generation and manipulation
- **scikit-learn** – for train/test split and standardization
- **Matplotlib** – to plot training metrics (optional)

---

### 🖼️ Quick Preview

![image](https://github.com/user-attachments/assets/045455c4-198f-4758-9aac-fbd679152654)



---
### ▶️ How to Run

1. Clone this repo or copy the script into a `.py` file.
2. Make sure you have the required libraries installed:
```bash
pip install tensorflow scikit-learn matplotlib
```
3. Run the script:
```bash
python neural.py
```

---

### 📈 Output

- Accuracy and loss for training and validation sets
- Final test set accuracy
- Line plots showing how accuracy and loss evolved over training epochs

---

### 📎 Notes

- You can easily swap in your own dataset by replacing the synthetic data generation block.
- Model architecture and training parameters are flexible and easy to adjust.
