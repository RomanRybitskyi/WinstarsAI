# MNIST Classifier Project

This project implements three different classifiers for the MNIST dataset of handwritten digits: Random Forest, Feed-Forward Neural Network (NN), and Convolutional Neural Network (CNN). Each classifier adheres to a common interface (`MnistClassifierInterface`) and is wrapped by a unified `MnistClassifier` class for consistent usage. The solution leverages scikit-learn for Random Forest and PyTorch for the neural networks, with support for GPU acceleration where available.

## Project Overview

The goal is to classify 28x28 grayscale images of handwritten digits (0-9) from the MNIST dataset using three distinct machine learning approaches:
1. **Random Forest**: An ensemble method using decision trees.
2. **Feed-Forward Neural Network**: A fully connected neural network with dropout regularization.
3. **Convolutional Neural Network**: A CNN designed to capture spatial patterns in images.

The code is structured to:
- Define an abstract interface for classifiers.
- Implement three classifier classes.
- Provide a wrapper class to unify their usage.
- Include a main script to train and evaluate all models.

## File Structure
- `main.py`: The main script containing all code (interface, classifiers, and execution logic).
- `README.md`: This file.
- `requirements.txt`: List of required Python libraries.
- `./data/`: Directory where MNIST data is automatically downloaded by PyTorch.

## Solution Explanation

### 1. `MnistClassifierInterface`
- **Purpose**: An abstract base class defining the contract for all classifiers.
- **Methods**:
  - `train(X_train, y_train)`: Trains the model on input images and labels.
  - `predict(X_test)`: Predicts labels for test images.
- **Implementation**: Uses Python’s `abc` module to enforce method implementation in subclasses.

### 2. `RFClassifier` (Random Forest)
- **Algorithm**: Builds 100 decision trees, using majority voting for predictions.
- **Input**: Flattens `[n_samples, 28, 28]` images to `[n_samples, 784]`.
- **Library**: scikit-learn (`RandomForestClassifier`).
- **Parameters**: `n_estimators=100` (trees), `random_state=42` (reproducibility).

### 3. `NNClassifier` (Feed-Forward Neural Network)
- **Architecture**: 
  - Flattens images to 784 features.
  - Layers: 784 → 256 (ReLU, 20% Dropout) → 128 (ReLU, 20% Dropout) → 10.
- **Training**: 10 epochs, batch size 64, Adam optimizer, CrossEntropyLoss.
- **Library**: PyTorch.

### 4. `CNNClassifier` (Convolutional Neural Network)
- **Architecture**:
  - Conv2d (1 → 16, 3x3, padding=1) → ReLU → MaxPool (2x2).
  - Conv2d (16 → 32, 3x3, padding=1) → ReLU → MaxPool (2x2).
  - Flatten (32x7x7 = 1568) → Linear (1568 → 128, ReLU) → Linear (128 → 10).
- **Input**: Reshapes `[n_samples, 28, 28]` to `[n_samples, 1, 28, 28]` (adds channel).
- **Training**: 10 epochs, batch size 64, Adam optimizer, CrossEntropyLoss.
- **Library**: PyTorch.

### 5. `MnistClassifier`
- **Purpose**: Wrapper class that selects and instantiates one of the above classifiers based on an algorithm parameter (`'rf'`, `'nn'`, `'cnn'`).
- **Usage**: Ensures consistent `train` and `predict` calls regardless of the underlying model.

### 6. Main Execution
- **Data Loading**: Downloads MNIST via `torchvision.datasets.MNIST`.
- **Execution**: Trains and evaluates all three models, printing accuracy for each.

## Setup Instructions

### Prerequisites
- **Python**: Version 3.7 or higher.
- **Operating System**: Windows, macOS, or Linux.
- **GPU (optional)**: For faster training of neural networks (requires CUDA-compatible GPU and drivers).

### Step-by-Step Setup
1. **Clone or Download the Project**
   - Clone this repository or download the files to a local directory.

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv .venv
   ```
   Activate it:

    Windows: `.venv\Scripts\activate`
    macOS/Linux: `source .venv/bin/activate`
   
3. **Install Dependencies**
   Ensure you have pip installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   This installs all required libraries listed in `requirements.txt`.

4. **Run the Code**
   From the project directory, execute:
   ```bash
   python main.py
   ```
   The script will:
    - Download MNIST data to `./data/ `(first run only).
    - Train and evaluate RF, NN, and CNN models.
    - Print accuracy for each.
