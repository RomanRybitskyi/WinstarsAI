# Importing all the necessary libraries and modules
import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


# Interface definition
class MnistClassifierInterface(ABC):
    """Abstract base class defining the interface for MNIST classifiers."""

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the classifier with training data.

        Args:
            X_train: Training images (numpy array of shape [n_samples, 28, 28])
            y_train: Training labels (numpy array of shape [n_samples])
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """Predict labels for test data.

        Args:
            X_test: Test images (numpy array of shape [n_samples, 28, 28])
        Returns:
            Predicted labels (numpy array of shape [n_samples])
        """
        pass


# Random Forest Classifier
class RFClassifier(MnistClassifierInterface):
    def __init__(self):
        """Initialize the Random Forest model."""
        # Create RandomForestClassifier with 100 trees and fixed random seed
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """Train the Random Forest model."""
        # Flatten images from [n_samples, 28, 28] to [n_samples, 784]
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        # Fit the model to the training data
        self.model.fit(X_train_flat, y_train)

    def predict(self, X_test):
        """Predict labels for test images."""
        # Flatten test images similarly
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        # Return predictions as numpy array
        return self.model.predict(X_test_flat)


# Feed-Forward Neural Network
class NNClassifier(MnistClassifierInterface):
    def __init__(self, batch_size=64):
        """Initialize the neural network model."""
        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size  # Number of samples per training batch

        # Define the network architecture using Sequential
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten [batch_size, 28, 28] to [batch_size, 784]
            nn.Linear(784, 256),  # Fully connected layer: 784 inputs -> 256 outputs
            nn.ReLU(),  # ReLU activation for nonlinearity
            nn.Dropout(0.2),  # 20% dropout to prevent overfitting
            nn.Linear(256, 128),  # Fully connected layer: 256 -> 128
            nn.ReLU(),  # ReLU activation
            nn.Dropout(0.2),  # Another 20% dropout
            nn.Linear(128, 10)  # Output layer: 128 -> 10 (one per digit)
        ).to(self.device)  # Move model to GPU/CPU

        self.criterion = nn.CrossEntropyLoss()  # Loss function for classification
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, X_train, y_train):
        """Train the neural network with batched data."""
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # Create dataset and dataloader for batching
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # shuffle=True: Randomizes data order each epoch

        self.model.train()  # Set model to training mode (enables dropout)
        for epoch in range(10):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()  # Clear previous gradients
                outputs = self.model(batch_X)  # Forward pass
                loss = self.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backward pass (compute gradients)
                self.optimizer.step()  # Update weights

    def predict(self, X_test):
        """Predict labels for test images."""
        self.model.eval()  # Set model to evaluation mode (disables dropout)
        X_test = torch.FloatTensor(X_test)  # Convert to tensor
        dataset = TensorDataset(X_test)  # Create dataset (no labels needed)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []  # Store predictions from all batches
        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)  # Forward pass
                _, predicted = torch.max(outputs.data, 1)  # Get class with max score
                predictions.append(predicted.cpu().numpy())  # Store predictions
        return np.concatenate(predictions)  # Combine into single array


# Convolutional Neural Network
class CNNClassifier(MnistClassifierInterface):
    def __init__(self, batch_size=64):
        """Initialize the CNN model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size  # Number of samples per batch

        # Define the CNN architecture
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 1 input channel -> 16 output channels
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(2),  # 2x2 max pooling (halves size)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16 -> 32 channels
            nn.ReLU(),  # ReLU activation
            nn.MaxPool2d(2),  # Halves size again
            nn.Flatten(),  # Flatten to [batch_size, 32*7*7]
            nn.Linear(32 * 7 * 7, 128),  # Fully connected: 1568 -> 128
            nn.ReLU(),  # ReLU activation
            nn.Linear(128, 10)  # Output layer: 128 -> 10
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()  # Classification loss
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, X_train, y_train):
        """Train the CNN with batched data."""
        # Reshape to include channel dimension: [n_samples, 1, 28, 28]
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)

        # Create dataset and dataloader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()  # Training mode
        for epoch in range(10):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        """Predict labels for test images."""
        self.model.eval()  # Evaluation mode
        X_test = X_test.reshape(-1, 1, 28, 28)  # Add channel dimension
        X_test = torch.FloatTensor(X_test)
        dataset = TensorDataset(X_test)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []
        with torch.no_grad():
            for batch_X, in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.append(predicted.cpu().numpy())
        return np.concatenate(predictions)


# Wrapper class to unify all classifiers
class MnistClassifier:
    """Wrapper class to provide a consistent interface for all classifiers."""

    def __init__(self, algorithm):
        """Initialize with the specified algorithm."""
        if algorithm == 'rf':
            self.classifier = RFClassifier()  # Random Forest
        elif algorithm == 'nn':
            self.classifier = NNClassifier()  # Feed-Forward NN
        elif algorithm == 'cnn':
            self.classifier = CNNClassifier()  # Convolutional NN
        else:
            raise ValueError("Algorithm must be 'rf', 'nn', or 'cnn'")

    def train(self, X_train, y_train):
        """Train the selected classifier."""
        self.classifier.train(X_train, y_train)

    def predict(self, X_test):
        """Predict using the selected classifier."""
        return self.classifier.predict(X_test)


if __name__ == "__main__":
    # Define data transformation: convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Extract numpy arrays from datasets
    X_train = train_dataset.data.numpy()
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    # Test all three models
    algorithms = ['rf', 'nn', 'cnn']
    for algo in algorithms:
        print(f"\nTraining {algo} model...")
        classifier = MnistClassifier(algo)  # Create classifier instance
        classifier.train(X_train, y_train)  # Train the model
        predictions = classifier.predict(X_test)  # Get predictions
        accuracy = np.mean(predictions == y_test)  # Calculate accuracy
        print(f"{algo} Accuracy: {accuracy:.4f}")

        # Clear GPU memory if using neural networks
        if algo in ['nn', 'cnn'] and torch.cuda.is_available():
            torch.cuda.empty_cache()
