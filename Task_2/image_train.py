import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch

# Define a class to train an image classification model
class ImageClassifierTrainer:
    # Initialize the trainer with a pre-trained model and preprocessing settings
    def __init__(self, num_classes=10):
        # Load a pre-trained ResNet34 model (34-layer Residual Network)
        self.model = models.resnet34(pretrained=True)
        # Modify the final fully connected (fc) layer to match the number of classes (e.g., 10 animals)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        # Set the device: use GPU (CUDA) if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the selected device for computation
        self.model.to(self.device)
        # Define a sequence of image transformations for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels (ResNet’s expected size)
            transforms.ToTensor(),  # Convert images to PyTorch tensors (HWC to CHW format)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

    # Train the model on the provided image dataset
    def train(self, data_path, epochs=10, batch_size=32):
        # Load the dataset from a folder structure
        dataset = ImageFolder(data_path, transform=self.transform)
        # Create a DataLoader to batch and shuffle the dataset for training
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Define the loss function (cross-entropy for multi-class classification)
        criterion = torch.nn.CrossEntropyLoss()
        # Use Adam optimizer with a learning rate of 0.001 to update model parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Set the model to training mode (enables dropout and batch normalization updates)
        self.model.train()
        # Train for the specified number of epochs
        for epoch in range(epochs):
            total_loss = 0  # Track cumulative loss for the epoch
            # Iterate over batches of images and labels
            for images, labels in loader:
                # Move images and labels to the device (CPU or GPU)
                images, labels = images.to(self.device), labels.to(self.device)
                # Clear previous gradients to avoid accumulation
                optimizer.zero_grad()
                # Forward pass: predict class logits from images
                outputs = self.model(images)
                # Compute the loss between predictions and true labels
                loss = criterion(outputs, labels)
                # Backward pass: calculate gradients of the loss with respect to model parameters
                loss.backward()
                # Update model weights using the optimizer
                optimizer.step()
                # Add batch loss to total for reporting
                total_loss += loss.item()
            # Print average loss for the epoch
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

    # Save the trained model’s weights to a file
    def save(self, path):
        # Save only the model’s state dictionary (weights) to the specified path
        torch.save(self.model.state_dict(), path)

# Main execution block to test the trainer
if __name__ == "__main__":
    # Create a trainer instance with 10 classes
    trainer = ImageClassifierTrainer(num_classes=10)
    # Train the model on the dataset located at the specified path
    trainer.train("data/archive/raw-img/")
    # Save the trained model weights to a file
    trainer.save("models/image_model.pth")