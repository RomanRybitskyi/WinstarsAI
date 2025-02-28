import torch
import torchvision.models as models  # Correct import for ResNet
import torchvision.transforms as transforms
from PIL import Image

# Define a class to predict animal classes from images using a trained model
class ImageClassifierPredictor:
    # Initialize the predictor with a trained model path and class names
    def __init__(self, model_path, class_names):
        # Load the ResNet34 architecture without pre-trained weights (we’ll load custom weights)
        self.model = models.resnet34(pretrained=False)
        # Adjust the final fully connected (fc) layer to match the number of classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_names))
        # Load the trained weights from the specified file (e.g., from image_train.py)
        self.model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode (disables dropout and batch normalization updates)
        self.model.eval()
        # Store the list of class names (e.g., ['dog', 'horse', ...]) for mapping predictions
        self.class_names = class_names
        # Define a sequence of image transformations for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224 pixels (ResNet’s expected size)
            transforms.ToTensor(),  # Convert image to PyTorch tensor (HWC to CHW, 0-255 to 0-1)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])

    # Predict the animal class of a given image
    def predict(self, image):
        # Apply transformations to the input image and convert to a tensor
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension for model input
        # Perform inference without gradient computation (saves memory and speeds up prediction)
        with torch.no_grad():
            # Run the model on the image tensor to get prediction logits
            outputs = self.model(img_tensor)
            # Get the predicted class index by finding the maximum logit value
            _, predicted = torch.max(outputs, 1)
        # Map the predicted index to the corresponding class name and return it
        return self.class_names[predicted.item()]

# Main execution block to test the predictor
if __name__ == "__main__":
    # Define the list of animal class names (must match training order)
    class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
    # Create a predictor instance with the trained model path and class names
    predictor = ImageClassifierPredictor("models/image_model.pth", class_names)
    # Load a test image from the specified path using PIL
    img = Image.open("data/archive/raw-img/gallina/21.jpeg")
    # Predict the animal class of the image
    result = predictor.predict(img)
    # Print the predicted class name
    print(f"Predicted animal: {result}")