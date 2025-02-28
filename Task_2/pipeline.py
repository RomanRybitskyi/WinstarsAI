from ner_inference import NERPredictor
from image_inference import ImageClassifierPredictor
from PIL import Image


# Define a class to combine NER and image classification for verification
class VerificationPipeline:
    # Initialize the pipeline with paths to the trained models and class names
    def __init__(self, ner_model_path, image_model_path, class_names):
        # Create an instance of the NER predictor using the trained NER model
        self.ner = NERPredictor(ner_model_path)
        # Create an instance of the image classifier using the trained image model and class names
        self.classifier = ImageClassifierPredictor(image_model_path, class_names)

    # Verify if an animal in the text matches the animal in the image
    def verify(self, text, image_path):
        # Extract animal names from the text using the NER predictor, convert to lowercase, and split into a list
        predicted_animals = self.ner.predict(text).lower().split()  # e.g., "chicken horse" → ['chicken', 'horse']
        # Handle the case where no animals are detected in the text
        if not predicted_animals:
            print("No animals detected in text")
            predicted_animals = []  # Ensure an empty list for consistency
        # Load the image from the specified path and convert it to RGB format
        image = Image.open(image_path).convert('RGB')
        # Predict the animal class of the image using the image classifier, convert to lowercase
        classified_animal = self.classifier.predict(image).lower()
        # Print debugging information to show extracted text animals and classified image animal
        print(f"Text animals: {predicted_animals}")
        print(f"Image animal: {classified_animal}")
        # Return True if the image animal is in the list of text animals, False otherwise
        return classified_animal in predicted_animals


# Main execution block to test the pipeline
if __name__ == "__main__":
    # Define the list of animal class names (must match the order used in image training)
    class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
    # Create a pipeline instance with paths to the trained NER and image models
    pipeline = VerificationPipeline(
        ner_model_path="models/ner_model/",  # Directory containing the trained NER model
        image_model_path="models/image_model.pth",  # File containing the trained image model weights
        class_names=class_names  # List of class names for image classification
    )

    # Test the pipeline with a sample text and image
    result = pipeline.verify(
        text="The chicken pecked the ground.",  # Text describing an animal scenario
        image_path="data/archive/raw-img/scoiattolo/OIP-4wH_ENOXEZduA-nNqIRC3gHaFv.jpeg"
    )
    # Print the verification result (True if animals match, False if not)
    print(f"Statement is {result}")