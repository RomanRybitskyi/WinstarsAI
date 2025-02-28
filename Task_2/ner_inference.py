import torch
from transformers import BertTokenizer, BertForTokenClassification


# Define a class to handle NER predictions using a pre-trained model
class NERPredictor:
    # Initialize the predictor with a path to the trained model
    def __init__(self, model_path):
        # Load the BERT tokenizer from the saved model directory
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # Load the trained BERT model for token classification from the saved directory
        self.model = BertForTokenClassification.from_pretrained(model_path)
        # Set the model to evaluation mode (disables dropout and batch normalization updates)
        self.model.eval()
        # Determine the device: use GPU (CUDA) if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move the model to the selected device (CPU or GPU)
        self.model.to(self.device)
        # Print confirmation of model loading and its number of labels (should be 2: 0=non-animal, 1=animal)
        print(f"Model loaded from {model_path}, num_labels: {self.model.num_labels}")

    # Predict animal entities in the given text
    def predict(self, text):
        # Tokenize the input text and prepare it for BERT (returns PyTorch tensors)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Move all input tensors (input_ids, attention_mask) to the selected device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Print the tokenized input for debugging (convert IDs back to readable tokens)
        print(f"Tokens: {self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

        # Perform inference without gradient computation (saves memory and speeds up prediction)
        with torch.no_grad():
            # Run the model on the tokenized input to get prediction logits
            outputs = self.model(**inputs)
        # Extract predictions by taking the argmax over the logits (0 or 1 for each token)
        predictions = torch.argmax(outputs.logits, dim=2)
        # Print the raw predictions for debugging
        print(f"Predictions: {predictions[0]}")

        # Extract the animal name(s) from the predictions
        return self._extract_entity(text, predictions)

    # Helper method to convert token predictions into a readable animal name
    def _extract_entity(self, text, predictions):
        # Tokenize the text again (without special tokens) to align with predictions
        tokens = self.tokenizer.tokenize(text)
        # Initialize an empty string to build the animal name
        animal_name = ""
        # Print token-prediction pairs for debugging, skipping [CLS] and [SEP]
        print("Token-Prediction pairs:")
        # Iterate over tokens and predictions, excluding [CLS] (index 0) and [SEP] (last index)
        for token, pred in zip(tokens, predictions[0][1:-1]):
            # Print each token and its predicted label (0 or 1)
            print(f"  {token}: {pred.item()}")
            # If the prediction is 1 (animal token)
            if pred == 1:
                # Handle subword tokens (e.g., "##ed") by removing "##" and appending
                if token.startswith("##"):
                    animal_name += token.replace("##", "")
                # For whole words, add a space before if not the first token, then append
                else:
                    animal_name += " " + token if animal_name else token
        # Return the extracted animal name, removing leading/trailing whitespace
        return animal_name.strip()


# Main execution block to test the predictor
if __name__ == "__main__":
    # Create an instance of NERPredictor with the trained model path
    predictor = NERPredictor("models/ner_model/")
    # Print a sample of the model's classifier weights to verify loading
    print("Weights after loading:", predictor.model.classifier.weight.data[0][:5])
    # Define a test sentence
    text = "The chicken pecked the ground."
    # Run prediction and get the extracted animal
    result = predictor.predict(text)
    # Print the final result
    print(f"Extracted animal: '{result}'")