from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
import json


# Define a custom Dataset class for NER data
class NERDataset(Dataset):
    # Initialize the dataset with a path to the JSON file, tokenizer, and max sequence length
    def __init__(self, data_path, tokenizer, max_len=128):
        # Load the JSON dataset file (contains [{"text": ..., "entities": [...]}, ...])
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer  # BERT tokenizer instance
        self.max_len = max_len  # Maximum sequence length for tokenized inputs

    # Return the total number of examples in the dataset
    def __len__(self):
        return len(self.data)

    # Fetch and preprocess a single example from the dataset
    def __getitem__(self, idx):
        # Extract text and entities from the dataset at the given index
        text = self.data[idx]["text"]  # e.g., "Horses need big fields."
        entities = self.data[idx]["entities"]  # e.g., [{"start": 0, "end": 6, "label": "ANIMAL"}]

        # Tokenize the text and prepare inputs for BERT (returns tensors with [CLS] and [SEP])
        encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_len,
                                  truncation=True, padding="max_length")
        # Initialize labels with zeros (0 = non-animal token)
        labels = [0] * self.max_len

        # Get tokenized version of the text (without [CLS] and [SEP] for label alignment)
        tokens = self.tokenizer.tokenize(text)  # e.g., ['horses', 'need', 'big', 'fields', '.']

        # Offset to account for [CLS] token at the start of the model's input
        token_start_offset = 1  # [CLS] is at position 0 in the final input_ids

        # Iterate over each entity in the example to assign labels
        for entity in entities:
            start, end = entity["start"], entity["end"]  # Character positions of the entity
            word = text[start:end]  # Extract the entity word (e.g., "Horses")
            # Calculate token positions, adjusting for [CLS]
            token_start = len(self.tokenizer.tokenize(text[:start])) + token_start_offset
            token_end = token_start + len(self.tokenizer.tokenize(word))

            # Debug prints to verify data processing
            print(f"Text: {text}")
            print(f"Entities: {entities}")
            print(f"Entity '{word}' (char {start}-{end}) -> Token {token_start}-{token_end}")

            # Assign label 1 to tokens corresponding to the entity
            for i in range(token_start, min(token_end, self.max_len)):
                labels[i] = 1  # 1 = animal token

        # Debug prints to show tokenized text and corresponding labels
        print(f"Tokens: {tokens}")
        print(f"Labels: {labels[:len(tokens) + 2]}")  # +2 for [CLS] and [SEP]
        print("-" * 50)

        # Return a dictionary with model inputs and labels as tensors
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # Token IDs (e.g., [101, 122, ...])
            "attention_mask": encoding["attention_mask"].squeeze(),  # 1s for real tokens, 0s for padding
            "labels": torch.tensor(labels)  # Labels aligned with input_ids
        }


# Define a class to handle NER model training
class NERTrainer:
    # Initialize the trainer with a pre-trained BERT model and device settings
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        # Load the BERT tokenizer for the specified model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # Load the BERT model for token classification with 2 labels (0 = non-animal, 1 = animal)
        self.model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        # Set device to CPU to avoid potential CUDA issues (can be changed to "cuda" if needed)
        self.device = torch.device("cpu")
        # Move the model to the specified device
        self.model.to(self.device)

    # Train the model on the provided dataset
    def train(self, train_data_path, epochs=10, batch_size=16):
        # Create a dataset instance with the training data
        dataset = NERDataset(train_data_path, self.tokenizer)
        # Create a DataLoader for batching and shuffling the data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Use Adam optimizer with a learning rate of 2e-5 (standard for fine-tuning BERT)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)

        # Set the model to training mode (enables dropout, etc.)
        self.model.train()
        # Train for the specified number of epochs
        for epoch in range(epochs):
            total_loss = 0
            # Iterate over batches of data
            for batch in loader:
                # Clear previous gradients
                optimizer.zero_grad()
                # Move batch tensors to the device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                # Forward pass through the model
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                # Compute the loss (cross-entropy over token predictions)
                loss = outputs.loss
                # Backward pass to compute gradients
                loss.backward()
                # Update model parameters
                optimizer.step()
                # Accumulate loss for reporting
                total_loss += loss.item()
            # Print average loss for the epoch
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

    # Save the trained model and tokenizer to a directory
    def save(self, path):
        self.model.save_pretrained(path)  # Save model weights and config
        self.tokenizer.save_pretrained(path)  # Save tokenizer files


# Main execution block
if __name__ == "__main__":
    # Create a trainer instance
    trainer = NERTrainer()
    # Train the model on the specified dataset
    trainer.train("data/archive/ner_data.json")

    # Switch to evaluation mode for inference (disables dropout)
    trainer.model.eval()
    # Perform inference without gradient computation
    with torch.no_grad():
        # Test with a sample sentence from the training data
        text = "Horses need big fields."
        # Tokenize the test sentence
        inputs = trainer.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length")
        # Move inputs to the device
        inputs = {k: v.to(trainer.device) for k, v in inputs.items()}
        # Run the model on the test input
        outputs = trainer.model(**inputs)
        logits = outputs.logits  # Raw prediction scores
        # Get predicted labels by taking the argmax over logits
        predictions = torch.argmax(logits, dim=2)

        # Print inference results for debugging
        print(f"\nInference Test (Training Example):")
        print(f"Tokens: {trainer.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        print(f"Logits: {logits[0]}")
        print(f"Predictions: {predictions[0]}")

    # Print a sample of classifier weights before saving
    print(f"Weights before saving: {trainer.model.classifier.weight.data[0][:5]}")
    # Save the trained model to the specified directory
    trainer.save("models/ner_model/")