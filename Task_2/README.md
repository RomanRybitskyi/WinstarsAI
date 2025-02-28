# Animal Verification Pipeline
This project implements a machine learning pipeline to verify if an animal mentioned in text matches the animal depicted in an image. It includes two main components: a Named Entity Recognition (NER) model using BERT to extract animal names from text, and an image classifier using ResNet34 to identify animals in images. The pipeline integrates these models to provide a unified verification result, leveraging Hugging Face’s `transformers` for NLP and PyTorch’s `torchvision` for computer vision, with GPU support where available.

## Project Overview

The goal is to build a system that:

1. Extracts animal names from text (e.g., "The chicken pecked the ground." → "chicken").
2. Classifies animals in images (e.g., a chicken image → "chicken").
3. Verifies if the image’s animal matches any animal in the text (e.g., "chicken" in text vs. "chicken" in image → `True`).

This is achieved using:

- NER Model: A transformer-based BERT model fine-tuned for token classification.
- Image Classifier: A pre-trained ResNet34 model fine-tuned on an animal dataset.
- Pipeline: A wrapper combining both models for consistent usage.

The code is structured to train, infer, and integrate these models, supporting 10 animal classes: `['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']`.

## File Structure
- `ner_train.py`: Trains the BERT-based NER model.
- `ner_inference.py`: Performs inference with the trained NER model.
- `image_train.py`: Trains the ResNet34 image classifier.
- `image_inference.py`: Performs inference with the trained image classifier.
- `pipeline.py`: Integrates NER and image classification for verification.
- `README.md`: This file.
- `requirements.txt`: List of required Python libraries.
- `data/archive/`: Directory for datasets (`ner_data.json` and `raw-img/`).
- `models/`: Directory for trained models (`ner_model/` and `image_model.pth`).

## Solution Explanation

### 1. `NERPredictor` (BERT-based NER Model)
- **Purpose**: Extracts animal names from text.
- **Algorithm**: Fine-tunes `bert-base-uncased` for token classification (0 = non-animal, 1 = animal).
- **Input**: Text sentences (e.g., "The chicken pecked.").
- **Training**:
    - Uses `ner_data.json` (50 annotated sentences).
    - Tokenizes text, aligns labels, trains for 10 epochs with Adam optimizer (`lr=2e-5`).
- **Inference**: Predicts token labels and reconstructs animal names (e.g., "chicken").
- **Library**: `transformers` (Hugging Face).

### 2. `ImageClassifierPredictor` (ResNet34 Image Classifier)
- **Purpose**: Classifies animals in images.
- **Algorithm**: Fine-tunes a pre-trained `resnet34` model on a 10-class animal dataset.
- **Input**:
    - Images preprocessed to 224x224 RGB tensors.
    - Flattened features fed to a modified fully connected layer (512 → 10 classes).
- **Training**:
    - Loads images via `ImageFolder` from 'raw-img/'.
    - Trains for 10 epochs with Adam optimizer (`lr=0.001`), CrossEntropyLoss.
- **Inference**: Predicts the top class (e.g., "squirrel").
- **Library**: 'torchvision' (PyTorch).

### 3. `VerificationPipeline`
- **Purpose**: Combines NER and image classification to verify matches.
- **Usage**:
    - Takes text and an image path as input.
    - Extracts animals from text (list) and classifies the image (single class).
    - Returns `True` if the image animal is in the text list, `False` otherwise.
- **Implementation**: Integrates `NERPredictor` and `ImageClassifierPredictor`.

### 4. Main Execution
- **Training**: Separate scripts (`ner_train.py`, `image_train.py`) train each model.
- **Inference**: Standalone scripts (`ner_inference.py`, `image_inference.py`) test individual components.
- **Pipeline**: `pipeline.py` runs the full verification process.

## Setup Instructions

### Prerequisites
- **Python**: Version 3.10 or higher.
- **Operating System**: Windows, macOS, or Linux.
- **GPU (optional)**: For faster training (requires CUDA-compatible GPU and drivers).

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

4. **Prepare the Dataset**
    - NER Dataset:
        * File: `data/archive/ner_data.json`
        * Provided as a JSON file with 50 examples (e.g., `{"text": "The chicken pecked.", "entities": [{"start": 4, "end": 11, "label": "ANIMAL"}`]}).
        * Save to `data/archive/ner_data.json`.
    - **Image Dataset**:
        * Folder: `data/archive/raw-img/`
        * Download Animals-10 from Kaggle.
        * Extract and rename subfolders to match `class_names` (e.g., `cane` → `dog`, `gallina` → `chicken`, `scoiattolo` → `squirrel`).
        Structure: `raw-img/dog/`, `raw-img/horse/`, etc., each with `.jpg` images.

5. **Train the Models**
    - **NER Model**:
    ```bash
    python ner_train.py
    ```
    Trains BERT, saves to models/ner_model/.

    - **Image Classifier**:
    ```bash
    python image_train.py
    ```

    Trains ResNet34, saves to models/image_model.pth.
6. **Run the Pipeline**
    ```bash
    python pipeline.py
    ```
    - Tests with `"The chicken pecked the ground."` and a squirrel image by default.
    - Edit `text` and `image_path` in `pipeline.py` for custom tests.
