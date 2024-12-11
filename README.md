# Vision Transformer Project

This repository contains the implementation and testing of Vision Transformer (ViT) models with optional convolutional stems for enhanced feature extraction. The project includes the following files:

## Files Overview

### 1. **`ViT.ipynb`**
   - **Purpose**: Implements a Vision Transformer model using the Hugging Face Transformers library.
   - **Key Components**:
     - **Model Definition**: Defines the Vision Transformer model.
     - **Training and Evaluation**:
       - Configuration of training arguments such as learning rate, number of epochs, and batch size.
       - Utilizes the `Trainer` class for handling training and evaluation.
     - **Output**: Saves the trained model parameters.
   - **Contribution**: I created a ViT model using the relevant code from the `transformer` library and trained it on a dataset.

### 2. **`ViTWithConvStem.ipynb`**
   - **Purpose**: Extends the Vision Transformer with a convolutional stem for initial feature extraction.
   - **Key Components**:
     - **Preprocessing**: Adds a convolutional block before the transformer layers.
     - **Model Training**:
       - Integrates convolutional operations into the training loop.
       - Tests the model on a validation dataset.
     - **Output**: Saves the model parameters with the convolutional stem.
   - **Contribution**: I designed the convolutional stem and integrated it into the Vision Transformer pipeline.

### 3. **test_ViT.ipynb`**  
   - **Purpose**: Evaluates the performance of the trained ViT model.
   - **Key Components**:
     - Loads the saved model parameters.
     - Computes evaluation metrics (accuracy, loss) on the test dataset.
     - Provides a detailed log of test results.
   - **Contribution**: I implemented the evaluation logic and metric computation.

### 3. **test_ViTwithConvStem.ipynb`**
   - **Purpose**: Evaluates the performance of the trained ViTWithConvStem model.
   - **Key Components**:
     - Loads the saved model parameters.
     - Computes evaluation metrics (accuracy, loss) on the test dataset.
     - Provides a detailed log of test results.
   - **Contribution**: I implemented the evaluation logic and metric computation.



## Running the Project

Follow the steps below to run the project:

### Prerequisites
- Python 3.7+
- Required libraries:
  - PyTorch
  - Transformers (Hugging Face)
  - Jupyter Notebook
  - Matplotlib (optional, for visualization)

Install dependencies with:
```bash
pip install torch transformers jupyter matplotlib
```

### Steps

1. **Train Vision Transformer**:
   - Open `ViT.ipynb` in Jupyter Notebook.
   - Run all cells to train the Vision Transformer model.
   - The trained model parameters will be saved as `vit_model`.

2. **Train Vision Transformer with Convolutional Stem**:
   - Open `ViTWithConvStem.ipynb` in Jupyter Notebook.
   - Run all cells to train the Vision Transformer with a convolutional stem.
   - The trained model parameters will be saved as `model_parameters_1202.pth`.

3. **Evaluate Models**:
   - Open `test_result.ipynb` in Jupyter Notebook.
   - Run all cells to evaluate the trained models on the test dataset.
   - The evaluation results, including accuracy and loss, will be displayed.

## Contributions
- I contributed to the project by setting up the model configurations, designing the convolutional stem, and implementing the training and evaluation pipelines. 



