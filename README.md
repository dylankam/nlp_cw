# Patronising and Condescending Language (PCL) Detection
This repository contains the code, model, and prediction files for the binary classification of Patronising and Condescending Language (PCL), originally presented in SemEval-2022 Task 4. 

## Overview
This project utilizes a pre-trained `RoBERTa` sequence classification model. To address the severe class imbalance in the dataset (9.5% positive instances), this approach utilizes a custom class-weighted Cross-Entropy loss function over the entire training dataset. This outperforms the standard baseline, which discards a large portion of the training data through random undersampling.

## Repository Structure
* `/BestModel/` - Contains the final, trained RoBERTa model files and the main Jupyter Notebook (`.ipynb`) used for data processing and training.
* `dev.txt` - Line-by-line binary predictions (0 or 1) aligned with the official dev set.
* `test.txt` - Line-by-line binary predictions (0 or 1) aligned with the official test set.
* `README.md` - Reproduction instructions and documentation.

## Datasets
To reproduce this project, the following official datasets must be placed in the root directory alongside the Jupyter Notebook. 
* **Main Dataset:** `dontpatronizeme_pcl.tsv`
* **Train Set IDs & Labels:** `train_semeval_parids-labels.csv`
* **Dev Set IDs & Labels:** `dev_semeval_parids-labels.csv`
* **Test Set:** `task4_test.tsv`

## Pre-trained Model Weights
Due to file size constraints, the primary `model.safetensors` file is hosted externally. 
To run inference without retraining the model, you need to download the weights from the link below. You must then take this and all files in the /BestModel/ directory EXCEPT THE .ipynb file, and place them inside a directory named `/Model/` alongisde the .ipynb file before executing the evaluation cells.

* **Download `model.safetensors` here:** [INSERT YOUR CLOUD STORAGE/ONEDRIVE/GOOGLE DRIVE LINK HERE]

## How to Reproduce the Model

### 1. Environment Setup
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:
`pip install torch transformers pandas numpy scikit-learn matplotlib seaborn`

### 2. File Placement
Ensure the official dataset files (listed above) are in the same working directory as the `.ipynb` notebook located in the `/BestModel/` folder.

### 3. Execution
Open the Jupyter Notebook. The notebook is structured sequentially:
1. **Data Preprocessing:** Loads the datasets and prepares the aligned dev and test DataFrames.
2. **Tokenization:** Uses the RoBERTa tokenizer (truncated to `max_length=128` to optimize VRAM and allow for larger batch sizes).
3. **Training:** Initializes the `Trainer` with the custom class-weighted loss function. *(Skip this section if you are loading the pre-trained weights).*
4. **Evaluation:** Generates the `dev.txt` and `test.txt` files and outputs the Precision, Recall, and F1 metrics.
