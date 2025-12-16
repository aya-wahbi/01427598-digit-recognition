# 01427598- Digit Recognition - MNIST Project

## Overview

This project implements a digit recognition pipeline using Convolutional Neural Networks (CNNs) to classify handwritten digits (0–9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The primary goal is to establish a working pipeline with a baseline model and evaluate its performance with macro-averaged metrics, including precision, recall, and F1-score.

### Goal
- **Error Metric**: 
  - Macro F1-Score
  - Validation accuracy.
- **Target**: 
  - Achieve **validation accuracy >98%** and **macro F1-score >0.90**.

---

## Results

| Metric             | Target Value | Validation Result | Test Result |
|--------------------|--------------|-------------------|-------------|
| **Validation Accuracy** | >98%       | 99.25%            | N/A         |
| **F1-Score (Macro)**     | >0.98      | 99.22%            | 99.31%      |
| **Precision (Macro)**    | >0.98        | 99.22%            | 99.31%      |
| **Recall (Macro)**       | >0.98        | 99.21%            | 99.31%      |

The results show exceptional performance across all metrics, comfortably exceeding the target values. The test results affirm that the model generalizes well to unseen data with near-perfect macro-average metrics. 

---

## Pipeline Description

The project pipeline consists of three main stages:

1. **Data Preprocessing**:
   - Input normalization: MNIST image pixel values are scaled to the range [0, 1].
   - Reshaping: Images are reshaped into \( 28 \times 28 \) grayscale inputs with a single channel.
   - Dataset splitting: MNIST data is divided into training, validation, and test sets.
   - Future possibilities: augmentation functionality (currently commented out still to be implemented and imporved upon).

2. **Model Training**:
   - A baseline CNN model with two convolutional blocks, a dense hidden layer, and a dropout layer was implemented.
   - **Optimizer**: Adam.
   - **Loss Function**: Categorical Crossentropy.
   - Callback mechanisms:
     - **ModelCheckpoint**: Saved the best-performing model based on validation accuracy.
     - **EarlyStopping**: Halted training when validation accuracy plateaued.
   - Training time: **5 minutes and 36 seconds**.

3. **Evaluation**:
   - The trained model was extensively evaluated on the test set.
   - Performance metrics included:
     - Test accuracy
     - Macro-averaged precision, recall, and F1-score.
   - Confusion matrix and classification report provided granular analysis.

---

## How to Run the Project

### Setup
1. Clone the repository:
   ```bash
   git clone <YOUR-REPO-URL>
   cd <PROJECT-FOLDER>

2. To train the baseline model:
python main.py --train

this also Saves the best-performing model in the saved_models folder.

3. To evaluate the trained model:
python main.py --evaluate

This evaluates the performance on the test set and logs evaluation metrics, confusion matrix, and classification report.

## Future Work / Challenges
- Data Augmentation: Investigate techniques for artificially expanding the MNIST dataset to further improve generalization.

