# 01427598- Digit Recognition - MNIST Project

# Digit Recognition with CNN and Data Augmentation
## Overview

This project implements an end-to-end solution for handwritten digit recognition using Convolutional Neural Networks (CNNs) on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Originally developed as a baseline model, the project later evolved to integrate data augmentation, batch normalization, and adaptive learning rate strategies. These enhancements have not only boosted accuracy but also demonstrated the benefits of robust training techniques, even on well-studied datasets.

## Project Goals

- **Validation Accuracy**: Achieve state-of-the-art accuracy on the MNIST validation set.
- **Macro-Averaged Metrics**: Attain near-perfect precision, recall, and F1-score (target > 0.99) for reliable performance evaluation.
- **Generalization**: Utilize data augmentation to enhance the network's ability to generalize and mitigate overfitting.
- **Model Comparison**: Compare the performance of baseline and refined architectures to balance training time and predictive performance.

## Final Performance Metrics

After several experiments, the following results were obtained for the different models:

| Model Name   | Validation Accuracy | Macro Precision | Macro Recall | Macro F1 Score | Training Time (s) |
|--------------|---------------------|-----------------|--------------|----------------|-------------------|
| Baseline_CNN | 0.99225             | 0.99222         | 0.99211      | 0.99216        | 336               |
| dataAugm_CNN | 0.99375             | 0.99377         | 0.99366      | 0.99371        | 165               |
| Refined_CNN  | 0.99575             | 0.99573         | 0.99569      | 0.99571        | 1047              |

*Notes*:
- The refined model achieved the highest validation accuracy and macro-averaged metrics while requiring more training time.

## Pipeline Description

The project pipeline consists of the following stages:

### 1. Data Preprocessing

- **Normalization & Reshaping**:  
  - MNIST image pixel values are scaled to the range \([0, 1]\).  
  - Images are reshaped into \(28 \times 28\) grayscale inputs with a single channel.
- **Dataset Splitting**:  
  - The dataset is partitioned into training, validation, and test sets.
- **Data Augmentation**:  
  - Random transformations (rotations, translations, and zooming) are applied to the training images to increase diversity and reduce the risk of overfitting.

### 2. Model Training

- **Baseline CNN**:  
  - A simple CNN architecture with convolutional layers, a dense hidden layer, and a dropout layer.
- **Refined CNN Enhancements**:  
  - **Batch Normalization**: Added after convolutional and dense layers to stabilize the training process.
  - **Learning Rate Scheduling**: Implemented using the `ReduceLROnPlateau` callback to dynamically lower the learning rate when improvements plateau.
- **Training Settings**:  
  - **Optimizer**: Adam  
  - **Loss Function**: Categorical Crossentropy  


First: 
```bash
pip install -r requirements.txt
```
Training is initiated with the command:
```bash
python main.py --train
```
The best-performing model from each experiment is automatically saved in the `saved_models` folder.

### 3. Evaluation

Models are evaluated on the test set using:
```bash
python main.py --evaluate
```
The evaluation process includes:
- Generating predictions on the test set.
- Computing a confusion matrix.
- Calculating macro-averaged metrics (precision, recall, and F1-score).
- Logging results to `results.csv`.

## Detailed Training Logs Summary

Below is an excerpt summarizing key milestones from the refined model training session:

- **Epoch 1**: Validation accuracy started at 0.98700.
- **Epoch 2**: Improved to 0.99200, and the model was saved.
- **Epoch 9**: Validation accuracy reached 0.99433.
- **Epoch 12**: The model achieved 0.99467.
- **Epoch 15**: Improved further to 0.99517.
- **Epoch 19**: The best validation accuracy of 0.99575 was recorded.
  
The refined model took approximately 1047 seconds to train, reflecting the additional computational load due to data augmentation and the advanced regularization strategies.

## Time Spent

The overall project—from initial setup and baseline development to model refinement and extensive evaluation—required approximately 40–45 hours. While the refined approach increased training time, the significant gains in accuracy and robustness justify this trade-off.

## Conclusion

This project demonstrates that even for a well-studied task like MNIST digit recognition, incremental improvements through data augmentation, batch normalization, and adaptive learning rate strategies can translate into meaningful performance gains. The refined CNN achieved nearly 99.6% validation accuracy along with outstanding macro-averaged metrics, underscoring the value of refined training pipelines for achieving competitive results. Future work may include exploring deeper architectures or ensemble techniques to further push the accuracy boundaries.