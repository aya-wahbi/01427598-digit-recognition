import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from data_preprocessing import load_data

def evaluate_model_with_macro_metrics(model_path, evaluation_data, evaluation_labels, log_to_console=True):
    """
    Evaluates a saved model using macro-averaged metrics (precision, recall, F1-score).

    Arguments:
    - model_path: File path to the saved model.
    - evaluation_data: Input data for testing.
    - evaluation_labels: True labels for the input data (can be one-hot or sparse).
    - log_to_console: If True, prints metrics and confusion matrix.

    Returns:
    - val_accuracy: Overall accuracy on the evaluation data.
    - macro_precision: Macro-averaged precision across all classes.
    - macro_recall: Macro-averaged recall across all classes.
    - macro_f1: Macro-averaged F1-Score across all classes.
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Generate predictions
    print("Making predictions...")
    predictions = model.predict(evaluation_data)
    predicted_classes = np.argmax(predictions, axis=1)

    # Convert one-hot labels to indices if necessary
    if evaluation_labels.ndim == 2 and evaluation_labels.shape[1] > 1:
        evaluation_labels = np.argmax(evaluation_labels, axis=1)  # Convert one-hot to indices

    # Calculate accuracy
    val_accuracy = np.mean(predicted_classes == evaluation_labels)

    # Macro-averaged metrics
    macro_precision = precision_score(evaluation_labels, predicted_classes, average="macro")
    macro_recall = recall_score(evaluation_labels, predicted_classes, average="macro")
    macro_f1 = f1_score(evaluation_labels, predicted_classes, average="macro")

    # Confusion matrix and classification report
    if log_to_console:
        print("\nConfusion Matrix:")
        print(confusion_matrix(evaluation_labels, predicted_classes))
        print("\nMacro-Averaged Metrics:")
        print(f"Precision (Macro-Averaged): {macro_precision:.4f}")
        print(f"Recall (Macro-Averaged): {macro_recall:.4f}")
        print(f"F1-Score (Macro-Averaged): {macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(evaluation_labels, predicted_classes, target_names=[str(i) for i in range(10)]))


    return val_accuracy, macro_precision, macro_recall, macro_f1

def main():
    """
    Evaluate the baseline model on the test data.
    """
    print("Loading evaluation data...")
    (_, _), (_, _), (test_data, test_labels) = load_data()

    model_path = "saved_models/final_baseline_cnn.h5"


    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the baseline model first.")
        return


    evaluate_model_with_macro_metrics(model_path, test_data, test_labels)

if __name__ == "__main__":
    main()