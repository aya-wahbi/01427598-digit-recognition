import os
import argparse
from train import main as train_model
from evaluate import evaluate_model_with_macro_metrics
from data_preprocessing import load_data

def main():
    """
    Main function to control the digit recognition pipeline (train/evaluate).
    """
    parser = argparse.ArgumentParser(description="Digit Recognition Pipeline")

    # Arguments for training and evaluation
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the baseline model and save it to the saved_models folder."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the baseline model using test data and display metrics."
    )

    args = parser.parse_args()

    # Step 1: Train the Model
    if args.train:
        print("\n--- Starting Training for the Baseline Model ---")
        train_model()  # Now always calls the baseline training function
        print("\nTraining completed. The baseline model is saved.\n")
    else:
        print("\nTraining skipped. Use '--train' to train the baseline model.\n")

    # Step 2: Evaluate the Model 
    if args.evaluate:
        print("\n--- Starting Evaluation for the Baseline Model ---")
        print("Loading evaluation data...")
        (_, _), (_, _), (test_data, test_labels) = load_data()

        # Path to saved baseline model
        model_path = "saved_models/final_baseline_cnn.h5"

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Please train the baseline model first.")
            return

        # Evaluate the model
        evaluate_model_with_macro_metrics(model_path, test_data, test_labels)
        print("\nEvaluation completed for the Baseline Model.\n")
    else:
        print("\nEvaluation skipped. Use '--evaluate' to evaluate the baseline model.\n")

if __name__ == "__main__":
    main()