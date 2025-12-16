import os
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from data_preprocessing import load_data  
from baseline_model import create_baseline_cnn
from evaluate import evaluate_model_with_macro_metrics
import csv

def save_results_to_csv(model_name, val_accuracy, precision, recall, f1_score, training_time):
    """
    Appends model evaluation results to results.csv file.

    Arguments:
    - model_name: Name of the trained model (e.g., "Baseline").
    - val_accuracy: The validation accuracy.
    - precision, recall, f1_score: Macro metrics from the evaluation.
    - training_time: Time taken to train the model (in seconds).
    """
    results_file = "results.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
       
        if not file_exists:
            writer.writerow([
                "Model_Name", "Validation_Accuracy", "Macro_Precision", 
                "Macro_Recall", "Macro_F1", "Training_Time(s)"
            ])
        
        
        writer.writerow([
            model_name, val_accuracy, precision, recall, f1_score, training_time
        ])

def main():
    """
    Main function to train and evaluate the baseline CNN model.
    """
    
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_data()
    
    print("\nCreating baseline model architecture...")
    model = create_baseline_cnn()
    model_name = "Baseline_CNN"

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join("saved_models", f"{model_name.lower()}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting training...")
    os.makedirs("saved_models", exist_ok=True)  # Ensure the saved_models folder exists
    start_time = time.time()

    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        batch_size=32,        
        epochs=20,             
        callbacks=callbacks,
        verbose=2              
    )

    training_time = time.time() - start_time  
    print(f"Training completed in {training_time:.2f} seconds.")  

  
    print("Saving the final model..")
    final_model_path = os.path.join("saved_models", f"final_{model_name.lower()}.h5")
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

 
    try:
        print("\nEvaluating the model..")
        val_accuracy, precision, recall, f1_score = evaluate_model_with_macro_metrics(
            final_model_path, val_data, val_labels
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return  

    save_results_to_csv(
        model_name=model_name,
        val_accuracy=val_accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        training_time=int(training_time)
    )
    print(f"Results for {model_name} logged in results.csv.")

if __name__ == "__main__":
    main()