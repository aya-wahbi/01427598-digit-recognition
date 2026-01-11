import os
import time
import csv
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_data, augment_data
from baseline_model import create_baseline_cnn
from evaluate import evaluate_model_with_macro_metrics

def save_results_to_csv(model_name, val_accuracy, precision, recall, f1_score, training_time):
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
    # Load the data: training, validation, and test sets
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_data()
    

    model = create_baseline_cnn()
    model_name = "Refined_CNN"

    # Define callbacks to help training:
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
        ),
        # Reduce the learning rate when a plateau in validation accuracy is detected
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    
    os.makedirs("saved_models", exist_ok=True)
    start_time = time.time()
    
    # Create augmented data generator
    train_generator = augment_data(train_data, train_labels, batch_size=32)
    steps_per_epoch = len(train_data) // 32
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(val_data, val_labels),
        epochs=20,
        callbacks=callbacks,
        verbose=2
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    

    final_model_path = os.path.join("saved_models", f"final_{model_name.lower()}.h5")
    model.save(final_model_path)

    
    try:
        print("\nEvaluating the refined model...")
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