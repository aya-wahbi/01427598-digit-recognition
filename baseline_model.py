from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    
def create_baseline_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates a simple CNN model for MNIST digit classification.
    
    :param input_shape: Shape of the input images (28, 28, 1 for MNIST).
    :param num_classes: Number of output classes (10 for MNIST digits).
    :return: A compiled Keras model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential(name="Baseline_CNN")  # Create a Sequential model

    # Convolutional Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name="Conv2D_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_1"))
    
    # Convolutional Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', name="Conv2D_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="MaxPool_2"))
    
    # Flatten and Dense Layers
    model.add(Flatten(name="Flatten"))
    model.add(Dense(128, activation='relu', name="Dense_1"))
    model.add(Dropout(0.5, name="Dropout"))  # Dropout to prevent overfitting
    model.add(Dense(num_classes, activation='softmax', name="Output"))

    # Correct the typo here: 'model', not 'mmodel'
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model