import os  # Imports the 'os' module, which provides a way of using operating system dependent functionality like reading or writing to the file system.
import numpy as np  # Imports the 'numpy' library and assigns it the alias 'np'. Numpy is used for numerical operations, especially with arrays.
import tensorflow as tf  # Imports the 'tensorflow' library and assigns it the alias 'tf'. TensorFlow is a powerful library for machine learning and artificial intelligence.
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # Imports the 'ImageDataGenerator' class from TensorFlow's Keras API, used for image augmentation and preprocessing.
from tensorflow.keras.models import Sequential  # type: ignore # Imports the 'Sequential' model type from Keras. A sequential model is a linear stack of layers.
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore # Imports different types of layers used to build the neural network.
from tensorflow.keras.optimizers import Adam  # type: ignore # Imports the 'Adam' optimizer, which is an algorithm for gradient-based optimization of stochastic objective functions.
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore # Imports 'ModelCheckpoint' to save the model at different points during training and 'EarlyStopping' to stop training when a monitored metric has stopped improving.
import matplotlib.pyplot as plt  # Imports the 'pyplot' module from the 'matplotlib' library for creating visualizations like plots and graphs.

# Constants - These are fixed values used throughout the script.
IMAGE_SIZE = (256, 256)  # Defines the size to which all images will be resized (256x256 pixels).
BATCH_SIZE = 32  # Sets the number of samples that will be propagated through the network at one time.
EPOCHS = 20  # Defines the number of times the learning algorithm will work through the entire training dataset.
NUM_CLASSES = 2  # Sets the number of categories for the crop model (e.g., Healthy vs Diseased).
ANIMAL_CLASSES = 3  # Sets the number of categories for the animal filter model (e.g., Dog, Cat, Human).

# Paths - These are string variables that hold the file paths for datasets and the saved model.
# The dataset directory should contain subfolders for each category of images.
DATASET_DIR = "dataset"  # Specifies the name of the directory where the dataset is located.
MODEL_PATH = "agricure_model.h5"  # Defines the filename for the saved trained model. The '.h5' extension is common for Keras models.


def create_model(input_shape, num_classes):  # Defines a function to create the neural network model.
    """Create a CNN model for classification"""  # This is a docstring, explaining what the function does.
    model = Sequential([  # Initializes a Sequential model, where layers are added one after another.
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  # 1st Convolutional layer: 32 filters, 3x3 kernel, 'relu' activation. 'input_shape' is specified for the first layer.
        MaxPooling2D(2, 2),  # 1st Pooling layer: Reduces the spatial dimensions (height and width) of the input volume.
        Conv2D(64, (3, 3), activation='relu'),  # 2nd Convolutional layer: 64 filters, 3x3 kernel, 'relu' activation.
        MaxPooling2D(2, 2),  # 2nd Pooling layer.
        Conv2D(128, (3, 3), activation='relu'),  # 3rd Convolutional layer: 128 filters, 3x3 kernel, 'relu' activation.
        MaxPooling2D(2, 2),  # 3rd Pooling layer.
        Conv2D(256, (3, 3), activation='relu'),  # 4th Convolutional layer: 256 filters, 3x3 kernel, 'relu' activation.
        MaxPooling2D(2, 2),  # 4th Pooling layer.
        Flatten(),  # Flattens the 3D output of the convolutional layers into a 1D vector.
        Dropout(0.5),  # Dropout layer: Randomly sets 50% of the input units to 0 at each update during training time, which helps prevent overfitting.
        Dense(512, activation='relu'),  # A fully connected (Dense) layer with 512 neurons and 'relu' activation.
        Dense(num_classes, activation='softmax')  # The output layer: A Dense layer with a number of neurons equal to 'num_classes'. 'softmax' activation is used for multi-class classification to output probabilities.
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),  # Configures the model for training. Sets the optimizer to Adam with a learning rate of 0.0001.
                  loss='categorical_crossentropy',  # Sets the loss function. 'categorical_crossentropy' is used for multi-class classification.
                  metrics=['accuracy'])  # Specifies the metric to be evaluated by the model during training and testing.
    return model  # Returns the compiled model.


def train_crop_model():  # Defines a function to train the crop disease detection model.
    """Train the main crop disease detection model"""  # Docstring for the function.
    # Data generators with augmentation - This creates modified versions of the images to make the model more robust.
    train_datagen = ImageDataGenerator(  # Creates an instance of ImageDataGenerator for the training data.
        rescale=1./255,  # Rescales the pixel values from [0, 255] to [0, 1] which is a common practice.
        rotation_range=40,  # Randomly rotates images by a degree in the range (-40, +40).
        width_shift_range=0.2,  # Randomly shifts images horizontally.
        height_shift_range=0.2,  # Randomly shifts images vertically.
        shear_range=0.2,  # Applies shearing transformations.
        zoom_range=0.2,  # Randomly zooms into images.
        horizontal_flip=True,  # Randomly flips images horizontally.
        fill_mode='nearest',  # Strategy for filling in newly created pixels, which can appear after a rotation or a width/height shift.
        validation_split=0.2  # Reserves 20% of the images for validation.
    )

    # Load datasets - Loads images from the directory.
    train_generator = train_datagen.flow_from_directory(  # Creates a generator for the training data.
        os.path.join(DATASET_DIR, "crops"),  # Path to the target directory (dataset/crops).
        target_size=IMAGE_SIZE,  # Resizes all images to the specified IMAGE_SIZE.
        batch_size=BATCH_SIZE,  # Number of images to yield from the generator per batch.
        class_mode='categorical',  # Determines the type of label arrays that are returned. 'categorical' will be 2D one-hot encoded labels.
        subset='training'  # Specifies that this is the training set.
    )

    validation_generator = train_datagen.flow_from_directory(  # Creates a generator for the validation data.
        os.path.join(DATASET_DIR, "crops"),  # Path to the target directory.
        target_size=IMAGE_SIZE,  # Resizes all images.
        batch_size=BATCH_SIZE,  # Batch size.
        class_mode='categorical',  # Label type.
        subset='validation'  # Specifies that this is the validation set.
    )

    # Create and train model - Builds the model and starts the training process.
    model = create_model(IMAGE_SIZE + (3,), NUM_CLASSES)  # Calls the 'create_model' function. The input shape is image size plus 3 for the RGB channels.

    callbacks = [  # A list of functions to be applied at given stages of the training procedure.
        ModelCheckpoint(MODEL_PATH, save_best_only=True),  # Saves the model after every epoch, but only if the model is the best one seen so far.
        EarlyStopping(patience=5, restore_best_weights=True)  # Stops training if the validation loss doesn't improve for 5 consecutive epochs.
    ]

    history = model.fit(  # Trains the model.
        train_generator,  # The generator for training data.
        steps_per_epoch=train_generator.samples // BATCH_SIZE,  # The number of batches to draw from the training generator for each epoch.
        epochs=EPOCHS,  # The total number of epochs to train for.
        validation_data=validation_generator,  # The generator for validation data.
        validation_steps=validation_generator.samples // BATCH_SIZE,  # The number of batches to draw from the validation generator for each epoch.
        callbacks=callbacks  # The list of callbacks to apply during training.
    )

    # Plot training history - Visualizes the training progress.
    plot_training_history(history)  # Calls the function to plot accuracy and loss.
    return model  # Returns the trained model.


def train_animal_filter():  # Defines a function to train the animal/human filter model.
    """Train a secondary model to filter out animals/humans"""  # Docstring for the function.
    animal_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Creates a data generator for the animal dataset, only rescaling the images.

    train_generator = animal_datagen.flow_from_directory(  # Creates a generator for the training data.
        os.path.join(DATASET_DIR, "animals"),  # Path to the animal dataset directory.
        target_size=IMAGE_SIZE,  # Resizes images.
        batch_size=BATCH_SIZE,  # Batch size.
        class_mode='categorical',  # Label type.
        subset='training'  # Specifies this is the training set.
    )

    validation_generator = animal_datagen.flow_from_directory(  # Creates a generator for the validation data.
        os.path.join(DATASET_DIR, "animals"),  # Path to the animal dataset directory.
        target_size=IMAGE_SIZE,  # Resizes images.
        batch_size=BATCH_SIZE,  # Batch size.
        class_mode='categorical',  # Label type.
        subset='validation'  # Specifies this is the validation set.
    )

    model = create_model(IMAGE_SIZE + (3,), ANIMAL_CLASSES)  # Creates the model for animal classification.

    model.fit(  # Trains the animal filter model.
        train_generator,  # Training data generator.
        steps_per_epoch=train_generator.samples // BATCH_SIZE,  # Steps per epoch for training.
        epochs=10,  # Trains for 10 epochs.
        validation_data=validation_generator,  # Validation data generator.
        validation_steps=validation_generator.samples // BATCH_SIZE  # Steps per epoch for validation.
    )

    return model  # Returns the trained animal filter model.


def plot_training_history(history):  # Defines a function to plot the training history.
    """Plot training and validation accuracy/loss"""  # Docstring for the function.
    acc = history.history['accuracy']  # Gets the training accuracy history.
    val_acc = history.history['val_accuracy']  # Gets the validation accuracy history.
    loss = history.history['loss']  # Gets the training loss history.
    val_loss = history.history['val_loss']  # Gets the validation loss history.

    epochs_range = range(len(acc))  # Creates a range of numbers for the x-axis (number of epochs).

    plt.figure(figsize=(12, 6))  # Creates a new figure with a specified size.
    plt.subplot(1, 2, 1)  # Creates a subplot in a 1x2 grid at position 1.
    plt.plot(epochs_range, acc, label='Training Accuracy')  # Plots training accuracy.
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # Plots validation accuracy.
    plt.legend(loc='lower right')  # Displays the legend at the lower right of the plot.
    plt.title('Training and Validation Accuracy')  # Sets the title of the plot.

    plt.subplot(1, 2, 2)  # Creates a subplot in a 1x2 grid at position 2.
    plt.plot(epochs_range, loss, label='Training Loss')  # Plots training loss.
    plt.plot(epochs_range, val_loss, label='Validation Loss')  # Plots validation loss.
    plt.legend(loc='upper right')  # Displays the legend at the upper right of the plot.
    plt.title('Training and Validation Loss')  # Sets the title of the plot.

    plt.savefig('training_history.png')  # Saves the plot as a PNG file.
    plt.close()  # Closes the figure to free up memory.


def predict_image(model, animal_model, image_path):  # Defines a function to make a prediction on a single image.
    """Predict if image is healthy/diseased crop or animal/human"""  # Docstring for the function.
    img = tf.keras.preprocessing.image.load_img(  # Loads an image from a file path.
        image_path, target_size=IMAGE_SIZE  # Specifies the path and resizes the image.
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Converts the image to a numpy array.
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Adds a batch dimension and rescales the pixel values.

    # First check if it's an animal/human
    animal_pred = animal_model.predict(img_array)  # Uses the animal model to make a prediction.
    animal_classes = ['dog', 'cat', 'human']  # A list of the animal classes.
    animal_prob = np.max(animal_pred)  # Gets the highest probability from the prediction.

    if animal_prob > 0.9:  # Checks if the model is highly confident that the image is an animal or human.
        return {  # Returns a dictionary with the prediction details.
            'type': 'animal',  # The type of prediction.
            'class': animal_classes[np.argmax(animal_pred)],  # The predicted class.
            'confidence': float(animal_prob)  # The confidence score.
        }

    # If not animal, check for crops
    crop_pred = model.predict(img_array)  # Uses the crop model to make a prediction.
    crop_classes = ['healthy', 'diseased']  # A list of the crop classes.
    return {  # Returns a dictionary with the prediction details.
        'type': 'crop',  # The type of prediction.
        'class': crop_classes[np.argmax(crop_pred)],  # The predicted class.
        'confidence': float(np.max(crop_pred))  # The confidence score.
    }


if __name__ == "__main__":  # This block of code will only run when the script is executed directly (not imported).
    # Train or load models
    if not os.path.exists(MODEL_PATH):  # Checks if the crop model file already exists.
        print("Training crop disease model...")  # Prints a message to the console.
        crop_model = train_crop_model()  # Calls the function to train the crop model.
        print("Training animal filter model...")  # Prints a message to the console.
        animal_model = train_animal_filter()  # Calls the function to train the animal filter model.
    else:  # If the model file exists:
        print("Loading existing models...")  # Prints a message to the console.
        crop_model = tf.keras.models.load_model(MODEL_PATH)  # Loads the pre-trained crop model.
        animal_model = create_model(IMAGE_SIZE + (3,), ANIMAL_CLASSES)  # Creates a new animal model structure.
        # Note: In a real-world application, you should save and load the trained animal model as well.

    # Test prediction
    test_image = "test_image.jpg"  # Defines the path to a test image.
    if os.path.exists(test_image):  # Checks if the test image file exists.
        prediction = predict_image(crop_model, animal_model, test_image)  # Calls the prediction function.
        print("\nPrediction Results:")  # Prints a header for the results.
        print(f"Type: {prediction['type']}")  # Prints the predicted type.
        print(f"Class: {prediction['class']}")  # Prints the predicted class.
        print(f"Confidence: {prediction['confidence']:.2%}")  # Prints the confidence score, formatted as a percentage.
    else:  # If the test image is not found:
        print(f"Test image {test_image} not found")  # Prints an error message.
