# import required libraries
import os
import numpy as np # type: ignore -for numerical operations
import tensorflow as tf # type: ignore building and training deep learning the model
from tensorflow import keras # type: ignore -high-level API for nueral networks
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore -for augumentation
from tensorflow.keras.models import Sequential # type: ignore -for linear stack of nueral network layers
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout # type: ignore -for layer CNN
from tensorflow.keras.optimizers import Adam # type: ignore -for optimization for training
from tensorflow.keras.callbacks import EarlyStopping # type: ignore -for training callbacks
from matplotlib.pyplot as plt # type: ignore -for plotting training history

# set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

#define the contant value
IMAGE_SIZE = (256, 256) #input image size
BATCH_SIZE = 32 #number of images to process in a batch
EPOCHS = 20 #number of full passes through the dataset
NUM_CLASSES = 2 #number of output classes for crops(diseases, healthy)
ANIMAL_CLASSES = 3 #number of animal classs(cat, dog, human)

#define the dataset directory and madel save paths
DATASET_DIR = "MACHINE LEARNING" #directory containing training
MODEL_PATH = "trained_model.h5" #path to save the trained model

#create a Convolutional Neural Network (CNN) model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), #filters, 3x3 kernel size,
        MaxPooling2D((2, 2)), #pooling layer to reduce dimensionality. Down sample features maps by 2
        Conv2D(64, (3, 3), activation='relu'), #increased to 64 filters to increase the brighterness of the image
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(), #flatten the 3D output to 1D
        Dense(512, activation='relu'), #fully connected layer with 512 neurons
        Dropout(0.5), #dropout to 50% of the neurons to prevent overfitting
        Dense(num_classes, activation='softmax') #output layer with softmax activation for multi-class classification
    ])
    return model
model.compile(optimizer=Adam(learning_rate=0.0001), #Adam optimizer with learning rate
              loss='categorical_crossentropy', #loss function for multi-class classification
              metrics=['accuracy']) #metric to evaluate the model
return model

def train_crop_model():
    # create data generators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255, #rescale pixel values to [0, 1]
        rotation_range=40, #randomly rotate images
        width_shift_range=0.2, #randomly shift images horizontally
        height_shift_range=0.2, #randomly shift images vertically
        shear_range=0.2, #randomly shear images
        zoom_range=0.2, #randomly zoom in on images
        horizontal_flip=True, #randomly flip images horizontally
        fill_mode='nearest' #fill missing pixels with nearest value
        validation_split=0.2 #split 20% of data for validation
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255) #only rescale for validation
     
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'), #directory containing training data
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical' #multi-class classification
        subset='training' #load only training subset
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'), #directory containing validation data
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
        subset='validation' #load only the validation subset
    )

   #create and train the model
    model = create_model(IMAGE_SIZE + (3,), NUM_CLASSES) #input shape for RGB images
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #stop training if validation loss does not improve for 5 epochs
        ModelCheckpoint(MODEL_PATH, save_best_only=True) #save the best model during training
    ]
    #train the model on the training datasets on validation datasets
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE, #number of batches per epoch
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE, #number of validation batches
        epochs=EPOCHS,
        callbacks=callbacks #callbacks for early stopping and model checkpointing
    )
    #plot training history
    plt.figure(figsize=(12, 6))

    return history