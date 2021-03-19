import os
import random
from shutil import copyfile
import tensorflow as tf
from matplotlib import pyplot as plt

# Data processing
def data_processing(train_dir, test_dir, target_size, batch_size, is_binary = False, is_augmentation = True)
    # Adding augmentation here
    if not is_augmentation:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1. / 255)
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1. / 255,
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True
        )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1. / 255)
    
    if not is_binary:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size = target_size,
            batch_size= batch_size,
            class_mode='categorical'
        )

        validation_generation = test_datagen.flow_from_directory(
            test_dir,
            target_size = target_size,
            batch_size = batch_size,
            class_mode= 'categorical'
        )
    else:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary'
        )
        validation_generation = test_datagen.flow_from_directory(
            test_dir,
            target_size = target_size,
            batch_size = batch_size,
            class_mode= 'binary'
        )

    return train_generator, validation_generation

# Make training set and test set from dataset
def make_dataset(dataset_dir, train_dir, test_dir, ratio):
    # Take lenght of files in dataset_dir then multiple by ratio
    # Then random pick to the lenght of ratio and copy to the train set
    file_len = len([name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir,name))])

    train_len = int(ratio * file_len)
    test_len = file_len - train_len

    #Random pick an image and move to the train_dataset

# Plot the training process
def plot(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_acc = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()