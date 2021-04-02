import os
import random
from shutil import copyfile
import tensorflow as tf
from matplotlib import pyplot as plt
import shutil

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
def make_dataset(dataset_path, train_path, test_path, ratio):
    """
    Take dataset path, training path, testing path and ratio
    to separate image from dataset path to train and test sets
    """
    # Take lenght of files in dataset_dir then multiple by ratio
    # Then random pick to the lenght of ratio and copy to the train set
    train = list()
    test = list()
    dataset_list = os.listdir(dataset_path)
    train = random.sample(dataset_list, int(ratio*len(dataset_list)))
    for xfile in os.listdir(dataset_path):
        if xfile not in train:
            test.append(xfile)
    #Copy filename to the train_dataset
    for fname in train:
        src = os.path.join(dataset_path, fname)
        dst = os.path.join(train_path, fname)
        shutil.copyfile(src, dst)
    for fname in test:
        src = os.path.join(dataset_path, fname)
        dst = os.path.join(test_path, fname)
        shutil.copyfile(src, dst)
        
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