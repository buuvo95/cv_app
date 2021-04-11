import os
import random
from shutil import copyfile
import tensorflow as tf
from matplotlib import pyplot as plt
import shutil

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