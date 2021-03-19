import tensorflow as tf
from model import *
from utils import *

# This parameter need to define:
    # - input_shape
    # - num_classes
    # - loss
    # - optimizers
    # - target_size
    # - batch_size
    # - steps_per_epoch
    # - epochs
    # - train_dir
    # - test_dir

# Define hyparameter
loss = 'categorical_crossentropy'
optimizers = tf.keras.optimizers.RMSprop(lr = 1e-4)
batch_size = 16
steps_per_epoch = 100
epochs = 30
validation_steps = 30

choosen = "Alex"

if choosen == "Alex":
    target_size = (256,256)
    model = AlexNet(input_shape, num_classes)
elif choosen == "Inception":
    target_size = (224,224)
    model = InceptionNet(input_shape, num_classes)
elif choosen == "VGG":
    target_size = (224,224)
    model = VGGNet(input_shape, num_classes)
elif choosen == "Xception":
    target_size = (299,299)
    model = XceptionNet(input_shape, num_classes)
else:
    target_size = (224,224)
    model = ResNet(input_shape, num_classes)

model.compile(
    loss = loss,
    optimizers = optimizers,
    metrics = ['acc']
)

train_generator, validation_generation = data_processing(
    train_dir,
    test_dir,
    target_size,
    batch_size,
    is_binary = False,
    is_augmentation = True
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generation,
    validation_steps=validation_steps
)

plot(history)