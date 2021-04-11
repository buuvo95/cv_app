import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class trainDatagen():
    def __init__(self, 
                 width_shift_range,
                 height_shift_range,
                 rotation_range,
                 zoom_range,
                 shear_range,
                 horizontal_flip,
                 vertical_flip):
        self. train_datagen = ImageDataGenerator(
            rescale = 1./255,
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            rotation_range = rotation_range,
            zoom_range = zoom_range,
            shear_range = shear_range,
            horizontal_flip = horizontal_flip,
            vertical_flip = vertical_flip,
            fill_mode = 'nearest'
            )
    def generator(self, train_dir, target_size, batch_size, class_mode):
        self.train_generator = self.train_datagen.flow_from_directory(
            train_dir,
            target_size = target_size,
            batch_size = batch_size,
            class_mode = class_mode
        )
        return self.train_generator

class testDatagen():
    def __init__(self):
        self.test_datagen = ImageDataGenerator(rescale = 1./255)
    def generator(self, test_dir, target_size, batch_size, class_mode):
        self.test_generator = self.test_datagen.flow_from_directory(
            test_dir,
            target_size = target_size,
            batch_size = batch_size,
            class_mode = class_mode
        )
        return self.test_generator