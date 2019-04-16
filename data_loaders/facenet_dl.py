import os
import tensorflow as tf
from root_dir import ROOT_DIR
from bases.data_loader_base import DataLoaderBase


class FaceNetDL(DataLoaderBase):

    def __init__(self, config=None):
        super(FaceNetDL, self).__init__(config)
        self.root_path = ROOT_DIR
        self.manga_dir = self.config.manga_dir
        self.input_shape = self.config.input_shape
        if self.config.input_channel == 1:
            self.color_mode = 'rgb'
        else:
            self.color_mode = 'grayscale'
        self.train_generator, self.validation_generator = self.generate_train_val()

    def generate_train_val(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.root_path, self.manga_dir, 'training'),
            target_size=(self.input_shape, self.input_shape),
            batch_size=self.config.batch_size,
            subset='training',
            color_mode=self.color_mode
        )
        val_generator = train_datagen.flow_from_directory(
            os.path.join(self.root_path, self.manga_dir, 'training'),
            target_size=(self.input_shape, self.input_shape),
            batch_size=self.config.batch_size,
            subset='validation',
            color_mode=self.color_mode
        )
        return train_generator, val_generator

    def get_train_data(self):
        return self.train_generator

    def get_test_data(self):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.root_path, self.manga_dir, 'testing'),
            target_size=(self.input_shape, self.input_shape),
            batch_size=self.config.batch_size,
            color_mode=self.color_mode
        )
        return test_generator

    def get_validation_data(self):
        return self.validation_generator
