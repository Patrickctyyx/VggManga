import os
import tensorflow as tf
from root_dir import ROOT_DIR
from bases.data_loader_base import DataLoaderBase


class VGGMangaDL(DataLoaderBase):

    def __init__(self, config=None):
        super(VGGMangaDL, self).__init__(config)
        self.root_path = ROOT_DIR

    def get_train_data(self):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.root_path, 'manga109/train'),
            target_size=(224, 224),
            batch_size=self.config.batch_size
        )
        return train_generator

    def get_test_data(self):
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )
        test_generator = test_datagen.flow_from_directory(
            os.path.join(self.root_path, 'manga109/test'),
            target_size=(224, 224),
            batch_size=self.config.batch_size
        )
        return test_generator

    def get_validation_data(self):
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255
        )
        val_generator = val_datagen.flow_from_directory(
            os.path.join(self.root_path, 'manga109/validation'),
            target_size=(224, 224),
            batch_size=self.config.batch_size
        )
        return val_generator
