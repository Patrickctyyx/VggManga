# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loaders.facenet_dl import FaceNetDL
from models.manga_facenet_model import MangaFaceNetModel
from trainers.vgg_manga_trainer import VGGMangaTrainer
from utils.config_utils import process_config, get_train_args
from utils.utils import mkdir_if_not_exist
import numpy as np
import tensorflow as tf
import cv2


def train_vgg_mnist():

    print('[INFO] 加载数据…')
    config = process_config('configs/vgg_mnist_config.json')
    dl = FaceNetDL(config=config)

    main_input = tf.keras.Input(shape=(config.input_shape, config.input_shape, 1))

    x = tf.keras.layers.Convolution2D(32, (3, 3), padding='same', name='conv1')(main_input)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    x = tf.keras.layers.Convolution2D(48, (3, 3), padding='same', name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn2')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    x = tf.keras.layers.Convolution2D(64, (3, 3), padding='same', name='conv3')(x)
    x = tf.keras.layers.BatchNormalization(name='bn3')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    x = tf.keras.layers.Flatten(name='fl')(x)

    x = tf.keras.layers.Dense(3168, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=main_input, outputs=x)

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('[INFO] 训练网络')
    model_save_path = 'experiments/mnist_facenet/checkpoints'
    mkdir_if_not_exist(model_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_path, 'mnist_facenet_weights.hdf5'),
        verbose=1, save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    model.fit_generator(
        dl.get_train_data(),
        epochs=config.num_epochs,
        validation_data=dl.get_validation_data(),
        callbacks=[cp_callback])
    print('[INFO] 训练完成…')

    print('[INFO] 测试模型…')
    model.evaluate_generator(dl.get_test_data())
    print('[INFO] 测试完成…')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train_vgg_mnist()
