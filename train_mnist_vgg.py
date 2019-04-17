# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.utils import mkdir_if_not_exist
import numpy as np
import tensorflow as tf


def train_vgg_mnist():

    print('[INFO] 加载数据…')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # x_train = np.resize(x_train, (x_train.shape[0], 48, 48, 3))
    # x_test = np.resize(x_test, (x_test.shape[0], 48, 48, 3))

    main_input = tf.keras.Input(shape=(28, 28, 1))

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
    x = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=main_input, outputs=x)

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('[INFO] 训练网络')
    model_save_path = 'experiments/mnist_model/checkpoints/'
    mkdir_if_not_exist(model_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_path, 'mnist_weights.hdf5'),
        verbose=1, save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)
    model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])

    print('[INFO] 训练完成…')
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_vgg_mnist()
