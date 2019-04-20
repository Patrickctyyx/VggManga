# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.utils import mkdir_if_not_exist
from utils.config_utils import process_config
from data_loaders.facenet_dl import FaceNetDL
import numpy as np
import tensorflow as tf


def train_vgg_mnist():

    print('[INFO] 加载数据…')
    config_str = 'configs/try_vgg_manga.json'
    config = process_config(config_str)

    np.random.seed(47)

    print('[INFO] 加载数据…')
    dl = FaceNetDL(config=config)

    base_model = tf.keras.applications.vgg16.VGG16(weights=None, include_top=False, input_shape=(224, 224, 1))
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    predictions = tf.keras.layers.Dense(2, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print('[INFO] 训练网络')
    model_save_path = 'experiments/try_vgg_manga/checkpoints/'
    mkdir_if_not_exist(model_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_save_path, 'mnist_weights.hdf5'),
        verbose=1, save_weights_only=False,
        monitor='loss',
        mode='min',
        save_best_only=True)
    model.fit_generator(
        dl.get_train_data(),
        epochs=config.num_epochs,
        verbose=2,
        validation_data=dl.get_validation_data(),
        callbacks=[cp_callback]
    )
    print('[INFO] 训练完成…')

    model.evaluate_generator(dl.get_test_data())
    print('[INFO] 测试完成…')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_vgg_mnist()
