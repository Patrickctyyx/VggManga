# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loaders.facenet_dl import FaceNetDL
from models.manga_facenet_model import MangaFaceNetModel
from trainers.vgg_manga_trainer import VGGMangaTrainer
from utils.config_utils import process_config, get_train_args
import numpy as np
import tensorflow as tf


def train_vgg_mnist():
    config_str = 'configs/vgg_mnist_config.json'
    config = process_config(config_str)

    print('[INFO] 加载数据…')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    x_train.resize((x_train.shape[0], 224, 224, 3))
    x_test.resize((x_test.shape[0], 224, 224, 3))

    # model = MangaFaceNetModel(config=config, use_vgg=True)
    model = tf.keras.applications.vgg16.VGG16(weights=None, classes=10)
    print('[INFO] 使用 VGG 作为骨架')

    print('[INFO] 训练网络')
    model.model.fit(x_train, y_train, epochs=5)
    print('[INFO] 训练完成…')
    model.model.evaluate(x_test, y_test)


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train_vgg_mnist()
