# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from data_loaders.facenet_dl import FaceNetDL
from models.manga_facenet_model import MangaFaceNetModel
from trainers.vgg_manga_trainer import VGGMangaTrainer
from utils.config_utils import process_config, get_train_args
import numpy as np


def train_vgg_manga():

    print('[INFO] 解析配置…')
    parser = None
    config = None
    model_path = None

    try:
        args, parser = get_train_args()
        config = process_config(args.config)
        model_path = args.pre_train
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] 参考: python main_train.py -c configs/simple_mnist_config.json')
        exit(0)

    np.random.seed(47)

    print('[INFO] 加载数据…')
    dl = FaceNetDL(config=config)

    print('[INFO] 构造网络…')
    if config.backbone == 'vgg':
        print('[INFO] 使用 VGG 作为骨架')
    elif config.backbone == 'alexnet':
        print('[INFO] 使用 AlexNet 作为骨架')
    else:
        print('[INFO] 使用多层 CNN 作为骨架')

    if model_path != 'None':
        model = MangaFaceNetModel(config=config, model_path=model_path)
    else:
        model = MangaFaceNetModel(config=config)

    print('[INFO] 训练网络')
    trainer = VGGMangaTrainer(
        model=model.model,
        data=[dl.get_train_data(), dl.get_validation_data()],
        config=config
    )
    trainer.train()
    print('[INFO] 训练完成…')


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    train_vgg_manga()
