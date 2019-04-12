# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from data_loaders.vgg_manga_dl import VGGMangaDL
from infers.vgg_manga_infer import VGGMangaInfer
from utils.config_utils import process_config, get_test_args


def test_vgg_manga():
    manga_dir = 'manga109_frame_face'
    print('[INFO] 解析配置…')
    parser = None
    config = None
    model_path = None

    try:
        args, parser = get_test_args()
        config = process_config(args.config)
        model_path = args.model
    except Exception as e:
        print('[Exception] 配置无效, %s' % e)
        if parser:
            parser.print_help()
        print('[Exception] 参考: python main_test.py -c configs/simple_mnist_config.json '
              '-m simple_mnist.weights.10-0.24.hdf5')
        exit(0)

    np.random.seed(47)

    print('[INFO] 加载数据…')
    dl = VGGMangaDL(config=config, manga_dir=manga_dir)
    test_generator = dl.get_test_data()

    print('[INFO] 测试模型…')
    infer = VGGMangaInfer(model_path, config)
    result = infer.model.evaluate_generator(test_generator)
    print(result)
    # print('[INFO] loss: %.4f, accuracy: %.4f' % (result[0], result[1]))
    print('[INFO] 测试完成…')

    # print('[INFO] 预测数据…')
    # infer = MnistInfer(model_path, config)
    # infer_label = np.argmax(infer.predict(test_data))
    # print ('[INFO] 真实Label: %s, 预测Label: %s' % (test_label, infer_label))

    # print ('[INFO] 预测完成…')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    test_vgg_manga()
