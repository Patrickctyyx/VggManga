import os

import tensorflow as tf
from bases.infer_base import InferBase


class VGGMangaInfer(InferBase):

    def __init__(self, name, config=None):
        super(VGGMangaInfer, self).__init__(config)
        self.model = self.load_model(name)

    def load_model(self, name):
        model = os.path.join(self.config.cp_dir, name)
        return tf.keras.models.load_model(model)

    def predict(self, data):
        return self.model.predict(data)
