import os
import tensorflow as tf

from bases.model_base import ModelBase


class VGGMangaModel(ModelBase):

    def __init__(self, config, model_path=None, fine_tune=False):
        super(VGGMangaModel, self).__init__(config)
        self.fine_tune = fine_tune
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):

        if self.fine_tune:
            base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                                           include_top=False,
                                                           input_shape=(224, 224, 3))

            x = base_model.output
            x = tf.keras.layers.Flatten(name='flatten')(x)
            x = tf.keras.layers.Dense(4096, activation='relu', name='fc1')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            predictions = tf.keras.layers.Dense(self.config.num_classes, activation='sigmoid', name='predictions')(x)

            model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

            for layer in base_model.layers:
                layer.trainable = False
        else:
            model = tf.keras.applications.vgg16.VGG16(weights=None, classes=self.config.num_classes)

        model.compile(optimizer='adam', loss='binary_crossentropy')

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, 'vgg16.png'), show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)
