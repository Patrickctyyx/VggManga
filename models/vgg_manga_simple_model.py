import os
import tensorflow as tf

from bases.model_base import ModelBase


class VGGMangaSimpleModel(ModelBase):

    def __init__(self, config, model_path=None):
        super(VGGMangaSimpleModel, self).__init__(config)
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):

        # conv layers
        main_input = tf.keras.Input(shape=(224, 224, 3), name='input')
        layer_1 = tf.keras.layers.Convolution2D(32, (3, 3), padding='same', activation='relu', name='conv1')(main_input)
        layer_1 = tf.keras.layers.BatchNormalization()(layer_1)
        layer_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(layer_1)

        layer_2 = tf.keras.layers.Convolution2D(32, (3, 3), padding='same', activation='relu', name='conv2')(layer_1)
        layer_2 = tf.keras.layers.BatchNormalization()(layer_2)
        layer_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(layer_2)

        layer_3 = tf.keras.layers.Convolution2D(64, (3, 3), padding='same', activation='relu', name='conv3')(layer_2)
        layer_3 = tf.keras.layers.BatchNormalization()(layer_3)
        layer_3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(layer_3)
        
        # fc layers
        x = tf.keras.layers.Flatten(name='flatten')(layer_3)
        x = tf.keras.layers.Dense(64, activation='relu', name='fc1')(x)
#         x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(2, activation='sigmoid', name='fc2')(x)

        model = tf.keras.models.Model(inputs=main_input, outputs=predictions)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, 'manga_simple.png'), show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)
