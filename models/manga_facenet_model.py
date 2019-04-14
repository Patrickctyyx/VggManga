import os
import tensorflow as tf

from bases.model_base import ModelBase


class MangaFaceNetModel(ModelBase):

    def __init__(self, config, model_path=None, with_bottom=False):
        super(MangaFaceNetModel, self).__init__(config)
        self.with_bottom = with_bottom
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):

        # conv layers
        main_input = tf.keras.Input(shape=(40, 40), name='input')
        layer_1 = tf.keras.layers.Convolution2D(32, (3, 3), padding='same', activation='relu', name='conv1')(main_input)
        layer_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(layer_1)
        layer_1 = tf.keras.layers.Dropout(0.25)(layer_1)

        layer_2 = tf.keras.layers.Convolution2D(64, (3, 3), padding='same', activation='relu', name='conv2')(layer_1)
        layer_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(layer_2)
        layer_2 = tf.keras.layers.Dropout(0.25)(layer_2)

        layer_3 = tf.keras.layers.Convolution2D(128, (3, 3), padding='same', activation='relu', name='conv3')(layer_2)

        layer_4 = tf.keras.layers.Convolution2D(128, (3, 3), padding='same', activation='relu', name='conv4')(layer_3)

        layer_5 = tf.keras.layers.Convolution2D(128, (3, 3), padding='same', activation='relu', name='conv5')(layer_4)
        layer_5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(layer_5)
        layer_5 = tf.keras.layers.Dropout(0.25)(layer_5)

        fl = tf.keras.layers.Flatten(name='flatten')(layer_5)
        # top branch
        x = tf.keras.layers.Dense(256, activation='relu', name='fc1')(fl)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions_top = tf.keras.layers.Dense(2, activation='softmax', name='fc2')(x)

        # bottom branch
        bottom = tf.keras.layers.Dense(256, activation='relu', name='fc1_bottom')(fl)
        bottom = tf.keras.layers.Dropout(0.5)(bottom)
        predictions_bottom = tf.keras.layers.Dense(4, name='fc2_bottom')(bottom)

        model = tf.keras.models.Model(inputs=main_input, outputs=[predictions_top, predictions_bottom])

        if self.with_bottom:
            losses = {
                'fc2': 'binary_crossentropy',
                'fc2_bottom': 'mean_squared_error'
            }
            loss_weights = {
                'fc2': 1.0, 'fc2_bottom': 1.0
            }
            model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, 'manga_facenet.png'),
                                  show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)
