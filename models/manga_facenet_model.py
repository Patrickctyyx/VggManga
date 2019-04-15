import os
import tensorflow as tf

from bases.model_base import ModelBase


class MangaFaceNetModel(ModelBase):

    def __init__(self, config, model_path=None, with_bottom=False, use_vgg=False):
        super(MangaFaceNetModel, self).__init__(config)
        self.with_bottom = with_bottom
        self.use_vgg = use_vgg
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):

        # conv layers
        if self.use_vgg:
            main_input, backbone = self.get_vgg_backbone()
        else:
            main_input, backbone = self.get_simple_backbone()

        fl = tf.keras.layers.Flatten(name='flatten')(backbone)
        # top branch
        x = tf.keras.layers.Dense(256, activation='relu', name='fc1')(fl)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions_top = tf.keras.layers.Dense(2, activation='softmax', name='fc2')(x)

        if self.with_bottom:
            # bottom branch
            bottom = tf.keras.layers.Dense(256, activation='relu', name='fc1_bottom')(fl)
            bottom = tf.keras.layers.Dropout(0.5)(bottom)
            predictions_bottom = tf.keras.layers.Dense(4, name='fc2_bottom')(bottom)

            model = tf.keras.models.Model(inputs=main_input, outputs=[predictions_top, predictions_bottom])

            losses = {
                'fc2': 'binary_crossentropy',
                'fc2_bottom': 'mean_squared_error'
            }
            loss_weights = {
                'fc2': 1.0, 'fc2_bottom': 1.0
            }
            model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights, metrics=['accuracy'])
        else:
            model = tf.keras.models.Model(inputs=main_input, outputs=predictions_top)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, 'manga_facenet.png'),
                                  show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)

    def get_simple_backbone(self):
        main_input = tf.keras.Input(shape=(40, 40, 1), name='input')
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

        return main_input, layer_5

    def get_vgg_backbone(self):
        main_input = tf.keras.Input(shape=(224, 224, 3), name='input')

        conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv1_1')(main_input)
        conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv1_2')(conv1_1)
        bn1 = tf.keras.layers.BatchNormalization()(conv1_2)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(bn1)

        conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv2_1')(pool1)
        conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv2_2')(conv2_1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2_2)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(bn2)

        conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv3_1')(pool2)
        conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv3_2')(conv3_1)
        conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv3_3')(conv3_2)
        bn3 = tf.keras.layers.BatchNormalization()(conv3_3)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(bn3)

        conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv4_1')(pool3)
        conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv4_2')(conv4_1)
        conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv4_3')(conv4_2)
        bn4 = tf.keras.layers.BatchNormalization()(conv4_3)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(bn4)

        conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv5_1')(pool4)
        conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv5_2')(conv5_1)
        conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         activation='relu',
                                         name='conv5_3')(conv5_2)
        bn5 = tf.keras.layers.BatchNormalization()(conv5_3)
        pool5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(bn5)
        return main_input, pool5
