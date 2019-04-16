import os
import tensorflow as tf

from bases.model_base import ModelBase


class MangaFaceNetModel(ModelBase):

    def __init__(self, config, model_path=None):
        super(MangaFaceNetModel, self).__init__(config)
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.build_model()

    def build_model(self):

        # conv layers
        main_input = tf.keras.Input(
            shape=(self.config.input_shape, self.config.input_shape, self.config.input_channel),
            name='input')
        if self.config.backbone == 'alexnet':
            backbone = self.get_alexnet_backbone(main_input)
        elif self.config.backbone == 'vgg':
            backbone = self.get_vgg_backbone(main_input)
        else:
            backbone = self.get_simple_backbone(main_input)

        fl = tf.keras.layers.Flatten(name='flatten')(backbone)
        # top branch
        x = tf.keras.layers.Dense(256, activation='relu', name='fc1')(fl)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions_top = tf.keras.layers.Dense(self.config.num_classes, activation='softmax', name='fc2')(x)

        if self.config.with_bottom:
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
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        tf.keras.utils.plot_model(model, to_file=os.path.join(self.config.img_dir, self.config.exp_name + '.png'),
                                  show_shapes=True)

        self.model = model

    def load_model(self, model_path):
        model = os.path.join(self.config.cp_dir, model_path)
        return tf.keras.models.load_model(model)

    def get_simple_backbone(self, main_input):

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

        return layer_5

    def get_vgg_backbone(self, main_input):

        conv1_1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                         name='conv1_1')(main_input)
        bn1_1 = tf.keras.layers.BatchNormalization()(conv1_1)
        relu1_1 = tf.keras.layers.ReLU()(bn1_1)
        conv1_2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                         name='conv1_2')(relu1_1)
        bn1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
        relu1_2 = tf.keras.layers.ReLU()(bn1_2)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(relu1_2)

        conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                         name='conv2_1')(pool1)
        bn2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
        relu2_1 = tf.keras.layers.ReLU()(bn2_1)
        conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                         name='conv2_2')(relu2_1)
        bn2_2 = tf.keras.layers.BatchNormalization()(conv2_2)
        relu2_2 = tf.keras.layers.ReLU()(bn2_2)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(relu2_2)

        conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         name='conv3_1')(pool2)
        bn3_1 = tf.keras.layers.BatchNormalization()(conv3_1)
        relu3_1 = tf.keras.layers.ReLU()(bn3_1)
        conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         name='conv3_2')(relu3_1)
        bn3_2 = tf.keras.layers.BatchNormalization()(conv3_2)
        relu3_2 = tf.keras.layers.ReLU()(bn3_2)
        conv3_3 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                         name='conv3_3')(relu3_2)
        bn3_3 = tf.keras.layers.BatchNormalization()(conv3_3)
        relu3_3 = tf.keras.layers.ReLU()(bn3_3)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(relu3_3)

        conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         name='conv4_1')(pool3)
        bn4_1 = tf.keras.layers.BatchNormalization()(conv4_1)
        relu4_1 = tf.keras.layers.ReLU()(bn4_1)
        conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         name='conv4_2')(relu4_1)
        bn4_2 = tf.keras.layers.BatchNormalization()(conv4_2)
        relu4_2 = tf.keras.layers.ReLU()(bn4_2)
        conv4_3 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         name='conv4_3')(relu4_2)
        bn4_3 = tf.keras.layers.BatchNormalization()(conv4_3)
        relu4_3 = tf.keras.layers.ReLU()(bn4_3)
        pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(relu4_3)

        conv5_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         name='conv5_1')(pool4)
        bn5_1 = tf.keras.layers.BatchNormalization()(conv5_1)
        relu5_1 = tf.keras.layers.ReLU()(bn5_1)
        conv5_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         name='conv5_2')(relu5_1)
        bn5_2 = tf.keras.layers.BatchNormalization()(conv5_2)
        relu5_2 = tf.keras.layers.ReLU()(bn5_2)
        conv5_3 = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                         name='conv5_3')(relu5_2)
        bn5_3 = tf.keras.layers.BatchNormalization()(conv5_3)
        relu5_3 = tf.keras.layers.ReLU()(bn5_3)
        pool5 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(relu5_3)
        return pool5

    def get_alexnet_backbone(self, main_input):

        conv_1 = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='same',
                                        name='conv1')(main_input)
        bn_1 = tf.keras.layers.BatchNormalization()(conv_1)
        relu_1 = tf.keras.layers.ReLU()(bn_1)
        pool_1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(relu_1)

        conv_2 = tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same',
                                        name='conv2')(pool_1)
        bn_2 = tf.keras.layers.BatchNormalization()(conv_2)
        relu_2 = tf.keras.layers.ReLU()(bn_2)
        pool_2 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(relu_2)

        conv_3 = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                                        name='conv3')(pool_2)
        bn_3 = tf.keras.layers.BatchNormalization()(conv_3)
        relu_3 = tf.keras.layers.ReLU()(bn_3)

        conv_4 = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same',
                                        name='conv4')(relu_3)
        bn_4 = tf.keras.layers.BatchNormalization()(conv_4)
        relu_4 = tf.keras.layers.ReLU()(bn_4)

        conv_5 = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                        name='conv5')(relu_4)
        bn_5 = tf.keras.layers.BatchNormalization()(conv_5)
        relu_5 = tf.keras.layers.ReLU()(bn_5)
        pool_5 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(relu_5)

        return pool_5
