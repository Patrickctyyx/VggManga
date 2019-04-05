import os
import warnings
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_recall_fscore_support

from bases.trainer_base import TrainerBase
import numpy as np

from utils.np_utils import prp_2_oh_array


class VGGMangaTrainer(TrainerBase):

    def __init__(self, model, data, config):
        super(VGGMangaTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []  # loss from validation
        self.val_acc = []  # accuracy from validation
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.cp_dir,
                                      '%s.weights.{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp_name),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=False
            )
        )

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config.tb_dir,
                write_images=True,
                write_graph=True
            )
        )

        self.callbacks.append(FPRMetricDetail())

    def train(self):
        history = self.model.fit_generator(
            self.data[0],
            epochs=self.config.num_epochs,
            verbose=2,
            validation_data=self.data[1],
            callbacks=self.callbacks
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])


class FPRMetricDetail(tf.keras.callbacks.Callback):
    """
    Output F, P, R
    """

    def on_epoch_end(self, batch, logs=None):
        val_x = self.validation_data[0]
        val_y = self.validation_data[1]

        pred_y = prp_2_oh_array(np.asarray(self.model.predict(val_x)))

        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        precision, recall, f_score, support = precision_recall_fscore_support(val_y, pred_y)

        for p, r, f, s in zip(precision, recall, f_score, support):
            print(" - val_f1: %0.4f - val_pre: %0.4f - val_rec: %0.4f - ins %s" % (f, p, r, s))
