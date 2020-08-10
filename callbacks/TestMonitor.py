import os
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging
from sklearn.metrics import cohen_kappa_score, accuracy_score

class STAccuracyMonitor(Callback):

    def __init__(self,
                test_generator=None,
                out_file=None,
                precision=4,
                verbose=0):
        super(STAccuracyMonitor, self).__init__()

        self.out_file = out_file
        self.precision = precision
        self.verbose = verbose
        self.acc_df = None

        if test_generator is None:
            logging.error("\nSTAccuracyMonitor: Test generator is empty. Skipping...", RuntimeError)
            return
        else:
            self.test_generator = test_generator

        if os.path.exists(self.out_file):
            logging.warning("\nSTAccuracyMonitor: Output file already exists. File will be overwritten.", RuntimeWarning)


    def on_train_begin(self, logs=None):
        df = pd.Dataframe({'Epoch': [],
                            'K': [],
                            'OA': []})
        self.acc_df = df

    
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.test_generator, steps=len(self.test_generator), verbose=self.verbose)

        pred_class_indices = np.argmax(preds[0], axis=1)
        test_class_indices = np.argmax(self.test_generator.labels[0], axis=1)

        K = cohen_kappa_score(test_class_indices, pred_class_indices, labels=None, weights=None, sample_weight=None)
        OA = accuracy_score(test_class_indices, pred_class_indices, normalize=True, sample_weight=None)
        K = round(K, self.precision)
        OA = round(OA, self.precision)

        acc_series = pd.Series({'Epoch': epoch, 'K': K, 'OA': OA})
        self.acc_df = self.acc_df.append(acc_series, ignore_index=True)

        if self.verbose > 0:
            print("\nSTAccuracyMonitor: K= {}, OA= {}".format(K, OA))

    def on_train_end(self, logs=None):
        try:
            self.acc_df.to_csv(self.out_file, index=False)
        except IOError as e:
            logging.error("STAccuracyMonitor: Writing dataframe failed: {}".format(e), RuntimeError)
        