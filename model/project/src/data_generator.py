
import numpy as np

from threading import Semaphore, Thread
from time import sleep
from random import choice, randint
from pdb import set_trace as pause
from tensorflow import keras
from src.sampler import augment_sample, labels2output_map

class DataGenerator(keras.utils.Sequence):

    def __init__(	self, data, model_stride,       \
                    batch_size			= 32,		\
                    dim 				= 204,       \
                    shuffle             = True,      \
                ):

        self._data = data
        self._batch_size = batch_size
        self._dim = dim
        self._model_stride = model_stride
        self._shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._data) / self._batch_size))

    def __getitem__(self, index):
        indexes = self._indexes[index*self._batch_size:(index+1)*self._batch_size]
        d = self._data[indexes]
        X,Y = [], []
        for item in d:
            x, y = self._process_data_item(item)
            X.append(x)
            Y.append(y)
        return np.array(X),np.array(Y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self._indexes = np.arange(len(self._data))
        if self._shuffle == True:
            np.random.shuffle(self._indexes)

    def _process_data_item(self,data_item):
        XX,llp,pts = augment_sample(data_item[0],data_item[1].pts,self._dim)
        YY = labels2output_map(llp,pts,self._dim,self._model_stride)
        return XX,YY