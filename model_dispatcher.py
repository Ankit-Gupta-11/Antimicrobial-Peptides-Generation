from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizer_v2 import rmsprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import pandas as pd

import config
# import hyperparameter_tunning


def get_model(model_name, vocab_size):

    if model_name == 'generation':
            
        print('Build model...')
        model = Sequential()
        # model.add(tf.keras.layers.GRU(512,  input_shape=(maxlen, len(chars)), return_sequences = True))
        model.add(LSTM(512, input_shape = (config.MAX_LEN, vocab_size), return_sequences = True))
        model.add(LSTM(256, return_sequences = True))
        model.add(LSTM(128, return_sequences = True))
        # model.add(LSTM(128, return_sequences = True))
        # model.add(tf.keras.layers.Dropout(0.2))
        model.add(Dense(vocab_size, activation = 'softmax'))
        model.compile(loss= config.LOSS, optimizer=config.OPTIMIZER , metrics = ['accuracy'])
        
        return model

    elif model_name == "prediction":
        pass

