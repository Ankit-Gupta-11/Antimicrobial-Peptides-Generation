import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import os  

import config
import model_dispatcher
import preprocessing

def run():
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    X, y, char_indices, indices_char = preprocessing.preprocessing(df)
    model = model_dispatcher.get_model("generation", len(char_indices))
    X_train, X_val, Y_train, Y_val = train_test_split(X, y , train_size = config.TRAIN_SIZE, random_state = 11)
    model.fit(X, y, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS)
    tf.keras.models.save_model(os.path.join(config.MODEL_PATH, "generation.h5"))
    
    return model 
