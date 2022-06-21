import os
from keras.optimizer_v2 import rmsprop

INPUTS = "./inputs"
OUTPUTS = "./outputs"
FILE_NAME = "AMP_dataset"
TRAIN_DATA_PATH = os.path.join(INPUTS, f"{FILE_NAME}.csv")

FOLDS = 5
MAX_LEN = 21
TRAIN_SIZE = 0.8
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001

LOSS = 'categorical_crossentropy'
OPTIMIZER = rmsprop.RMSProp(learning_rate=LR)

MODEL_PATH = "./models"

THRESHOLD = 0.5