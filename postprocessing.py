import pandas as pd   
import numpy as np   

import config   
import training
import preprocessing


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    ls = []
    for i in range(preds.shape[0]):
        a = np.log(preds[i]) / temperature
        exp_preds = np.exp(a)
        a = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, a, size = 1)
        probas = np.asarray(probas[0])
        residue = indices_char[np.argmax(probas, axis = -1).astype(int)]
        
        if residue == '\n':
            break
        else:
            ls.append(residue)
    

    return ls

if __name__ == '__main__':
        
    novel_seq = []
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    x, y, char_indices, indices_char = preprocessing.preprocessing(df)
    model = training.run()

    for i in range(x.shape[0]):
        pred = model.predict(x[i].reshape(1, config.MAX_LEN, len(char_indices)))
        pred = sample(pred.reshape(config.MAX_LEN, len(char_indices)))
        novel_seq.append(pred)