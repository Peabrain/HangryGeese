import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv3D, Conv2D, MaxPool2D, ReLU, Add, Lambda, LayerNormalization
import keras
import numpy as np
import pickle

#weights_txt = 
#weights = pickle.loads(weights_txt)
#for i in weights:
#    print(i.shape)
model = load_model('./model_target')
weights = model.get_weights()

x_as_bytes = pickle.dumps(weights)

f = open('weights.txt', 'w')
f.write(str(x_as_bytes))
f.close()

#print(x_as_bytes)

#for i in range(len(weights)):
#    weights[i].tofile('weights_' + str(i) + '.txt')
#    with open('weights_' + str(i) + '.txt', 'wb') as f:
#        np.savetxt(f, np.column_stack(weights[i]), fmt='%1.10f')

