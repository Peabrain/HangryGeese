from typing import Deque
import _pickle as cPickle
import gc
import json
import numpy as np
import os
import glob
import random
from typing import Any

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
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

dir_dict = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}

playground_shape = (11,11)
target_shape = (11,11)

def transfer_weights_partially(source, target, lr=0.5):
    wts = source.get_weights()
    twts = target.get_weights()

    for i in range(len(wts)):
        twts[i] = lr * wts[i] + (1-lr) * twts[i]
    target.set_weights(twts)

def masked_mse(args):
    y_true, y_pred, mask = args
    loss = (y_true - y_pred) ** 2
    loss *= mask
    return K.sum(loss,axis=-1)

def add_rl_loss_to_model(model):
    num_actions = model.output.shape[1]
    y_pred = model.output
    y_true = keras.Input(name='y_true',shape=(num_actions,))
    mask = keras.Input(name='mask',shape=(num_actions,))
    loss_out = Lambda(masked_mse,output_shape=(1,0),name='loss')([y_true,y_pred,mask])
    trainable_model = Model(inputs=[model.input,y_true,mask], outputs=loss_out)
    trainable_model.compile(optimizer='adam', loss=lambda yt,yp:yp)
    return trainable_model
    

def createModel():
    dropout = 0.0

    i0 = keras.Input(shape=(playground_shape[0], playground_shape[1], 9))
#    x = LayerNormalization()(i0)
    x = i0

#    x = Conv2D(filters=64, kernel_size=(3,3,3), activation='linear', padding='valid')(x)
#    x = keras.layers.Reshape((9, 9, 64))(x)
    x = Conv2D(filters=16, kernel_size=(3,3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
    x = ReLU()(x)
#    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
#    x = ReLU()(x)
#    x = tf.keras.layers.Concatenate(axis=3)([x0, x1, x2])
#    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
#    x = ReLU()(x)
#    x = Conv2D(filters=128, kernel_size=(3,3), padding='valid')(x)
#    x = ReLU()(x)
#    x = MaxPool2D(2)(x)
#    x = Dropout(dropout)(x)
#    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
#    x = ReLU()(x)
#    x = MaxPool2D(2)(x)
#    x = Dropout(dropout)(x)
#    x = Conv2D(filters=128, kernel_size=(3,3), padding='same')(x)
#    x = ReLU()(x)
#    x = MaxPool2D(2)(x)
    x = Dropout(dropout)(x)
#    x = Conv2D(filters=64, kernel_size=(3,3), activation='linear', padding='same')(x)
#    x = Dropout(dropout)(x)    
#    x = MaxPool2D(pool_size=2)(x)
#    x = Conv2D(filters=8, kernel_size=(3,3), activation='linear', padding='same',use_bias=False)(x)
#    x = Dropout(dropout)(x)
#    x = Conv2D(filters=8, kernel_size=(3,3), activation='linear', padding='same')(x)
#    x = Dropout(dropout)(x)
    x = Flatten()(x)
#    x0 = Dense(128, activation='linear')(x)
#    x = Dropout(dropout)(x0)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dropout(dropout)(x)
#    x = Add()((x,x0))
#    x = Dense(32, activation='linear')(x)
#    x = Dropout(dropout)(x)
#    x = Dense(32, activation='linear')(x)
#    x = Dropout(dropout)(x)
    o0 = Dense(3, activation='linear')(x)
    model = Model(i0,o0)
#    opt = keras.optimizers.Adam()#.minimize(loss)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def getPlayGround(playersData,foodData,lastActions):
    playground = np.zeros((playground_shape[0],playground_shape[1]), dtype=np.uint8)
    y_p = playersData[0][0] // playground.shape[1]
    x_p = playersData[0][0] % playground.shape[1]

    for f in foodData:
        y_ = f // playground.shape[1]
        x_ = f % playground.shape[1]
        playground[y_,x_] = 8

    for g in range(len(playersData)):
        e = (g != 0) * 4
        go = playersData[g]
        for l in range(1,len(go)):
            y = go[l] // playground.shape[1]
            x = go[l] % playground.shape[1]
            playground[y,x] = 1 * (l < (len(go) - 1)) + (l == 0 or l == (len(go) - 1)) * 2 + e

    playground = np.tile(playground, (3,3))
    playground = playground[(y_p + playground_shape[0]) - 5:(y_p + playground_shape[0]) - 5 + playground_shape[0],(x_p + playground_shape[1]) - 5:(x_p + playground_shape[1]) - 5 + playground_shape[1]]

    if lastActions == dir_dict['SOUTH']:
        playground = np.rot90(playground,1,(0,1))
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['EAST']:
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['WEST']:
        playground = np.rot90(playground,-1,(0,1))

    return playground

class Game:
    def init(self):
        self.player = [[random.randint(0,playground_shape[0] * playground_shape[1] - 1)]]
        self.last_action = dir_dict['NORTH']
#        self.next_action = dir_dict['NORTH']
        self.food = list([np.random.randint(playground_shape[0] * playground_shape[1]) for i in range(2)])
    
    def do_action(self,action_rel):

        action = (self.last_action + action_rel - 1) % 4

        y = self.player[0][0] // playground_shape[1]
        x = self.player[0][0] % playground_shape[1]
        if action == 0:
            y = (y - 1) % playground_shape[0]
        elif action == 2:
            y = (y + 1) % playground_shape[0]
        if action == 1:
            x = (x - 1) % playground_shape[1]
        elif action == 3:
            x = (x + 1) % playground_shape[1]

        new_pos = y * playground_shape[1] + x

        if new_pos in self.player[0]:
            reward = -1
            self.player[0].remove(self.player[0][-1])
            self.player[0] = [new_pos] + self.player[0]
        else:
            if new_pos in self.food:
                self.player[0] = [new_pos] + self.player[0]
                self.food.remove(new_pos)
                reward = 1
                self.food = self.food + [random.randint(0,playground_shape[0] * playground_shape[1] - 1)]
            else:
                self.player[0].remove(self.player[0][-1])
                self.player[0] = [new_pos] + self.player[0]
                reward = -0.01


            self.last_action = action

        playground = getPlayGround(self.player,self.food,action_rel)
        return playground, reward
    def get_playground(self):
        return getPlayGround(self.player,self.food,self.last_action)


class Step:
    def __init__(self,st,stn,at,lat,rt,done):
        self.st = st
        self.stn = stn
        self.at = at
        self.lat = lat
        self.rt = rt
        self.done = done

def play_game(eps,model,memory):
    game = Game()
    game.init()
    count = 100
    done = False
    rewards = 0
    while count > 0 and done == False:
        playground_t = game.get_playground()
        if random.random() < eps:
            action = np.random.randint(3)
        else:
            target_vector = model.predict(keras.backend.one_hot([playground_t],num_classes=9))[0]
#            target_vector = model.predict(np.array([playground_t]))[0]
            action = np.argmax(target_vector)

        lat = game.last_action

        playground_tn, reward = game.do_action(action)
        if reward == -1:
            done = True
        elif reward == 1:
            rewards += 1

        step = Step(st = playground_t,stn = playground_tn, at = action, lat = lat, rt = reward, done = done)
        memory.append(step)
        count -= 1

    return rewards

eps = 1.0
model_ = createModel()
model_target = createModel()
model_training = add_rl_loss_to_model(model_)
transfer_weights_partially(model_,model_target,1)
model_target.summary()
memory = Deque()
rewards = 0
for i in range(10000):
    rewards += play_game(eps,model_target,memory)
print(rewards)

sample_num = 100

for k in range(100000):
    sample = random.sample(memory,sample_num)

#    target_vectors = model_.predict(np.array([i.st for i in sample]))
#    fut_actions = model_target.predict(np.array([i.stn for i in sample]))
    target_vectors = model_.predict(keras.backend.one_hot([i.st for i in sample],num_classes=9))
    fut_actions = model_target.predict(keras.backend.one_hot([i.stn for i in sample],num_classes=9))

    X = np.zeros((sample_num,target_shape[0],target_shape[1]),dtype=np.uint8)
    Y = np.zeros((sample_num,3))
    M = np.zeros((sample_num,3))


    for i in range(sample_num):
        j = sample[i]

        target_vector, fut_action = target_vectors[i].copy(), fut_actions[i].copy()
        target = j.rt
        if not j.done:
            target += 0.95 * np.max(fut_action)

        target_vector[j.at] = target
        mask = target_vector.copy() * 0
        mask[j.at] = 1

        X[i] = j.st
        Y[i] = target_vector
        M[i] = mask

    model_training.fit([keras.backend.one_hot(X,num_classes=9),Y,M],Y,epochs=1,verbose=0)#,validation_split=0.1)
#    model_training.fit([X,Y,M],Y,epochs=1,verbose=0)#,validation_split=0.1)

#    memory.pop()
    for i in range(1):
        play_game(eps,model_target,memory)

    if k % 50 == 0:
        transfer_weights_partially(model_,model_target,1)
        rewards = 0
        for i in range(10):
            rewards += play_game(0,model_target,memory)
        print(k,rewards,eps)
        model_target.save('./model_target')

    eps *= 0.99999
    eps = np.maximum(0.1,eps)


#for i in memory:
#    print(i.st,i.at,i.lat,i.rt)