import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import tensorflow.keras as keras
#import keras
#import keras.backend as K

from typing import Deque
import _pickle as cPickle
import collections
import itertools
import gc
import json
import numpy as np
import os
import glob
import random
from typing import Any
import copy

#from keras.models import Sequential, load_model, Model
#from keras.layers import Dense, Dropout, Flatten, Conv3D, Conv2D, MaxPool2D, ReLU, Add, Lambda, LayerNormalization
#from keras import optimizers
#import tensorflow as tf

#from sklearn.model_selection import train_test_split

dir_dict = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}
dir_ = ['NORTH', 'WEST', 'SOUTH', 'EAST']

playground_shape = (7,11)
target_shape = (11,11)

memorySize = 3
player_num = 2

#def transfer_weights_partially(source, target, lr=0.5):
#    target.set_weights(source.get_weigths())
#    return
#    wts = source.get_weights()
#    twts = target.get_weights()

#    for i in range(len(wts)):
#        twts[i] = lr * wts[i] + (1-lr) * twts[i]
#    target.set_weights(twts)

def masked_mse(args):
    y_true, y_pred, mask = args
    loss = (y_true - y_pred) ** 2
    loss *= mask
    return loss # K.sum(loss,axis=0)

def add_rl_loss_to_model(model):
    num_actions = model.output.shape[1]
    y_pred = model.output
    y_true = keras.layers.Input(name='y_true',shape=(num_actions,))
    mask = keras.layers.Input(name='mask',shape=(num_actions,))
    loss_out = keras.layers.Lambda(masked_mse,output_shape=(3,),name='loss')([y_true,y_pred,mask])
    trainable_model = keras.models.Model(inputs=[model.input,y_true,mask], outputs=loss_out)
#    opt = optimizers.Adam(learning_rate=0.0001)
    trainable_model.compile(optimizer='adam', loss=lambda yt,yp:yp)
    return trainable_model

def res(x):    
    x0 = keras.layers.Dense(32)(x)
    x0 = keras.layers.ReLU()(x0)
    x0 = keras.layers.Dense(32)(x0)
    x0 = keras.layers.ReLU()(x0)
    x = keras.layers.Add()([x0,x])
    return x

class MyModel(keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv10 = keras.layers.Conv3D(filters=32, kernel_size=(1,3,3), padding='same',activation='relu',use_bias=True)
        self.conv11 = keras.layers.Conv3D(filters=32, kernel_size=(1,3,3), padding='same',activation='relu',use_bias=True)
#        self.relu1 = keras.layers.ReLU()
        self.maxpool1 = keras.layers.MaxPool3D((1,2,2))
        self.conv20 = keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), padding='same',activation='relu',use_bias=True)
        self.conv21 = keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), padding='same',activation='relu',use_bias=True)
#        self.relu2 = keras.layers.ReLU()
        self.maxpool2 = keras.layers.MaxPool3D((1,2,2))
        self.conv30 = keras.layers.Conv3D(filters=256, kernel_size=(1,2,2), padding='same',activation='relu',use_bias=True)
        self.conv31 = keras.layers.Conv3D(filters=256, kernel_size=(1,2,2), padding='valid',activation='relu',use_bias=True)
#        self.relu3 = keras.layers.ReLU()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256,activation='relu')
#        self.relu4 = keras.layers.ReLU()
        self.dense2 = keras.layers.Dense(256,activation='relu')
#        self.relu5 = keras.layers.ReLU()

        self.convlstm10 = keras.layers.ConvLSTM2D(32,kernel_size=(3,3),return_sequences=True)
        self.convlstm11 = keras.layers.ConvLSTM2D(32,kernel_size=(3,3),return_sequences=True)
        self.convlstm12 = keras.layers.ConvLSTM2D(32,kernel_size=(3,3),return_sequences=True)
        self.convlstm13 = keras.layers.ConvLSTM2D(32,kernel_size=(3,3),return_sequences=True)
        self.convlstm14 = keras.layers.ConvLSTM2D(32,kernel_size=(3,3),return_sequences=False)

        self.V = keras.layers.Dense(1,activation=None)
        self.A = keras.layers.Dense(3,activation=None)

    def mm(self,state):
        x = self.convlstm10(state)
        x = self.convlstm11(x)
        x = self.convlstm12(x)
        x = self.convlstm13(x)
        x = self.convlstm14(x)
        return x
        x = self.conv10(state)
        x = self.conv11(x)
#        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv20(x)        
        x = self.conv21(x)
#        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv30(x)
        x = self.conv31(x)
#        x = self.relu3(x)
        return x

    def call(self,state):
        x = self.mm(state)
        x = self.flatten(x)
        x = self.dense1(x)
#        x = self.relu4(x)
        x = self.dense2(x)
#        x = self.relu5(x)
        A = self.A(x)
        V = self.V(x)
        Q = (V + (A - tf.math.reduce_mean(A,axis=1,keepdims=True)))
        return Q

    def advantage(self,state):
        x = self.mm(state)
        x = self.flatten(x)
        x = self.dense1(x)
#        x = self.relu4(x)
        x = self.dense2(x)
#        x = self.relu5(x)
        A = self.A(x)
        return A

def createModel():
    dropout = 0.3

    i0 = keras.layers.Input(shape=(memorySize,target_shape[0], target_shape[1],9))
    i1 = keras.layers.Input(shape=(201,))
    x_ = keras.layers.Flatten()(i1)
    x_ = keras.layers.Dense(128)(x_)
    x = i0
#    x = keras.layers.Reshape((memorySize * target_shape[0] * target_shape[1],1))(x)
#    x = keras.layers.Embedding(9,8)(x)
#    x = keras.layers.Reshape((memorySize,target_shape[0], target_shape[1],8))(x)
#    x = tf.transpose(x, [0, 2, 3, 1])
    x = keras.layers.Conv3D(filters=32, kernel_size=(1,3,3), padding='same',activation=None,use_bias=True)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)#,return_sequences=False)(x)
#    x = keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), padding='same',activation=None,use_bias=False)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)#,return_sequences=False)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool3D((1,2,2))(x)
    x = keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), padding='same',activation=None,use_bias=True)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)#,return_sequences=False)(x)
#    x = keras.layers.Conv3D(filters=128, kernel_size=(1,3,3), padding='same',activation='linear',use_bias=True)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)#,return_sequences=False)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.Conv3D(filters=128, kernel_size=(1,3,3), padding='same',activation='linear',use_bias=True)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)#,return_sequences=False)(x)
    x = keras.layers.Conv3D(filters=256, kernel_size=(1,2,2), padding='valid',activation=None,use_bias=True)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)#,return_sequences=False)(x)
    x = keras.layers.ReLU()(x)
#    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.Conv3D(filters=16, kernel_size=(1,3,3), padding='same',activation='linear')(x)#,return_sequences=False)(x)
#    x = keras.layers.Conv3D(filters=16, kernel_size=(1,3,3), padding='same',activation='linear')(x)#,return_sequences=False)(x)
#    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.Conv3D(filters=16, kernel_size=(1,2,2), padding='valid',activation='linear')(x)#,return_sequences=False)(x)
#    x = keras.layers.Conv3D(filters=16, kernel_size=(1,2,2), padding='valid',activation='linear')(x)#,return_sequences=False)(x)
#    x = keras.layers.Reshape((memorySize,256))(x)
#    print(x.shape)
#    x = keras.layers.LSTM(8)(x)
#    print(x.shape)
#    x = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same',return_sequences=True,activation='linear')(x)
#    x = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same',return_sequences=True,activation='linear')(x)
#    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.Conv3D(filters=64, kernel_size=(2,3,3), padding='same',activation='linear')(x)#,return_sequences=False)(x)
#    x = keras.layers.ReLU()(x)
#    x = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.ConvLSTM2D(filters=32, kernel_size=(3,3), padding='same',return_sequences=True)(x)
#    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same')(x)#,return_sequences=True)(x)
#    x = keras.layers.MaxPool3D()(x)
#    x = keras.layers.ConvLSTM2D(filters=256, kernel_size=(3,3), padding='same')(x)
#    x = keras.layers.ReLU()(x)
#    x = keras.layers.Reshape(target_shape=(memorySize,256))(x)
#    x = keras.layers.ConvLSTM2D(16)(x)
#    x = tf.keras.layers.Conv1D(filters=4,kernel_size=3,padding='valid')(x)
#    x = tf.keras.layers.Dropout(dropout)(x)
    print(x.shape)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate(axis=1)([x, x_])
    x = keras.layers.Dense(256)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(256)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)
    x = keras.layers.ReLU()(x)
#    x = res(x)
#    x = res(x)
#    x = tf.keras.layers.Dropout(dropout)(x)
#    x = res(x)
#    x = tf.keras.layers.Dropout(dropout)(x)
    o0 = keras.layers.Dense(3, activation=None)(x)#,kernel_initializer='zeros',bias_initializer='zeros')(x)
    model = keras.models.Model(inputs=[i1,i0],outputs=[o0])
#    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def getPlayGround(playersData,foodData,lastActions,act_player,printf=False):
    playground = np.zeros((playground_shape[0],playground_shape[1]), dtype=np.uint8)
    if len(playersData[act_player]) == 0:
        return np.zeros((target_shape[0],target_shape[1]), dtype=np.uint8)
    y_p = playersData[act_player][0] // playground.shape[1]
    x_p = playersData[act_player][0] % playground.shape[1]

    for f in foodData:
        y_ = f // playground.shape[1]
        x_ = f % playground.shape[1]
        playground[y_,x_] = 8

    for g in range(len(playersData)):
        e = 0 # (g != act_player) * 4
        go = playersData[g]
        if len(go) == 0:
            continue
        for l in range(1,len(go)):
            y = go[l] // playground.shape[1]
            x = go[l] % playground.shape[1]
            playground[y,x] = 1 + e
        y = go[len(go) - 1] // playground.shape[1]
        x = go[len(go) - 1] % playground.shape[1]
        playground[y,x] = 2 + e
        y = go[0] // playground.shape[1]
        x = go[0] % playground.shape[1]
        playground[y,x] = 3 + e

    playground = np.tile(playground, (3,3))
    playground = playground[(y_p + playground_shape[0]) - target_shape[0] // 2:(y_p + playground_shape[0]) - target_shape[0] // 2 + target_shape[0],(x_p + playground_shape[1]) - target_shape[1] // 2:(x_p + playground_shape[1]) - target_shape[1] // 2 + target_shape[1]]

    if act_player == 0 and len(playersData[0]) == 2 and printf == True:
        print(playground,dir_[lastActions],playersData[0])

    if lastActions == dir_dict['SOUTH']:
        playground = np.rot90(playground,2,(0,1))
    elif lastActions == dir_dict['EAST']:
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['WEST']:
        playground = np.rot90(playground,3,(0,1))

    if act_player == 0 and len(playersData[0]) == 2 and printf == True:
        print(playground)
        print('---------------------------------------------------------------')

    return playground

class Game:
    def init(self):
        self.player = list(range(playground_shape[0] * playground_shape[1]))
        random.shuffle(self.player)
        self.player = self.player[:player_num]
        self.player = [[i] for i in self.player]
        self.last_action = [dir_dict['NORTH']] * len(self.player)
        self.food = list(range(playground_shape[0] * playground_shape[1]))
        pl = []
        for i in self.player:
            pl += i
        for i in pl:
            self.food.remove(i)
        random.shuffle(self.food)
        self.food = self.food[:2]
        self.round = 0
        self.MemorySteps = [np.zeros((memorySize,target_shape[0],target_shape[1]),dtype=np.uint8)] * player_num
        self.MemoryStepsNext = [np.zeros((memorySize,target_shape[0],target_shape[1]),dtype=np.uint8)] * player_num
    
    def do_action(self,actions_rel,reward):
        winner = 0
        self.round += 1
        new_pos = [None] * len(self.player)
        playground = [None] * len(self.player)
#        dones = [False] * len(self.player)
        actions = [None] * len(self.player)

#        for i in range(len(actions_rel)):
#            if len(self.player[i]) == 0:
#                dones[i] = True


        for i in range(len(actions_rel)):
            if len(self.player[i]) > 0:
                actions[i] = (self.last_action[i] + actions_rel[i] - 1) % 4

                y = self.player[i][0] // playground_shape[1]
                x = self.player[i][0] % playground_shape[1]
                if actions[i] == dir_dict['NORTH']:
                    y = (y - 1) % playground_shape[0]
                elif actions[i] == dir_dict['SOUTH']:
                    y = (y + 1) % playground_shape[0]
                if actions[i] == dir_dict['WEST']:
                    x = (x - 1) % playground_shape[1]
                elif actions[i] == dir_dict['EAST']:
                    x = (x + 1) % playground_shape[1]

                new_pos[i] = y * playground_shape[1] + x

        for i in range(len(actions_rel)):
            if len(self.player[i]) > 0:
                if new_pos[i] in self.food:
                    self.player[i] = [new_pos[i]] + self.player[i]
#                    reward[i] = 1
                else:
                    self.player[i].remove(self.player[i][-1])
                    self.player[i] = [new_pos[i]] + self.player[i]
                self.last_action[i] = actions[i]

        new_food = 0
        killed = False
        for i in range(len(actions_rel)):
            if len(self.player[i]) > 0:
                new_player_set = []
                for j in range(len(self.player)):
                    new_player_set += self.player[j][int(i == j):]
                if self.player[i][0] in new_player_set:
                    reward[i] = -2 # len(self.player[i]) # len(self.player[i])#(self.round * 2 + len(self.player[i]))
                    killed = True
#                    dones[i] = True
                    self.player[i] = []
                elif self.player[i][0] in self.food:
                    self.food.remove(self.player[i][0])
                    reward[i] = 1 # len(self.player[i])
                    new_food += 1
                else:
                    reward[i] = -0.0

#        if killed == True:
#            for i in range(len(actions_rel)):
#                if len(self.player[i]) > 0:
#                    reward[i] += 1


        if len(self.food) < 2:
            if random.random() < 0.3:
#        for i in range(new_food):
                nf = list(range(playground_shape[0] * playground_shape[1]))
                pl = []
                for j in self.player:
                    pl += j
                pl = set(pl)
                for j in pl:
                    nf.remove(j)
                for j in self.food:
                    nf.remove(j)
                random.shuffle(nf)
                self.food = self.food + [nf[0]]


        pl_count = 0
        pl_count_idx = 0
        for i in range(len(self.player)):
            if len(self.player[i]) > 0: # and (reward[i] == -1 or reward[i] >= 1):
                pl_count += 1
                pl_count_idx = i
        if pl_count == 1 or self.round == 200:
            winner = self.round * 2
#            reward[pl_count_idx] = self.round * 2 + len(self.player[pl_count_idx])
#            dones[pl_count_idx] = True

        return winner

    def get_playground(self,player_idx):
        return getPlayGround(self.player,self.food,self.last_action[player_idx],player_idx)


class Step:
    def __init__(self,st,stn,at,rt,done,round):
        self.st = st
        self.stn = stn
        self.at = at
        self.rt = rt
        self.done = done
        self.round = round
        self.prio = 1 if at == 0 else 2

def play_game(eps,model,memory,game_=None,pr=False):
    if game_ is None:
        game = Game()
        game.init()
    else:
        game = copy.deepcopy(game_)
    rewards = np.zeros((len(game.player)))
    dones = np.zeros((player_num))
    winner = 0
    while game.round < 200 and np.sum(dones) < player_num:
        action = [None] * len(game.player)

        playground_t = np.array([getPlayGround(game.player,game.food,game.last_action[player_idx],player_idx) for player_idx in range(len(game.player))])
        R = np.full((len(game.player)),game.round)
        for i in range(len(game.player)):
            for j in range(memorySize - 1):
                game.MemorySteps[i][j] = game.MemorySteps[i][j + 1]
            game.MemorySteps[i][memorySize - 1] = playground_t[i]
        if random.random() < eps:
            target_vectors = np.random.random((len(game.player),3))
        else:     
#            target_vectors = model.advantage(keras.backend.one_hot(np.array(game.MemorySteps),num_classes=9))
            target_vectors = model.advantage(np.expand_dims(np.array(game.MemorySteps),axis=4).astype(np.float16))
        action = np.argmax(target_vectors,axis=1)

        if pr == True:
            print('state')
            print(game.MemorySteps,target_vectors,R)
            return 0,0,0

        reward = np.zeros((len(game.player)))
        winner = game.do_action(action,reward)
        for i in range(len(game.player)):
            if dones[i] == 0:
#                game_state  = ([game.player[(i + j) % len(game.player)] for j in range(len(game.player))],game.food,game.last_action[i])
#                if reward[i] >= 1:
#                rewards[i] += reward[i]
                if (not game.player[i]) or (winner != 0):
                    dones[i] = 1

                playground_tn = getPlayGround(game.player,game.food,game.last_action[i],i)
                for j in range(memorySize - 1):
                    game.MemoryStepsNext[i][j] = game.MemoryStepsNext[i][j + 1]
                game.MemoryStepsNext[i][memorySize - 1] = playground_tn
                rt = reward[i]# + winner
                d = reward[i] < 0 or winner != 0 # or reward[i] < 0 # reward[i] != 0
#                if winner > 0:
#                    rt = len(game.player[i]) + winner
#                elif reward[i] == -1:
#                    rt = -1
#                else:
#                    rt = reward[i]
                step = Step(st = game.MemorySteps[i].copy(),stn = game.MemoryStepsNext[i].copy(), at = action[i], rt = rt, done = d,round=game.round)
                if reward[i] < 0:
                    reward[i] = 0
                if not (memory is None):
                    if rt == 0:
                        memory.append(step)
                    else:
                        memory.appendleft(step)
        rewards = rewards + reward


    return np.array(rewards),game.round,winner

def my_main():
    sample_num = 100
    eps = 1.0
    model_ = MyModel() # keras.models.load_model('./model_target') # createModel()
    model_.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model_target = MyModel()
    model_target.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model_.build((None,memorySize,11,11,1))
    model_target.build((None,memorySize,11,11,1))

    model_.summary()
    for i in range(len(model_.layers)):
        model_target.layers[i].set_weights(model_.layers[i].get_weights())


#    model_.save_weights('weights')
#    model_target.load_weights('weights')

#    model_target.set_weights(model_.get_weigths())

#    model_training = add_rl_loss_to_model(model_)
#    model_target = keras.models.clone_model(model_)
#    transfer_weights_partially(model_,model_target,1)
#    model_target.summary()
    memory = Deque()
    rewards = np.zeros((player_num))
    rounds = 0
    winners = 0

#    if os.path.exists('memory.file'):
#        filehandler = open('memory.file', 'rb') 
#        memory = cPickle.load(filehandler)
#        filehandler.close()
#        print('memory.file loaded')
#    else:

    for i in range(10000):
        re,ro,wi = play_game(eps,model_target,memory)
        rewards = rewards + re
        rounds += ro
        winners += wi
    print(rewards)
#    eps = 0.93
#    filehandler = open('memory.file', 'wb') 
#    cPickle.dump(memory,filehandler)
#    filehandler.close()

    k = 0
    while True:
        k += 1
        for middle in range(len(memory)):
            if memory[middle].rt == 0:
                break
#        print(middle,len(memory))
        sample_rt = collections.deque(itertools.islice(memory, 0, middle))
        sample_no_rt = collections.deque(itertools.islice(memory, middle, len(memory)))
        sample = random.sample(sample_rt,sample_num * 20 // 100) + random.sample(sample_no_rt,sample_num * 80 // 100)

    #    target_vectors = model_.predict(np.array([i.st for i in sample]))
    #    fut_actions = model_target.predict(np.array([i.stn for i in sample]))

        X = np.array([i.st for i in sample],dtype=np.float16)
        X = np.expand_dims(X,axis=4)
#        X = keras.backend.one_hot(X,num_classes=9)
#        R = keras.backend.one_hot(np.array([i.round for i in sample]),num_classes=201)
        R1 = keras.backend.one_hot(np.array([i.round + 1 for i in sample]),num_classes=201)
#        X = np.array([i.st for i in sample])

        target_vectors = model_.advantage(X).numpy()
        y_ = np.array([i.stn for i in sample],dtype=np.float16)
#        y_ = keras.backend.one_hot(y_,num_classes=9)
        fut_actions = model_target.advantage(np.expand_dims(y_,axis=4)).numpy()
#        fut_actions = np.max(fut_actions,axis=1)
#        fut_actions = model_target.predict(np.array([i.stn for i in sample]))

#        print(target_vectors)

#        X = np.zeros((sample_num,target_shape[0],target_shape[1],9))
        R = np.zeros((sample_num,1))
        Y = np.zeros((sample_num,3))
#        M = np.zeros((sample_num,3))

        alpha = 0.5
        gamma = 0.95
        for i in range(sample_num):
            j = sample[i]

            target_vector, fut_action = target_vectors[i].copy(), fut_actions[i].copy()
            target = j.rt
            if not j.done:
                target = (np.max(fut_action) * gamma + target)# * alpha - target_vector[j.at]# - target_vector[j.at]

            target_vector[j.at] = target
            mask = target_vector.copy() * 0
            mask[j.at] = 1

#            X[i] = i.st
            R[i] = j.round
            Y[i] = target_vector
#            M[i] = mask

#        model_training.fit([keras.backend.one_hot(R,num_classes=201),X,Y,M],Y,epochs=1,verbose=0)#,validation_split=0.1)
        R = keras.backend.one_hot(R,num_classes=201)
        R = np.reshape(R,(R.shape[0],R.shape[2]))
        model_.train_on_batch(X,Y)#,validation_split=0.1)

        if len(memory) >= 50000:
#            memory = random.sample(memory,90000)
            memory.pop()
        for i in range(1):
            play_game(eps,model_target,memory)

        if k % 100 == 0:
            print('')

            for i in range(len(model_.layers)):
                s = model_.layers[i].get_weights()
                if len(s) > 0:                   
                    h = [None] * len(s)
                    t = model_target.layers[i].get_weights()
                    for n in range(len(s)):
                        h_ = np.array(s[n]) * 0.5 + np.array(t[n]) * 0.5
                        h[n] = h_

                    model_target.layers[i].set_weights(h)
#            model_target.set_weights(model_.get_weigths())
#            transfer_weights_partially(model_,model_target,1)

            games = [None] * 20
            for i in range(len(games)):
                games[i] = Game()
                games[i].init()

            rewards = [] # np.zeros((player_num))
            rounds = 0
            winners = 0
            print('+', end='',flush=True)
            for i in range(len(games)):
                re,ro,wi = play_game(-1,model_target,None,games[i])
                rewards.append(re)
                rounds += ro
                winners += wi
            rewards = np.array(rewards,dtype=int)
            print(' test: ',k // 100,np.median(rewards),np.max(rewards),eps)
#            model_target.save('./model_target')
        else:
            print('*', end='',flush=True)

        eps *= 0.99999
        eps = np.maximum(0.1,eps)


    #for i in memory:
    #    print(i.st,i.at,i.lat,i.rt)


my_main()