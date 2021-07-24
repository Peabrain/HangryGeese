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
    
#from keras.models import Sequential, load_model, Model
#from keras.layers import Dense, Dropout, Flatten, Conv3D, Conv2D, MaxPool2D, ReLU, Add, Lambda, LayerNormalization
from tensorflow import keras
from keras import optimizers
#import tensorflow as tf
import keras.backend as K

#from sklearn.model_selection import train_test_split

dir_dict = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}
dir_ = ['NORTH', 'WEST', 'SOUTH', 'EAST']

playground_shape = (7,11)
target_shape = (11,11)

memorySize = 3

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
    y_true = keras.layers.Input(name='y_true',shape=(num_actions,))
    mask = keras.layers.Input(name='mask',shape=(num_actions,))
    loss_out = keras.layers.Lambda(masked_mse,output_shape=(1,0),name='loss')([y_true,y_pred,mask])
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

def createModel():
    dropout = 0.3

    i0 = keras.layers.Input(shape=(memorySize,target_shape[0], target_shape[1],1))
    x = i0
#    x = tf.transpose(x, [0, 2, 3, 1])
    x = keras.layers.ConvLSTM2D(filters=64, kernel_size=(3,3), padding='same',return_sequences=True)(x)
    x = keras.layers.MaxPool3D((1,2,2))(x)
#    x = keras.layers.ReLU()(x)
    x = keras.layers.ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same',return_sequences=True)(x)
    x = keras.layers.MaxPool3D((1,2,2))(x)
    print(x.shape)
#    x = keras.layers.ConvLSTM2D(filters=128, kernel_size=(3,3), padding='same')(x)#,return_sequences=True)(x)
#    x = keras.layers.MaxPool3D()(x)
#    x = keras.layers.ConvLSTM2D(filters=256, kernel_size=(3,3), padding='same')(x)
#    x = keras.layers.ReLU()(x)
#    x = tf.keras.layers.Reshape(target_shape=(target_shape[0] * target_shape[1],32))(x)
#    x = tf.keras.layers.Conv1D(filters=4,kernel_size=3,padding='valid')(x)
#    x = tf.keras.layers.Dropout(dropout)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.ReLU()(x)    
    x = keras.layers.Dense(128)(x)
    x = keras.layers.ReLU()(x)
#    x = res(x)
#    x = res(x)
#    x = tf.keras.layers.Dropout(dropout)(x)
#    x = res(x)
#    x = tf.keras.layers.Dropout(dropout)(x)
    o0 = keras.layers.Dense(3, activation='linear')(x)
    model = keras.models.Model(i0,o0)
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
        e = (g != act_player) * 4
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
        self.player = self.player[:4]
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
        self.MemorySteps = [np.zeros((memorySize,target_shape[0],target_shape[1]),dtype=np.uint8)] * 4
        self.MemoryStepsNext = [np.zeros((memorySize,target_shape[0],target_shape[1]),dtype=np.uint8)] * 4
    
    def do_action(self,actions_rel):
        self.round += 1
        reward = [None] * len(self.player)
        new_pos = [None] * len(self.player)
        playground = [None] * len(self.player)
        dones = [False] * len(self.player)
        actions = [None] * len(self.player)

        for i in range(len(actions_rel)):
            if len(self.player[i]) == 0:
                dones[i] = True


        for i in range(len(actions_rel)):
            if len(self.player[i]) > 0:
                action_rel = actions_rel[i]
                actions[i] = (self.last_action[i] + action_rel - 1) % 4

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
                    reward[i] = -len(self.player[i])#(self.round * 2 + len(self.player[i]))
                    killed = True
                    dones[i] = True
                    self.player[i] = []
                elif self.player[i][0] in self.food:
                    self.food.remove(self.player[i][0])
                    reward[i] = 1
                    new_food += 1
                else:
                    reward[i] = -0.01

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
            if not reward[i] is None and (reward[i] == -1 or reward[i] >= 1):
                pl_count += 1
                pl_count_idx = i
        if pl_count == 1:
            reward[pl_count_idx] = self.round * 2 + len(self.player[pl_count_idx])
            dones[pl_count_idx] = True

        return reward, dones

    def get_playground(self,player_idx):
        if len(self.player[player_idx]) > 0:
            return getPlayGround([self.player[(i - player_idx) % len(self.player)] for i in range(len(self.player))],self.food,self.last_action[player_idx])
        else:
            return np.zeros(shape=target_shape, dtype=np.uint8)


class Step:
    def __init__(self,st,stn,at,rt,done):
        self.st = st
        self.stn = stn
        self.at = at
        self.rt = rt
        self.done = done

def play_game(eps,model,memory):
    game = Game()
    game.init()
    count = 200
    rewards = [0] * len(game.player)
    done_count = 0
    while count > 0 and done_count != len(game.player):
        action = [None] * len(game.player)

        playground_t = [getPlayGround(game.player,game.food,game.last_action[player_idx],player_idx) for player_idx in range(len(game.player))]
        for i in range(len(game.player)):
            if len(game.player[i]) > 0:
                if game.MemorySteps[i] is None:
                    for j in range(memorySize):
                        game.MemorySteps[i,j] = playground_t[i]
                else:
                    for j in range(memorySize - 1):
                        game.MemorySteps[i][j] = game.MemorySteps[i][j + 1]
                    game.MemorySteps[i][memorySize - 1] = playground_t[i]
                if random.random() < eps:
                    action[i] = np.random.randint(3)
                else:
#                    print(game.MemorySteps[i].shape)
                    X = np.expand_dims(game.MemorySteps[i],axis=0)
#                    target_vector = model.predict(keras.backend.one_hot(X,num_classes=9))[0]
                    target_vector = model.predict(X / 8)[0]
                    action[i] = np.argmax(target_vector)
#                print(game.MemorySteps[i])

        reward, dones = game.do_action(action)
        done_count = 0
        for i in range(len(game.player)):
            if not (reward[i] is None):
#                game_state  = ([game.player[(i + j) % len(game.player)] for j in range(len(game.player))],game.food,game.last_action[i])
                if reward[i] >= 1:
                    rewards[i] += reward[i]
                if dones[i] == True:
                    done_count += 1

                playground_tn = getPlayGround(game.player,game.food,game.last_action[i],i)
                if game.MemoryStepsNext[i] is None:
                    for j in range(memorySize):
                        game.MemoryStepsNext[i][j] = playground_tn
                else:
                    for j in range(memorySize - 1):
                        game.MemoryStepsNext[i][j] = game.MemoryStepsNext[i][j + 1]
                    game.MemoryStepsNext[i][memorySize - 1] = playground_tn
                step = Step(st = game.MemorySteps[i].copy(),stn = game.MemoryStepsNext[i].copy(), at = action[i], rt = reward[i], done = dones[i])
                memory.append(step)
        count -= 1

    return np.array(rewards,dtype=np.int32)

def my_main():
    eps = 1.0
    model_ = createModel() # keras.models.load_model('./model_target') # createModel()
    model_target = createModel()
    model_training = add_rl_loss_to_model(model_)
    transfer_weights_partially(model_,model_target,1)
    model_target.summary()
    memory = Deque()
    rewards = np.zeros((4))

#    if os.path.exists('memory.file'):
#        filehandler = open('memory.file', 'rb') 
#        memory = cPickle.load(filehandler)
#        filehandler.close()
#        print('memory.file loaded')
#    else:
    for i in range(10000):
        rewards += play_game(eps,model_target,memory)
    print(rewards)
#    filehandler = open('memory.file', 'wb') 
#    cPickle.dump(memory,filehandler)
#    filehandler.close()

    sample_num = 100

    for k in range(100000):
        sample = random.sample(memory,sample_num)

    #    target_vectors = model_.predict(np.array([i.st for i in sample]))
    #    fut_actions = model_target.predict(np.array([i.stn for i in sample]))

#        X = keras.backend.one_hot(np.array([i.st for i in sample]),num_classes=9)
        X = np.array([i.st for i in sample])

        target_vectors = model_.predict(X / 8)
#        fut_actions = model_target.predict(keras.backend.one_hot([i.stn for i in sample],num_classes=9))
        fut_actions = model_target.predict(np.array([i.stn for i in sample]) / 8)

#        X = np.zeros((sample_num,target_shape[0],target_shape[1],9))
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

#            X[i] = i.st
            Y[i] = target_vector
            M[i] = mask

        model_training.fit([X / 8,Y,M],Y,epochs=1,verbose=0)#,validation_split=0.1)

        memory.pop()
        for i in range(1):
            play_game(eps,model_target,memory)

        if (k + 1) % 50 == 0:
            print('')
            transfer_weights_partially(model_,model_target,0.25)
            rewards = np.zeros((4),dtype=np.int32)
            for i in range(1):
                print('+', end='',flush=True)
                rewards += play_game(0,model_target,memory)
            print(' test: ',k // 50,rewards,np.max(rewards),np.sum(rewards),eps)
            model_target.save('./model_target')
        else:
            print('*', end='',flush=True)

        eps *= 0.99999
        eps = np.maximum(0.1,eps)


    #for i in memory:
    #    print(i.st,i.at,i.lat,i.rt)


my_main()