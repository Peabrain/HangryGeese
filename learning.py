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

    i0 = keras.Input(shape=(playground_shape[0], playground_shape[1], 3))
#    x = LayerNormalization()(i0)
    x = i0

#    x = Conv2D(filters=64, kernel_size=(3,3,3), activation='linear', padding='valid')(x)
#    x = keras.layers.Reshape((9, 9, 64))(x)
    x0 = Conv2D(filters=16, kernel_size=(3,3), padding='same')(x)
    x0 = ReLU()(x)
    x1 = Conv2D(filters=16, kernel_size=(5,5), padding='same')(x)
    x1 = ReLU()(x)
    x2 = Conv2D(filters=16, kernel_size=(7,7), padding='same')(x)
    x2 = ReLU()(x)
    x = tf.keras.layers.Concatenate(axis=3)([x0, x1, x2])
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(x)
    x = ReLU()(x)
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
        playground[y_,x_] = 3

    for g in range(len(playersData)):
        e = (g != 0) * 4
        go = playersData[g]
        for l in range(1,len(go)):
            y = go[l] // playground.shape[1]
            x = go[l] % playground.shape[1]
            playground[y,x] = 1 # * (l < (len(go) - 1)) + (l == 0 or l == (len(go) - 1)) * 2 + e
        y = go[0] // playground.shape[1]
        x = go[0] % playground.shape[1]
        playground[y,x] = 2 # * (l < (len(go) - 1)) + (l == 0 or l == (len(go) - 1)) * 2 + e

    playground = np.tile(playground, (3,3))
    playground = playground[(y_p + playground_shape[0]) - 2:(y_p + playground_shape[0]) - 2 + playground_shape[0],(x_p + playground_shape[1]) - 2:(x_p + playground_shape[1]) - 2 + playground_shape[1]]

    if lastActions == dir_dict['SOUTH']:
        playground = np.rot90(playground,1,(0,1))
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['EAST']:
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['WEST']:
        playground = np.rot90(playground,-1,(0,1))

    return playground


class rewarding:
    def getPlayGround(self,playersData,foodData,lastActions,playground):
    #    playground = np.zeros((7,11), dtype=np.uint8)
        y_p = playersData[0][0] // playground.shape[1]
        x_p = playersData[0][0] % playground.shape[1]
        for g in range(len(playersData)):
            e = (g != 0) * 4
            go = playersData[g]
            for l in range(len(go)):
                y = go[l] // playground.shape[1]
                x = go[l] % playground.shape[1]
                playground[y,x] = 1# * (l < (len(go) - 1)) + (l == 0 or l == (len(go) - 1)) * 2 + e

        for f in foodData:
            y_ = f // playground.shape[1]
            x_ = f % playground.shape[1]
            playground[y_,x_] = 2

        playground = np.tile(playground, (3,3))
        playground = playground[(y_p + playground.shape[0]) - 2:(y_p + playground.shape[0]) - 2 + playground.shape[1],(x_p + playground.shape[1]) - 2:(x_p + playground.shape[1]) - 2 + playground.shape[1]]

        if lastActions == dir_dict['SOUTH']:
            playground = np.rot90(playground,1,(0,1))
            playground = np.rot90(playground,1,(0,1))
        elif lastActions == dir_dict['EAST']:
            playground = np.rot90(playground,1,(0,1))
        elif lastActions == dir_dict['WEST']:
            playground = np.rot90(playground,-1,(0,1))
    
        return int(playersData[0][0] in foodData)

    #    return playground

    def getRewards(self,model,players,food,last_action,depth_max):
        playground = np.zeros((playground_shape[0],playground_shape[1]),dtype=np.uint8)
        self.getPlayGround(players,food,last_action,playground)
        X = keras.backend.one_hot(playground,num_classes=8)
        X = np.expand_dims(X,axis=0)
        X = np.expand_dims(X,axis=4)
        return model.predict(X)
    #    return np.zeros((3))
        depth = 0

        N = (3 * 3 ** (depth_max - 1) - 1) // (3 - 1)

        states = [None] * N
        playgrounds = np.zeros((N, playground_shape[0], playground_shape[1]),dtype=np.uint8)
        R = np.zeros((N, 3))

        start = 0
        end = 0
        states[end] = (players.copy(), food.copy(), last_action, 0, -1, depth, end, 0)
        end = end + 1


        while len(states) > 0:
            (players_, food_, last_action_, reward_, pre_step, depth, pos, dir_) = states[start]
            if depth + 1 == depth_max:
                break

            self.getPlayGround(players_,food_,last_action_,playgrounds[pos])

            for i in range(3):
                p = [[m for m in n] for n in players_]
                food_c = [m for m in food_]
                next_action = (last_action_ + i - 1) % 4                   

                y = p[0][0] // playground_shape[1]
                x = p[0][0] % playground_shape[1]
                if next_action == 0:
                    y = (y - 1) % playground_shape[0]
                elif next_action == 2:
                    y = (y + 1) % playground_shape[0]
                if next_action == 1:
                    x = (x - 1) % playground_shape[1]
                elif next_action == 3:
                    x = (x + 1) % playground_shape[1]

                if p[0][0] in food:
                    food_c.remove(p[0][0])
                    if len(food) < 2:
                        p[0] = [p[0][0]] + p[0][0:len(p[0])]
                        food_c.append(random.randint(0,76))

                p[0][0] = y * playground_shape[1] + x

                states[end] = (p, food_c, next_action, R[pos, i], start, depth + 1, end, i)
                end = end + 1

            start = start + 1

        X = keras.backend.one_hot(playgrounds,num_classes=8)
        X = np.expand_dims(X,axis=4)

        pred__ = model.predict(X)

        rew = np.zeros((end))
        prev = np.zeros((end))

        for i in range(end - 1,0,-1):
            (players_, food_, last_action_, reward_, pre_step, depth, pos, dir_) = states[i]
            rew[i] = pred__[pos,dir_]
            prev[i] = pre_step

        g = 1
        a = 0.5
        max_prev = np.max(prev)
        while max_prev > -1:
            w0 = np.where(prev == max_prev)
            w1 = np.where(prev == max_prev - 1)
    #        print(w0)
            ma = np.max(rew[w0])
            rew[w1] = rew[w1] + (g * ma - rew[w1]) * a
            max_prev = max_prev - 1

        re = np.zeros((3))
        for i in range(1,4):
            (players_, food_, last_action_, reward_, pre_step, depth, pos, dir_) = states[i]
            re[i - 1] = rew[i]

        for (players_, food_, last_action_, reward_, pre_step, depth, pos, dir_) in states:
            del players_
            del food_

        return re

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
            target_vector = model.predict(keras.backend.one_hot([playground_t],num_classes=3))[0]
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


def build(memory,model_,eps):
    batch_size = 10
    player = [[random.randint(0,playground_shape[0] * playground_shape[1] - 1)] for i in range(batch_size)]
    last_action = [dir_dict['NORTH']] * batch_size
    next_action = [dir_dict['NORTH']] * batch_size
    food = [list(np.random.randint(playground_shape[0] * playground_shape[1], size=(1))) for i in range(batch_size)]
    rew = rewarding()
    panelty = [-0.1] * batch_size
#    print('build: ', end = '', flush=True)
    final_score = [0] * batch_size
    for i in range(len(memory) // batch_size):
        reward = 0
#        if i % 100 == 0:
#        print(i % 10, end = '', flush=True)
        playground_t = np.zeros((batch_size,target_shape[0],target_shape[1]), dtype=np.uint8)
        r_t = [rew.getPlayGround([player[j]],food[j],last_action[j],playground_t[j]) for j in range(batch_size)]

        done = [0] * batch_size        
        if random.random() < eps:
            a = np.random.randint(3, size=(batch_size))
        else:
            t = model_.predict(keras.backend.one_hot(playground_t,num_classes=3))
            a = np.argmax(t,axis=1)

        for j in range(batch_size):
            if r_t[j] == 1:
                food[j].remove(player[j][0])
                r_t[j] = 1# + panelty[j]
                panelty[j] = 0
                final_score[j] += 1
                if len(food[j]) == 0:
#                    r_t[j] = 10
                    done[j] = 1
            else:
                r_t[j] = panelty[j]


#            panelty[j] -= 0.01
            next_action[j] = (last_action[j] + a[j] - 1) % 4

            y = player[j][0] // playground_shape[1]
            x = player[j][0] % playground_shape[1]
            if next_action[j] == 0:
                y = (y - 1) % playground_shape[0]
            elif next_action[j] == 2:
                y = (y + 1) % playground_shape[0]
            if next_action[j] == 1:
                x = (x - 1) % playground_shape[1]
            elif next_action[j] == 3:
                x = (x + 1) % playground_shape[1]
            player[j][0] = y * playground_shape[1] + x

            playground_tn = np.zeros((playground_shape[1],playground_shape[1]), dtype=np.uint8)
            rew.getPlayGround([player[j]],food[j],last_action[j],playground_tn)
            memory[i * batch_size + j] = (playground_t[j].copy(),r_t[j],a[j],playground_tn.copy(),done[j])
            last_action[j] = next_action[j]

            if len(food[j]) == 0:
                food[j] = list(np.random.randint(77, size=10))
#    print()

def temp():
    eps = 1.0
    eps_factor = 0.99999
    rew = rewarding()

#    if os.path.exists(r"my.model"):
#        model_ = keras.models.load_model(r"my.model")
#    else:
    model_ = createModel()
    model_target = createModel()
    model_training = add_rl_loss_to_model(model_)
    transfer_weights_partially(model_,model_target,1)



    if os.path.exists(r"data.pickle"):
        with open(r"data.pickle", "rb") as input_file:
            memory = cPickle.load(input_file)
            print('file loaded')
    else:
        memory = [None] * 50000
        build(memory,model_target,eps)

    u = 0
    mean_ = np.zeros((20))
    mean_idx = 0
    while True:
        u = u + 1
        sample_num = 128 #len(memory) // 10
        sample = list(range(sample_num))
        random.shuffle(sample)
        sample = sample[:sample_num]

        sample = random.sample(memory,sample_num)
    
        target_vectors = model_.predict(keras.backend.one_hot([st for (st,_,_,_,_) in sample],num_classes=3))
        fut_actions = model_target.predict(keras.backend.one_hot([stn for (_,_,_,stn,_) in sample],num_classes=3))

        X = np.zeros((sample_num,target_shape[1],target_shape[1]),dtype=np.uint8)
        Y = np.zeros((sample_num,3))
        M = np.zeros((sample_num,3))
        

        for i in range(sample_num):
            (x,r,a,xn,done) = sample[i]

            target_vector, fut_action = target_vectors[i].copy(), fut_actions[i].copy()
            target = r
            if not done:
                target += 0.95 * np.max(fut_action)

            target_vector[a] = target
            mask = target_vector.copy() * 0
            mask[a] = 1

            X[i] = x
            Y[i] = target_vector
            M[i] = mask

        model_training.fit([keras.backend.one_hot(X,num_classes=3),Y,M],Y,epochs=1,verbose=0)#,validation_split=0.1)

        if u % 10 == 0:
            transfer_weights_partially(model_,model_target,1)
            player = [int(random.randint(0,77-1))]
            last_action = dir_dict['NORTH']
            food = list(np.random.randint(0,77-1, size=10))
            reward = 0
            for i in range(100):
                playground_t = np.zeros((target_shape[0],target_shape[1]), dtype=np.uint8)
                r_t = rew.getPlayGround([player],food,last_action,playground_t)
                target_vectors = model_target.predict(keras.backend.one_hot([playground_t],num_classes=3))[0]

                if r_t == 1:
                    reward = reward + 1
                    food.remove(player[0])
                    if len(food) == 0:
                        food = list(np.random.randint(0,77-1, size=10))

                next_action = (last_action + np.argmax(target_vectors) - 1) % 4
                y = player[0] // playground_shape[1]
                x = player[0] % playground_shape[1]
                if next_action == 0:
                    y = (y - 1) % playground_shape[0]
                elif next_action == 2:
                    y = (y + 1) % playground_shape[0]
                if next_action == 1:
                    x = (x - 1) % playground_shape[1]
                elif next_action == 3:
                    x = (x + 1) % playground_shape[1]
                player[0] = y * playground_shape[1] + x

            mean_[mean_idx] = reward
            mean_idx = (mean_idx + 1) % len(mean_)
            print('try: ', u,' epsilon: ',eps, 'collected: ', reward, ' mean: ',np.median(mean_))

        eps *= eps_factor
        eps = np.maximum(0.1,eps)

        memory_ = [None] * 100
        build(memory_,model_target,eps)

        r = list(range(len(memory)))
        random.shuffle(r)
        r = r[:len(memory_)]
        for f in range(len(r)):
            memory[r[f]] = memory_[f]

#        build(memory,model_target,eps)

#        with open(r"data.pickle", "wb") as output_file:
#            cPickle.dump(memory, output_file)
#        model_.save(r"my.model")

    return
""""""
#temp()
eps = 1.0

model_ = createModel()
model_target = createModel()
model_training = add_rl_loss_to_model(model_)
transfer_weights_partially(model_,model_target,1)
memory = Deque()
rewards = 0
for i in range(10000):
    rewards += play_game(eps,model_target,memory)
print(rewards)

sample_num = 100

for k in range(100000):
    sample = random.sample(memory,sample_num)

    target_vectors = model_.predict(keras.backend.one_hot([i.st for i in sample],num_classes=3))
    fut_actions = model_target.predict(keras.backend.one_hot([i.stn for i in sample],num_classes=3))

    X = np.zeros((sample_num,target_shape[1],target_shape[1]),dtype=np.uint8)
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

    model_training.fit([keras.backend.one_hot(X,num_classes=3),Y,M],Y,epochs=1,verbose=0)#,validation_split=0.1)

    eps *= 0.99999
    eps = np.maximum(0.1,eps)

#    memory.pop()
    for i in range(1):
        play_game(eps,model_target,memory)

    if k % 50 == 0:
        transfer_weights_partially(model_,model_target,1)
        rewards = 0
        for i in range(10):
            rewards += play_game(0,model_target,memory)
        print(rewards,eps)


#for i in memory:
#    print(i.st,i.at,i.lat,i.rt)