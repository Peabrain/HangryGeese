import json
import numpy as np
import os
import glob
import random

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv3D, Conv2D, MaxPool2D, Add, Lambda, LayerNormalization
import keras
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

dir_dict = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}

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
    trainable_model.compile(optimizer=keras.optimizers.Adam(0.0001), loss=lambda yt,yp:yp)
    return trainable_model
    

def createModel():
    dropout = 0.4

    i0 = keras.Input(shape=(11, 11, 8))
#    x = LayerNormalization()(i0)
    x = i0
#    x = keras.layers.Reshape((11, 11, 8))(i0)
#    x = Conv3D(filters=64, kernel_size=(3,3,1), input_shape=(11, 11, 1, 8), activation='linear', padding='valid')(x)
#    x = keras.layers.Reshape((9, 9, 64))(x)
    x = Conv2D(filters=8, kernel_size=(3,3), input_shape=(11, 11, 8), data_format="channels_last", activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)
#    x = Conv2D(filters=64, kernel_size=(3,3), activation='linear', padding='same')(x)
#    x = Dropout(dropout)(x)    
#    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)
#    x = Conv2D(filters=8, kernel_size=(3,3), activation='linear', padding='same')(x)
#    x = Dropout(dropout)(x)
    x = Flatten()(x)
#    x0 = Dense(8, activation='relu')(x)
#    x = Dropout(dropout)(x0)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
#    x = Add()((x,x0))
#    x = Dense(32, activation='linear')(x)
#    x = Dropout(dropout)(x)
#    x = Dense(32, activation='linear')(x)
#    x = Dropout(dropout)(x)
    o0 = Dense(3, activation='linear')(x)
    model = Model(i0,o0)
    opt = keras.optimizers.Adam(0.0001)#.minimize(loss)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return model

    i0 = keras.Input(shape=(11, 11, 8, 1))
    x = Conv3D(filters=64, kernel_size=(5,5,8), input_shape=(11, 11, 8, 1), activation='linear', padding='valid')(i0)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filters=64, kernel_size=(5,5), activation='linear', padding='same')(x)
    x = Dropout(dropout)(x)
    x = Conv2D(filters=64, kernel_size=(5,5), activation='linear', padding='same')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='linear')(x)
    x = Dropout(dropout)(x)
    o0 = Dense(3, activation='linear')(x)
    model = Model(i0,o0)
    opt = keras.optimizers.Adam(0.001)#.minimize(loss)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return model

    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=(5,5,8), input_shape=(11, 11, 8, 1), activation='linear', padding='valid'))
    #model.add(Conv3D(filters=256, kernel_size=(3,3,4), activation='tanh', padding='valid'))
    model.add(keras.layers.Reshape((7, 7, 64)))
    model.add(Dropout(dropout))
    #model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='linear', padding='same'))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='linear', padding='same'))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='linear', padding='same'))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='linear', padding='same'))
    #model.add(Dropout(dropout))
    #model.add(MaxPool2D(pool_size=2))
    #model.add(Conv2D(filters=256, kernel_size=(3,3), input_shape=(11, 11, 1), activation='tanh', padding='valid'))
    #model.add(Dropout(dropout))
    #model.add(MaxPool2D(pool_size=2))
    #model.add(Conv2D(filters=256, kernel_size=(3,3), input_shape=(11, 11, 1), activation='tanh', padding='valid'))
    #model.add(Dropout(dropout))
    #model.add(MaxPool2D(pool_size=2))
    #model.add(MaxPool2D(pool_size=2))#, data_format='channels_first'))
    #model.add(Conv3D(filters=256, kernel_size=3, activation='tanh', padding='valid'))
    #model.add(Dropout(dropout))
    #model.add(MaxPool2D(pool_size=2))#, data_format='channels_first'))
    #model.add(Conv3D(filters=256, kernel_size=3, activation='tanh', padding='valid'))
    #model.add(MaxPool2D(pool_size=2))#, data_format='channels_first'))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='selu'))
    #model.add(Dropout(dropout))
    model.add(Dense(32, activation='selu'))
    #model.add(Dropout(dropout))
    model.add(Dense(32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='selu'))
    #model.add(Dropout(dropout))
    model.add(Dense(32, activation='selu'))
    #model.add(Dropout(dropout))
    model.add(Dense(32, activation='selu'))
    model.add(Dropout(dropout))
    model.add(Dense(3, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])
    return model

def getPlayGround(playersData,foodData,lastActions):
    playground = np.zeros((7,11), dtype=np.uint8)
    y_p = playersData[0][0] // 11
    x_p = playersData[0][0] % 11
    for g in range(len(playersData)):
        e = (g != 0) * 4
        go = playersData[g]
        for l in range(len(go)):
            y = go[l] // 11
            x = go[l] % 11
            playground[y,x] = 1 * (l < (len(go) - 1)) + (l == 0 or l == (len(go) - 1)) * 2 + e

    for f in foodData:
        y_ = f // 11
        x_ = f % 11
        playground[y_,x_] = 8

    playground = np.tile(playground, (3,3))
    playground = playground[(y_p + 7) - 5:(y_p + 7) - 5 + 11,(x_p + 11) - 5:(x_p + 11) - 5 + 11]

    if lastActions == dir_dict['SOUTH']:
        playground = np.rot90(playground,1,(0,1))
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['EAST']:
        playground = np.rot90(playground,1,(0,1))
    elif lastActions == dir_dict['WEST']:
        playground = np.rot90(playground,-1,(0,1))
    
    return playground

def getDataFromJSON(file_name):
    file = open(file_name)
    y = json.load(file)
    file.close()

    conf = y['configuration']
    info = y['info']
    rewards = y['rewards']
    specs = y['specification']
    statuses = y['statuses']
    steps = y['steps']

    teamNames = info['TeamNames']
#    handyRL = teamNames.index('HandyRL')

    print(len(steps))

    train_len = 0

    playground_teamnames = [None] * (200 * 4)
    playground_X_0 = np.zeros((200 * 4,11,11), dtype=np.uint8)
    playground_X_1 = np.zeros((200 * 4,1), dtype=np.uint8)
    playground_Y_0 = np.zeros((200 * 4,1), dtype=np.uint8)



    for step_act in range(1,len(steps) - 1):
        actions_last = [None] * 4
        actions_next = [None] * 4

        step = steps[step_act][0]
        observ = step['observation']
        geese = observ['geese']
        food = observ['food']

        for i in range(4):
            actions_last[i] = steps[step_act][i]['action']
        for i in range(4):
            actions_next[i] = steps[step_act + 1][i]['action']

        playground = np.zeros((4,7,11), dtype=np.uint8)

        for i in range(len(geese)):
            if len(geese[i]) > 0:
                playg = getPlayGround([geese[i], geese[(i + 1) % len(geese)], geese[(i + 2) % len(geese)], geese[(i + 3) % len(geese)]], food, actions_last[i])
                playground_X_0[train_len] = playg
                playground_X_1[train_len] = dir_dict[actions_last[i]]
                playground_Y_0[train_len] = (dir_dict[actions_next[i]] - dir_dict[actions_last[i]]) % 3
                playground_teamnames[train_len] = teamNames[i]
                train_len = train_len + 1

    playground_X_0 = playground_X_0[:train_len]
    playground_X_1 = playground_X_1[:train_len]
    playground_Y_0 = playground_Y_0[:train_len]
    playground_teamnames = playground_teamnames[:train_len]

    playground_X_0.shape = (playground_X_0.shape[0], playground_X_0.shape[1], playground_X_0.shape[2], 1)

    print('Dataset_len: ',train_len)
    
    return playground_X_0, playground_X_1, playground_Y_0, playground_teamnames

def run():
    dir = 'GooseLuck'
    filenames = glob.glob('GooseLuck/*.json') + glob.glob('HandyRL/*.json') + glob.glob('GooseBumps/*.json')

    playground_X_0 = [None] * len(filenames)
    playground_X_1 = [None] * len(filenames)
    playground_Y_0 = [None] * len(filenames)
    playground_teamnames = [None] * len(filenames)

    for i in range(len(filenames)):
        playground_X_0[i], playground_X_1[i], playground_Y_0[i], playground_teamnames[i] = getDataFromJSON(filenames[i])

    print(len(playground_X_0))
    playground_X_0 = np.vstack(playground_X_0)
    playground_X_1 = np.vstack(playground_X_1)
    playground_Y_0 = np.vstack(playground_Y_0)
    playground_teamnames = np.array([s for t in playground_teamnames for s in t])
    names, idxs = np.unique(playground_teamnames,return_index=True)
    names_dict = zip(names,list(range(len(names))))
    names_dict = dict(names_dict)
    playground_teamnames = np.array([names_dict[i] for i in playground_teamnames])

    ra = list(range(len(playground_X_0)))
    random.shuffle(ra)

    ran_l = int(len(playground_X_0) * 0.8)

    X_train = playground_X_0[ra[:ran_l]]
    X_test = playground_X_0[ra[ran_l:]]
    y_train = playground_Y_0[ra[:ran_l]]
    y_test = playground_Y_0[ra[ran_l:]]
    names_train = playground_teamnames[ra[:ran_l]]
    names_test = playground_teamnames[ra[ran_l:]]

    model = createModel()

    X_train = np.array(keras.backend.one_hot(X_train,num_classes=3))
    X_test = np.array(keras.backend.one_hot(X_test,num_classes=3))
    print(X_train.shape)
#    X_train = np.squeeze(X_train,axis=3)
#    X_test = np.squeeze(X_test,axis=3)
#    X_train = np.expand_dims(X_train,axis=4)
#    X_test = np.expand_dims(X_test,axis=4)
    model.fit(X_train, np.array(y_train), epochs=100, batch_size=100, verbose=1,validation_split=0.1)

    names_test_unique, idxs = np.unique(names_test,return_index=True)
    for i in names_test_unique:
        a = np.where(names_test == i)
        X = X_test[a]
        y = y_test[np.where(names_test == i)]
        y_ = model.predict(X)
        k = np.sum(np.argmax(y_,axis=1) == np.array(y).flatten())
        print('team:',names[i], len(np.array(y)), k / len(np.array(y)))
    #    print(names_test)
    #loss, accuracy = model.evaluate(X_test, np.array(y_test))
    #print('Test:')
    #print('Loss: %s\nAccuracy: %s' % (loss, accuracy))
    
def getRewards1(model,players,food,last_action):
    depth = 0

    start = 0
    end = 0
    states = [None] * 1000
    states[end] = (players.copy(), food.copy(), last_action, 0, -1, depth)
    end = end + 1

    while len(states) > 0:
        R = [0,0,0,0]
        (players_, food_, last_action_, reward_, pre_step, depth) = states[start]
        if depth == 3:
            break

        playground = getPlayGround(players_,food_,last_action_)

        X = keras.backend.one_hot(playground,num_classes=8)
        X = np.expand_dims(X, axis=0)
        X = np.expand_dims(X,axis=4)

        pred_ = model.predict(X)[0]
        for i in range(3):
            p = [[m for m in n] for n in players_]
            food_c = [m for m in food]
            next_action = (last_action_ + i - 1) % 4

            y = p[0][0] // 11
            x = p[0][0] % 11
            if next_action == 0:
                y = (y - 1) % 7
            elif next_action == 2:
                y = (y + 1) % 7
            if next_action == 1:
                x = (x - 1) % 11
            elif next_action == 3:
                x = (x + 1) % 11
            p[0][0] = y * 11 + x

            states[end] = (p, food_c, next_action, pred_[i], start, depth + 1)
            end = end + 1

        start = start + 1

#    print(states[:end])

    rew = np.zeros((end))
    prev = np.zeros((end))

    for i in range(end - 1,0,-1):
        (players_, food_, last_action_, reward_, pre_step, depth) = states[i]
        rew[i] = reward_
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

#    for i in range(end - 1,0,-1):
#        (players_, food_, last_action_, reward_, pre_step, depth) = states[i]
#        (players_p, food_p, last_action_p, reward_p, pre_step_p, depth_p) = states[i - 1]
#        states[pre_step] = (players_p, food_p, last_action_p, reward_p + reward_, pre_step_p, depth_p)

#    print(states[:end])

    re = np.zeros((3))
    for i in range(1,4):
        (players_, food_, last_action_, reward_, pre_step, depth) = states[i]
        re[i - 1] = reward_
#    print(re)
#    mo = (last_action - 1 + np.argmax(np.array(re))) % 4
#    print(re)
    return re #mo, re[np.argmax(np.array(re))]

def getRewards(model,players,food,last_action):
    depth = 0
    depth_max = 4

    start = 0
    end = 0
    states = [None] * 1000
    states[end] = (players.copy(), food.copy(), last_action, 0, -1, depth, end, 0)
    end = end + 1

    N = (3 * 3 ** (depth_max - 1) - 1) // (3 - 1)
    playgrounds = np.zeros((N, 11, 11),dtype=np.uint8)
    pred_ = np.full((N),-1)
    R = np.zeros((N, 3))
    r = np.zeros((N, 3))
    while len(states) > 0:
        (players_, food_, last_action_, reward_, pre_step, depth, pos, dir_) = states[start]
        if depth + 1 == depth_max:
            break

        playgrounds[pos] = getPlayGround(players_,food_,last_action_)

        for i in range(3):
            p = [[m for m in n] for n in players_]
            food_c = [m for m in food]
            next_action = (last_action_ + i - 1) % 4                   

            y = p[0][0] // 11
            x = p[0][0] % 11
            if next_action == 0:
                y = (y - 1) % 7
            elif next_action == 2:
                y = (y + 1) % 7
            if next_action == 1:
                x = (x - 1) % 11
            elif next_action == 3:
                x = (x + 1) % 11
            p[0][0] = y * 11 + x

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

#    for i in range(end - 1,0,-1):
#        (players_, food_, last_action_, reward_, pre_step, depth) = states[i]
#        (players_p, food_p, last_action_p, reward_p, pre_step_p, depth_p) = states[i - 1]
#        states[pre_step] = (players_p, food_p, last_action_p, reward_p + reward_, pre_step_p, depth_p)

#    print(states[:end])

    re = np.zeros((3))
    for i in range(1,4):
        (players_, food_, last_action_, reward_, pre_step, depth, pos, dir_) = states[i]
        re[i - 1] = rew[i]
#    print(re)
#    mo = (last_action - 1 + np.argmax(np.array(re))) % 4
#    print(re)
    return re #mo, re[np.argmax(np.array(re))]




def temp():
    model_ = createModel()
    model = add_rl_loss_to_model(model_)

    for kk in range(200):
#        learnData = [None] * 500 * 200
    #   print(food)
        player = [int(0)]
        last_action = dir_dict['NORTH']

        S = np.zeros((500 * 200,11,11))
        R = np.zeros((500 * 200,3))
        M = np.zeros((500 * 200))
        SW = np.ones((500 * 200))
        pos = 0
        pos_s = pos
        print('round: ', end = '', flush=True)
        for rou in range(10+kk):
            print(rou % 10, end = '', flush=True)
            pp = 0
            food = list(range(77))
            random.shuffle(food)
            food = food[:10]
            pred__ = np.zeros((3))
            #    for i in range(1000):
            for i in range(200):
                playground = getPlayGround([player],food,last_action)
                pred_ = getRewards(model_,[player],food,last_action)
#                quit()
        #    while pos < len(learnData):
#                playground = getPlayGround([player],food,last_action)

#                X = keras.backend.one_hot(playground,num_classes=8)
#                X = np.expand_dims(X, axis=0)
#                X = np.expand_dims(X,axis=4)

#                pred_ = model_.predict(X)
                if random.random() >= 0.2:
#                    pred_ = np.squeeze(pred_,axis=0)
                    pred = np.argmax(pred_)
                else:
                    pred = random.randint(0,2)
                    pred_ = np.zeros((3))
                next_action = (last_action - 1 + pred) % 4

                #print(playground)
            #    print(pred)
            #    print(next_action)

                y = player[0] // 11
                x = player[0] % 11
                if next_action == 0:
                    player[0] = ((y - 1) % 7) * 11 + x
                elif next_action == 2:
                    player[0] = ((y + 1) % 7) * 11 + x
                if next_action == 1:
                    player[0] = y * 11 + (x - 1) % 11
                elif next_action == 3:
                    player[0] = y * 11 + (x + 1) % 11

            #    print(player[0])
                pred__ = pred__ - pp
                pp = pp + 0.1
                if player[0] in food:
                    food.remove(player[0])
                    pred__[pred] = 1
                    last_reward_pos = pos + 1


                S[pos] = playground
                R[pos] = pred__
                M[pos] = pred
                pos = pos + 1

                    #            print('reward at: ',food)
                last_action = next_action
                if len(food) == 0:
                    break    #print(r)
        #            print('end round:',rou)
        #        if len(food) == 0:
        #            break
        print('')
        if pos == pos_s:
            print('no data')
            continue
#        print(pos)

        pos = last_reward_pos

        pp = 0
        g = 1
        a = 0.5
        for i in range(pos - 1,pos_s,-1):
            j = i - 1
            R[j] = R[j] + (g * np.max(R[j + 1]) - R[j]) * a

        R[pos_s:pos] = np.gradient(R[pos_s:pos],axis=0)
        SW[:pos_s] = SW[:pos_s] * 0.9
        SW[pos_s:pos] = 1.0
        pos_s = pos

        X_train = S[:pos] #np.array([x for (x,_,_) in learnData[:pos]])
        y_train = R[:pos]# - R[0:pos - 1] #np.array([y for (_,y,_) in learnData[:pos]])
        m_train = M[:pos] #np.array([m for (_,_,m) in learnData[:pos]])

        X_train = keras.backend.one_hot(X_train,num_classes=8)    
        mask = keras.backend.one_hot(m_train,num_classes=3)
        X_train = np.expand_dims(X_train,axis=4)
    #    y = np.expand_dims(y,axis=1)

        model.fit([X_train,y_train,mask],y_train,batch_size=10,epochs=10,verbose=1,validation_split=0.1)

        food = list(range(77))
        random.shuffle(food)
        food = food[:10]
        print(food)

        for i in range(50):
            pred_ = getRewards(model_,[player],food,last_action)
#            playground = getPlayGround([player],food,last_action)

 #           X = keras.backend.one_hot(playground,num_classes=8)
#            X = np.expand_dims(X, axis=0)
#            X = np.expand_dims(X,axis=4)

#            pred_ = model_.predict(X)[0]
            pred = np.argmax(pred_)
            next_action = (last_action - 1 + pred) % 4

            #print(playground)
        #    print(pred)
        #    print(next_action)

            y = player[0] // 11
            x = player[0] % 11
            if next_action == 0:
                player[0] = ((y - 1) % 7) * 11 + x
            elif next_action == 2:
                player[0] = ((y + 1) % 7) * 11 + x
            if next_action == 1:
                player[0] = y * 11 + (x - 1) % 11
            elif next_action == 3:
                player[0] = y * 11 + (x + 1) % 11

        #    print(player[0])
            if player[0] in food:
                print((10 - len(food)) % 10, end = '', flush=True)
                food.remove(player[0])

            last_action = next_action  
            if len(food) == 0:
                print('Done:',i, end = '', flush=True)
                break    #print(r)
        print('')
""""""
temp()