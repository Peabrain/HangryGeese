import json
import numpy as np
import os
import glob
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv3D, Conv2D, MaxPool2D, Add
from keras.utils import np_utils, plot_model
import keras

from sklearn.model_selection import train_test_split

dir_dict = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}

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

        for i in range(4):
            actions_last[i] = steps[step_act][i]['action']
        for i in range(4):
            actions_next[i] = steps[step_act + 1][i]['action']

        playground = np.zeros((4,7,11), dtype=np.uint8)
        for g in range(4):
            go = geese[g]
            for l in range(len(go)):
                y = go[l] // 11
                x = go[l] % 11
                playground[g,y,x] = 1 * (l < (len(go) - 1)) + (l == 0 or l == (len(go) - 1)) * 2

        food = observ['food']
        playground__ = np.sum(playground,axis=0)
        for g in range(4):
            go = geese[g]
            if len(go) == 0:# or g != handyRL:
                continue
            y = (go[0] // 11)
            x = (go[0] % 11)
            r = list(range(4))
            r.remove(g)
            p = np.where(playground[r] != 0)
            playg = playground__.copy()
            playg[p[1],p[2]] = playg[p[1],p[2]] + 4

            for f in food:
                y_ = f // 11
                x_ = f % 11
                playg[y_,x_] = 8

            playg = np.tile(playg, (3,3))
            playg = playg[(y + 7) - 5:(y + 7) - 5 + 11,(x + 11) - 5:(x + 11) - 5 + 11]

            if actions_last[g] == 'SOUTH':
                playg = np.rot90(playg,1,(0,1))
                playg = np.rot90(playg,1,(0,1))
            elif actions_last[g] == 'EAST':
                playg = np.rot90(playg,1,(0,1))
            elif actions_last[g] == 'WEST':
                playg = np.rot90(playg,-1,(0,1))

        #    playg = playg[2:9]

            playground_X_0[train_len] = playg
            playground_X_1[train_len] = dir_dict[actions_last[g]]
            playground_Y_0[train_len] = (dir_dict[actions_next[g]] - dir_dict[actions_last[g]]) % 3
            playground_teamnames[train_len] = teamNames[g]
            train_len = train_len + 1

    playground_X_0 = playground_X_0[:train_len]
    playground_X_1 = playground_X_1[:train_len]
    playground_Y_0 = playground_Y_0[:train_len]
    playground_teamnames = playground_teamnames[:train_len]

    playground_X_0.shape = (playground_X_0.shape[0], playground_X_0.shape[1], playground_X_0.shape[2], 1)

    print('Dataset_len: ',train_len)
    
    return playground_X_0, playground_X_1, playground_Y_0, playground_teamnames

dir = 'GooseLuck'
filenames = glob.glob('GooseLuck/*.json') + glob.glob('HandyRL/*.json') + glob.glob('GooseBumps/*.json')

playground_X_0 = [None] * len(filenames)
playground_X_1 = [None] * len(filenames)
playground_Y_0 = [None] * len(filenames)
playground_teamnames = [None] * len(filenames)

for i in range(len(filenames)):
    playground_X_0[i], playground_X_1[i], playground_Y_0[i], playground_teamnames[i] = getDataFromJSON(filenames[i])

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

dropout = 0.4
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

X_train = keras.backend.one_hot(X_train,num_classes=8)
X_test = keras.backend.one_hot(X_test,num_classes=8)
print(X_train.shape)
X_train = np.squeeze(X_train,axis=3)
X_test = np.squeeze(X_test,axis=3)
X_train = np.expand_dims(X_train,axis=4)
X_test = np.expand_dims(X_test,axis=4)
model.fit(X_train, np.array(y_train), epochs=100, batch_size=128, verbose=1,validation_split=0.1)

names_test_unique, idxs = np.unique(names_test,return_index=True)
for i in names_test_unique:
    X = X_test[np.where(names_test == i)]
    y = y_test[np.where(names_test == i)]
    y_ = model.predict(X)
    k = np.sum(np.argmax(y_,axis=1) == np.array(y).flatten())
    print('team:',names[i], len(np.array(y)), k / len(np.array(y)))
#    print(names_test)
#loss, accuracy = model.evaluate(X_test, np.array(y_test))
#print('Test:')
#print('Loss: %s\nAccuracy: %s' % (loss, accuracy))
 