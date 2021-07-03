from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

import numpy as np
import random

last_dir = np.array([5,5,5,5])

def fill(matrix,x,y,depth):
    if x >= matrix.shape[1] or x < 0 or y >= matrix.shape[0] or y < 0 or matrix[y,x] == 2:
        return 0
#    print(matrix.shape[0],matrix.shape[1],matrix.shape[0] * matrix.shape[1])
    list_ = np.zeros((matrix.shape[0] * matrix.shape[1],3),dtype=np.int)
    start = 0
    end = 0
    list_[end] = [x,y,depth]
    matrix[y,x] = 255
    end = end + 1
    while end != start:
        x_,y_,d_ = list_[start]
        start = start + 1
        d_ = d_ - 1
        if d_ == 0:
            continue
        else:
            if y_ - 1 >= 0 and matrix[y_ - 1,x_] < 2:
                matrix[y_ - 1,x_] = 255
                list_[end] = [x_,y_ - 1,d_]
                end = end + 1                
            if y_ + 1 < matrix.shape[0] and matrix[y_ + 1,x_] < 2:
                matrix[y_ + 1,x_] = 255        
                list_[end] = [x_,y_ + 1,d_]
                end = end + 1      
            if x_ - 1 >= 0 and matrix[y_,x_ - 1] < 2:
                matrix[y_,x_ - 1] = 255        
                list_[end] = [x_ - 1,y_,d_]
                end = end + 1                
            if x_ + 1 < matrix.shape[1] and matrix[y_,x_ + 1] < 2:
                matrix[y_,x_ + 1] = 255        
                list_[end] = [x_ + 1,y_,d_]
                end = end + 1          
    end = np.sum(matrix == 255)      
    return end


    if depth == 0 or x < 0 or x >= 33 or y < 0 or y >= 33 or matrix[y,x] == 2:
        return 0
    su = 0
    matrix[y,x] = 1
    su = 1 + fill(matrix,x - 1,y,depth - 1) + fill(matrix,x,y - 1,depth - 1) + fill(matrix,x + 1,y,depth - 1) + fill(matrix,x,y + 1,depth - 1)
    return su

def path(matrix_):
    matrix = np.tile(matrix_, (3,3))
    matrix[0,:] = 2
    matrix[matrix.shape[0] - 1,:] = 2
    matrix[:,0] = 2
    matrix[:,matrix.shape[1] - 1] = 2
    data = np.zeros((matrix.shape[0],matrix.shape[1],4))
    pos_food = np.where(matrix == 1)
    pos_food = zip(pos_food[0],pos_food[1])

    
    
    for y,x in pos_food:
        a_n = fill(matrix.copy(),x,y-1,16)
        a_e = fill(matrix.copy(),x-1,y,16)
        a_s = fill(matrix.copy(),x,y+1,16)
        a_w = fill(matrix.copy(),x+2,y,16)
#        print(a_n,a_e,a_s,a_w)
        data[y,x,0] = a_s
        data[y,x,2] = a_n
        data[y,x,1] = a_w
        data[y,x,3] = a_e

    pos_danger = np.where(matrix == 2)
    pos_danger = zip(pos_danger[0],pos_danger[1])
    for i,j in pos_danger:
        data[i,j] = -1000
        
    for k in range(20):
        r = [(i,j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i,j] != 1 and matrix[i,j] != 2]
        random.shuffle(r)
        for i,j in r:
            if i - 1 >= 0:
                data[i,j,0] = np.sum(data[i - 1,j]) / 8
            if i + 1 < matrix.shape[0]:
                data[i,j,2] = np.sum(data[i + 1,j]) / 8
            if j - 1 >= 0:
               data[i,j,1] = np.sum(data[i,j - 1]) / 8
            if j + 1 < matrix.shape[1]:
                data[i,j,3] = np.sum(data[i,j + 1]) / 8
            
    p = data[matrix_.shape[0]:matrix_.shape[0] * 2,matrix_.shape[1]:matrix_.shape[1] * 2]
    return p

def markov(playground,y_,x_,h,w):
    matrix = np.zeros((playground.shape[0],playground.shape[1]),dtype=np.float32)
    deep = 0
    deep_max = 2
    list_ = np.zeros((11 * 7 * 10,4),dtype=np.int32)
    start = 0
    end = 0
    list_[end] = [y_,x_,deep,1]
    end = end + 1
    
    while end != start:
        y,x,d,l_ = list_[start]
        start = start + 1

        l = 0
        if playground[(y - 1) % matrix.shape[0],x] < 2:
            l = l + 1
        if playground[(y + 1) % matrix.shape[0],x] < 2:
            l = l + 1
        if playground[y,(x - 1) % matrix.shape[1]] < 2:
            l = l + 1
        if playground[y,(x + 1) % matrix.shape[1]] < 2:
            l = l + 1
        if l == 0:
            continue
        l = 1 / l       
        l = l * l_
#        l =  # (1.0 - 1.0 / float(d + 1)**1)
#        print(y,x,l)
        if matrix[(y - 1) % matrix.shape[0],x] == 0 and playground[(y - 1) % matrix.shape[0],x] < 2 and d + 1 < deep_max:
            matrix[(y - 1) % matrix.shape[0],x] = l
            list_[end] = [(y - 1) % matrix.shape[0],x,d + 1,l]
            end = end + 1
        if matrix[(y + 1) % matrix.shape[0],x] == 0 and playground[(y + 1) % matrix.shape[0],x] < 2 and d + 1 < deep_max:
            matrix[(y + 1) % matrix.shape[0],x] = l
            list_[end] = [(y + 1) % matrix.shape[0],x,d + 1,l]
            end = end + 1
        if matrix[y,(x - 1) % matrix.shape[1]] == 0 and playground[y,(x - 1) % matrix.shape[1]] < 2 and d + 1 < deep_max:
            matrix[y,(x - 1) % matrix.shape[1]] = l
            list_[end] = [y,(x - 1) % matrix.shape[1],d + 1,l]
            end = end + 1
        if matrix[y,(x + 1) % matrix.shape[1]] == 0 and playground[y,(x + 1) % matrix.shape[1]] < 2 and d + 1 < deep_max:
            matrix[y,(x + 1) % matrix.shape[1]] = l
            list_[end] = [y,(x + 1) % matrix.shape[1],d + 1,l]
            end = end + 1
#        print(matrix)     
    return matrix

def path_new(matrix_,enemy_,player_row,player_column,player_index,goose_len):
    enemy = np.tile(enemy_, (3,3))
    matrix = np.tile(matrix_, (3,3))
    matrix[0,:] = 2
    matrix[matrix.shape[0] - 1,:] = 2
    matrix[:,0] = 2
    matrix[:,matrix.shape[1] - 1] = 2
    data = np.zeros((matrix.shape[0],matrix.shape[1],4))
    pos_food = np.where(matrix == 1)
    pos_food = zip(pos_food[0],pos_food[1])

    p = 1
    for y,x in pos_food:
        a_n = fill(matrix.copy(),x,y - 1,16)
        a_e = fill(matrix.copy(),x - 1,y,16)
        a_s = fill(matrix.copy(),x,y + 1,16)
        a_w = fill(matrix.copy(),x + 1,y,16)
        l = (a_w + a_s + a_e + a_n)
#        data[y,x] = 2
        if l > 0:
            data[y,x,0] = data[y,x,0] * p + (a_s + a_w + a_e) / l
            data[y,x,2] = data[y,x,2] * p + (a_n + a_w + a_e) / l
            data[y,x,1] = data[y,x,1] * p + (a_w + a_n + a_s) / l
            data[y,x,3] = data[y,x,3] * p + (a_e + a_n + a_s) / l
        else:
            data[y,x] = 2

#    for j in range(matrix.shape[0]):
#        for i in range(matrix.shape[1]):
#            data[j,i] = data[j,i] * enemy[j,i]

    pos_danger = np.where((matrix >= 3) & (matrix != player_index + 3))
    pos_danger = zip(pos_danger[0],pos_danger[1])
    k = 2.0
    for i,j in pos_danger:
#        if i - 1 >= 0:
#            data[i - 1,j,2] = data[i - 1,j,2] - k
#        if i + 1 < matrix.shape[0]:
#            data[i + 1,j,0] = data[i + 1,j,0] - k
#        if j - 1 >= 0:
#            data[i,j - 1,3] = data[i,j - 1,3] - k
#        if j + 1 < matrix.shape[1]:
#            data[i,j + 1,1] = data[i,j + 1,1] - k
        if i - 1 >= 0 and matrix[i - 1,j] != player_index + 3:
            data[i - 1,j] = -k
        if i + 1 < matrix.shape[0] and matrix[i + 1,j] != player_index + 3:
            data[i + 1,j] = -k
        if j - 1 >= 0 and matrix[i,j - 1] != player_index + 3:
            data[i,j - 1] = -k
        if j + 1 < matrix.shape[1] and matrix[i,j + 1] != player_index + 3:
            data[i,j + 1] = -k
#        data[i,j] = -k

#    print(player_row + matrix_.shape[0],player_column + matrix_.shape[1],goose_len)
    a_w = fill(matrix.copy(), player_column + matrix_.shape[1] - 1, player_row + matrix_.shape[0], goose_len + 1)
    a_n = fill(matrix.copy(), player_column + matrix_.shape[1], player_row + matrix_.shape[0] - 1, goose_len + 1)
    a_e = fill(matrix.copy(), player_column + matrix_.shape[1] + 1, player_row + matrix_.shape[0], goose_len + 1)
    a_s = fill(matrix.copy(), player_column + matrix_.shape[1], player_row + matrix_.shape[0] + 1, goose_len + 1)

    l_ch = [a_n, a_w, a_s, a_e]
#    print(l_ch)

    g = 0.1
    a = 0.01
    for k in range(20):
        r = [(i,j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if (matrix[i,j] < 2 or matrix[i,j] == player_index + 3)]
        random.shuffle(r)
        for i,j in r:
            if i - 1 >= 0 and matrix[i - 1,j] < 2:
                data[i,j,0] = data[i,j,0] + (g * np.max(data[i - 1,j]) - data[i,j,0]) * a
            if i + 1 < matrix.shape[0] and matrix[i + 1,j] < 2:
                data[i,j,2] = data[i,j,2] + (g * np.max(data[i + 1,j]) - data[i,j,2]) * a
            if j - 1 >= 0 and matrix[i,j - 1] < 2:
               data[i,j,1] = data[i,j,1] + (g * np.max(data[i,j - 1]) - data[i,j,1]) * a
            if j + 1 < matrix.shape[1] and matrix[i,j + 1] < 2:
                data[i,j,3] = data[i,j,3] + (g * np.max(data[i,j + 1]) - data[i,j,3]) * a

    p = data[matrix_.shape[0]:matrix_.shape[0] * 2,matrix_.shape[1]:matrix_.shape[1] * 2]

    return p, l_ch


def agent(obs_dict, config_dict):

    
    
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)

    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    player_row = player_row % 7
    player_column = player_column % 11
      
    playground = np.zeros((7,11),dtype=np.uint8)
    for i in range(len(observation.geese)):
        goose = observation.geese[i]
        if len(goose) > 0:
            for j in range(0,len(goose)):
                r,c = row_col(goose[j], configuration.columns)
                r = r % 7
                c = c % 11
                playground[r,c] = 2
            r,c = row_col(goose[0], configuration.columns)
            r = r % 7
            c = c % 11
            playground[r,c] = 3 + i

    matrix_ = np.zeros((playground.shape[0],playground.shape[1]),dtype=np.float32)
    for i in range(len(observation.geese)):
        if i != player_index:
            enemy_goose = observation.geese[i]
            if len(enemy_goose) > 0:
                enemy_head = enemy_goose[0]
                enemy_row, enemy_column = row_col(enemy_head, configuration.columns)
                enemy_row = enemy_row % 7
                enemy_column = enemy_column % 11                   
                m = markov(playground,enemy_row,enemy_column,playground.shape[0],playground.shape[1])
                matrix_ = np.maximum(matrix_,m)

    matrix_ = 1 - matrix_
#    matrix_ = np.abs(matrix_ - 1)

    for food in observation.food:
        r, c = row_col(food, configuration.columns)
        r = r % 7
        c = c % 11
        playground[r,c] = 1

    r, l_ch = path_new(playground,matrix_,player_row,player_column,player_index,len(player_goose))

    #print(r)
#    print(o)

            

    ori = np.array([Action.NORTH.name,Action.WEST.name,Action.SOUTH.name,Action.EAST.name,'*'])

#    if len(player_goose) == 1:
#        if (last_dir[player_index] + 2) % 4 == np.argmax(r[player_row,player_column]):
#            r[player_row,player_column,np.argmax(r[player_row,player_column])] = -1000
    dir_v = np.array([  [(player_row - 1) % playground.shape[0],player_column],
                        [player_row,(player_column - 1) % playground.shape[1]],
                        [(player_row + 1) % playground.shape[0],player_column],
                        [player_row,(player_column + 1) % playground.shape[1]]
                    ])

#    print(playground)
    l_ch_ = np.array(l_ch) >= len(player_goose)

    if np.sum(l_ch_ == True) == 0 or np.sum(r[player_row,player_column] == 0) == 4:
        l_ch[(last_dir[player_index] + 2) % 4] = 0
        dir_ = np.argmax(l_ch)
    else:
        min_ = np.min(r[player_row,player_column]) - 1
        dir_ = -1
        steps = 0
        for i in range(4):
            if min_ < r[player_row,player_column,i] and playground[dir_v[i,0],dir_v[i,1]] < 2 and l_ch_[i] == True and (last_dir[player_index] + 2) % 4 != i:
                min_ = r[player_row,player_column,i]
                dir_ = i
        if dir_ == -1:
            dir_ = 0
    last_dir[player_index] = dir_
    dir_ = ori[dir_]
    print(player_index,dir_,len(player_goose),l_ch,l_ch_,r[player_row,player_column])
    return dir_

"""
playground = np.array([ [2,0,0,0,0,0,0,0,0,3,2],
                        [0,0,0,0,1,0,0,0,2,0,0],
                        [0,0,0,3,0,0,0,0,4,0,0],
                        [0,2,2,2,0,0,0,2,0,0,1],
                        [2,2,2,2,0,2,2,2,2,2,2],
                        [0,2,2,3,0,0,3,0,0,0,0],
                        [2,0,0,0,0,0,0,0,2,0,0],
                        ],dtype=np.uint8)


matrix_ = np.zeros((playground.shape[0],playground.shape[1]),dtype=np.float32)
#markov(matrix_,playground,0,0,playground.shape[0],playground.shape[1])
for j,i in [(5,3),(5,6),(0,9)]:
    m = markov(playground,j,i,playground.shape[0],playground.shape[1])
    matrix_ = np.maximum(matrix_,m)
matrix_ = 1 - matrix_

print(playground)
print(matrix_)
playground[2,3] = 0
r = path_new(playground,matrix_)
print(r[2,3])
"""