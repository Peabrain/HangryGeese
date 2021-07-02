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
    matrix[y,x] = 2
    end = end + 1
    while end != start:
        x_,y_,d_ = list_[start]
        start = start + 1
        if d_ == 0:
            continue
        else:
            d_ = d_ - 1
            if y_ - 1 >= 0 and matrix[y_ - 1,x_] != 2:
                matrix[y_ - 1,x_] = 2          
                list_[end] = [x_,y_ - 1,d_]
                end = end + 1                
            if y_ + 1 < matrix.shape[0] and matrix[y_ + 1,x_] != 2:
                matrix[y_ + 1,x_] = 2          
                list_[end] = [x_,y_ + 1,d_]
                end = end + 1      
            if x_ - 1 >= 0 and matrix[y_,x_ - 1] != 2:
                matrix[y_,x_ - 1] = 2          
                list_[end] = [x_ - 1,y_,d_]
                end = end + 1                
            if x_ + 1 < matrix.shape[1] and matrix[y_,x_ + 1] != 2:
                matrix[y_,x_ + 1] = 2          
                list_[end] = [x_ + 1,y_,d_]
                end = end + 1                
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

def agent(obs_dict, config_dict):

    
    
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    
    playground = np.zeros((7,11),dtype=np.uint8)
    for i in range(len(observation.geese)):
        goose = observation.geese[i]
        if i == observation.index:
            start = 1
        else:
            start = 0
        for j in range(start,len(goose)):
            r,c = row_col(goose[j], configuration.columns)
            r = r % 7
            c = c % 11
            playground[r,c] = 2

    for food in observation.food:
        r, c = row_col(food, configuration.columns)
        r = r % 7
        c = c % 11
        playground[r,c] = 1
    
    r = path(playground)
            
    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration.columns)
    player_row = player_row % 7
    player_column = player_column % 11

    ori = np.array([Action.NORTH.name,Action.WEST.name,Action.SOUTH.name,Action.EAST.name,'*'])

    if len(player_goose) == 1:
        if (last_dir[player_index] + 2) % 4 == np.argmax(r[player_row,player_column]):
            r[player_row,player_column,np.argmax(r[player_row,player_column])] = -1000
    dir_ = ori[np.argmax(r[player_row,player_column])]
    last_dir[player_index] = np.argmax(r[player_row,player_column])
    print(player_index,dir_)
    return dir_

"""
mat = np.array([[2,2,2,2,2,2,2,2],
                [2,0,2,0,2,0,0,2],
                [2,0,0,0,2,0,0,2],
                [2,0,2,2,2,0,0,2],
                [2,0,2,0,0,0,0,2],
                [2,0,2,2,2,0,0,2],
                [2,0,0,0,2,0,0,2],
                [2,2,2,2,2,2,2,2],
                ],dtype=np.uint8)

print(mat)
print(fill(mat,1,1,4))
print(mat)
"""
