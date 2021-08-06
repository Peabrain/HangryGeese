import numpy as np

dir_dict = {'NORTH': 0, 'WEST': 1, 'SOUTH': 2, 'EAST': 3}
dir_ = ['NORTH', 'WEST', 'SOUTH', 'EAST']

class STEP:
    def __init__(self):
        self.players = np.zeros((2,8),dtype=np.uint16)
        self.food = np.full((2),-1,dtype=np.int8)
        self.last_dir = np.full((self.players.shape[0]),-1,dtype=np.int8)
        self.value = np.full((self.players.shape[0]),0,dtype=np.float)

    def copy(self,s):
        self.players = s.players.copy()
        self.food = s.food.copy()
        self.last_dir = s.last_dir.copy()
        self.value = s.value.copy()

    def randomize(self):
        r = list(np.random.choice(list(range(77)),size=(self.players.shape[0] + self.food.shape[0])))
        for i in range(self.players.shape[0]):
            self.players[i,0] = r[i]
            self.players[i,0] |= self.players[i,0] << 8
            x = (self.players[i,0] & 255) % 11
            y = (self.players[i,0] & 255) // 11
            self.players[i,1 + y] |= 1 << x
        for i in range(self.food.shape[0]):
            self.food[i] = r[self.players.shape[0] + i]
    
    def calc_values(self):
        values = np.full((7,11),255,dtype=np.uint8)
        for i in self.food:
            pass

        pass


s = STEP()
s.randomize()
print(s.players,s.food,s.last_dir,s.value)
