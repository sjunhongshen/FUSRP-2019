from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import binvox_rw

p = 400
dim = 512
sampling_ratio = 4500
with open('test_1_hemi.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

Coords=[]
pts = np.transpose(np.nonzero(model.data))
for i in range(len(pts)):
    if ((i%sampling_ratio)==0):
        Coords.append(pts[i])
        
Coords = np.array(Coords)
RoI = np.ndarray(shape = (p,2,3))
r = [None]*p
n= np.zeros(shape = (p,3))
x_init = np.random.randint(dim, size=p)
y_init = np.random.randint(dim, size=p)
z_init = np.random.randint(300, size=p)    
         
class LeafLocs:
    def __init__(self, x, y, z, inf, reg):
        self.x = x
        self.y = y
        self.z = z
        self.inf= inf
        self.reg = reg
    
    def influence(self):
        for i in range (len(self)):
            for j in range (2):
                if j == 0:
                    k = self[0].inf*(j-1)
                elif j == 1:
                    k = self[0].inf*(j)
                self[i].reg[j][:] = [self[i].x + k, self[i].y + k, self[i].z + k]
                    
                    
    def Opt(self):
        miss = [None]*(len(Coords))
        while len(miss) > 1:
            LeafLocs.influence(self)
            miss =  LeafLocs.DomCheck(self)
            rp = LeafLocs.Repeat(self)
            move = LeafLocs.MovingP(self)
            mp = np.array(move + [i for i in rp if i not in move])
            if len(miss) == 0:
                break
            k=0
            for [i,j] in mp:
                self[i].x = miss[k][0]
                self[i].y = miss[k][1]
                self[i].z = miss[k][2]
                k += 1
                if k>=len(miss):
                    break

    def Repeat(self):
        repeat = [[0,0]]
        for i in range (len(self)):
            for j in [x for x in range(len(self)) if x != i]:
                if self[i].reg[0][0] <= self[j].x <= self[i].reg[1][0] and self[i].reg[0][1]\
                <= self[j].y <= self[i].reg[1][1] and self[i].reg[0][2]<= self[j].z <= self[i].reg[1][2]:
                    test = 0
                    for [m,n] in repeat:
                        if [j,i] == [m,n]:
                            test =1
                    if test ==0:
                        repeat.append([i,j])
        print("Repeat = " + str(len(repeat)))
        return repeat
        
    def DomCheck(self):
        miss = []
        for [x,y,z] in Coords:
            flag = 1
            for l in range(len(self)):
                if self[l].reg[0][0] <= x <= self[l].reg[1][0] and\
                self[l].reg[0][1]<= y <= self[l].reg[1][1] and \
                self[l].reg[0][2]<= z <= self[l].reg[1][2]:
                    flag =1
                    break
                else:
                    flag = 0
            if flag == 0:
                miss.append([x,y,z])
        print ("Miss = " + str(len(miss)))
        return miss
    
    def MovingP(self):
        mp = []
        for i in range(p):
            fl = 0
            for [x,y,z] in Coords:
                if r[i].reg[0][0] <= x <= r[i].reg[1][0] and r[i].reg[0][1]<= y <= r[i].reg[1][1] and r[i].reg[0][2]<= z <= r[i].reg[1][2]:
                    fl =1
            if fl == 0:
                mp.append([i,0])
        return mp

for i in range(p):
    r[i]= LeafLocs(x_init[i], y_init[i], z_init[i], 2, RoI[i])
LeafLocs.Opt(r)

for i in range (p):
    n[i]= [r[i].x, r[i].y, r[i].z]


#ax.scatter3D(n[:,0], n[:,1], n[:,2], c= n[:,2], cmap='Blues')

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(n[:,0], n[:,1], n[:,2], 'k.', alpha = 0.8)