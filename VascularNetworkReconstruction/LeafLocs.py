# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import binvox_rw
#import itertools
#from _SimAnneal import SA

p = 70
dim = 10
#Ewith open('test_1_hemi.binvox', 'rb') as f:
 #    model = binvox_rw.read_as_3d_array(f)

C = np.array(Pts)
lam = []
for [x,y,z] in C:
    k = [int(x*dim), int(y*dim), int(z*dim)]
    lam.append(k)
    
Coords = np.array(lam)
lc = len(Coords)
RoI = np.ndarray(shape = (p,2,3))
r = [None]*p
num= int (lc/p)
x_init = np.random.randint(dim, size=p)
y_init = np.random.randint(dim, size=p)
z_init = np.random.randint(dim, size=p)

    
        
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
        m = [None]*(len(Coords))
        while len(m) > 1:
            LeafLocs.influence(self)
            m =  LeafLocs.DomCheck(self)
            rp = LeafLocs.Repeat(self)
            if len(m) == 0:
                break
            k=0
            for [i,j] in rp:
                self[i].x = m[k][0]
                self[i].y = m[k][1]
                self[i].z = m[k][2]
                k += 1
                if k>=len(m):
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
        #print(repeat)
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

for i in range(p):
    r[i]= LeafLocs(x_init[i], y_init[i], z_init[i], 2, RoI[i])
LeafLocs.Opt(r)

sp = []
for i in range(p):
    fl = 0
    for [x,y,z] in Coords:
        if r[i].reg[0][0] <= x <= r[i].reg[1][0] and r[i].reg[0][1]<= y <= r[i].reg[1][1] and r[i].reg[0][2]<= z <= r[i].reg[1][2]:
            fl =1
    if fl == 1:
        sp.append([r[i].x, r[i].y, r[i].z])
 
Rn= np.ndarray(shape = (len(sp),2,3))
n= np.array(sp)
rn= [None]*len(sp)

for i in range(len(sp)):
    rn[i]= LeafLocs(sp[i][0], sp[i][1], sp[i][2], 2, Rn[i])

LeafLocs.influence(rn)
rep = LeafLocs.Repeat(rn)


#ax.scatter3D(n[:,0], n[:,1], n[:,2], c= n[:,2], cmap='Blues')

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(n[:,0], n[:,1], n[:,2], 'k.', alpha = 0.8)