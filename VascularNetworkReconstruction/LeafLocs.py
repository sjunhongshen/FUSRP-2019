from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #for reading csv files if needed
import itertools
#from _SimAnneal import SA (may use this later)
dim =20         
pts = 50
n = np.ndarray(shape = (pts,3))
r = [None]*pts
Coords = []
RoI = np.ndarray(shape = (pts,2,3))
x_init = np.random.randint(dim, size=pts)
y_init = np.random.randint(dim, size=pts)
z_init = np.random.randint(dim, size=pts)
xp = np.arange(dim)
yp = np.arange(dim)
zp = np.arange(dim)
Coords.append(xp)
Coords.append(yp)
Coords.append(zp)

        
class LeafLocs:
    def __init__(self, x, y, z, inf, reg):
        self.x = x
        self.y = y
        self.z = z
        self.inf= inf
        self.reg = reg
        
# Calculates regions of influence for each point
    def influence(self):
        for i in range (pts):
            for j in range (2):
                if j == 0:
                    k = r[0].inf*(j-1)
                elif j == 1:
                    k = r[0].inf*(j)
                r[i].reg[j][:] = [r[i].x + k, r[i].y + k, r[i].z + k]
                    
                    
    def Opt(self):
        m = [None]*(dim**3)
        while len(m) > 0:
            LeafLocs.influence(r)
            m =  LeafLocs.DomCheck(r)
            if len(m) == 0:
                break
            rp = LeafLocs.Repeat(r)
            k=0
            for [i,j] in rp:
                r[i].x = m[k][0]
                r[i].y = m[k][1]
                r[i].z = m[k][2]
                k += 1
                if k>=len(m):
                    break

# Finds overlapping points                
    def Repeat(self):
        repeat = [[0,0]]
        for i in range (pts):
            for j in [x for x in range(pts) if x != i]:
                if r[i].reg[0][0] <= r[j].x <= r[i].reg[1][0] and r[i].reg[0][1]\
                <= r[j].y <= r[i].reg[1][1] and r[i].reg[0][2]<= r[j].z <= r[i].reg[1][2]:
                    test = 0
                    for [m,n] in repeat:
                        if [j,i] == [m,n]:
                            test =1
                    if test ==0:
                        repeat.append([i,j])
        #print(repeat)
        print("Repeat = " + str(len(repeat)))
        return repeat
 
# Finds points in the domain not covered by sampled points        
    def DomCheck(self):
        miss = []
        for element in itertools.product(*Coords):
            flag = 1
            for l in range(pts):
                if r[l].reg[0][0] <= element[0] <= r[l].reg[1][0] and r[l].reg[0][1]<= element[1] <= r[l].reg[1][1] and r[l].reg[0][2]<= element[2] <= r[l].reg[1][2]:
                    flag =1
                    break
                else:
                    flag = 0
            if flag == 0:
                miss.append(element)
        print ("Miss = " + str(len(miss)))
        return miss

for i in range(pts):
    r[i]= LeafLocs(x_init[i], y_init[i], z_init[i], 5, RoI[i])
LeafLocs.Opt(r)

for i in range(pts):
    n[i] = [r[i].x, r[i].y, r[i].z]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(n[:,0], n[:,1], n[:,2], c= n[:,2], cmap='Blues')
