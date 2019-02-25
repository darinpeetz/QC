# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 12:04:29 2018

@author: Darin
"""

import numpy as np
# Creates a uniform mesh with a crack midway up the right edge of length = cr_sz

# Input
filename = 'Test_Small.inp'
Nx = 20
Ny = 20
sx = 5
sy = 5
Lx = 1
Ly = 1
cr_sz = 0.01
E = 20e9
A = 1e-4
I = 1e-9
h = 1e-2
limit = 350e12
mode = 'max_tensile'

# Calculations
dx = Lx/Nx
dy = Ly/Ny

x = np.linspace(0, Lx, Nx+1)
y = np.linspace(0, Ly, Ny+1)
Nodes = np.hstack([np.tile(x,(Ny+1,1)).reshape(-1,1), np.tile(y.reshape(-1,1),(1,Nx+1)).reshape(-1,1)])

Elements = []
for j in range(Ny+1):
    for i in range(Nx+1):
        nd = i + (Nx+1)*j
        if i < Nx:
            Elements.append([nd, nd + 1])
        if i < Nx and j < Ny:
            Elements.append([nd, nd + Nx + 2])
        if j < Ny:
            Elements.append([nd, nd + Nx + 1])
        if i > 0 and j < Ny:
            Elements.append([nd, nd + Nx])
Elements = np.array(Elements)
Centroids = np.zeros((Elements.shape[0],2))
Centroids[:,0] = Nodes[Elements,0].mean(axis=1)
Centroids[:,1] = Nodes[Elements,1].mean(axis=1)
remove = np.where(np.logical_and(Centroids[:,0] < 0.149, np.logical_and(Centroids[:,1] < 0.515, Centroids[:,1] > 0.505)))
#remove2 = np.where(np.logical_and(Centroids[:,0] > 0.851, np.logical_and(Centroids[:,1] < 0.52, Centroids[:,1] > 0.51)))
#remove = np.union1d(remove, remove2)
#Elements = np.delete(Elements, remove, axis=0)

cElements = []
for j in range(Ny/sy):
    for i in range(Nx/sx):
        nodes = [i + (Nx/sx+1)*j, i+1 + (Nx/sx+1)*j, i+1 + (Nx/sx+1)*(j+1), i + (Nx/sx+1)*(j+1)]
        # Lower right
        cElements.append([nodes[0], nodes[1], nodes[2]])
        # Upper left
        cElements.append([nodes[0], nodes[2], nodes[3]])
cElements = np.array(cElements)
cNodes = (np.tile(np.arange(0, Nx+1, sx).reshape(1,-1), ((Ny/sy)+1,1)) + 
 (Nx+1) * np.tile(np.arange(0, Ny+1, sy).reshape(-1,1), (1,(Nx/sx)+1))).reshape(-1)

Associate = cNodes.shape[0] * np.ones(Nodes.shape[0], dtype=int)
Associate[cNodes] = np.arange(cNodes.shape[0])
Associate_old = Associate.copy()
while Associate.max() == cNodes.shape[0]:
    for el in Elements:
        if Associate_old[el].max() == cNodes.shape[0] and Associate_old[el].min() < cNodes.shape[0]:
            Associate[el] = Associate_old[el].min()
    Associate_old = Associate.copy()
    
NodeElem = [[] for i in range(cNodes.shape[0])]
for i, el in enumerate(cElements):
    for nd in el:
        NodeElem[nd].append(i)

for nd in range(Associate.shape[0]):
    for el in NodeElem[Associate[nd]]:
        edges = np.hstack([cElements[el].reshape(-1,1), np.roll(cElements[el],-1).reshape(-1,1)])
        pVec = Nodes[nd,:] - Nodes[cNodes[edges[:,0]],:]
        eVec = Nodes[cNodes[edges[:,1]],:] - Nodes[cNodes[edges[:,0]],:]
        cross = eVec[:,0] * pVec[:,1] - eVec[:,1] * pVec[:,0]
        if cross.min() > -1e-6 * cross.max():
            Associate[nd] = el
            break
    
with open(filename, 'w') as fh:
    # Header information
    fh.write('*Test model for evaluating functionality of QC solver\n')
    
    # Nodes
    fh.write('*Node\n')
    for nd in range(Nodes.shape[0]):
        fh.write('%6i,%6.6g,%6.6g\n'%(nd+1, Nodes[nd,0], Nodes[nd,1]))
    fh.write('*/Node\n')
    
    # Elements
    fh.write('*Elements\n')
    fh.write('mode=%s\n'%mode)
    for el in range(Elements.shape[0]):
        fh.write('%6i,%6i,%6i,%10.8g,%10.8g,%10.8g,%10.8g,%10.8g\n'%(el+1, Elements[el,0]+1,
                                                       Elements[el,1]+1, E, A, I, h, limit))
    fh.write('*/Elements\n')
    
    # Continuum nodes
    fh.write('*cNode\n%3i' % (cNodes[0]+1))
    for i, nd in enumerate(cNodes[1:]):
        if (i+1) % 16 == 0:
            fh.write('\n%3i'%(nd+1))
        else:
            fh.write(', %3i'%(nd+1))
    fh.write('\n*/cNodes\n')
    
    # Continuum elements
    fh.write('*cElements\n')
    for i, el in enumerate(cElements):
        fh.write('%6i,%6i,%6i,%6i\n'%(i+1, el[0]+1, el[1]+1, el[2]+1))
    fh.write('*/cElements\n')
    
    # Association between discrete and continuum elements
    fh.write('*D2C\n%3i' % (Associate[0]+1))
    for i, d2c in enumerate(Associate[1:]):
        if (i+1) % 16 == 0:
            fh.write('\n%3i'%(d2c+1))
        else:
            fh.write(', %3i'%(d2c+1))
    fh.write('\n*/D2C\n')
    
    # Node sets
    offset = (Nx+1)*10
    
    # Bottom edge
    fh.write('*Nset\nBOT\n%3i' % (1))
    for num, i in enumerate(range(1,Nx+1)):
        if (num+1) % 16 == 0:
            fh.write('\n%3i'%(i+1))
        else:
            fh.write(', %3i'%(i+1))
    fh.write('\n')
    
    fh.write('BOT_M\n%3i\n' % ((Nx+1)/2))
    
    # Top edge
    Ld = Lx
    fh.write('TOP\n%6i' % (Nodes.shape[0]-Nx))
    for num, i in enumerate(range(Nodes.shape[0]-Nx, Nodes.shape[0])):
        if (num+1) % 16 == 0:
            fh.write('\n%6i'%(i+1))
        else:
            fh.write(', %6i'%(i+1))
    fh.write('\n')
    
    fh.write('*/Nset\n')
    
    fh.write('*BC\n')
    fh.write('BOT, 1, 0\n')
    fh.write('BOT, 2, 0\n')
#    fh.write('TOP, 1, 0.05\n')
    fh.write('TOP, 2, 0.01\n')
    fh.write('*/BC\n')
    
    fh.write('*TRACK\n')
    fh.write('TOP,R,2\n')
    fh.write('*/TRACK\n')