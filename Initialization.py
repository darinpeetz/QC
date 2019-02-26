# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 11:37:07 2018

@author: Darin
"""

import numpy as np

def Activate(line):
    """Checks if we are reading nodes, elements, or something else
    
    Parameters
    ----------
    line : string
        The line to check for activation
        
    Returns
    -------
    active : string
        'nodes' if we are going to read nodes, 'elems' if we are going to read
        elements, 'prop' if properties, or 'none' otherwise
    """

    active = 'none'
    if line[:5].upper() == '*NODE':
        active = 'nodes'
    elif line[:6].upper() == '*CNODE':
        active = 'cnodes'
    elif line[:8].upper() == '*ELEMENT':
        active = 'elems'
    elif line[:9].upper() == '*CELEMENT':
        active = 'celems'
    elif line[:4].upper() == '*D2C':
        active = 'd2c'
    elif line[:5].upper() == '*NSET':
        active = 'nset'
    elif line[:6].upper() == '*ELSET':
        active = 'elset'
    elif line[:5].upper() == '*LOAD':
        active = 'load'
    elif line[:3].upper() == '*BC':
        active = 'bc'
    elif line[:6].upper() == '*TRACK':
        active = 'track'
    else:
        print("Unknown item in input file: %s"%line)
    
    return active
    
def Read_Mesh(filename):
    """Reads a mesh from a .inp file
    
    Parameters
    ----------
    filename : string
        name of the file
        
    Returns
    -------
    Nodes : array_like
        Coordinates of every node in the mesh
    cNodes : array_like
        List of nodes that exist in continuum mesh
    Elements : array_like
        All elements in the mesh
    cElements : array_like
        Continuum element connectivity
    El_Properites : array_like
        Element properties
    d2c : array_like
        List of continuum elements that each node lies in
    FracMode : string
        Fracture mode to use
    NSet : dict
        All node sets
    ElSet : dict
        All element sets
    Loads : list of dict
        Description of all external loads
    BCs : list of dict
        Description of all boundary conditions
    Track : list of dict
        What to track in the simulation
    """
    
    Nodes = []
    cNodes = []
    Elements = []
    cElements = []
    El_Properties = []
    d2c = []
    ElSet = {}
    NSet = {}
    Loads = []
    BCs = []
    Track = []
    name = 'none'
    FracMode = 'max_tensile'
    with open(filename, 'r') as fh:
        active = 'none'
        for ln, line in enumerate(fh):
            if line[0] == '#':
                pass
            elif line[0:2] == '*/':
                active = None
            elif line[0] == '*':
                active = Activate(line)

            elif active == 'nodes':
                Nodes.append([float(x) for x in line.split(',')[1:]])

            elif active == 'elems':
                if line[:4] == 'mode':
                    FracMode = line.split('=')[-1].strip()
                else:
                    Elements.append([int(x)-1 for x in line.split(',')[1:3]])
                    El_Properties.append([float(x) for x in line.split(',')[3:]])
                    
            elif active == 'cnodes':
                cNodes += [int(x)-1 for x in line.split(',')]
                
            elif active == 'celems':
                cElements.append([int(x)-1 for x in line.split(',')[1:]])
                
            elif active == 'd2c':
                d2c += [int(x)-1 for x in line.split(',')]
                
            elif active == 'nset':
                try:
                    int(line.strip(' ')[0]) 
                    NSet[name] += [int(x)-1 for x in line.split(',')]
                except ValueError:
                    name = line.strip('\n')
                    NSet[name] = []
            elif active == 'elset':
                try:
                    int(line[0]) 
                    ElSet[name] += [int(x)-1 for x in line.split(',')]
                except ValueError:
                    name = line.strip('\n')
                    ElSet[name] = []
                    
            elif active == 'load':
                temp = line.split(',')
                Loads.append({'Nodes':temp[0], 'dof':int(temp[1])-1,
                             'val':float(temp[2])})
                
            elif active == 'bc': 
                temp = line.split(',')
                BCs.append({'Nodes':temp[0], 'dof':int(temp[1])-1,
                             'val':float(temp[2])})
    
            elif active == 'track':
                temp = line.split(',')
                Track.append({'Nodes':temp[0], 'type':temp[1].upper(),
                              'dof':[int(d)-1 for d in temp[2:]]})
    
            else:
                print("Ignoring line %i: %s"%(ln,line))
        
    return (np.array(Nodes), np.array(cNodes), np.array(Elements), np.array(cElements),
            np.array(El_Properties), np.array(d2c),
            FracMode, NSet, ElSet, Loads, BCs, Track)
