# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 14:03:42 2018

@author: Darin
"""

import numpy as np

def Orientation(Nodes, DShape):
    """Calculates angles and lengths of element
    
    Parameters
    ----------
    Nodes : array_like
        Nodal coordinates for a given element
    DShape : array_like
        Deflected coordinates for a given element
        
    Returns
    -------
    L : scalar
        Deformed length
    L0 : scalar
        Undeformed length
    cosb : scalar
        Cosine of angle between corotated axis and x-axis
    sinb : scalar
        Sine of angle between corotated axis and x-axis
    t_loc : array_like
        Local rotations at each node
    """
    
    # Original configuration
    vec0 = Nodes[1,:] - Nodes[0,:]
    L0 = np.sqrt(vec0[0]**2 + vec0[1]**2)
    
    # Deformed configuration
    vec = DShape[1,:] - DShape[0,:]
    L = np.sqrt(vec[0]**2 + vec[1]**2)
    
#    cosb0 = vec0[0] / L
#    sinb0 = vec0[1] / L
#    
#    cost1 = np.cos(DShape[0,2])
#    sint1 = np.sin(DShape[0,2])
#    
#    cost2 = np.cos(DShape[1,2])
#    sint2 = np.sin(DShape[1,2])
#    
#    cosb1 = cosb0*cost1 - sinb0*sint1
#    sinb1 = sinb0*cost1 + cosb0*sint1
#    cosb2 = cosb0*cost2 - sinb0*sint2
#    sinb2 = sinb0*cost2 + cosb0*sint2
    
    # Angles of rotation (b0=undeformed, b=deformed, b1=undeformed+t1, b2=undefosrmed+t2)
    b0 = np.arctan2(vec0[1], vec0[0])
    b1 = b0 + DShape[0,2]
    b2 = b0 + DShape[1,2]
    
    cosb1 = np.cos(b1)
    sinb1 = np.sin(b1)
    
    cosb2 = np.cos(b2)
    sinb2 = np.sin(b2)
    
    cosb = vec[0] / L
    sinb = vec[1] / L
    
    # Local rotation relative to new deformed axis
    t_loc = np.array([np.arctan2(cosb*sinb1 - sinb*cosb1, cosb*cosb1 + sinb*sinb1),
                      np.arctan2(cosb*sinb2 - sinb*cosb2, cosb*cosb2 + sinb*sinb2)])
    
    return L, L0, cosb, sinb, t_loc

def Forces_Int(L, L0, t_loc, E, A, I):
    """Calculates internal forces of an element
    
    Parameters
    ----------
    L : scalar
        Deformed length
    L0 : scalar
        Undeformed length
    t_loc : array_like
        Local rotations at each node
    E : scalar
        Young's modulus
    A : scalar
        Cross-sectional area
    I : scalar
        Moment of inertia
        
    Returns
    -------
    N : scalar
        Axial Force
    M : array_like
        Moments at each node
    """
    
    # Axial strain and normal force
    e_ax = (L**2 - L0**2) / (L + L0)
    N = E * A * e_ax / L0
    
    # Moments
    M = 2*E*I/L0 * np.dot(np.array([[2, 1], [1, 2]]), t_loc)
    
    return N, M
    
def Local_Assembly(Nodes, DShape, E, A, I):
    """Assembles the local tangent stiffness matrix and internal force vector
    
    Parameters
    ----------
    Nodes : array_like
        Nodal coordinates for a given element
    DShape : array_like
        Deflected coordinates for a given element
    E : scalar
        Young's modulus
    A : scalar
        Cross-sectional area
    I : scalar
        Moment of inertia
        
    Returns
    -------
    K : array_like
        Tangent stiffness matrix
    F : array_like
        Force vector in global orientation
    """
    
    # Get angles and lengths of element
    L, L0, cosb, sinb, t_loc = Orientation(Nodes, DShape)
    
    # Get internal forces
    N, M = Forces_Int(L, L0, t_loc, E, A, I)
    
    sinb_L = sinb/L
    cosb_L = cosb/L
    r = np.array([-cosb, -sinb, 0, cosb, sinb, 0])
    z = np.array([sinb, -cosb, 0, -sinb, cosb, 0])
    B = np.array([[-cosb, -sinb, 0, cosb, sinb, 0],
                  [-sinb_L, cosb_L, 1, sinb_L, -cosb_L, 0],
                  [-sinb_L, cosb_L, 0, sinb_L, -cosb_L, 1]])
    
    # Force vector
    F = np.dot(np.array([N, M[0], M[1]]), B)
    
    r2 = I/A
    C = E*A/L0 * np.array([[1, 0, 0], [0, 4*r2, 2*r2], [0, 2*r2, 4*r2]])
    
    # Standard stiffness matrix
    K1 = np.dot(B.T, np.dot(C, B))
    
    # Nonlinear stiffness matrix
    rz = np.outer(r,z)
    K2 = N/L*np.outer(z,z) + M.sum()/L**2 * (rz + rz.T)
    
    return K1+K2, F

def Fracture_Check(Nodes, DShape, E, A, I, h, limit, mode='max_tensile'):
    """Checks to see if an element has broken
    
    Parameters
    ----------
    Nodes : array_like
        Nodal coordinates for a given element
    DShape : array_like
        Deflected coordinates for a given element
    E : scalar
        Young's modulus
    A : scalar
        Cross-sectional area
    I : scalar
        Moment of inertia
    limit : scalar
        Fracture criteria
    mode : string
        Mode of fracture to evaluate (e.g. maximum tensile stress)
        
    Returns
    -------
    ratio : float
        Ratio of failure measure to failure limit (>1 means breaking)
    """
    
    # Get angles and lengths of element
    L, L0, cosb, sinb, t_loc = Orientation(Nodes, DShape)
    
    # Get internal forces
    N, M = Forces_Int(L, L0, t_loc, E, A, I)
    
    if mode == 'max_tensile':
        # Check maximum tensile stress for fracture
        stress = N/A + np.abs(M).max()*h/2/I
        return max(stress / limit, 0)
    else:
        msg = "Unknown fracture mode specified: %s"%mode
        raise ValueError(msg)
    
def K_FD(Nodes, DShape, E, A, I, delta=1e-6):
    """Assembles the local tangent stiffness matrix using finite difference on
    the force vector
    
    Parameters
    ----------
    Nodes : array_like
        Nodal coordinates for a given element
    DShape : array_like
        Deflected coordinates for a given element
    E : scalar
        Young's modulus
    A : scalar
        Cross-sectional area
    I : scalar
        Moment of inertia
    delta : scalar, optional
        Finite difference step size
        
    Returns
    -------
    K : array_like
        Tangent stiffness matrix
    F : array_like
        Force vector in global orientation
    """
    
    K = np.zeros((DShape.size, DShape.size))
    for i in range(DShape.size):
        u = DShape.copy().reshape(-1)
        u[i] += delta
        up = Local_Assembly(Nodes, u.reshape(-1,3), E, A, I)[1]
        
        u = DShape.copy().reshape(-1)
        u[i] -= delta
        down = Local_Assembly(Nodes, u.reshape(-1,3), E, A, I)[1]
        K[:,i] = (up-down)/2/delta
        
    return K