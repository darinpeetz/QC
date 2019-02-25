# -*- coding: utf-8 -*-
"""
Created on Tue May 08 11:26:00 2018

@author: Darin
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import pyamg
from scipy.special import comb
from os import getcwd, path, makedirs, unlink, listdir
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
from PIL import ImageDraw

from Initialization import Read_Mesh
from Element import Local_Assembly, Fracture_Check

class Solver:
    """Provides functionality to solve the beam QC problem
    """
            
    def __init__(self, filename, Folder=None, processes=None):
        """Constructor from input file. Calls Read_Mesh then Setup()
        
        Parameters
        ----------
        filename : string
            Input filename
        Folder : string
            Directory where to store outputs
        """
        
        (Nodes, cNodes, Elements, cElements, El_Properties, d2c, FracMode, NSet,
            ElSet, Loads, BCs, Track) = Read_Mesh(filename)
        self.Setup(Nodes, cNodes, Elements, cElements, El_Properties, d2c,
                   FracMode, NSet, ElSet, Loads, BCs, Track, Folder)
        
        (np.array(Nodes), np.array(cNodes), np.array(Elements), np.array(cElements),
            np.array(El_Properties), np.array(d2c),
            FracMode, NSet, ElSet, Loads, BCs, Track)
        
    def Setup(self, Nodes, cNodes, Elements, cElements, El_Properties, d2c, 
              FracMode, NSet, ElSet, Loads, BCs, Track, Folder=None):
        """Constructor
        
        Parameters
        ----------
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
        Steps : list of boundary information at each step
            Provides all boundary condition information
        Amplitude : list of amplitude information
            Provides amplitudes to be implemented in the steps
        NSet : dict
            All node sets
        ElSet : dict
            All element sets
        Folder : string
            Directory where to store outputs
        """
        self.Setup_Directory(Folder)
                   
        # Mesh
        self.Nodes = Nodes
        self.cNodes = cNodes
        self.Elements = Elements
        self.cElements = cElements
        self.E = El_Properties[:,0]
        self.E_min = 1e-10*self.E.min()
        self.A = El_Properties[:,1]
        self.I = El_Properties[:,2]
        self.h = El_Properties[:,3]
        self.limit = El_Properties[:,4]
        self.d2c = d2c
        self.fracMode = FracMode
        self.brk = np.zeros(self.Elements.shape[0], dtype=float)
        self.NSet = NSet
        self.ElSet = ElSet
#        self.Loads = Loads
#        self.BCs = BCs
        
        self.dim = Nodes.shape[1]
        self.ndof = self.dim + int(comb(self.dim, 2))
        
        self.FixDof = []
        self.FixDisp = []
        for bc in BCs:
            self.FixDof.append(self.ndof * np.array(self.NSet[bc['Nodes']]) + bc['dof'])
            self.FixDisp.append(bc['val'] * np.ones(len(self.NSet[bc['Nodes']])))
        if self.FixDof:
            self.FixDof = np.concatenate(self.FixDof)
            self.FixDisp = np.concatenate(self.FixDisp)
            self.FreeDof = np.setdiff1d(np.arange(self.ndof * Nodes.shape[0]), self.FixDof)
        else:
            raise ValueError('No Dirichlet BC specified')
        
        self.LoadDof = []
        self.LoadVals = []
        for load in Loads:
            self.LoadDof.append(self.ndof * np.array(self.NSet[load['Nodes']]) + load['dof'])
            self.LoadVals.append(load['val'] * np.ones(len(self.NSet[load['Nodes']])))
        if self.LoadDof:
            self.LoadDof = np.concatenate(self.LoadDof)
            self.LoadVals = np.concatenate(self.LoadVals)
            
        self.Track = {'time':[],'data':[]}
        for track in Track:
            dof = (self.ndof * np.array(self.NSet[track['Nodes']]).reshape(-1,1) +
                   np.array(track['dof']).reshape(1,-1))
            self.Track['data'].append({'dof':dof.reshape(-1), 'type':track['type'], 'val':[]})
        
        self.RHS = np.zeros(Nodes.shape[0]*(self.dim+1), dtype=float)
        self.u = self.RHS.copy()
        self.u = self.u.copy()
        
        # Solver characteristics
        self.step = 0
        self.iter_max = 300
        self.sub_iter_max = 20
        self.t = 0.
        self.t_max = 1.
        self.dt = 1e-2
        self.dt_min = 1e-12
        self.dt_max = 1e-2
        self.ftol = 5e-3
        self.ctol = 1e-2
        self.flux = {}
        
        # Save the mesh information for ease of use later
        np.save(self.Folder + "\\Mesh.npy", {'Elements':self.Elements,
                                             'Nodes':self.Nodes,
                                             'ElSet':self.ElSet,
                                             'NSet':self.NSet,
                                             'BCs':BCs,
                                             'Loads':Loads})
    
    def Resume(self, filename, step=0):
        """ Picks up where a previous solution left off
        
        Parameters
        ----------
        filename : string
            file containing last step information from other solution
        
        Returns
        -------
        None
        """
        data = np.load(filename).item()
        self.u = data['uphi']
        self.RHS = data['RHS']
        self.t = data['time']
        self.step = step
        
    def Setup_Directory(self, Folder):
        """ Prepares output directory for storing results
        
        Parameters
        ----------
        Folder : string
            Directory where to store outputs
        
        Returns
        -------
        None
        """
        
        if Folder is None:
            self.Folder = getcwd() + '\\Steps'
        else:
            self.Folder = Folder
        if not path.isdir(self.Folder):
            makedirs(self.Folder)
        else:
            for filename in listdir(self.Folder):
                file_path = path.join(self.Folder, filename)
                if path.isfile(file_path):
                    unlink(file_path)
        
    def Global_Assembly(self):
        """Assembles the global tangent stiffness matrix and internal force vector
        
        Parameters
        ----------
        
        Returns
        -------
        K : sparse matrix
            Tangent stiffness matrix
        """

        init = False
        if not hasattr(self,'Ki'):
            init = True
            self.Ki = []
            self.Kj = []
        self.Kk = []

        DShape = self.u.copy().reshape(-1,3)
        DShape[:,:2] += self.Nodes
        self.RHS.fill(0)
        for el in range(self.Elements.shape[0]):
            if self.brk[el] < 1.:
                Nodes = self.Nodes[self.Elements[el,:],:]
                Coords = DShape[self.Elements[el,:],:]
                K_el, F_el = Local_Assembly(Nodes, Coords, self.E[el], self.A[el], self.I[el])
                                        
                dof = 3*self.Elements[el,:].reshape(-1,1) + np.arange(3).reshape(1,-1)
                self.RHS[dof.reshape(-1)] -= F_el
            
                if init:
                    dof = dof.reshape(-1,1)
                    dof = np.tile(dof, (1, dof.shape[0]))
                    self.Ki += dof.reshape(-1).tolist()
                    self.Kj += dof.T.reshape(-1).tolist()
                self.Kk += K_el.reshape(-1).tolist()
            else:
                self.Kk += (self.E_min * np.identity(6).reshape(-1)).tolist()
      
        K = sparse.csr_matrix((self.Kk, (self.Ki, self.Kj)))
        
        return K
    
    def Map_Back(self, x, Nodes):
        """Maps from mapped to parent coordinates
        
        Parameters
        ----------
        x : array_like
            Coordinates in mapped domain
        Nodes : array_like
            Nodal coordinates
        
        Returns
        -------
        pc : array_like
            Coorindates in parent domain
        """
        
        d = Nodes[1:,:] - Nodes[0,:]
        mat = np.array([[d[1,1], -d[1,0]], [-d[0,1], d[0,0]]])
        pc = np.dot(mat, x-Nodes[0,:])
        
        return pc / (d[0,0]*d[1,1] - d[0,1]*d[1,0])
        
    def Setup_QC(self):
        """Sets up additional information for QC solve
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        self.N = np.zeros((self.Nodes.shape[0], 3))
        for nd in range(self.Nodes.shape[0]):
            cnd = self.cNodes[self.cElements[self.d2c[nd]]]
            pc = self.Map_Back(self.Nodes[nd,:], self.Nodes[cnd,:])
            self.N[nd,:] = [1 - pc[0] - pc[1], pc[0], pc[1]]
            
        self.cFixDof = []
        for dof in self.FixDof:
            if dof / 3 in self.cNodes:
                self.cFixDof.append(3*np.where(self.cNodes==(dof/3))[0][0] + dof % 3)
        self.cFixDof = np.array(self.cFixDof)
        self.cFreeDof = np.setdiff1d(np.arange(3*self.cNodes.size), self.cFixDof)
        
        self.cu = np.zeros(3*self.cNodes.size)
        
    def Cont_Assembly(self):
        """Assembles the QC RHS and tangent stiffness
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        DShape = self.u.copy().reshape(-1,3)
        DShape[:,:2] += self.Nodes
        
        cRHS = np.zeros(3*self.cNodes.size)
        for nd in range(self.Nodes.shape[0]):
            cdof = 3*self.cElements[self.d2c[nd],:].reshape(-1,1) + np.arange(3).reshape(1,-1)
            cdof = cdof.reshape(-1)
            ddof = 3*nd + np.arange(3)
            cRHS[cdof] += (self.RHS[ddof].reshape(1,-1)*self.N[nd,:].reshape(-1,1)).reshape(-1)
        
        Ki = []
        Kj = []
        Kk = []
        RHS = np.zeros_like(self.RHS)
        for el in range(self.Elements.shape[0]):
            if self.brk[el] < 1.:
                dnd = self.Elements[el,:]
                Nodes = self.Nodes[dnd,:]
                Coords = DShape[dnd,:]
                K_el, F_el = Local_Assembly(Nodes, Coords, self.E[el], self.A[el], self.I[el])
                RHS[(3*dnd.reshape(-1,1) + np.arange(3).reshape(1,-1)).reshape(-1)] -= F_el

                for i, dnd1 in enumerate(dnd): # Loop over each discrete node
                    for j, cnd1 in enumerate(self.cElements[self.d2c[dnd1],:]): # Loop over each continuum node associated with node 1
                        cdof1 = 3*cnd1 + np.arange(3)
                        N1 = self.N[dnd1, j]
                        cRHS[cdof1] -= N1 * F_el[3*i:3*(i+1)]
                        for k, dnd2 in enumerate(dnd): # Again for each discrete node
                            for l, cnd2 in enumerate(self.cElements[self.d2c[dnd2],:]): # Loop over each continuum node associated with node 2
                                cdof2 = 3*cnd2 + np.arange(3)
                                N2 = self.N[dnd2, l]
                                ki, kj = np.meshgrid(cdof1, cdof2)
                                Ki.append(ki.reshape(-1))
                                Kj.append(kj.reshape(-1))
                                Kk.append(N1*N2*K_el[3*i:3*(i+1), 3*k:3*(k+1)].reshape(-1))
      
        K = sparse.csr_matrix((np.concatenate(Kk), (np.concatenate(Ki), np.concatenate(Kj))))
        
        return K, cRHS
    
    def Continuum2Discrete(self, continuum, discrete, hold=None):
        """Get discrete displacements from QC solution
        
        Parameters
        ----------
        continuum : array_like
            values in the continuum mesh
        discrete : array_like
            values in the discrete mesh
        
        Returns
        -------
        None
        """
        
        if hold is not None:
            held = discrete[hold]
            
        for nd in range(self.Nodes.shape[0]):
            cdof = 3*self.cElements[self.d2c[nd],:].reshape(1,-1) + np.arange(3).reshape(-1,1)
            ddof = 3*nd + np.arange(3)
            discrete[ddof] = np.dot(continuum[cdof],self.N[nd])        
            
        if hold is not None:
            discrete[hold] = held

    def Save_Status(self):
        """Saves current status of the solver
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        np.save(self.Folder + "\\Step_%i.npy"%self.step, {'u':self.u,
                                                          'RHS':self.RHS,
                                                          'time':self.t})
        
    def Increment(self):
        """Increments the solver one step forward
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        self.Save_Status()
        self.step += 1
        self.u_old = self.u.copy()
        self.RHS_old = self.RHS.copy()
        self.brk_old = self.brk.copy()
        self.t += self.dt
        if self.LoadVals:
            self.Load = self.t/self.t_max * self.LoadVals
        else:
            self.Load = []
        self.u[self.FixDof] = self.t/self.t_max * self.FixDisp
            
    def Reduce_Step(self, ratio=0.5):
        """Reduces the step size in the case of non-convergence
        
        Parameters
        ----------
        ratio : scalar, optional
            Ratio to reduce step size by
        
        Returns
        -------
        None
        """
        
        print "Reducing step size from %1.2g to %1.2g"%(self.dt, self.dt*ratio)
        self.t -= self.dt
            
        if self.dt > self.dt_min:
            self.dt *= ratio
        else:
            return
            
        self.t += self.dt
        
        self.uphi = self.u_old.copy()
        self.RHS = self.RHS_old.copy()
        self.brk = self.brk_old.copy()
        self.u[self.FixDof] = (self.t/self.t_max) * self.FixDisp
        if self.LoadVals:
            self.Load = self.t/self.t_max * self.LoadVals
        
    def Solve_QC(self):
        """Solve QC system
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.cu_old = self.cu.copy()
        
        for self.iter in range(self.iter_max):
            K, cRHS = self.Cont_Assembly()
            
            du = spla.spsolve(K[self.cFreeDof[:,np.newaxis],self.cFreeDof], cRHS[self.cFreeDof])
            self.cu[self.cFreeDof] += du
            self.Continuum2Discrete(self.cu, self.u, hold=self.FixDof)
                
            conv = self.Convergence_QC(du, cRHS)
            if conv:
                return
        raise ValueError('Too many iterations')
        
    def Solve(self):
        """Solve fully discrete system
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        for self.iter in range(self.iter_max):
            K = self.Global_Assembly()
            self.RHS[self.LoadDof] += self.Load
            
#            Nullspace = np.zeros((3*self.Nodes.shape[0],3))
#            Nullspace[::3,0]  = 1
#            Nullspace[1::3,1] = 1
#            Nullspace[::3,2]  = -self.Nodes[:,1]
#            Nullspace[1::3,2] =  self.Nodes[:,0]
#            Nullspace[2::3,2] =  np.pi/2
#            Nullspace = np.linalg.solve(np.linalg.cholesky(np.dot(Nullspace.T,Nullspace)),Nullspace.T).T
#            K[self.FixDof] = 0
#            K = K.T
#            K[self.FixDof] = 0
#            K = K.T
#            K[self.FixDof[:,np.newaxis], self.FixDof] = sparse.identity(self.FixDof.size)
#            K.eliminate_zeros()
#            
#            ml = pyamg.smoothed_aggregation_solver(K.tobsr(blocksize=(3, 3)),
#                                   B = Nullspace, max_coarse = 80, max_levels=8,
#                                   presmoother='block_jacobi', postsmoother='block_jacobi',
#                                   strength=('symmetric',{'theta':0.003}),
#                                   smooth=('jacobi',{'degree':2}),keep=True,
#                                   coarse_solver='splu')
#            res = []
#            du = ml.solve(self.RHS, residuals=res, tol=1e-8, maxiter=1000)
#            du[self.FixDof] = 0
#            self.u += du
            
            du = spla.spsolve(K[self.FreeDof[:,np.newaxis], self.FreeDof],
                              self.RHS[self.FreeDof])
            self.u[self.FreeDof] += du
                
            conv = self.Convergence(du)
            if conv:
                return
        raise ValueError('Too many iterations')
            
    def Break_Elements(self):
        """Break any elements that have exceeded fracture criteria
        
        Parameters
        ----------
        None
        
        Returns
        -------
        max_rat : float
            Maximum ratio of failure measure to failure criterion
        """
        
        DShape = self.u.copy().reshape(-1,3)
        DShape[:,:2] += self.Nodes
        for el in range(self.Elements.shape[0]):
            if self.brk[el] < 1.:
                Nodes = self.Nodes[self.Elements[el,:],:]
                Coords = DShape[self.Elements[el,:],:]
                self.brk[el] = Fracture_Check(Nodes, Coords, self.E[el], self.A[el],
                                     self.I[el], self.h[el], self.limit[el], self.fracMode)
        
        return
        
    def Convergence(self, du):
        """Check if nonlinear iterations have converged
        
        Parameters
        ----------
        du : array_like
            Change to field variables in last increment
        section : string
            Which subset of problem is being updated ('UU', 'PP', or 'ALL')
        hold : boolean
            Flag indicating that this is is a correction step and criteria are different
        
        Returns
        -------
        converged : bool
            True if iterations have converged, false otherwise
        """
        
        
        if self.iter == 0:
            self.flux = np.sum(np.abs(self.RHS[self.FreeDof]))
        else:
            self.flux += np.sum(np.abs(self.RHS[self.FreeDof]))
#            self.flux[section] = max(np.sum(np.abs(self.RHS[subset])), self.flux[section])
            
        if self.flux == 0:
            force_check = True
        else:
            force_check = np.max(np.abs(self.RHS[self.FreeDof])) < self.ftol * self.flux/(self.iter+1)
            
        increment = self.u[self.FreeDof] - self.u_old[self.FreeDof]
        if np.max(np.abs(increment)) == 0:
            corr_check = True
        else:
            corr_check = np.max(np.abs(du)) < self.ctol * np.max(np.abs(increment)) or np.max(abs(du)) < 1e-12

        print "It: %i, Force: %i, Corr: %i"%(self.iter, force_check, corr_check)
#        print "Time since last convergence check: %1.4g"%(time.time() - self.t0)
        self.t0 = time.time()
        return force_check and corr_check
    
    def Convergence_QC(self, du, cRHS):
        """Check if nonlinear iterations have converged in coarse system
        
        Parameters
        ----------
        du : array_like
            Change to field variables in last increment
        section : string
            Which subset of problem is being updated ('UU', 'PP', or 'ALL')
        hold : boolean
            Flag indicating that this is is a correction step and criteria are different
        
        Returns
        -------
        converged : bool
            True if iterations have converged, false otherwise
        """
        
        
        if self.iter == 0:
            self.flux = np.sum(np.abs(cRHS[self.cFreeDof]))
        else:
            self.flux += np.sum(np.abs(cRHS[self.cFreeDof]))
#            self.flux = max(np.sum(np.abs(cRHS[self.cFreeDof])), self.flux
            
        if self.flux == 0:
            force_check = True
        else:
            force_check = np.max(np.abs(cRHS[self.cFreeDof])) < self.ftol * self.flux/(self.iter+1)
            
        increment = self.cu[self.cFreeDof] - self.cu_old[self.cFreeDof]
        if np.max(np.abs(increment)) == 0:
            corr_check = True
        else:
            corr_check = np.max(np.abs(du)) < self.ctol * np.max(np.abs(increment)) or np.max(abs(du)) < 1e-12

        print "It: %i, Force: %i, Corr: %i"%(self.iter, force_check, corr_check)
#        print "Time since last convergence check: %1.4g"%(time.time() - self.t0)
        self.t0 = time.time()
        return force_check and corr_check
        
    def run(self, plot_frequency=np.Inf, method='QC'):
        """Run the phase field solver
        
        Parameters
        ----------
        plot_frequncy : scalar
            How often to plot the displaced shape and damage status
        method : string
            Either 'Discrete' or 'QC' to indicate how to solve the problem
        
        Returns
        -------
        None
        """
        
        if method == 'QC':
            self.Setup_QC()
        self.plot()
        Disp = []
        Reaction = []
        while self.t < self.t_max:
            self.Increment()
            if method == 'QC':
                self.Solve_QC()
            elif method == 'Discrete':
                self.Solve()
                
            unbroken = self.brk < 1.
            brk = self.brk.copy()
            cnt_old = unbroken.sum()
            self.Break_Elements()
            cnt_new = np.sum(self.brk < 1.)
            
            if cnt_new < cnt_old:
                break_it = 0
                while True:
                    if np.max(self.brk[unbroken]) > max(3. - 0.1 * break_it, 1.):
                        ratio = 0.9*np.min(np.maximum(1 - brk[unbroken], 1e-16) /
                                           np.maximum(self.brk[unbroken] - brk[unbroken], 1e-16))
                        print ratio
                        self.Reduce_Step(ratio=min(ratio,0.1))
                        break_it = 0
                        brk = self.brk.copy()
                    
                    if method == 'QC':
                        self.Solve_QC()
                    elif method == 'Discrete':
                        self.Solve()
                    unbroken = self.brk < 1.
                    cnt_old = cnt_new
                    brk[unbroken] = self.brk[unbroken]
                    self.Break_Elements()
                    cnt_new = np.sum(self.brk < 1.)
                    
                    break_it += 1
                    print break_it, cnt_new, np.max(self.brk[unbroken])
                    if cnt_new == cnt_old:
                        break
            else:
                self.dt *= min(10, self.dt_max/self.dt)
#                change = self.brk[unbroken] - self.brk_old[unbroken]
#                ratio = 0.8*np.min((1-self.brk[unbroken])/change)
#                if ratio > 2:
#                    self.dt *= min(min(ratio, 10), self.dt_max/self.dt)
                
            print "Time: ", self.t
                
            self.Track['time'].append(self.t)
            for track in self.Track['data']:
                if track['type'] == 'U':
                    track['val'].append(np.linalg.norm(self.u[track['dof']]))
                elif track['type'] == 'R':
                    track['val'].append(np.linalg.norm(self.RHS[track['dof']]))
                
            if True or int(self.t / plot_frequency) > int((self.t - self.dt) / plot_frequency):
                self.plot()
        
#        self.Save_Status()
        return Disp, Reaction
        
    def plot(self, data=None, sf=0.8):
        """Plot the current displacements and damage status
        
        Parameters
        ----------
        amp : scalar, optional
            Amplification factor
        data : dict, optional
            Name-value pair of information to plot
        sf : scalar
            Scale factor
        
        Returns
        -------
        None
        """
        if data is None:
            data = {'Displacements': self.u}
        for key in data.keys():
            vals = data[key].reshape(-1,3)[:,:2]
            if np.linalg.norm(data[key]) == 0:
                scale = 1
            else:
                scale = sf*(self.Nodes.max()-self.Nodes.min())/np.abs(vals).max()
            DShape = self.Nodes + scale*vals
            DShape = np.append(DShape, [[None, None]], axis=0)
            Elements = np.hstack([self.Elements, self.Nodes.shape[0]*np.ones((self.Elements.shape[0],1),dtype=int)])
            whole = Elements[self.brk <  1.,:].reshape(-1)
            broke = Elements[self.brk >= 1.,:].reshape(-1)
            
            plt.figure(key,figsize=(12,12))
            plt.clf()
            plt.plot(DShape[whole,0], DShape[whole,1],c='b')
            plt.plot(DShape[broke,0], DShape[broke,1],c='r')
            plt.axis('equal')
            plt.title('Scale: %1.6g'%scale)
            plt.show()
            plt.pause(0.05)
            
        
if __name__ == "__main__":
    
    solver = Solver('Test_Small.inp', getcwd() + '\\Test')
    
#    import cProfile
#    cProfile.run('solver.run(1e-10)', 'prof_stats')
    solver.run(method='QC')