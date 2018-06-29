#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:28:15 2018

@author: Simon

2D panel code test
"""

import numpy as np
from numpy.linalg import norm, solve

class Vortex:
    """
    Class for a discrete vortex
    """
    
    def __init__(self, point, strength, core = False):
        """
        2D option: specify the radius of the vortex core
        """
        self.point = point
        self.strength = strength
        self.core = core
        if core:
            self.omega = self.strength/(2*np.pi*core**2)
        
    def velind(self, point):
        """
        Calculates the induced velocity of the vortex on a point
        """
        
        point = np.asarray(point)
        relpos = point-self.point
        r = norm(relpos)
        if np.isclose(r, 0): #Check whether r is zero so no infinities are present
            return 0
        mat = np.array([[0,1],[-1,0]])
        #Check whether we have a 2D vortex
        if self.core:
            if r<self.core: #is it in the core?
                v_ind = self.strength/(2*np.pi*(self.core**2))*mat.dot(relpos)
            else:
                v_ind = self.strength/(2*np.pi*r**2)*mat.dot(relpos)
        else:
           v_ind = self.strength/(2*np.pi*r**2)*mat.dot(relpos) 
        
        return v_ind
        
class Wake:
    """
    Class for a wake which consists of multiple vortices
    """
    
    def __init__(self, vortexlist = None):
        if type(vortexlist) == type(None):
            self.vortices = []
        else:
            self.vortices = vortexlist
            
    def add_vortex(self, vortex):
        """
        Wrapper around the built-in list append method to add a vorcex to the wake
        """
        self.vortices.append(vortex)
        
    def velind(self, point):
        """
        Calculates the induced velocity due to the wake on a point
        """
        v_ind = 0
        for vortex in self.vortices:
            v_ind += vortex.velind(point)
        return v_ind
    
    def wake_rollup(self, plate, pos, dt = 0.1):
        """
        Calculates the movement of the wake due to the self-induced velocities 
        and those from an airfoil (plate) at a position pos from the 
        starting point
        """
        #Calcuate induced velocities
        vellist = []
        for vortex in self.vortices:
            v_ind = 0
            v_ind += self.velind(vortex.point)
            v_ind += plate.vel_ind(vortex.point-pos)
            vellist.append(v_ind)
        #Update positions
        for i, vortex in enumerate(self.vortices):
            vortex.point += vellist[i]*dt
    
class Panel:
    """
    Class for a single panel
    """
    
    def __init__(self, start, end, strength):
        start = np.asarray(start)
        end = np.asarray(end)
        self.start = start
        
        self.end = end
        
        vect = end-start
        
        self.vpoint = start + 0.25*(vect)
        
        self.cpoint = start + 0.75*(vect)
        
        self.vortex = Vortex(self.vpoint, strength)
        
        beta = np.arctan2(vect[1],vect[0])
        
        self.normal = np.array([-np.sin(beta), np.cos(beta)])
        
        self.tangential = np.array([np.cos(beta), np.sin(beta)])
        
        self.length = np.linalg.norm(vect)
    
    def velind(self, point):
        '''
        Calculate the induced velocity at a point
        '''
        return self.vortex.velind(point)
        
       
class Plate:
    """
    Class for a plate in a flow which can be unsteady
    """
    
    def __init__(self, coordlist, connect = False, unsteady = False, vortexpoint = None, dt = 0.1):
        """
        If unsteady is true a vortexpoint has to be given
        """
        panellist = []
        if connect:
            end = 0
        else:
            end = 1
        
        for i in range(len(coordlist)-end):
            panel = Panel(coordlist[i], coordlist[i+1], 1)
            panellist.append(panel)
        
        self.panels = panellist
        self.dim = len(panellist)
        self.Construct_matrix()
        if unsteady: #Do all the nescesarry steps for the unsteady part
            if type(vortexpoint) == type(None):
                raise RuntimeError("Please define a vortexpoint in the input")
            self.Construct_unsteady(vortexpoint)
            self.vortexpoint = vortexpoint
            self.dt = 0.1

    def Update_geo(self, coordlist, connect = False):
        """
        Updates the geometry of the plate
        """
        panellist = []
        if connect:
            end = 0
        else:
            end = 1
        for i in range(len(coordlist)-end):
            panel = Panel(coordlist[i], coordlist[i+1], 1)
            panellist.append(panel)
        self.panels = panellist
        self.dim = len(panellist)
        self.Construct_matrix()
        
    def Construct_matrix(self):
        """
        Construct the influence coefficient matrix
        """

        A = np.zeros((self.dim, self.dim))
        for i, panel_i in enumerate(self.panels):
            for j, panel_j in enumerate(self.panels):
                A[i,j] = np.dot(panel_j.velind(panel_i.cpoint),panel_i.normal)
        self.influences = A
        
    def Construct_unsteady(self, vortexpoint):
        """
        Construct the influence coefficient matrix for the unsteady cases
        vortexpoint is the unknown vortex'location
        """
        #extending the influence matrix
        A = np.ones((self.dim + 1, self.dim + 1))
        A[:-1,:-1] = self.influences
        u_vortex = Vortex(vortexpoint, 1)
        for i, panel in enumerate(self.panels):
            A[i,-1] = np.dot(u_vortex.velind(panel.cpoint), panel.normal)
        self.u_influences = A
        
    def solve_circs(self, V, unsteady = False, wake = None, prevcirc = None, rot = None, append = False):
        """
        Solve for the circulations for a certain velocity
        """
        
        V = np.asarray(V)
        if unsteady:
            B = np.zeros(self.dim+1)
            B[-1] = np.sum(prevcirc)
            V_tot = V + rot.T
            for i, panel in enumerate(self.panels):
                wake_vel = wake.velind(panel.cpoint)
                B[i] = np.dot(-(V_tot[i]+wake_vel), panel.normal)
            circs = solve(self.u_influences, B)
        else:
            B = np.zeros(self.dim)
            for i, panel in enumerate(self.panels):
                B[i] = np.dot(-V, panel.normal)
            
            circs = solve(self.influences, B)
        if append:
            self.circs = circs
        return circs
        
    def c_lift(self, V):
        """
        Calculate the lift
        """
        try:
            circs = self.circs[:-1]
            
        except AttributeError:
            V = np.asarray(V)
            V_norm = np.linalg.norm(V)
            circs = self.solve_circs(V)
        return np.sum(V_norm*circs)*2
        
    def lift(self, V, rho, unsteady = False, wake = None, prevcirc = None, rot = None):
        """
        Calculates the lift force by calling the pressure vectors function
        """
        plist = []
        if unsteady:
            V = V + rot.T
        
        coordv = np.array([0,1])
        try:
            if unsteady:
                circs = self.circs[:-1]
            else:
                circs = self.circs
        except AttributeError:
            if unsteady:
                circs = self.solve_circs(V, unsteady = unsteady, wake = wake, prevcirc = prevcirc, rot = rot)[:,-1]
            else:
                circs = self.solve_circs(V)
        for i, panel in enumerate(self.panels):
            dp = rho* (np.dot(V,panel.tangential)*circs[i]/panel.length + (np.sum(circs[:i+1])-np.sum(prevcirc[:i+1]))/self.dt)
            plist.append(dp*panel.length*np.dot(panel.normal, coordv))
        plist = np.array(plist)
        
        return np.sum(plist)
    
    def apply_circs(self,V):
        """
        Applies the circulations to the panels in panellist
        """
        try:
            for i, panel in enumerate(self.panels):
                panel.vortex.strength = self.circs[i]
            
        except AttributeError:
           circs = self.solve_circs(V)
           for i, panel in enumerate(self.panels):
               panel.vortex.strength = circs[i] 
        
    def pressurefield(self, V, rho, region, velfield = False, resolution = 0.01):
        """
        Calculates the pressurefield in a region
        There are two options:
            1: Input a velocity field (handy if you need the velocityfield separately)
            2: Calculate a velocity field 
        Make sure the circultions are applied with the apply_circs method
        
        NOTE: it works but the vortices make the pressure drop really low at the vortex locations.
        """
        if np.any(velfield):
            print("Using input velocity field")
        else:
            print("Calculating the velocity field")
            grid, velfield = self.velocityfield(V, region, resolution)
            
        velocity = velfield[:,:,0]**2+velfield[:,:,1]**2
        return 0.5*rho*(np.linalg.norm(V)**2 - velocity)
        
    def pressurevectors(self, V, rho):
        """
        Calculates the pressure on each panel element.
        
        Make sure the circulations are applied with the apply_circs method
        """
        Press = np.zeros((self.dim,2))
        V_n = np.linalg.norm(V)
        for i, panel in enumerate(self.panels):
            Press[i] = V_n*rho*panel.vortex.strength/panel.length*panel.normal
        
        return Press
    
    def velocityfield(self, V, region, resolution = 0.01):
        """
        Calculates the velocityfield for a region
        Make sure the circultions are applied with the apply_circs method
        
        region = grid where velocities are evaluated
        """
        region = np.asarray(region)
        dim = ((region[1] - region[0])/resolution).astype(int)
        #dim = tuple(np.append(dim, 2))
        velocity = np.zeros((dim[1],dim[0],2))
        #create grid
        xcoords = np.linspace(region[0,0], region[1,0], num = dim[0])
        ycoords = np.linspace(region[0,1], region[1,1], num = dim[1])
        #Calcualte induced velocities
        for i, x in enumerate(xcoords):
            for j, y in enumerate(ycoords):
                ind_vel = 0
                for panel in self.panels:
                    ind_vel += panel.velind([x,y])
                velocity[j,i,:] += ind_vel+V
                
        return (xcoords, ycoords), velocity

    def vel_ind(self, point):
        """
        Calculates the induced velocity from the whole plate on a point
        """
        v_ind = 0
        for panel in self.panels:
            v_ind += panel.velind(point)
        return v_ind

    def unit_circs(self):
        """
        Sets all circulations to 1
        """
        for panel in self.panels:
            panel.vortex.strength = 1
        