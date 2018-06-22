#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:28:15 2018

@author: Simon

2D panel code test
"""

import numpy as np


class Panel:
    """
    Class for a single panel
    """
    
    def __init__(self, start, end, strength):
        start = np.asarray(start)
        end = np.asarray(end)
        self.start = start
        
        self.end = end
        
        self.strength = strength
        
        vect = end-start
        
        self.vpoint = start + 0.25*(vect)
        
        self.cpoint = start + 0.75*(vect)
        
        beta = np.arctan2(vect[1],vect[0])
        
        self.normal = np.array([-np.sin(beta), np.cos(beta)])
        
        self.length = np.linalg.norm(vect)
    
    def velind(self, point, r = 0.005):
        '''
        Calculate the induced velocity at a point
        '''
        point = np.asarray(point)
        relpos = point-self.vpoint
        
        r_sq = np.linalg.norm(relpos)**2
        mat = np.array([[0,1],[-1,0]])
        
        v_ind_r = self.strength/(2*np.pi*r_sq)*mat.dot(relpos)
        #v_ind_R = self.strength/(2*np.pi*(r**2))*mat.dot(relpos)
        return v_ind_r
        
       
class Plate:
    """
    Class for a plate
    """
    
    def __init__(self, coordlist, connect = False):
        
        panellist = []
        if connect:
            end = 0
        else:
            end=1
        
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
        
    def solve_circs(self, V):
        """
        Solve for the circulaations for a certain velocity
        """
        V = np.asarray(V)
        B = np.zeros(self.dim)
        for i, panel in enumerate(self.panels):
            B[i] = np.dot(-V, panel.normal)
        
        circs = np.linalg.solve(self.influences, B)
        return circs
        
    def c_lift(self, V):
        """
        Calculate the lift
        """
        V = np.asarray(V)
        V_norm = np.linalg.norm(V)
        circs = self.solve_circs(V)
        return np.sum(V_norm*circs)*2
        
        
    def apply_circs(self,V):
        """
        Applies the circulations to the panels in panellist
        """
        try:
            for i, panel in enumerate(self.panels):
                panel.strength = self.circs[i]
            
        except AttributeError:
           circs = self.solve_circs(V)
           for i, panel in enumerate(self.panels):
               panel.strength = circs[i] 
        
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
            Press[i] = V_n*rho*panel.strength/panel.length*panel.normal
        
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
                
        return (xcoords, ycoords),velocity

        

    def unit_circs(self):
        """
        Sets all circulations to 1
        """
        for panel in self.panels:
            panel.strength = 1
        