#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:33:00 2018

@author: Simon

Testing ground for Panel code
"""

import numpy as np
from Panel_blade import Panel, Plate, Vortex
import matplotlib.pyplot as plt
from matplotlib import cm
import types

def naca_camber(x, m, p):
    """
    Calculates the camber line for a NACA-4 series airfoil
    An airfoil is given by mpxx
    This funcion only uses the first two digits, m and p
    
    x is assumed to be the non dimensional quantity x/c
    """
    m /= 100.
    p /= 10.
    
    camber = np.where(x<p, m/(p**2)*(2*p*x-x**2), m/((1-p)**2)*((1-2*p)+2*p*x-x**2))
    
    return camber

def flap(N, xf, af, profile = None, profileargs = None):
    """
    Creates a flat plate with a flap hinged at 'xf' with an angle of 'af' (in degrees)
    It will be discretised in N panels
    
    Inputs
    ------
    
    N = int, amount of panels
    xf = float, nndimensional location of the flap
    af = float, angle of the flap in degrees, downward is considered positive
    
    If a profile function is given the needed arguments can be given in profileargs
    """
    af *= np.pi/180
    N += 1 #One point more than the amount of panels is needed
    plate = np.zeros((N,2))
    #calculating the x-coordinates
    Nflat = int(N*xf)
    Nflap = N - Nflat
    flat = np.linspace(0,xf, Nflat+1)
    flap0 = np.linspace(xf, 1, Nflap)
    if type(profile) == type(None):
        flap = np.cos(af)*(flap0-flap0[0])
        y_flat = np.zeros(Nflat+1)
        y_flap = -np.sin(af)*(flap0-flap0[0])
    elif type(profile) == types.FunctionType:
        if type(profileargs) == type(None):
            flap = np.cos(af)*(flap0-flap0[0])
            y_flat = profile(flat, *profileargs)
            y_flap = profile(flap0, *profileargs)-np.sin(af)*(flap0-flap0[0])
        else:
            flap = np.cos(af)*(flap0-flap0[0])
            y_flat = profile(flat)
            y_flap = profile(flap0)-np.sin(af)*(flap0-flap0[0])
    else:
        raise ValueError("{} is not a function or 'None'".format(profile))
        
    plate[:,0] = np.concatenate((flat, flap[1:]+flap0[0]))
    plate[:,1] = np.concatenate((y_flat, y_flap[1:]))
    return plate
    
    

rho = 1.225
data = np.genfromtxt('naca2412.dat', skip_header=1)

N_panels = 15
x = np.linspace(0,1,num = N_panels+1)
coordlist_test = np.zeros((N_panels+1,2))
coordlist_test[:,0] = x
coordlist_test[:,1] = naca_camber(x,2,4)
coords = flap(N_panels, 0.75, 20)

alpha = 2*np.pi/180
V = 1*np.array([np.cos(alpha), np.sin(alpha)])
foil = Plate(coords)
cl_xfoil = 0.4969

#Extract the coordinates of the points where the pressures should attach
P_coords = np.zeros((foil.dim, 2))
normals = np.zeros((foil.dim, 2))

for i, panel in enumerate(foil.panels):
    P_coords[i] = panel.vpoint
    normals[i] = panel.normal

print(foil.c_lift(V))

circs = foil.solve_circs(V, append = True)
foil.apply_circs(V)
Press = foil.pressurevectors(V, rho)

region = np.array([[-0.25,-0.5],[1.25,0.5]])

grid, velfield = foil.velocityfield(V, region, resolution = 0.005)
pressfield = foil.pressurefield(V, rho, region, velfield = velfield, resolution = 0.005)


#Plot streamplot

fig = plt.figure(figsize=(6,4.5))
ax = fig.add_subplot(111)
ax.plot(coords[:,0], coords[:,1], '-k')
"""
plt.axis('equal')
ax.quiver(P_coords[:,0], P_coords[:,1], normals[:,0], normals[:,1])
"""

speed = np.sqrt(velfield[:,:,0]**2 + velfield[:,:,1]**2)
lw = speed/speed.max()
norm = cm.colors.Normalize(vmax=abs(pressfield).max(), vmin=-abs(pressfield).max())
#ax.imshow(pressfield, origin='lower', extent = (-0.25,1.25, -0.5, 0.5), alpha = 0.5, norm = norm)
ax.streamplot(grid[0], grid[1], velfield[:,:,0], velfield[:,:,1])

