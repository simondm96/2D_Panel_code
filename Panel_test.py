#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:33:00 2018

@author: Simon

Testing ground for Panel code
"""

import numpy as np
from Panel_blade import Panel, Plate
import matplotlib.pyplot as plt
from matplotlib import cm

rho = 1.225
data = np.genfromtxt('naca2412.dat', skip_header=1)

N_panels = 10

coordlist_test = np.zeros((N_panels+1,2))
coordlist_test[:,0] = np.linspace(0,1,num = N_panels+1)

coords = coordlist_test

alpha = 5*np.pi/180
V = 1*np.array([np.cos(alpha), np.sin(alpha)])
foil = Plate(coordlist_test)
cl_xfoil = 0.4969

#Extract the coordinates of the points where the pressures should attach
P_coords = np.zeros((foil.dim,2))
print(P_coords.shape)
for i, panel in enumerate(foil.panels):
    P_coords[i] = panel.vpoint

print(foil.c_lift(V))

circs = foil.solve_circs(V)
foil.apply_circs(V)
Press = foil.pressurevectors(V, rho)

region = np.array([[-0.25,-0.5],[1.25,0.5]])
"""
grid, velfield = foil.velocityfield(V, region, resolution = 0.0005)
pressfield = foil.pressurefield(V, region, velfield = velfield, resolution = 0.0005)
"""

#Plot streamplot

fig = plt.figure(figsize=(6,4.5))
ax = fig.add_subplot(111)
ax.plot(coords[:,0], coords[:,1], '-k')
"""
speed = np.sqrt(velfield[:,:,0]**2 + velfield[:,:,1]**2)
lw = speed/speed.max()
norm = cm.colors.Normalize(vmax=abs(pressfield).max(), vmin=-abs(pressfield).max())
#ax.imshow(pressfield, origin='lower', extent = (-0.25,1.25, -0.5, 0.5), alpha = 0.5, norm = norm)
ax.streamplot(grid[0], grid[1], velfield[:,:,0], velfield[:,:,1])
"""
ax.quiver(P_coords[:,0], P_coords[:,1], Press[:,0], Press[:,1])