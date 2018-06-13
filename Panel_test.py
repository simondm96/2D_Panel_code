#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:33:00 2018

@author: Simon

Testing ground for Panel code
"""

import numpy as np
from Panel_blade import Panel, Plate

data = np.genfromtxt('naca2412.dat', skip_header=1)

coordlist_test = [[0,0],[0.5,0], [0.75,0], [1,-0.1]]

alpha = 2*np.pi/180
V = np.array([np.cos(alpha), np.sin(alpha)])
foil = Plate(coordlist_test)
cl_xfoil = 0.4969


print(foil.c_lift(V))