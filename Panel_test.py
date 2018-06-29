#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:33:00 2018

@author: Simon

Testing ground for Panel code
"""

import numpy as np
from Panel_blade import Panel, Plate, Vortex, Wake
import matplotlib.pyplot as plt
from matplotlib import cm
import types
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern'], 'size':10})
rc('text', usetex=True)

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
    
    
def sine_motion(k):
    """
    Simulating a flat plate in a sinusoidal pitching motion.
    """
    #Setting up the input variables
    N_panels = 4
    unsteady = True
    x = np.linspace(0,1,num = N_panels+1)
    coordlist = np.zeros((N_panels+1,2))
    coordlist[:,0] = x
    Q = 1 #Speed
    rho = 1.225 #Density (for pressure)
    #k = 0.02 #Reduced frequency
    omega = k*2*Q
    alpha_0 = 2*np.pi/180 #Mean vaue of the motion
    Amp = 2*np.pi/180 #Amplitude of the motion
    
    q_inf = 0.5*rho*Q**2
    
    T = 6*np.pi/omega #Total time in seconds for two oscillations
    dt = 0.06/omega #Timestep in seconds
    
    t = np.arange(0,T+dt, dt)
    
    #Pitch angle and derivative
    theta = Amp*np.sin(omega*t) + alpha_0
    thetadot = Amp*omega*np.cos(omega*t)
    
    #X and Z-possition and derivatives (speeds)
    xdot = - Q
    zdot = 0
    
    X = xdot*t
    Z = zdot*t
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    vel = np.array([xdot, zdot])
    cllist = []
    #1st timestep needs to be outside the loop because we define our foil along the way
    circ = 0
    
    wake = Wake()
    prevcircs = circ
    mat = np.array([[cos[0], -sin[0]], [sin[0], cos[0]]])
    rotpart = thetadot[0] * np.array([-coordlist[:,1],coordlist[:,0]])
    vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt) #0.25 distance traveled by the TE
    
    #Define foil
    foil = Plate(coordlist, unsteady = unsteady, vortexpoint = vpoint)
    
    v_plate = np.dot(mat, -vel)
    circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = 0, rot = rotpart, append = True)
    cl = foil.lift(v_plate, rho, prevcirc = np.zeros(N_panels), rot = rotpart)/(q_inf)
    cllist.append(cl)
    
    pos = np.array([X[0], Z[0]])
    new_vortex = Vortex(pos+ vpoint, circ[-1])
    wake.add_vortex(new_vortex)
    wake.wake_rollup(foil, pos, dt)
    for i, time in enumerate(t):
        #We calculated the first timestep outside of the loop, this makes indexing readable
        if i==0:
            continue
        pos = np.array([X[i], Z[i]])
        prevcircs = circ
        mat = np.array([[cos[i], -sin[i]], [sin[i], cos[i]]])
        rotpart = thetadot[i] * np.array([-coordlist[:,1],coordlist[:,0]])
        v_plate = np.dot(mat, vel)
        vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt)
        foil.Construct_unsteady(vpoint)
        circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = prevcircs, rot = rotpart, append = True)
        cl = foil.lift(v_plate, rho,  prevcirc = prevcircs, rot = rotpart)/(q_inf)
        cllist.append(cl)
        pos = np.array([X[i], Z[i]])
        new_vortex = Vortex(pos + vpoint, circ[-1])
        wake.add_vortex(new_vortex)
        wake.wake_rollup(foil, pos, dt)
        
    cllist = np.array(cllist)
    
    return theta, cllist
    
    
def flat_plate_steady():
    """
    Simulating a flat plate in steady flow
    """
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
    
def flap_flapping(k):
    """
    Simulating a flat plate with a flap flapping in a sinusoidal motion.
    """
    #Setting up the input variables
    N_panels = 4
    unsteady = True
    Q = 1 #Speed
    rho = 1.225 #Density (for pressure)
    #k = 0.02 #Reduced frequency
    xflap = 0.75
    omega = k*2*Q/(1-xflap)
    alpha_0 = 10*np.pi/180 #Mean vaue of the motion
    Amp = 10*np.pi/180 #Amplitude of the motion
    
    q_inf = 0.5*rho*Q**2
    
    T = 6*np.pi/omega #Total time in seconds for three oscillations
    dt = 0.06/omega #Timestep in seconds
    
    t = np.arange(0,T+dt, dt)
    
    #Flap pitching motion
    flapping = Amp*np.sin(omega*t-np.pi/2) + alpha_0
    #Pitch angle and derivative
    theta = np.zeros(t.shape)
    thetadot = np.zeros(t.shape)
    coordlist = flap(N_panels, xflap, flapping[0]*180/np.pi)
    
    #X and Z-possition and derivatives (speeds)
    xdot = - Q
    zdot = 0
    
    X = xdot*t
    Z = zdot*t
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    vel = np.array([xdot, zdot])
    cllist = []
    #1st timestep needs to be outside the loop because we define our foil along the way
    circ = 0
    
    wake = Wake()
    prevcircs = circ
    mat = np.array([[cos[0], -sin[0]], [sin[0], cos[0]]])
    rotpart = thetadot[0] * np.array([-coordlist[:,1],coordlist[:,0]])
    vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt) #0.25 distance traveled by the TE
    
    #Define foil
    foil = Plate(coordlist, unsteady = unsteady, vortexpoint = vpoint)
    
    v_plate = np.dot(mat, -vel)
    circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = 0, rot = rotpart, append = True)
    cl = foil.lift(v_plate, rho, prevcirc = np.zeros(N_panels), rot = rotpart)/(q_inf)
    cllist.append(cl)
    
    pos = np.array([X[0], Z[0]])
    new_vortex = Vortex(pos+ vpoint, circ[-1])
    wake.add_vortex(new_vortex)
    wake.wake_rollup(foil, pos, dt)
    for i, time in enumerate(t):
        #We calculated the first timestep outside of the loop, this makes indexing readable
        if i==0:
            continue
        oldcoords = coordlist
        coordlist = flap(N_panels, xflap, flapping[i]*180/np.pi)
        diffcoords = (coordlist - oldcoords)/dt
        #Update foil geometry
        foil.Update_geo(coordlist)
        pos = np.array([X[i], Z[i]])
        prevcircs = circ
        mat = np.array([[cos[i], -sin[i]], [sin[i], cos[i]]])
        rotpart = thetadot[i] * np.array([-coordlist[:,1],coordlist[:,0]-diffcoords[:,1]])
        v_plate = np.dot(mat, vel)
        vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt)
        foil.Construct_unsteady(vpoint)
        circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = prevcircs, rot = rotpart, append = True)
        cl = foil.lift(v_plate, rho,  prevcirc = prevcircs, rot = rotpart)/(q_inf)
        cllist.append(cl)
        pos = np.array([X[i], Z[i]])
        new_vortex = Vortex(pos + vpoint, circ[-1])
        wake.add_vortex(new_vortex)
        wake.wake_rollup(foil, pos, dt)
        
    cllist = np.array(cllist)
    
    return flapping, cllist

def gust():
    """
    Simulating a flat plate with a flap flapping in a sinusoidal motion.
    """
    #Setting up the input variables
    N_panels = 4
    unsteady = True
    U_inf = 1 #Speed
    V_inf = 0.2 #upwards gust
    rho = 1.225 #Density (for pressure)
    t_gust = 5
    T = 20 #Total time in seconds 
    dt = 0.1 #Timestep in seconds
    
    t = np.arange(0,T+dt, dt)
    
    #Flap pitching motion
    x = np.linspace(0,1,num = N_panels+1)
    coordlist = np.zeros((N_panels+1,2))
    coordlist[:,0] = x
    
    #Theta mmotion
    theta = np.zeros(t.shape)
    thetadot = np.zeros(t.shape)
    #X and Z-possition and derivatives (speeds)
    xdot = U_inf
    zdot = np.where(t<t_gust, 0, V_inf)
    q_inf = 0.5*rho*(xdot**2+zdot**2)
    
    X = xdot*t
    Z = zdot*t
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    vel = np.array([xdot, zdot])
    cllist = []
    #1st timestep needs to be outside the loop because we define our foil along the way
    circ = 0
    
    wake = Wake()
    prevcircs = circ
    mat = np.array([[cos[0], -sin[0]], [sin[0], cos[0]]])
    rotpart = thetadot[0] * np.array([-coordlist[:,1],coordlist[:,0]])
    vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt) #0.25 distance traveled by the TE
    
    #Define foil
    foil = Plate(coordlist, unsteady = unsteady, vortexpoint = vpoint)
    
    v_plate = np.dot(mat, -vel)
    circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = 0, rot = rotpart, append = True)
    cl = foil.lift(v_plate, rho, prevcirc = np.zeros(N_panels), rot = rotpart)/(q_inf[0])
    cllist.append(cl)
    
    pos = np.array([X[0], Z[0]])
    new_vortex = Vortex(pos+ vpoint, circ[-1])
    wake.add_vortex(new_vortex)
    wake.wake_rollup(foil, pos, dt)
    for i, time in enumerate(t):
        #We calculated the first timestep outside of the loop, this makes indexing readable
        if i==0:
            continue
        pos = np.array([X[i], Z[i]])
        prevcircs = circ
        mat = np.array([[cos[i], -sin[i]], [sin[i], cos[i]]])
        rotpart = thetadot[i] * np.array([-coordlist[:,1],coordlist[:,0]])
        v_plate = np.dot(mat, vel)
        vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt)
        foil.Construct_unsteady(vpoint)
        circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = prevcircs, rot = rotpart, append = True)
        cl = foil.lift(v_plate, rho,  prevcirc = prevcircs, rot = rotpart)/(q_inf[i])
        cllist.append(cl)
        pos = np.array([X[i], Z[i]])
        new_vortex = Vortex(pos + vpoint, circ[-1])
        wake.add_vortex(new_vortex)
        wake.wake_rollup(foil, pos, dt)
        
    cllist = np.array(cllist)
    
    return t, cllist

def vortex_conv(start_pos):
    """
    Simulating a flat plate with a flap flapping in a sinusoidal motion.
    """
    #Setting up the input variables
    N_panels = 4
    unsteady = True
    U_inf = 1 #Speed
    rho = 1.225 #Density (for pressure)
    T = 20 #Total time in seconds 
    dt = 0.1 #Timestep in seconds
    
    t = np.arange(0,T+dt, dt)

    x = np.linspace(0,1,num = N_panels+1)
    coordlist = np.zeros((N_panels+1,2))
    coordlist[:,0] = x
    #Theta
    theta = np.zeros(t.shape)
    thetadot = np.zeros(t.shape)
    #X and Z-possition and derivatives (speeds)
    xdot = U_inf
    zdot = 0
    q_inf = 0.5*rho*(xdot**2+zdot**2)
    
    X = xdot*t
    Z = zdot*t
    
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    vel = np.array([xdot, zdot])
    cllist = []
    #1st timestep needs to be outside the loop because we define our foil along the way
    circ = 0
    
    wake = Wake()
    prevcircs = circ
    mat = np.array([[cos[0], -sin[0]], [sin[0], cos[0]]])
    rotpart = thetadot[0] * np.array([-coordlist[:,1],coordlist[:,0]])
    vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt) #0.25 distance traveled by the TE
    
    #Define foil
    foil = Plate(coordlist, unsteady = unsteady, vortexpoint = vpoint)
    
    v_plate = np.dot(mat, -vel)
    circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = 0, rot = rotpart, append = True)
    cl = foil.lift(v_plate, rho, prevcirc = np.zeros(N_panels), rot = rotpart)/(q_inf[0])
    cllist.append(cl)
    
    pos = np.array([X[0], Z[0]])
    new_vortex = Vortex(pos+ vpoint, circ[-1])
    wake.add_vortex(new_vortex)
    wake.wake_rollup(foil, pos, dt)
    for i, time in enumerate(t):
        #We calculated the first timestep outside of the loop, this makes indexing readable
        if i==0:
            continue
        pos = np.array([X[i], Z[i]])
        prevcircs = circ
        mat = np.array([[cos[i], -sin[i]], [sin[i], cos[i]]])
        rotpart = thetadot[i] * np.array([-coordlist[:,1],coordlist[:,0]])
        v_plate = np.dot(mat, vel)
        vpoint = coordlist[-1] + 0.25*(coordlist[-1] - vel*dt - rotpart[:,-1]*dt)
        foil.Construct_unsteady(vpoint)
        circ = foil.solve_circs(v_plate, unsteady = unsteady, wake = wake, prevcirc = prevcircs, rot = rotpart, append = True)
        cl = foil.lift(v_plate, rho,  prevcirc = prevcircs, rot = rotpart)/(q_inf[i])
        cllist.append(cl)
        pos = np.array([X[i], Z[i]])
        new_vortex = Vortex(pos + vpoint, circ[-1])
        wake.add_vortex(new_vortex)
        wake.wake_rollup(foil, pos, dt)
        
    cllist = np.array(cllist)
    
    return t, cllist

"""
ks = [0.02, 0.05, 0.1]
clist = []
thetalist = []
for k in ks:
    theta, cllist = sine_motion(k)
    clist.append(cllist)
    thetalist.append(theta)


    
fig = plt.figure(figsize=(6,4.5))
ax = fig.add_subplot(111)
for i in range(3):
    
    ax.plot(thetalist[i]*180/np.pi, clist[i], label = str(ks[i]))
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$C_L$")
ax.legend()
plt.savefig("sine_motion.eps")
"""

theta, cllist = flap_flapping(0.02)

fig = plt.figure(figsize=(6,4.5))
ax = fig.add_subplot(111)
ax.plot(theta*180/np.pi, cllist)
ax.set_xlabel(r"Flap deflection angle $\beta$")
ax.set_ylabel(r"$C_L$")

plt.savefig("flap_motion.eps")