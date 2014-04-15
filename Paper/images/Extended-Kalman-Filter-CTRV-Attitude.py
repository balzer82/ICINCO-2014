# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import Image as ImageDisp
from sympy import Symbol, symbols, Matrix, sin, cos, latex, Plot
from sympy.interactive import printing
printing.init_printing()
%pylab inline --no-import-all

# <headingcell level=1>

# Adaptive Extended Kalman Filter Implementation for Constant Turn Rate and Velocity (CTRV) Vehicle Model with Attitude Estimation in Python

# <markdowncell>

# Situation covered: You have an velocity sensor which measures the vehicle speed ($v$) in heading direction ($\psi$) and a yaw rate sensor ($\dot \psi$) which both have to fused with the position ($x$ & $y$) from a GPS sensor in loosely coupled way.

# <headingcell level=2>

# State Vector - Constant Turn Rate and Velocity Vehicle Model (CTRV) + Roll and Pitch Estimation

# <markdowncell>

# $$x_k= \left[\begin{matrix}x\\y\\\psi\\v\\\dot\psi\\\phi\\\dot\phi\\\Theta\\\dot\Theta\end{matrix}\right] = \left[ \matrix{ \text{Position X} \\ \text{Position Y} \\ \text{Heading} \\ \text{Velocity} \\ \text{Yaw Rate} \\ \text{Pitch} \\ \text{Pitchrate} \\ \text{Roll} \\ \text{Rollrate}} \right]$$

# <codecell>

numstates=9 # States

# <codecell>

dt = 1.0/50.0 # Sample Rate of the Measurements is 50Hz
dtGPS=1.0/10.0 # Sample Rate of GPS is 10Hz

# <markdowncell>

# All symbolic calculations are made with [Sympy](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-5-Sympy.ipynb). Thanks!

# <codecell>

vs, psis, dpsis, dts, xs, ys, phis, dphis, thetas, dthetas, Lats, Lons = \
 symbols('v \psi \dot\psi T x y \phi \dot\phi \Theta \dot\Theta Lat Lon')

As = Matrix([[xs+(vs/dpsis)*(sin(psis+dpsis*dts)-sin(psis))],
             [ys+(vs/dpsis)*(-cos(psis+dpsis*dts)+cos(psis))],
             [psis+dpsis*dts],
             [vs],
             [dpsis],
             [phis+dphis*dts],
             [dphis],
             [thetas+dthetas*dts],
             [dthetas]])
state = Matrix([xs,ys,psis,vs,dpsis,phis,dphis,thetas,dthetas])

# <headingcell level=2>

# Initial Uncertainty

# <markdowncell>

# Initialized with $0$ means you are pretty sure where the vehicle starts

# <codecell>

P = 1000.0*np.eye(numstates)
print(P.shape)

fig = plt.figure(figsize=(numstates/2, numstates/2))
im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Initial Covariance Matrix $P$')
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(10))
# set the locations and labels of the yticks
plt.yticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(10))
# set the locations and labels of the yticks
plt.xticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

plt.xlim([-0.5,8.5])
plt.ylim([8.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()

# <headingcell level=2>

# Dynamic Matrix

# <markdowncell>

# This formulas calculate how the state is evolving from one to the next time step

# <codecell>

As

# <codecell>

print latex(As)

# <headingcell level=3>

# Calculate the Jacobian of the Dynamic Matrix with respect to the state vector

# <codecell>

state

# <codecell>

As.jacobian(state)

# <markdowncell>

# It has to be computed on every filter step because it consists of state variables.

# <headingcell level=2>

# Control Input

# <markdowncell>

# Matrix G is the Jacobian of the Dynamic Matrix with respect to control (the translation velocity $v$ and the rotational
# velocity $\dot \psi$).

# <codecell>

control = Matrix([vs,dpsis,dphis,dthetas])
control

# <headingcell level=3>

# Calculate the Jacobian of the Dynamic Matrix with Respect to the Control

# <codecell>

Gs=As.jacobian(control)
Gs

# <markdowncell>

# It has to be computed on every filter step because it consists of state variables.

# <codecell>

print latex(Gs)

# <headingcell level=2>

# Process Noise Covariance Matrix $Q$

# <codecell>

svQs, syQs, spQs, srQs = \
 symbols('\sigma_v \sigma_{\dot\psi} \sigma_{\dot\phi} \sigma_{\dot\Theta}')

Qs = Matrix([[svQs**2, 0.0, 0.0, 0.0],
             [0.0, syQs**2, 0.0, 0.0],
             [0.0, 0.0, spQs**2, 0.0],
             [0.0, 0.0, 0.0, srQs**2]])
Qs

# <markdowncell>

# Matrix Q is the expected noise on the State.
# 
# One method is based on the interpretation of the matrix as the weight of the dynamics prediction from the state equations
# Q relative to the measurements.
# 
# As you can see in [Schubert, R., Adam, C., Obst, M., Mattern, N., Leonhardt, V., & Wanielik, G. (2011). Empirical evaluation of vehicular models for ego motion estimation. 2011 IEEE Intelligent Vehicles Symposium (IV), 534–539. doi:10.1109/IVS.2011.5940526] one can assume the velocity process noise for a vehicle with $\sigma_v=1.5m/s$ and the yaw rate process noise with $\sigma_\psi=0.29rad/s$, when a timestep takes 0.02s (50Hz).

# <codecell>

control

# <codecell>

amax = 5.0    # m/s2
yawrateaccmax = 100.0  # Grad/s2
pitchrateaccmax=300.0  # Grad/s2
rollrateaccmax =1000.0 # Grad/s2

svQ = (amax*dt)      # Velocity
syQ = (yawrateaccmax*np.pi/180.0*dt)   # Yawrate
spQ = (pitchrateaccmax*np.pi/180.0*dt)   # Pitchrate
srQ = (rollrateaccmax*np.pi/180.0*dt) # Rollrate

Q = np.matrix([[svQ**2, 0.0, 0.0, 0.0],
               [0.0, syQ**2, 0.0, 0.0],
               [0.0, 0.0, spQ**2, 0.0],
               [0.0, 0.0, 0.0, srQ**2]])

# <codecell>

print svQ, syQ, spQ, srQ

# <codecell>

fig = plt.figure(figsize=(4, 4))
im = plt.imshow(Q, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Process Noise Covariance Matrix $Q$')
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(5))
# set the locations and labels of the yticks
plt.yticks(np.arange(4), \
           ('$v$', '$\dot \psi$', '$\dot \phi$', '$\dot \Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(5))
# set the locations and labels of the yticks
plt.xticks(np.arange(4), \
           ('$v$', '$\dot \psi$', '$\dot \phi$', '$\dot \Theta$'),\
           fontsize=22)

plt.xlim([-0.5,3.5])
plt.ylim([3.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()

# <headingcell level=2>

# Real Measurements

# <codecell>

#path = './../RaspberryPi-CarPC/TinkerDataLogger/DataLogs/2014/'
#datafile = path+'2014-02-21-002-Data.csv'
datafile = '2014-03-26-000-Data.csv'

date, \
time, \
millis, \
ax, \
ay, \
az, \
rollrate, \
pitchrate, \
yawrate, \
roll, \
pitch, \
yaw, \
speed, \
course, \
latitude, \
longitude, \
altitude, \
pdop, \
hdop, \
vdop, \
epe, \
fix, \
satellites_view, \
satellites_used, \
temp = np.loadtxt(datafile, delimiter=',', unpack=True, skiprows=1)

print('Read \'%s\' successfully.' % datafile)

# A course of 0° means the Car is traveling north bound
# and 90° means it is traveling east bound.
# In the Calculation following, East is Zero and North is 90°
# We need an offset.
course =(-course+90.0)

# <codecell>

# Display GPS Heatmap from Disk
gpsheatmap = ImageDisp(filename='2014-03-26-000-Map.png')
gpsheatmap

# <markdowncell>

# Map tiles by [Stamen Design](http://stamen.com/), under [CC BY 3.0](http://creativecommons.org/licenses/by/3.0/). Data by [OpenStreetMap](http://www.openstreetmap.org/), under [CC BY SA](http://creativecommons.org/licenses/by-sa/3.0/).

# <headingcell level=3>

# EPE

# <codecell>

plt.figure(figsize=(16,3))
plt.plot(epe, label='$EPE$ from GNSS modul', marker='*', markevery=50)
plt.ylabel('$EPE$ in $(m)$')
plt.xlabel('Filterstep $k$')
plt.legend(loc='best')
#plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.png', dpi=72, transparent=True, bbox_inches='tight')
plt.savefig('Extended-Kalman-Filter-CTRV-EPE.eps', bbox_inches='tight')

# <headingcell level=3>

# Static Gain

# <codecell>

pitch = pitch - 10.107
roll = roll - 1.157
#ax = ax - 1.757
rollrate = rollrate + 1.58
pitchrate = pitchrate+2.897

# <headingcell level=2>

# Roll-/Pitch-/Yawrate

# <codecell>

plt.figure(figsize=(14,4))
#plt.plot(roll)
plt.plot(pitchrate)
plt.plot(rollrate)
plt.plot(yawrate)

# <headingcell level=2>

# Roll-/Pitch-/Yawacceleration

# <codecell>

plt.figure(figsize=(14,4))
#plt.plot(roll)
plt.semilogy(np.diff(pitchrate)/dt, label='$\mathrm{d}\dot\phi/\mathrm{d}t$')
plt.plot(np.diff(rollrate)/dt, label='$\mathrm{d}\dot\Theta/\mathrm{d}t$')
plt.plot(np.diff(yawrate)/dt, label='$\mathrm{d}\dot\psi/\mathrm{d}t$')
plt.legend()

# <codecell>

# clamp speed and yawrate to zero while standing still
#speed[speed<3.0]=0.0
#yawrate[speed<3.0]=0.0

# <headingcell level=3>

# Lat/Lon to Meters

# <codecell>

RadiusEarth = 6378388.0 # m
arc= 2.0*np.pi*RadiusEarth/360.0 # m/°

dx = arc * np.cos(latitude*np.pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
dy = arc * np.hstack((0.0, np.diff(latitude))) # in m

mx = np.cumsum(dx)
my = np.cumsum(dy)

ds = np.sqrt(dx**2+dy**2)

GPS=np.hstack((True, (np.diff(ds)>0.0).astype('bool'))) # GPS Trigger for Kalman Filter

# <headingcell level=2>

# Measurement Noise Covariance Matrix $R$ (Adaptive)

# <codecell>

spxs, spys, srs, sps = \
 symbols('\sigma_x \sigma_y \sigma_\phi \sigma_\Theta')

Rs = Matrix([[spxs**2, 0.0, 0.0, 0.0],
             [0.0, spys**2, 0.0, 0.0],
             [0.0, 0.0, srs**2, 0.0],
             [0.0, 0.0, 0.0, sps**2]])
Rs

# <codecell>

print latex(Rs)

# <markdowncell>

# "In practical use, the uncertainty estimates take on the significance of relative weights of state estimates and measurements. So it is not so much important that uncertainty is absolutely correct as it is that it be relatively consistent across all models" - Kelly, A. (1994). A 3D state space formulation of a navigation Kalman filter for autonomous vehicles, (May). Retrieved from http://oai.dtic.mil/oai/oai?verb=getRecord&metadataPrefix=html&identifier=ADA282853

# <codecell>

R = np.matrix([[6.0**2, 0.0, 0.0, 0.0],
               [0.0, 6.0**2, 0.0, 0.0],
               [0.0, 0.0, 100.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

# <headingcell level=3>

# Position

# <markdowncell>

# $R$ is just initialized here. In the Kalman Filter Step it will calculated dynamically with the $EPE$ (Estimated Position Error) from the GPS signal as well as depending on the $speed$, like proposed in [Wender, S. (2008). Multisensorsystem zur erweiterten Fahrzeugumfelderfassung. Retrieved from http://vts.uni-ulm.de/docs/2008/6605/vts_6605_9026.pdf P.108].

# <markdowncell>

# $\sigma_p^2 = \sigma_\text{speed}^2 + \sigma_\text{EPE}^2$
# 
# with 
# 
# $\sigma_v = (v+\epsilon)^{-\xi}$
# 
# $\sigma_\text{EPE} = \zeta \cdot EPE$

# <codecell>

epsilon = 0.1
xi      = 500.0
zeta    = 50.0

spspeed=xi/((speed/3.6)+epsilon)
spepe=zeta*epe
sp = (spspeed)**2 + (spepe)**2

# <codecell>

plt.figure(figsize=(6,2))
plt.semilogy(spspeed**2, label='$\sigma_{x/y}$ from speed', marker='*', markevery=150)
plt.semilogy(spepe**2, label='$\sigma_{x/y}$ from EPE', marker='x', markevery=150)
plt.semilogy(sp, label='Res.', marker='o', markevery=150)
plt.ylabel('Values for $R$ Matrix')
plt.xlabel('Filterstep $k$')
#plt.legend(loc='best')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0.)
#plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.png', dpi=72, transparent=True, bbox_inches='tight')
plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.eps', bbox_inches='tight')

# <headingcell level=3>

# Attitude

# <markdowncell>

# Because the estimation of Roll and Pitch is only valid for quasistatic situations (which is not valid for a moving vehicle), the values for the measured rotations are dynamically chosen.

# <markdowncell>

# Uncertainty should be high when car is moving and very low, when the vehicle is standing still

# <codecell>

plt.figure(figsize=(12,4))
plt.plot(np.sqrt(ax**2+ay**2+(az+9.806)**2), label='$\sqrt{a_x^2+a_y^2+(a_z+9{,}81)^2}$')
plt.plot(speed/10, label='$1/10 \cdot \mathrm{speed}$', alpha=0.6)
plt.axhline(3, color='k', alpha=0.5)
plt.legend(loc='best')

# <codecell>

sroll = (200.0+500.0*np.sqrt(ax**2+ay**2+(az+9.806)**2))**2
spitch= (200.0+500.0*np.sqrt(ax**2+ay**2+(az+9.806)**2))**2

# <codecell>

plt.figure(figsize=(6,2))
plt.semilogy(sroll, label='$\sigma_{\Theta}$', marker='o', markevery=150, alpha=0.6)
plt.semilogy(spitch, label='$\sigma_{\phi}$', marker='*', markevery=150, alpha=0.6)
plt.ylabel('Values for $R$ Matrix')
plt.xlabel('Filterstep $k$')
plt.legend(bbox_to_anchor=(0.0, 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
#plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.png', dpi=72, transparent=True, bbox_inches='tight')
plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R2.eps', bbox_inches='tight')

# <headingcell level=2>

# Measurement Function H

# <markdowncell>

# Matrix H is the Jacobian of the Measurement function h with respect to the state.

# <codecell>

hs = Matrix([[xs],
             [ys],
             [phis],
             [thetas]])
Hs=hs.jacobian(state)
Hs

# <headingcell level=3>

# Identity Matrix

# <codecell>

I = np.eye(numstates)
print(I, I.shape)

# <headingcell level=2>

# Initial State

# <codecell>

x = np.matrix([[mx[0], my[0], course[0]/180.0*np.pi-0.18, speed[0]/3.6+0.001, yawrate[0]/180.0*np.pi, \
                0.0, pitchrate[0]/180.0*np.pi, \
                0.0, rollrate[0]/180.0*np.pi]]).T
print(x, x.shape)

U=float(np.cos(x[2])*x[3])
V=float(np.sin(x[2])*x[3])

plt.quiver(x[0], x[1], U, V)
plt.scatter(float(x[0]), float(x[1]), s=100)
plt.title('Initial Location')
plt.axis('equal')

# <headingcell level=3>

# Put everything together as a measurement vector

# <codecell>

measurements = np.vstack((mx, my, pitch/180.0*np.pi, roll/180.0*np.pi))
# Lenth of the measurement
m = measurements.shape[1]
print(measurements.shape)

# <codecell>


# <codecell>

# Preallocation for Plotting
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
Zx = []
Zy = []
P0 = []
P1 = []
P2 = []
P3 = []
P4 = []
P5 = []
P6 = []
P7 = []
P8 = []
K0 = []
K1 = []
K2 = []
K3 = []
K4 = []
dstate=[]

# <headingcell level=2>

# Extended Kalman Filter

# <markdowncell>

# ![Extended Kalman Filter Step](https://raw.github.com/balzer82/Kalman/master/Extended-Kalman-Filter-Step.png)

# <markdowncell>

# $$x_k= \left[\begin{matrix}x\\y\\\psi\\v\\\dot\psi\\\phi\\\dot\phi\\\Theta\\\dot\Theta\end{matrix}\right] = \left[ \matrix{ \text{Position X} \\ \text{Position Y} \\ \text{Heading} \\ \text{Velocity} \\ \text{Yaw Rate} \\ \text{Pitch} \\ \text{Pitchrate} \\ \text{Roll} \\ \text{Rollrate}} \right] =  \underbrace{\begin{matrix}x[0] \\ x[1] \\ x[2] \\ x[3] \\ x[4] \\ x[5] \\ x[6] \\ x[7] \\ x[8] \end{matrix}}_{\textrm{Python Nomenclature}}$$

# <codecell>

for filterstep in range(m):
    
    # Data (Control)
    vt=speed[filterstep]/3.6
    yat=yawrate[filterstep]/180.0*np.pi
    pit=pitchrate[filterstep]/180.0*np.pi
    rot=rollrate[filterstep]/180.0*np.pi
    
   
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # see "Dynamic Matrix"
    if np.abs(yat)<0.0001: # Driving straight
        x[0] = x[0] + vt*dt * np.cos(x[2])
        x[1] = x[1] + vt*dt * np.sin(x[2])
        x[2] = x[2]
        x[3] = vt
        x[4] = 0.0000001 # avoid numerical issues in Jacobians
        x[5] = x[5] + pit*dt
        x[6] = pit
        x[7] = x[7] + rot*dt
        x[8] = rot
        dstate.append(0)
    else: # otherwise
        x[0] = x[0] + (vt/yat) * (np.sin(yat*dt+x[2]) - np.sin(x[2]))
        x[1] = x[1] + (vt/yat) * (-np.cos(yat*dt+x[2])+ np.cos(x[2]))
        x[2] = (x[2] + yat*dt + np.pi) % (2.0*np.pi) - np.pi
        x[3] = vt
        x[4] = yat
        x[5] = x[5] + pit*dt
        x[6] = pit
        x[7] = x[7] + rot*dt
        x[8] = rot
        dstate.append(1)
    
    # Calculate the Jacobian of the Dynamic Matrix A
    # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
    a13 = float((x[3]/x[4]) * (np.cos(x[4]*dt+x[2]) - np.cos(x[2])))
    a14 = float((1.0/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a15 = float((dt*x[3]/x[4])*np.cos(x[4]*dt+x[2]) - (x[3]/x[4])*a14)
    a23 = float((x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a24 = float((1.0/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    a25 = float((dt*x[3]/x[4])*np.sin(x[4]*dt+x[2]) - (x[3]/x[4])*a24)
    JA = np.matrix([[1.0, 0.0, a13, a14, a15, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, a23, a24, a25, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    # Calculate the Jacobian of the Control Input G
    # see "Calculate the Jacobian of the Dynamic Matrix with Respect to the Control"
    g11 = float(1.0/x[4]*(-np.sin(x[2])+np.sin(dt*x[4]+x[2])))
    g12 = float(dt*x[3]/x[4]*np.cos(dt*x[4]+x[2]) - x[3]/x[4]*g11)
    g21 = float(1.0/x[4]*(np.cos(x[2])-np.cos(dt*x[4]+x[2])))
    g22 = float(dt*x[3]/x[4]*np.sin(dt*x[4]+x[2]) - x[3]/x[4]*g21)
    JG = np.matrix([[g11, g12, 0.0, 0.0],
                    [g21, g22, 0.0, 0.0],
                    [0.0, dt, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, dt, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 1.0]])
    
    # Project the error covariance ahead
    P = JA*P*JA.T + JG*Q*JG.T
    
    
    # Measurement Update (Correction)
    # ===============================
    
    # Measurement Function
    hx = np.matrix([[float(x[0])],
                    [float(x[1])],
                    [float(x[5])],
                    [float(x[7])]])
    
    # Because GPS is sampled with 10Hz and the other Measurements, as well as
    # the filter are sampled with 50Hz, one have to wait for correction until
    # there is a new GPS Measurement
    
    if GPS[filterstep]:
        # Calculate the Jacobian of the Measurement Function
        # see "Measurement Matrix H"
        JH = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    else:
        JH = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])        
    
    
    # Calculate R with Data from the GPS Signal itself
    # and raise it when standing still
    R[0,0] = sp[filterstep]
    R[1,1] = sp[filterstep]
    R[2,2] = spitch[filterstep]
    R[3,3] = sroll[filterstep]
    
    S = JH*P*JH.T + R
    K = (P*JH.T) * np.linalg.inv(S)

    # Update the estimate via
    z = measurements[:,filterstep].reshape(JH.shape[0],1)
    y = z - (hx)                         # Innovation or Residual
    x = x + (K*y)
    
    # Update the error covariance
    P = (I - (K*JH))*P
    

    # Save states for Plotting
    x0.append(float(x[0]))
    x1.append(float(x[1]))
    x2.append(float(x[2]))
    x3.append(float(x[3]))
    x4.append(float(x[4]))
    x5.append(float(x[5]))
    x6.append(float(x[6]))
    x7.append(float(x[7]))
    x8.append(float(x[8]))
    
    P0.append(float(P[0,0]))
    P1.append(float(P[1,1]))
    P2.append(float(P[2,2]))
    P3.append(float(P[3,3]))
    P4.append(float(P[4,4]))
    P5.append(float(P[5,5]))
    P6.append(float(P[6,6]))
    P7.append(float(P[7,7]))
    P8.append(float(P[8,8]))
    
    Zx.append(float(z[0]))
    Zy.append(float(z[1]))    
    
    K0.append(float(K[0,0]))
    K1.append(float(K[1,0]))
    K2.append(float(K[2,0]))
    K3.append(float(K[3,0]))
    K4.append(float(K[4,0]))

# <codecell>


# <headingcell level=2>

# Plots

# <codecell>

%pylab inline --no-import-all

# <headingcell level=3>

# Uncertainty

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.semilogy(range(m),P0, label='$x$')
plt.step(range(m),P1, label='$y$')
plt.step(range(m),P2, label='$\psi$')
plt.step(range(m),P3, label='$v$')
plt.step(range(m),P4, label='$\dot \psi$')
plt.step(range(m),P5, label='$\phi$')
plt.step(range(m),P6, label='$\dot \phi$')
plt.step(range(m),P7, label='$\Theta$')
plt.step(range(m),P8, label='$\dot \Theta$')

plt.xlabel('Filter Step')
plt.ylabel('')
plt.title('Uncertainty (Elements from Matrix $P$)')
#plt.legend(loc='best',prop={'size':22})
plt.legend(bbox_to_anchor=(0., 0.9, 1., .06), loc=3,
       ncol=9, mode="expand", borderaxespad=0.,prop={'size':22})
plt.savefig('Covariance-Matrix-Verlauf.eps', bbox_inches='tight')

# <codecell>

fig = plt.figure(figsize=(numstates/2, numstates/2))
im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
#plt.title('Covariance Matrix $P$ (after %i Filter Steps)' % (m))
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(10))
# set the locations and labels of the yticks
plt.yticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(10))
# set the locations and labels of the yticks
plt.xticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

plt.xlim([-0.5,8.5])
plt.ylim([8.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.savefig('Covariance-Matrix-imshow-P.eps', bbox_inches='tight')

# <headingcell level=3>

# Kalman Gains

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.step(range(len(measurements[0])),K0, label='$x$')
plt.step(range(len(measurements[0])),K1, label='$y$')
plt.step(range(len(measurements[0])),K2, label='$\psi$')
plt.step(range(len(measurements[0])),K3, label='$v$')
plt.step(range(len(measurements[0])),K4, label='$\dot \psi$')

plt.xlabel('Filter Step')
plt.ylabel('')

plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
#plt.legend(prop={'size':18})
plt.legend(bbox_to_anchor=(0., 0., 1., .102), loc=3,
       ncol=5, mode="expand", borderaxespad=0.,prop={'size':22})
plt.ylim([-0.4,0.4])

# <headingcell level=2>

# State Vector

# <codecell>

fig = plt.figure(figsize=(8,numstates))

# Course
plt.subplot(511)
plt.step(range(len(measurements[0])),(course+180.0)%(360.0)-180.0, label='$\psi$ (GPS)', marker='o', markevery=150, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x2,180.0/np.pi), label='$\psi$', marker='*', markevery=140)
plt.ylabel('Course $^\circ$')
plt.yticks(np.arange(-180, 181, 45))
plt.ylim([-200,200])
plt.legend(bbox_to_anchor=(0.0, 0., 0.4, .06), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
#plt.title('State Estimates $x_k$')

# Velocity
plt.subplot(512)
plt.step(range(len(measurements[0])),speed, label='$v$ (GPS)', marker='o', markevery=150, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x3,3.6), label='$v$', marker='*', markevery=140)
plt.ylabel('Velocity $km/h$')
#plt.ylim([0, 30])
plt.legend(loc='best',prop={'size':12})

# Yawrate
plt.subplot(513)
plt.step(range(len(measurements[0])),yawrate, label='$\dot \psi$ (IMU)', marker='o', markevery=150, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x4,180.0/np.pi), label='$\dot \psi$', marker='*', markevery=140)
plt.ylabel('Yaw Rate $^\circ/s$')
plt.ylim([-50.0, 50.0])
plt.legend(bbox_to_anchor=(0.6, 0., 0.4, .06), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)

# Pitch
plt.subplot(514)
plt.step(range(len(measurements[0])),pitch, label='$\phi$ (IMU)', marker='o', markevery=150, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x5,180.0/np.pi), label='$\phi$', marker='*', markevery=140)
#plt.step(range(len(measurements[0])),pitchacc*180.0/np.pi, label='$\phi$ (Acc)', alpha=0.5)
plt.ylabel('Pitch $^\circ$')
plt.ylim([-25.0, 25.0])
#plt.legend(loc='best',prop={'size':12})
plt.legend(bbox_to_anchor=(0.0, 0., 0.4, .06), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)

# Roll
plt.subplot(515)
plt.step(range(len(measurements[0])),roll, label='$\Theta$ (IMU)', marker='o', markevery=150, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x7,180.0/np.pi), label='$\Theta$', marker='*', markevery=140)
#plt.step(range(len(measurements[0])),rollacc*180.0/np.pi, label='$\Theta$ (Acc)', alpha=0.5)
plt.ylabel('Roll $^\circ$')
plt.ylim([-25.0, 25.0])
#plt.legend(loc='best',prop={'size':12})
plt.legend(bbox_to_anchor=(0.6, 0., 0.4, .06), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Filter Step')

plt.savefig('Extended-Kalman-Filter-CTRV-Attitude-State-Estimates.eps', bbox_inches='tight')

# <codecell>


# <headingcell level=2>

# Position x/y

# <codecell>

#%pylab --no-import-all

# <codecell>

fig = plt.figure(figsize=(10,6))

# EKF State
qscale= 0.5*np.divide(x3[::5],np.max(x3))+0.1
plt.quiver(x0[::5],x1[::5],np.cos(x2[::5]), np.sin(x2[::5]), color='#94C600', units='xy', width=0.01, scale=qscale)
plt.plot(x0[::5],x1[::5], label='EKF Position Estimation', color='k', alpha=0.5)

# Measurements
plt.scatter(mx[::50],my[::50], s=120, label='GNSS Measurements (every 10th)',\
            c=sp[::50], cmap='autumn_r', norm=matplotlib.colors.LogNorm())
cbar=plt.colorbar()
cbar.ax.set_ylabel(u'Adaptive Measurement Noise Covariance', rotation=270)
cbar.ax.set_xlabel(u'$m^2$')

# Annotations
plt.annotate('see Fig. 11', xy=(110, 150), xytext=(120, 50),fontsize=16, ha='center',
            arrowprops=dict(facecolor='k', shrink=0.05))

bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="w", lw=2)
t = plt.text(450, 290, "Driving Direction", ha="center", va="center", rotation=-32,
            size=12,
            bbox=bbox_props)

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Position of EKF state $x_k$, GNSS measurements and uncertainty $R$ (color)')
plt.legend(loc='best')
plt.axis('equal')
#plt.tight_layout()

#plt.show()
plt.savefig('Extended-Kalman-Filter-CTRV-Position.eps', bbox_inches='tight')

# <codecell>

fig = plt.figure(figsize=(6,3))

# EKF State
plt.plot(x0,x1, label='EKF Position Estimation', c='k')

# Measurements
plt.scatter(mx[::5],my[::5], s=40, label='GNSS Measurements',\
            c=sp[::5], cmap='autumn_r', norm=matplotlib.colors.LogNorm())
cbar=plt.colorbar()
cbar.ax.set_ylabel(u'$\sigma^2_x$ and $\sigma^2_y$ values in $R$', rotation=270)
cbar.ax.set_xlabel(u'')

plt.annotate('high uncertainty in $R$', xy=(103, 174), xytext=(103, 172), fontsize=12,
            arrowprops=dict(facecolor='k', shrink=0.05), ha='center'
            )

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
#plt.title('Position')
plt.legend(loc='upper left')
plt.axis('equal')
#plt.tight_layout()
plt.xlim([96, 108])
plt.ylim([172, 178])
#plt.show()
plt.savefig('Extended-Kalman-Filter-CTRV-Position-Detail.eps', bbox_inches='tight')

# <codecell>

fig = plt.figure(figsize=(6,3))

# EKF State
plt.plot(x0,x1, label=u'verbesserte Positionsschätzung', c='k')

# Measurements
plt.scatter(mx[::5],my[::5], s=40, label='GPS Positionsmessungen',\
            c=sp[::5], cmap='autumn_r', norm=matplotlib.colors.LogNorm())

bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="w", lw=2)
t = plt.text(85, 175, u"Hinfahrt", ha="center", va="center", rotation=66,
            size=12,
            bbox=bbox_props)

bbox_props = dict(boxstyle="larrow,pad=0.3", ec="w", lw=2)
t = plt.text(140, 165, u"Rückfahrt", ha="center", va="center", rotation=-20,
            size=12,
            bbox=bbox_props)


plt.xlabel('X [m]')
plt.ylabel('Y [m]')
#plt.title('Position')
plt.legend(loc='best')
plt.axis('equal')
#plt.tight_layout()
plt.xlim([90, 138])
plt.ylim([145, 185])
#plt.show()
plt.savefig('EKF.png', bbox_inches='tight',dpi=150)

# <headingcell level=3>

# Convert back from Meters to Lat/Lon (WGS84)

# <codecell>

latekf = latitude[0] + np.divide(x1,arc)
lonekf = longitude[0]+ np.divide(x0,np.multiply(arc,np.cos(latitude*np.pi/180.0)))

# <headingcell level=2>

# Position Lat/Lon

# <codecell>

fig = plt.figure(figsize=(10,4.5))

# EKF State
plt.plot(lonekf[::5],latekf[::5], label='EKF Position Estimation', color='k', alpha=0.5)

# Measurements
plt.scatter(longitude[::50],latitude[::50], s=50, label='GNSS Measurements (every 10th)',\
            c=sp[::50], cmap='autumn_r', norm=matplotlib.colors.LogNorm())
cbar=plt.colorbar()
cbar.ax.set_ylabel(u'Adaptive Measurement Noise Covariance', rotation=270)
cbar.ax.set_xlabel(u'$m^2$')

# Annotations
plt.annotate('see Fig. 11', xy=(13.794, 51.041), xytext=(13.795, 51.04),fontsize=16, ha='center',
            arrowprops=dict(facecolor='k', shrink=0.05))

bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="w", lw=2)
t = plt.text(13.793, 51.041, "Driving Direction", ha="center", va="center", rotation=65,
            size=12,
            bbox=bbox_props)
bbox_props = dict(boxstyle="larrow,pad=0.3", ec="w", lw=2)
t = plt.text(13.7985, 51.0405, "Driving Direction", ha="center", va="center", rotation=-35,
            size=12,
            bbox=bbox_props)

plt.xlabel('Longitude [$^\circ$]')
plt.ylabel('Latitude [$^\circ$]')
plt.title('Position of EKF state $x_k$, GNSS measurements and uncertainty $R$ (color)')
plt.legend(loc='best')
#plt.axis('equal')
#plt.tight_layout()

# xticks
locs,labels = plt.xticks()
plt.xticks(locs, map(lambda x: "%.3f" % x, locs))

# ytikcs
locs,labels = plt.yticks()
plt.yticks(locs, map(lambda x: "%.4f" % x, locs))

#plt.show()
plt.savefig('Extended-Kalman-Filter-CTRV-Position.eps', bbox_inches='tight')

# <codecell>

print('Done.')

# <headingcell level=1>

# Conclusion

# <markdowncell>

# As you can see, complicated analytic calculation of the Jacobian Matrices, but it works pretty well.
# 
# Let's take a look at the trajectory on Google Earth:

# <headingcell level=2>

# Write Google Earth KML

# <markdowncell>

# Coordinates and timestamps to be used to locate the car model in time and space
# The value can be expressed as yyyy-mm-ddThh:mm:sszzzzzz, where T is the separator between the date and the time, and the time zone is either Z (for UTC) or zzzzzz, which represents ±hh:mm in relation to UTC.

# <codecell>

import datetime
car={}
car['when']=[]
car['coord']=[]
car['gps']=[]
for i in range(len(millis)):
    d=datetime.datetime.fromtimestamp(millis[i]/1000.0)
    car["when"].append(d.strftime("%Y-%m-%dT%H:%M:%SZ"))
    car["coord"].append((lonekf[i], latekf[i], 0))
    car["gps"].append((longitude[i], latitude[i], 0))

# <codecell>

from simplekml import Kml, Model, AltitudeMode, Orientation, Scale

# <codecell>

# The model path and scale variables
car_dae = r'http://simplekml.googlecode.com/hg/samples/resources/car-model.dae'
car_scale = 1.0

# Create the KML document
kml = Kml(name=d.strftime("%Y-%m-%d %H:%M"), open=1)

# Create the model
model_car = Model(altitudemode=AltitudeMode.clamptoground,
                            orientation=Orientation(heading=75.0),
                            scale=Scale(x=car_scale, y=car_scale, z=car_scale))

# Create the track
trk = kml.newgxtrack(name="EKF", altitudemode=AltitudeMode.clamptoground,
                     description="State Estimation from Extended Kalman Filter with CTRV Model")
gps = kml.newgxtrack(name="GPS", altitudemode=AltitudeMode.clamptoground,
                     description="Original GPS Measurements")

# Attach the model to the track
trk.model = model_car
#gps.model = model_car

trk.model.link.href = car_dae
#gps.model.link.href = car_dae

# Add all the information to the track
trk.newwhen(car["when"])
trk.newgxcoord(car["coord"])

gps.newwhen(car["when"][::5])
gps.newgxcoord((car["gps"][::5]))

# Style of the Track
trk.iconstyle.icon.href = ""
trk.labelstyle.scale = 1
trk.linestyle.width = 10
trk.linestyle.color = '7f00ff00' # aabbggrr

gps.iconstyle.icon.href = ""
gps.labelstyle.scale = 0
gps.linestyle.width = 4
gps.linestyle.color = '7fff0000'


# Saving
#kml.save("Extended-Kalman-Filter-CTRV.kml")
kml.savekmz("Extended-Kalman-Filter-CTRV-Adaptive.kmz")

# <codecell>

print('Exported KMZ File for Google Earth')

# <markdowncell>

# Works just fine!

