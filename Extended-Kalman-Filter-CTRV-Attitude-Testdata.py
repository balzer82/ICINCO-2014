# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, latex, Plot
from sympy.interactive import printing
printing.init_printing()
%pylab inline --no-import-all

# <headingcell level=1>

# Adaptive Extended Kalman Filter Implementation for Constant Turn Rate and Velocity (CTRV) Vehicle Model with Attitude Estimation for Testdata

# <markdowncell>

# Situation covered: You have an velocity sensor which measures the vehicle speed ($v$) in heading direction ($\psi$) and a yaw rate sensor ($\dot \psi$) which both have to fused with the position ($x$ & $y$) from a GPS sensor.

# <headingcell level=2>

# State Vector - Constant Turn Rate and Velocity Vehicle Model (CTRV) + Roll and Pitch Estimation

# <markdowncell>

# $$x_k= \left[\begin{matrix}x\\y\\\psi\\v\\\dot\psi\\\phi\\\dot\phi\\\Theta\\\dot\Theta\end{matrix}\right] = \left[ \matrix{ \text{Position X} \\ \text{Position Y} \\ \text{Heading} \\ \text{Velocity} \\ \text{Yaw Rate} \\ \text{Pitch} \\ \text{Pitchrate} \\ \text{Roll} \\ \text{Rollrate}} \right]$$

# <codecell>

adaptive = True

# <codecell>

numstates=9 # States

# <codecell>

dt = 1.0/10.0 # Sample Rate of the Measurements is 10Hz
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
yawrateaccmax = 50.0  # Grad/s2

svQ = (amax*dt)      # Velocity
syQ = (yawrateaccmax*np.pi/180.0*dt)   # Yawrate
spQ = (yawrateaccmax*np.pi/180.0*dt)   # Pitchrate
srQ = (2*yawrateaccmax*np.pi/180.0*dt) # Rollrate

Q = np.matrix([[svQ**2, 0.0, 0.0, 0.0],
               [0.0, syQ**2, 0.0, 0.0],
               [0.0, 0.0, spQ**2, 0.0],
               [0.0, 0.0, 0.0, srQ**2]])

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
datafile = 'testdata.csv'

xout, \
yout, \
speed, \
ax, \
GNSSx, \
GNSSy, \
epe = np.loadtxt(datafile, delimiter=',', unpack=True)

val = len(xout)

ay = np.zeros(val)
az = np.zeros(val)-9.81
rollrate = np.zeros(val)
pitchrate = np.zeros(val)
roll = np.zeros(val)
pitch = np.zeros(val)
yaw = np.zeros(val)
yawrate = np.zeros(val)
course = np.zeros(val)
altitude = np.zeros(val)

print('Read \'%s\' successfully.' % datafile)

# <codecell>


# <headingcell level=3>

# Static Gain

# <codecell>

pitchrate = pitchrate - np.mean(pitchrate)
rollrate = rollrate - np.mean(rollrate)

# <codecell>

# clamp speed and yawrate to zero while standing still
speed[speed<1.0]=0.0
yawrate[speed<1.0]=0.0

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

epsilon = 1.5
xi      = 100.0
zeta    = 10.0

spspeed=xi/((speed/3.6)+epsilon)
spepe=zeta*epe
sp = spspeed**2 + spepe**2

# <codecell>

plt.figure(figsize=(16,6))
plt.semilogy(spspeed**2, label='$\sigma_P$ from $speed$', marker='*', markevery=50)
plt.semilogy(spepe**2, label='$\sigma_P$ from $EPE$', marker='x', markevery=50)
plt.semilogy(sp, label='Resulting $\sigma_{x,y}$ value', marker='o', markevery=50)
plt.ylabel('Values for $R$ Matrix')
plt.xlabel('Filterstep $k$')
plt.legend(loc='best')
#plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.png', dpi=72, transparent=True, bbox_inches='tight')
plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.eps', bbox_inches='tight')

# <headingcell level=3>

# Attitude

# <markdowncell>

# Because the estimation of Roll and Pitch is only valid for quasistatic situations (which is not valid for a moving vehicle), the values for the measured rotation $\sigma_r$ is very high.

# <markdowncell>

# Uncertainty should be high when car is moving and very low, when the vehicle is standing still

# <codecell>

sroll = (0.1*((speed/3.6)+10.0))**2
spitch= (10.0*((speed/3.6)+10.0))**2

# <codecell>

plt.figure(figsize=(16,6))
plt.semilogy(sroll, label='$\sigma_{\Theta}$ value', marker='o', markevery=50)
plt.semilogy(spitch, label='$\sigma_{\phi}$ value', marker='x', markevery=50)
plt.ylabel('Values for $R$ Matrix')
plt.xlabel('Filterstep $k$')
plt.legend()
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

# <headingcell level=3>

# Roll & Pitch from Acceleration

# <markdowncell>

# As mentioned in Euston, M., Coote, P., & Mahony, R. (2008). A complementary filter for attitude estimation of a fixed-wing UAV. Intelligent Robots and …, 340–345. doi:10.1109/IROS.2008.4650766, a raw estimate of roll and pitch can be determined from acceleration with respect to gravity.

# <codecell>

aye = speed/3.6*-yawrate/180.0*np.pi
axe = np.hstack((0.0, -np.diff(speed/3.6)/dtGPS))

rollacc = np.arctan2(-(ay-aye), -az)
pitchacc= -np.arctan2(-(ax-axe), np.sqrt((ay-aye)**2+az**2))

# <codecell>


# <headingcell level=2>

# Initial State

# <codecell>

x = np.matrix([[0.0, 0.0, course[0]/180.0*np.pi, speed[0]/3.6+0.001, yawrate[0]/180.0*np.pi, \
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

measurements = np.vstack((GNSSx, GNSSy, pitchacc, rollacc))
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
        x[4] = -0.0000001 # avoid numerical issues in Jacobians
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
    
    # Calculate the Jacobian of the Measurement Function
    # see "Measurement Matrix H"
    H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    if adaptive:
        # Calculate R with Data from the GPS Signal itself
        # and raise it when standing still
        R[0,0] = sp[filterstep]
        R[1,1] = sp[filterstep]
        R[2,2] = spitch[filterstep]
        R[3,3] = sroll[filterstep]
        
    
    S = H*P*H.T + R
    K = (P*H.T) * np.linalg.inv(S)

    # Update the estimate via
    Z = measurements[:,filterstep].reshape(H.shape[0],1)
    y = Z - (hx)                         # Innovation or Residual
    x = x + (K*y)
    
    # Update the error covariance
    P = (I - (K*H))*P


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
    
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))    
    
    K0.append(float(K[0,0]))
    K1.append(float(K[1,0]))
    K2.append(float(K[2,0]))
    K3.append(float(K[3,0]))
    K4.append(float(K[4,0]))

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
plt.legend(loc='best',prop={'size':22})
#plt.savefig('Covariance-Matrix-Verlauf.eps', bbox_inches='tight')

# <codecell>

fig = plt.figure(figsize=(numstates, numstates))
im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Covariance Matrix $P$ (after %i Filter Steps)' % (m))
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
#plt.savefig('Covariance-Matrix-imshow-P.eps', bbox_inches='tight')

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
plt.legend(prop={'size':18})
plt.ylim([-1.0,1.0])

# <headingcell level=2>

# State Vector

# <codecell>

fig = plt.figure(figsize=(16,2*numstates))

# Course
plt.subplot(511)
plt.step(range(len(measurements[0])),(course+180.0)%(360.0)-180.0, label='$\psi$ (GPS)', marker='o', markevery=50, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x2,180.0/np.pi), label='$\psi$', marker='*', markevery=40)
plt.ylabel('Course $^\circ$')
plt.yticks(np.arange(-180, 181, 45))
plt.ylim([-200,200])
plt.legend(loc='best',prop={'size':16})
plt.title('State Estimates $x_k$')

# Velocity
plt.subplot(512)
plt.step(range(len(measurements[0])),speed, label='$v$ (GPS)', marker='o', markevery=50, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x3,3.6), label='$v$', marker='*', markevery=40)
plt.ylabel('Velocity $km/h$')
#plt.ylim([0, 30])
plt.legend(loc='best',prop={'size':16})

# Yawrate
plt.subplot(513)
plt.step(range(len(measurements[0])),yawrate, label='$\dot \psi$ (IMU)', marker='o', markevery=50, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x4,180.0/np.pi), label='$\dot \psi$', marker='*', markevery=40)
plt.ylabel('Yaw Rate $^\circ/s$')
plt.ylim([-6.0, 6.0])
plt.legend(loc='best',prop={'size':16})

# Pitch
plt.subplot(514)
plt.step(range(len(measurements[0])),pitchrate, label='$\dot \phi$ (IMU)', marker='o', markevery=50, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x5,180.0/np.pi), label='$\phi$', marker='*', markevery=40)
#plt.step(range(len(measurements[0])),pitchacc*180.0/np.pi, label='$\phi$ (Acc)', alpha=0.5)
plt.ylabel('Pitch$^\circ$')
plt.ylim([-20.0, 20.0])
plt.legend(loc='best',prop={'size':16})

# Roll
plt.subplot(515)
plt.step(range(len(measurements[0])),rollrate, label='$\dot \Theta$ (IMU)', marker='o', markevery=50, alpha=0.6)
plt.step(range(len(measurements[0])),np.multiply(x7,180.0/np.pi), label='$\Theta$', marker='*', markevery=40)
#plt.step(range(len(measurements[0])),rollacc*180.0/np.pi, label='$\Theta$ (Acc)', alpha=0.5)
plt.ylabel('Roll$^\circ$')
#plt.ylim([-20.0, 20.0])
plt.legend(loc='best',prop={'size':16})
plt.xlabel('Filter Step')

#plt.savefig('Extended-Kalman-Filter-CTRV-Attitude-State-Estimates.eps', bbox_inches='tight')

# <codecell>


# <headingcell level=2>

# Position x/y

# <codecell>

#%pylab --no-import-all

# <codecell>

fig = plt.figure(figsize=(8,3))

# Ground Truth
plt.plot(xout, yout, color='k', label='Ground Truth')

# EKF State
#qscale= 0.5*np.divide(x3,np.max(x3))+0.1
#plt.quiver(x0,x1,np.cos(x2), np.sin(x2), color='#94C600', units='xy', width=0.01, scale=qscale)
plt.plot(x0,x1, c='b', label='EKF Position')

# Measurements
if adaptive:
    plt.scatter(GNSSx,GNSSy, s=50, label='GNSS Measurements', c=sp, cmap='autumn_r')
    cbar=plt.colorbar()
    cbar.ax.set_ylabel(u'$\sigma^2_x$ and $\sigma^2_y$', rotation=270)
    #cbar.ax.set_xlabel(u'm')
    #plt.title('Performance of adaptive EKF with values of measurement uncertainty $R$ (color)')
else:
    plt.scatter(GNSSx,GNSSy, s=20, label='GPS Measurements', c='g')
    #plt.title('Performance of EKF with static $R$')


plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.legend(loc='best')
plt.axis('equal')
plt.xlim([-10, np.max(xout)+10])
#plt.tight_layout()

# Annotations
bbox_props = dict(boxstyle="square,pad=0.8", fc="k", ec="w", lw=2)
t = plt.text(340, -40, "B", ha="left", va="center", rotation=0,
            size=14, color='grey',
            bbox=bbox_props)

plt.annotate('stop', xy=(144, -20), xytext=(110, -60), fontsize=16,
            arrowprops=dict(facecolor='k', shrink=0.05), ha='center',
            )

#plt.show()
if adaptive:
    plt.savefig('Extended-Kalman-Filter-CTRV-Position-Testdata.eps', bbox_inches='tight')
else:
    plt.savefig('Extended-Kalman-Filter-CTRV-Position-Testdata-NonAdaptive.eps', bbox_inches='tight')

# <headingcell level=1>

# Calculate the Error

# <codecell>

CTEy = (x1 - yout)
CTEx = (x0 - xout)

sCTEy = np.sum(CTEy**2)
sCTEx = np.sum(CTEx**2)

# <codecell>

fig = plt.figure(figsize=(8,3))
plt.plot(xout, CTEx, label='$CTE_x$, $\Sigma(CTE_x^2)$=%.0f$m^2$' % sCTEx)
plt.plot(xout, CTEy, label='$CTE_y$, $\Sigma(CTE_y^2)$=%.0f$m^2$' % sCTEy)
plt.xlabel('X [m]')
plt.ylabel('CTE [m]')
if adaptive:
    plt.title(r'$CTE$ of Position Estimation for $\epsilon=$ %.2f, $\xi=$ %.2f, $\zeta=$ %.2f' % (epsilon, xi, zeta))
else:
    plt.title('$CTE$ of Position Estimation with static $R$')
plt.legend(loc='best')
plt.axis('equal')
plt.xlim([-10, np.max(xout)+10])

# Annotations
bbox_props = dict(boxstyle="square,pad=0.8", fc="k", ec="w", lw=2)
t = plt.text(340, -40, "B", ha="left", va="center", rotation=0,
            size=14, color='grey',
            bbox=bbox_props)

plt.annotate('stop', xy=(144, -20), xytext=(110, -60), fontsize=16,
            arrowprops=dict(facecolor='k', shrink=0.05), ha='center',
            )

#plt.show()
if adaptive:
    plt.savefig('Extended-Kalman-Filter-CTRV-CTE-Testdata.eps', bbox_inches='tight')
else:
    plt.savefig('Extended-Kalman-Filter-CTRV-CTE-Testdata-NonAdaptive.eps', bbox_inches='tight')

# <markdowncell>

# Typical Values for CTE (several runs with different datasets from `DataGenerator.py`

# <markdowncell>

# Boxplot from http://stackoverflow.com/questions/20365122/how-to-make-a-grouped-boxplot-graph-in-matplotlib

# <codecell>

CTE = {}
CTE['Adaptive'] = {}
CTE['Non-Adaptive']={}

CTE['Non-Adaptive']['X'] = (1616.0, 1521.0, 1403.0, 1028.0, 1689.0, 1515.0)
CTE['Non-Adaptive']['Y'] = (4760.0,4397.0,5248.0, 5007.0, 4770.0, 5340.0)

CTE['Adaptive']['X'] = (853.0, 692.0, 635.0, 704.0, 750.0, 1003.0)
CTE['Adaptive']['Y'] = (1757.0, 1493.0, 2920.0, 1796.0, 1739.0, 2257.0)

# <codecell>

fig, axes = plt.subplots(ncols=2, sharey=True)

for ax, name in zip(axes, ['Non-Adaptive', 'Adaptive']):
    ax.boxplot([CTE[name][item] for item in ['X', 'Y']])
    ax.set(xticklabels=['X', 'Y'], xlabel=name)
    ax.set(ylabel='$\Sigma$ CTE $(m^2)$')
    ax.margins(0.05) # Optional

plt.savefig('CTE-Adaptive-NonAdaptive-Boxplot.eps', bbox_inches='tight')

# <codecell>

print('CTE reduced in X: %.0f%%' % (100.0-100.0*np.mean(CTE['Adaptive']['X']) / np.mean(CTE['Non-Adaptive']['X'])))
print('CTE reduced in Y: %.0f%%' % (100.0-100.0*np.mean(CTE['Adaptive']['Y']) / np.mean(CTE['Non-Adaptive']['Y'])))

# <codecell>


