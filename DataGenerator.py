# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
%pylab inline --no-import-all

# <headingcell level=1>

# Generating Vehicle Data + GNSS

# <codecell>


# <codecell>

dt = 0.1    # Timestep s

# <codecell>

def move(x, yawc, axc, dt):
    if yawc==0.0: # Driving straight
        x[0] = x[0] + x[2]*dt * np.cos(x[3])
        x[1] = x[1] + x[2]*dt * np.sin(x[3])
        x[2] = x[2] + axc*dt
        x[3] = x[3]
    else: # otherwise
        x[0] = x[0] + (x[2]/yawc) * (np.sin(yawc*dt+x[3]) - np.sin(x[3]))
        x[1] = x[1] + (x[2]/yawc) * (-np.cos(yawc*dt+x[3])+ np.cos(x[3]))
        x[2] = x[2] + axc*dt
        x[3] = (x[3] + yawc*dt + np.pi) % (2.0*np.pi) - np.pi
    return x

# <codecell>

def GPS(x,sp):
    GNSSx = x[0] + np.sqrt(sp)*(np.random.randn(1)-0.5)
    GNSSy = x[1] + np.sqrt(sp)*(np.random.randn(1)-0.5)
    GNSScourse = x[3] + np.sqrt(sp/200.0)*(np.random.randn(1)-0.5)
    return GNSSx, GNSSy, GNSScourse

# <codecell>

x = np.array([[0.0],[0.0],[20/3.6],[0.0]])
t = 0.0

xout=[]
yout=[]
vout=[]
aout=[]
yawout=[]
courseout=[]
EPEout=[]
GNSSxout=[]
GNSSyout=[]

# Breaking
while x[2]>0.0:
    a = -0.6      # acceleration m/s2
    yaw= 0.0      # yawrate in rad/s
    x = move(x, yaw, a,dt)
    
    sp = 2.0
    GNSSx, GNSSy, GNSScourse = GPS(x, sp)
    
    xout.append(float(x[0]))
    yout.append(float(x[1]))
    vout.append(float(x[2]))
    courseout.append(float(x[3]))
    yawout.append(yaw)
    aout.append(a)
    EPEout.append(sp)
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt


# Waiting    
twait = t+10.0
while t<twait:
    a =  0.0      # acceleration m/s2
    yaw= 0.0      # yawrate in rad/s
    x = move(x, yaw, a,dt)
    
    sp = 20.0
    GNSSx, GNSSy, GNSScourse = GPS(x, sp)
    
    xout.append(float(x[0]))
    yout.append(float(x[1]))
    vout.append(float(x[2]))
    courseout.append(GNSScourse)
    aout.append(a)
    yawout.append(yaw)    
    EPEout.append(sp)
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt
    
# Accelerating until driving 50km/h
while x[2]<(50.0/3.6):
    a =  2.0      # acceleration m/s2
    yaw= 0.0      # yawrate in rad/s
    x = move(x, yaw, a,dt)
    
    sp = 2.0
    GNSSx, GNSSy, GNSScourse = GPS(x, sp)
    
    xout.append(float(x[0]))
    yout.append(float(x[1]))
    vout.append(float(x[2]))
    courseout.append(GNSScourse)
    aout.append(a)
    yawout.append(yaw)    
    EPEout.append(sp)
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt


# Passing a Building    
tdrive = t+10.0
tbadGPS = t+5.0
tbadGPSend=t+8.0
while t<tdrive:
    a =  0.0      # acceleration m/s2
    yaw= 0.0      # yawrate in rad/s
    x = move(x, yaw, a,dt)
    
    if t>tbadGPS and t<tbadGPSend:
        sp = 25.0
        GNSSx, GNSSy, GNSScourse = GPS(x, sp)
        GNSSy+=15.0
    else:
        sp = 2.0
        GNSSx, GNSSy, GNSScourse = GPS(x, sp)
    
    xout.append(float(x[0]))
    yout.append(float(x[1]))
    vout.append(float(x[2]))
    courseout.append(GNSScourse)
    aout.append(a)
    yawout.append(yaw)    
    EPEout.append(sp)
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt
    
# Cornering
tdrive = t+11.0
while t<tdrive:
    a =  0.0      # acceleration m/s2
    yaw = 0.8*np.sin(t)      # yawrate in rad/s
    x   = move(x, yaw, a,dt)
    
    sp = 2.0
    GNSSx, GNSSy, GNSScourse = GPS(x, sp)
    
    xout.append(float(x[0]))
    yout.append(float(x[1]))
    vout.append(float(x[2]))
    courseout.append(GNSScourse)
    aout.append(a)
    yawout.append(yaw)    
    EPEout.append(sp)
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt

# <headingcell level=3>

# Physical Data for the Test Data Set

# <codecell>

plt.figure(figsize=(16,2.5))
plt.subplot(151)
plt.plot(xout, yout)
plt.axis('equal')
plt.title('Trajectory')
plt.subplot(152)
plt.plot(vout)
plt.title('Velocity')
plt.subplot(153)
plt.plot(aout)
plt.title('Acceleration')
plt.subplot(154)
plt.plot(courseout)
plt.title('Course')
plt.subplot(155)
plt.plot(yawout)
plt.title('Yawrate')

# <headingcell level=3>

# Position

# <codecell>

plt.figure(figsize=(8,3))
plt.plot(xout, yout, color='k', label='Ground Truth')
plt.scatter(GNSSxout, GNSSyout, label='GNSS measurements')
plt.legend(loc='best')
plt.axis('equal')
plt.xlabel('Position X ($m$)')
plt.xlim([-10, np.max(xout)+10])
plt.ylabel('Position Y ($m$)')
#plt.title('View from top with building (B)')

# Annotations
bbox_props = dict(boxstyle="square,pad=0.9", fc="k", ec="w", lw=2)
t = plt.text(160, -30, "B", ha="left", va="center", rotation=0,
            size=16, color='grey',
            bbox=bbox_props)

plt.annotate('stop', xy=(30, -20), xytext=(60, -50), fontsize=16,
            arrowprops=dict(facecolor='k', shrink=0.05), ha='center',
            )

plt.savefig('Testdata.eps', bbox_inches='tight')

# <headingcell level=1>

# Maximum Likelihood Estimator for EPE

# <codecell>

n=10
epeout=[]
for i in range(len(EPEout)):
    if i<n:
        epeout.append(2.0)
    else:
        epeout.append(np.mean(EPEout[(i-n):i]))

# <codecell>

plt.figure(figsize=(8,3))
plt.plot(xout, epeout, color='k', label='Estimated Position Error')
plt.legend(loc='best')
plt.xlabel('Position X ($m$)')
plt.xlim([0, np.max(xout)])
plt.ylim([0, np.max(EPEout)+10])
plt.ylabel('EPE ($m$)')
plt.savefig('Testdata-EPE.eps', bbox_inches='tight')

# <headingcell level=2>

# Dump to Disk

# <codecell>

with open('testdata.csv', 'wb') as thefile:
    for i in range(len(xout)):
       thefile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (xout[i], yout[i], 3.6*vout[i], -aout[i], courseout[i], yawout[i], GNSSxout[i], GNSSyout[i], epeout[i]))

# <codecell>


# <codecell>


