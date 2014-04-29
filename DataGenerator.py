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

dt = 0.1     # Timestep s

# <codecell>

def move(x,v,a,dt):
    x = x + v*dt + 0.5*a*dt**2
    v = v + a*dt
    y = 0.0
    return x,y,v

# <codecell>

def GPS(x,y,sp):
    GNSSx = x + np.sqrt(sp)*(np.random.randn(1)-0.5)
    GNSSy = y + np.sqrt(sp)*(np.random.randn(1)-0.5)
    return GNSSx, GNSSy

# <codecell>

v = 50.0/3.6  # velocity m/s
x = 0.0       # position m
t = 0.0

xout=[]
yout=[]
vout=[]
aout=[]
EPEout=[]
GNSSxout=[]
GNSSyout=[]

# Breaking and cornering
while v>0.0:
    a = -0.6      # acceleration m/s2
    x,y,v = move(x,v,a,dt)
    
    sp = 2.0
    GNSSx, GNSSy = GPS(x,y,sp)
    
    xout.append(x)
    yout.append(y)
    vout.append(v)
    aout.append(a)
    EPEout.append(sp)
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt


# Waiting    
twait = t+20.0
while t<twait:
    a = 0.0      # acceleration m/s2
    x,y,v = move(x,v,a,dt)

    sp = 20.0
    GNSSx, GNSSy = GPS(x,y,sp)
    
    xout.append(x)
    yout.append(y)
    vout.append(v)
    aout.append(a)
    EPEout.append(sp)    
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt
    
# Accelerating
while v<(50.0/3.6):
    a = 4.0      # acceleration m/s2
    x,y,v = move(x,v,a,dt)

    sp = 2.0
    GNSSx, GNSSy = GPS(x,y,sp)
    
    xout.append(x)
    yout.append(y)    
    vout.append(v)
    aout.append(a)
    EPEout.append(sp)    
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt


# Passing a Building    
tdrive = t+25.0
tbadGPS = t+10.0
tbadGPSend=t+12.0
while t<tdrive:
    a = 0.0      # acceleration m/s2
    x,y,v = move(x,v,a,dt)

    if t>tbadGPS and t<tbadGPSend:
        sp = 25.0
        GNSSx, GNSSy = GPS(x,y,sp)
        GNSSy+=15.0
    else:
        sp = 2.0
        GNSSx, GNSSy = GPS(x,y,sp)
    
    xout.append(x)
    yout.append(y)    
    vout.append(v)
    aout.append(a) 
    EPEout.append(sp)    
    GNSSxout.append(GNSSx)
    GNSSyout.append(GNSSy)    
    t+=dt   

# <codecell>

plt.figure(figsize=(8,3))
plt.plot(xout, yout, color='k', label='Ground Truth')
plt.scatter(GNSSxout, GNSSyout, label='GNSS measurements')
plt.legend(loc='best')
plt.axis('equal')
plt.xlabel('Position X ($m$)')
plt.xlim([-10, np.max(xout)+10])
plt.ylabel('Position Y ($m$)')
plt.title('View from top with building (B)')

# Annotations
bbox_props = dict(boxstyle="square,pad=0.8", fc="k", ec="w", lw=2)
t = plt.text(340, -40, "B", ha="left", va="center", rotation=0,
            size=14, color='grey',
            bbox=bbox_props)

plt.annotate('stop', xy=(144, -20), xytext=(110, -60), fontsize=16,
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
       thefile.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % (xout[i], yout[i], 3.6*vout[i], -aout[i], GNSSxout[i], GNSSyout[i], epeout[i]))

# <codecell>


# <codecell>


