# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
%pylab inline --no-import-all

# <headingcell level=1>

# Adaptive Values for Measurement Covariance Matrix $R$

# <markdowncell>

# $\sigma_p^2 = c \cdot \sigma_\text{speed}^2 + \sigma_\text{EPE}^2$
# 
# $\sigma_v = (v+\epsilon)^{-\xi}$
# 
# $\sigma_\text{EPE} = \zeta \cdot EPE$

# <codecell>

vrange=np.arange(0.0, 20.01, 0.01)
eperange=np.arange(0.0, 8.0, 1)
sp=np.empty([eperange.size,vrange.size])

epsilon = 0.1
xi      = 500.0
zeta    = 50.0

for vi, v in enumerate(vrange):
    for epei, epe in enumerate(eperange):
        sv=xi/(v+epsilon)
        sepe=zeta*epe
        
        sp[epei,vi] = sv**2 + sepe**2

# <codecell>

# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label

V=np.linspace(0,1e5,5+1)

plt.figure(figsize=(5,2))
CS = plt.contourf(vrange, eperange, sp, V, cmap=plt.get_cmap('binary'), extend='max')
cbar=plt.colorbar(CS)
cbar.ax.set_ylabel(u'$\sigma_x^2$ and $\sigma_y^2$ values', rotation=270)
cbar.ax.set_xlabel(u'')

tstr = r'$R$ with $\epsilon=$ %.2f, $\xi=$ %.2f, $\zeta=$ %.2f' % (epsilon, xi, zeta)
plt.title(tstr, size=12)
plt.xlabel('speed $v$ in $m/s$')
plt.ylabel('$EPE$ in $m$')

#fname = 'R-%s-%s-%s' % (epsilon, xi, zeta)
#fname = fname.replace('.','-')
plt.savefig('R.eps', bbox_inches='tight')

# <markdowncell>

# Awesome!

