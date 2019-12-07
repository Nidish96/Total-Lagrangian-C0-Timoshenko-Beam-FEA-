#!/usr/bin/python
###############################################################################################
# FILE: 1_simple_cases.py                                                                     #
# Written by Nidish Narayanaa Balaji                                                          #
# Last Modified on 3 Dec 2019                                                                 #
#                                                                                             #
# The current script considers a test cantilever beam with five different loading cases where #
# the different strain measures are compared. Coarse-stepped continuation is employed for     #
# convergence.                                                                                #
###############################################################################################
import numpy as np
from numpy import pi
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.optimize as so
import scipy.sparse as ss
import STATICS as nr
import SOLVERS as ns
import pdb
reload(nr)
reload(ns)

###############################################################################################
# PHYSICAL PROPTERIES                                                                         #
###############################################################################################
E = 210e9
nu = 0.33
G = E/(2.0*(1.0+nu))
L = 1.0
R = 0.5
props = type('', (), {})()

# # Circular Section
# A = pi*R**2
# I2 = pi*R**4/4.0
# I4 = pi*R**6/8.0
# props.bfunc = lambda y: 2*np.sqrt(R**2-y**2)

# Square Section
A = (2*R)**2
I2 = 4*R**4/3
I4 = 4*R**6/5
props.bfunc = lambda y: 2*R

props.EA = E*A
props.GA = G*A
props.EI2 = E*I2
props.EI4 = E*I4
props.E = E
props.G = G
props.Npy = 8  # Section quadrature
props.yrange = [-R, R]
###############################################################################################

###############################################################################################
# USER INPUT SECTION                                                                          #
###############################################################################################
# CHOOSE STRAIN MEASURES TO COMPARE HERE
STRAINMEASURES = {  # Dictionary of Strain Measures (we identify the following explicitly)
    -100: 'Integrated Section Strain',
    -1: 'Almansi Strain',
    -0.5: 'Swaiger',
    0: 'Log Strain',
    0.5: 'Cauchy Strain',
    1: 'Green-Lagrange Strain',
    100: 'Kuhn Strain'}
smeasures = [-100, -1, -0.5, 0, 0.5, 1, 1.5, 100]

# CHOOSE TEST CASE HERE
# 1: Cantilever Beam with Transverse Load At Tip
# 2: Cantilever Beam with Moment Load At Tip
# 3: Cantilever Beam with Axial Load At Tip
# 4: Cantilever Beam with Transverse Follower Load At Tip
# 5: Cantilever Beam with Axial Follower Load At Tip
cas = 5

# CHOOSE FORCING AMPLITUDE HERE
# High levels Are Suggested for Each Case
# Decimal overflow seems to happen some where along the iterates for a few strains (especially negative exponents and log-strain).
if cas == 1:
    famp = 4e9
elif cas == 2:
    famp = 4e9
elif cas == 3:
    famp = 1e10  # Tension
elif cas == 4:
    famp = 5e9
elif cas == 5:
    famp = 1e10  # Tension
# The current solver properties have large step sizes since final state is more
# important here than path. Therefore this can't be used for comperissive loads
# since the curve will try to cross over bifurcation points with large step sizes
# and fail


# PLOTTING PARAMETERS
aflag = 1  # Set to 1 to plot analytic (linear) solutions

# SOLVER AND CONTINUATION PARAMETERS
opt = type('', (), {})()
opt.ITMAX = 50
opt.relethr = 1e-16
opt.absethr = 1e-16
opt.absrthr = 1e-10
opt.dsmin = 1.0
opt.dsmax = famp
opt.b = 0.5
opt.maxsteps = 10
opt.minl = 0.01
opt.maxl = famp
###############################################################################################

###############################################################################################
# Numerical Solution                                                                          #
###############################################################################################
# Finite Element Scheme
Nn = 11  # Fixed to 11 nodes
No = 5  # Quadrature Points per element
Nd = Nn*3  # Number of DOFs
Xnds = np.linspace(0, L, Nn)

# Boundary Conditions
bcs = (0, 1, 2)  # Fixed Dof's: (ux, uy, theta_y) at x = 0
btrans = ss.csr_matrix(ss.eye(Nd))
btrans = btrans[:, np.setdiff1d(range(Nd), bcs)]

# Loading Setup
fshape = np.zeros(Nd)  # Non-Follower Loads
Flwds = nr.FLWLDS()  # Follower Loads
u0 = np.zeros(btrans.shape[1])
if cas == 1:
    #########################
    # Tip-Loaded Cantilever #
    #########################
    fshape[-2] = 1.0
    rdofs = range(1, Nd, 3)  # Transverse Displacements
    yt = 'Y'

    def x_an(x): return [x*0]  # Analytic Solution (EB)

    def u_an(x): return famp*x**2*(3.0*L-x)/(6.0*props.EI2), famp*x*(2.0*L-x)/(2.0*props.EI2)  # Analytic Solution (EB)
elif cas == 2:
    ##############################
    # Cantilever with tip moment #
    ##############################
    fshape[-1] = 1.0
    rdofs = range(1, Nd, 3)  # Transverse Displacements
    yt = 'Y'

    def x_an(x): return [x*0]  # Analytic Solution (EB)

    def u_an(x): return famp*x**2/(2.0*props.EI2), famp*x/props.EI2  # Analytical Solution (EB)
elif cas == 3:
    ######################
    # Axially loaded bar #
    ######################
    fshape[-3] = 1.0
    rdofs = range(0, Nd, 3)  # Axial Displacements
    yt = 'Y'

    def x_an(x): return [famp*x/props.EA]  # Analytic Solution (EB)

    def u_an(x): return [famp*x/props.EA]  # Analytical Solution (Linear Bar)
elif cas == 4:
    ############################################
    # Cantilever with transverse follower load #
    ############################################
    Flwds.Nl += 1
    Flwds.nd.append(Nn-1)
    Flwds.F.append(np.array([0.0, 1.0]))

    rdofs = range(1, Nd, 3)  # Transverse Displacements
    yt = 'Y'

    def x_an(x): return [x*0]  # Analytic Solution (EB)

    def u_an(x): return famp*x**2*(3.0*L-x)/(6.0*props.EI2), famp*x*(2.0*L-x)/(2.0*props.EI2)  # Analytic Solution (EB)
elif cas == 5:
    #######################################
    # Cantilever with Axial Follower Load #
    #######################################
    Flwds.Nl += 1
    Flwds.nd.append(Nn-1)
    Flwds.F.append(np.array([1.0, 0.0]))

    rdofs = range(0, Nd, 3)  # Axial Displacements
    yt = 'X'

    def x_an(x): return [x*0]  # Analytic Solution (EB)

    def u_an(x): return [famp*x/props.EA]  # Analytical Solution (Linear Bar)

u0 = btrans[rdofs, :].T.dot(u_an(Xnds)[0])/famp


# Function Handle for Continuation
def func(u, l, d3=0, smeasure=-100): return nr.STATICRESFN(Xnds, np.hstack((u, l)), fshape, btrans, No, props, NDFls=Flwds, d3=d3, smeasure=smeasure)


uphs = np.zeros((Nd, len(smeasures)))
for i in range(len(smeasures)):
    try:
        # Continuation (to converge to large amplitudes)
        X, lam, _ = ns.CONTINUESOLS(lambda u, l: func(u, l, smeasure=smeasures[i]), u0*0.01,
                                    10.0, famp, famp/5, opt, adapt=1, ALfn=ns.ARCORTHOGFN)
        # Exact Point solution for comparison
        us = ns.SPNEWTONSOLVER(lambda u: func(u, famp, smeasure=smeasures[i])[0:2], X[-1], opt)
        u0 = X[0]
        uphs[:, i] = btrans.dot(us.x)
        print('%.1f %d %d' % (smeasures[i], us.success, us.status))
    except Exception as inst:
        uphs[:, i] = btrans.dot(us.x*0)
        print('dnan')
###############################################################################################

###############################################################################################
# Plotting                                                                                    #
###############################################################################################
xg = np.linspace(0.0, L, Nn*10)
plt.ion()
fig, ax = plt.subplots(num=cas, clear=True)
ax.plot(xg, xg*0, 'k--', label='_nolegend_')

for i in range(len(smeasures)):
    if smeasures[i] in STRAINMEASURES.keys():
        lt = STRAINMEASURES[smeasures[i]]
    else:
        lt = 'Seth-Hill: %.2f' % (smeasures[i])
    ax.plot(Xnds+uphs[0::3, i], uphs[rdofs, i], 'o-', label=lt)
if aflag:
    ax.plot(xg+x_an(xg)[0], u_an(xg)[0], 'k:', label='Analytical (Linear) Solution')
ax.legend(loc='upper left')
plt.xlim([0, np.max(np.ceil((Xnds+np.max(uphs[0::3, :], 1))*10)/10)])
plt.grid()

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
plt.xlabel('X Coordinate (m)')
plt.ylabel('%s Deflection (m)' % (yt))

fig.savefig('./FIGS/VALIDATION_F%d_CASE%d.pdf' % (famp, cas), dpi=100)
###############################################################################################

print 'Total Error: %e' % (np.sum(np.diff(uphs)**2))
