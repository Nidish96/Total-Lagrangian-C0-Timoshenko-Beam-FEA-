 #!/usr/bin/python
###############################################################################################
# FILE: validate.py                                                                           #
# Written by Nidish Narayanaa Balaji on 17 Nov 2018                                           #
#                                                                                             #
# The current script considers a test cantilever beam with three different loading cases      #
# (corresponding to a concentrated transverse, axial, & moment applied at the free end).      #
# The convergence against the linear solution for the bar and Euler-Bernouilli approximations #
# is demonstrated.                                                                            #
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

# Physical Properties
E = 210e9
nu = 0.33
G = E/(2.0*(1.0+nu))
R = 1e-2
L = 1.0
A = pi*R**2
I2 = pi*R**4/4.0
I4 = pi*R**6/8.0
props = type('', (), {})()
props.EA = E*A
props.GA = G*A
props.EI2 = E*I2
props.EI4 = E*I4

props.E = E
props.G = G
props.Npy = 8  # Section quadrature
props.bfunc = lambda y: 2*np.sqrt(R**2-y**2)
props.yrange = [-R, R]

###############################################################################################
# USER INPUT SECTION                                                                          #
###############################################################################################
cas = 1  # CHOOSE TEST CASE HERE
# 1: Cantilever Beam with Transverse Load At Tip
# 2: Cantilever Beam with Moment Load At Tip
# 3: Cantilever Beam with Axial Load At Tip
famp = 1e8   # FORCING AMPLITUDE: High levels by Case [1: 1e4, 2: , 3: 1e8]
famp = 1e4
smeasures = [-1, 1, 0, 2]   # Choose Strain Measures to compare:
# -1: integrated Green-Lagrange;
# 0: log;
# 1: Green-Lagrange (numerical);
# <other decimals>: arbitrary Seth-Hill strain exponents
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

nr.FLWFRC_N
# Loading Setup
fshape = np.zeros(Nd)
u0 = np.zeros(btrans.shape[1])
if cas == 1:  # Tip-Loaded Cantilever
    fshape[-2] = 1.0
    rdofs = range(1, Nd, 3)  # Transverse Displacements
    yt = 'Y'

    def x_an(x): return [x*0]  # Analytic Solution (EB)

    def u_an(x): return famp*x**2*(3.0*L-x)/(6.0*props.EI2), famp*x*(2.0*L-x)/(2.0*props.EI2)  # Analytic Solution (EB)
elif cas == 2:  # Cantilever with tip moment
    fshape[-1] = 1.0
    rdofs = range(1, Nd, 3)  # Transverse Displacements
    yt = 'Y'

    def x_an(x): return [x*0]  # Analytic Solution (EB)

    def u_an(x): return famp*x**2/(2.0*props.EI2), famp*x/props.EI2  # Analytical Solution (EB)
elif cas == 3:  # Axially loaded bar
    fshape[-3] = 1.0
    rdofs = range(0, Nd, 3)  # Axial Displacements
    yt = 'Y'

    def x_an(x): return [famp*x/props.EA]  # Analytic Solution (EB)

    def u_an(x): return [famp*x/props.EA]  # Analytical Solution (Linear Bar)
u0 = btrans[rdofs, :].T.dot(u_an(Xnds)[0])/famp
# Solver options
opt = type('', (), {})()
opt.ITMAX = 50
opt.relethr = 1e-16
opt.absethr = 1e-16
opt.absrthr = 1e-10
opt.dsmin = 1.0
opt.dsmax = famp/2
opt.b = 0.5
opt.maxsteps = 10000
opt.minl = 0.01
opt.maxl = famp


# Function Handle for Continuation
def func(u, l, d3=0, smeasure=-1): return nr.STATICRESFN(Xnds, np.hstack((u, l)), fshape, btrans, No, props, d3=d3, smeasure=smeasure)


uphs = np.zeros((Nd, len(smeasures)))
for i in range(len(smeasures)):
    try:
        # Continuation (to converge to large amplitudes)
        X, lam, _ = ns.CONTINUESOLS(lambda u, l: func(u, l, smeasure=smeasures[i]), u0*0.01,
                                    10.0, famp, famp/5, opt, adapt=1, ALfn=ns.ARCORTHOGFN)
        # Exact Point solution for comparison
        us = ns.SPNEWTONSOLVER(lambda u: nr.STATICRESFN(Xnds, np.hstack((u, famp)), fshape,
                                                        btrans, No, props,
                                                        smeasure=smeasures[i])[0:2], X[-1], opt)
        u0 = X[0]
        uphs[:, i] = btrans.dot(us.x)
        print('%d %d %d' % (smeasures[i], us.success, us.status))
    except Exception as inst:
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
    if smeasures[i] in nr.STRAINMEASURES.keys():
        lt = nr.STRAINMEASURES[smeasures[i]]
    else:
        lt = 'Seth-Hill: %.2f' % (smeasures[i])
    ax.plot(Xnds+uphs[0::3, i], uphs[rdofs, i], 'o-', label=lt)
ax.plot(xg+x_an(xg)[0], u_an(xg)[0], 'k:', label='Analytical (Linear) Solution')
ax.legend(loc='upper left')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
plt.xlabel('X Coordinate (m)')
plt.ylabel('%s Deflection (m)' % (yt))

# fig.savefig('./FIGS/VALIDATION_F%d_CASE%d.pdf' % (famp, cas), dpi=100)
###############################################################################################
