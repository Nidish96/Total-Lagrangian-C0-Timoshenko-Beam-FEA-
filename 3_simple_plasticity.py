#!/usr/bin/python
###############################################################################################
# FILE: 1_simple_cases.py                                                                     #
# Written by Nidish Narayanaa Balaji                                                          #
# Last Modified on 3 Dec 2019                                                                 #
#                                                                                             #
# The current script considers a test cantilever beam with five different loading cases where #
# the different strain measures are compared.                                                 #
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
# smeasures = [-100, -1, -0.5, 0, 0.5, 1, 2, 100]
smeasure = 1


famp = 1000

# SOLVER AND CONTINUATION PARAMETERS
opt = type('', (), {})()
opt.ITMAX = 50
opt.relethr = 1e-16
opt.absethr = 1e-16
opt.absrthr = 1e-10
opt.dsmin = 1.0
opt.dsmax = famp
opt.b = 0.5
opt.maxsteps = 10000
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

######################
# Axially loaded bar #
######################
fshape[-3] = 1.0
rdofs = range(0, Nd, 3)  # Axial Displacements
yt = 'Y'


def x_an(x): return [famp*x/props.EA]  # Analytic Solution (EB)


def u_an(x): return [famp*x/props.EA]  # Analytical Solution (Linear Bar)


u0 = btrans[rdofs, :].T.dot(u_an(Xnds)[0])/famp


# Function Handle for Linear Elastic Material
def func(u, l, d3=0, smeasure=1): return nr.STATICRESFN(Xnds, np.hstack((u, l)), fshape, btrans, No, props, NDFls=Flwds, d3=d3, smeasure=smeasure)


# Function Handle for Elasto-Plastic Material
def func_ep(u, l, d3=0, smeasure=1): return nr.STATICRESFN_PLAN(Xnds, np.hstack((u, l)), fshape, btrans, No, props, NDFls=Flwds, d3=d3, smeasure=smeasure)


############
# SOLUTION #
############
famp = 10
us = ns.SPNEWTONSOLVER(lambda u: func(u, famp, smeasure=smeasure)[0:2], u0, opt)


# usp = ns.SPNEWTONSOLVER(lambda u: func_ep(u, famp, smeasure=smeasure)[0:2], u0, opt)
usp = so.fsolve(lambda u: func_ep(u, famp)[0], u0)


plt.ion()
plt.figure()
plt.clf()
plt.plot(Xnds, btrans[0::3, :].dot(us.x), 'o-')
plt.plot(Xnds, btrans[0::3, :].dot(usp), '+-')
