#!/usr/bin/python
#############################################################################################
# FILE: 2_simpsupp_buckling.py                                                              #
# Written by Nidish Narayanaa Balaji                                                        #
# Last Modified by N. N. Balaji on 1 Dec 2019                                               #
#                                                                                           #
# This considers the buckling cases (follower & nonfollower) for the simply supported beam. #
#############################################################################################
import numpy as np
import pdb
from numpy import pi
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy.sparse as ss
import scipy.optimize as so
import scipy.linalg as sn
import sparse as sp
import STATICS as nr
import SOLVERS as ns
import DYNAMICS as nd
reload(ns)
reload(nr)

###############################################################################################
# PHYSICAL PROPTERIES                                                                         #
###############################################################################################
rho = 7800.0
E = 210e9
nu = 0.33
G = E/(2.0*(1.0+nu))
R = 1e-3/2  # 2.5 cm
L = 30e-2  # 30 cm
props = type('', (), {})()

# # CIRCULAR SECTION
# A = pi*R**2
# I2 = pi*R**4/4.0
# I4 = pi*R**6/8.0
# props.bfunc = lambda y: 2*np.sqrt(R**2-y**2)

# Square Section
b = 2.5e-2  # 0.5 mm
A = 2*R*b
I2 = 2*R**3*b/3
I4 = 2*R**5*b/5
props.bfunc = lambda y: b


props.EA = E*A
props.GA = G*A
props.EI2 = E*I2
props.EI4 = E*I4
props.RA = rho*A
props.RI2 = rho*I2

props.E = E
props.G = G
props.Npy = 8  # Section quadrature
props.yrange = [-R, R]

# CRITICAL LOADS (ANALYTICAL: EULER BUCKLING)
Pcrits = pi**2*E*I2/L**2*(np.arange(1, 4))**2
print('Euler Buckling Predictions: %e, %e, %e' % tuple(Pcrits[:]))

###############################################################################################

###############################################################################################
# USER INPUT SECTION                                                                          #
###############################################################################################
# CHOOSE STRAIN MEASURE TO DEPLOY
STRAINMEASURES = {  # Dictionary of Strain Measures (we identify the following explicitly)
    -100: 'Integrated Section Strain',
    -1: 'Almansi Strain',
    -0.5: 'Swaiger Strain',
    0: 'Log Strain',
    0.5: 'Cauchy Strain',
    1: 'Green-Lagrange Strain',
    100: 'Kuhn Strain'}
smeasure = -0.5

# -100: [ 48.77507664 205.23509523 504.53466168]
# -1  : [ 48.78051495 205.33141713 505.11776897]
# 1   : [ 48.77505654 205.23513865 504.53492132]

# Forcing Type
Follower_Forcing = False

# SOLVER AND CONTINUATION PARAMETERS
opt = type('', (), {})()
opt.ITMAX = 20
opt.relethr = 1e-16
opt.absethr = 1e-16
opt.absrthr = 1e-10
opt.dsmin = 0.25
opt.dsmax = 1e10
opt.b = 0.5
opt.maxsteps = 10000

# CALCULATE OR LOAD DATA
CALC = True

# PLOT OR NOT
PLOT = True
###############################################################################################

###############################################################################################
# Numerical Solution                                                                          #
###############################################################################################
# Finite Element Scheme
Nn = 11
Nd = Nn*3
No = 5
Xnds = np.linspace(0.0, L, Nn)

# Non-follower force vector
fshape = np.zeros(Nd, dtype=float)
if not(Follower_Forcing):
    fshape[-3] = -1.0

# Boundary Conditions (homogeneous)
bcs = (0, 1, (Nn-1)*3+1)
btrans = ss.csr_matrix(ss.eye(Nd))
btrans = btrans[:, np.setdiff1d(range(Nd), bcs)]

u0 = np.zeros(btrans.shape[1])

# Follower force options
Flwds = nr.FLWLDS()
if Follower_Forcing:
    Flwds.Nl += 1
    Flwds.nd.append(Nn-1)
    Flwds.F.append(np.array([-1.0, 0.0]))


# Function Handle for Continuation - Follower Loading Case
def func(u, l, d3=0, smeasure=smeasure):
    return nr.STATICRESFN(Xnds, np.hstack((u, l)), fshape, btrans, No, props,
                          d3=d3, smeasure=smeasure, NDFls=Flwds)


if CALC:
    ######################################################################################
    # CONTINUATION OF PRIMARY BRANCH
    ######################################################################################
    lstart = 10
    lend = 550
    print('%e, %e' % (lstart, lend))

    opt.minl = min(lstart, lend)
    opt.maxl = max(lstart, lend)
    # pdb.set_trace()
    unf = ns.SPNEWTONSOLVER(lambda u: func(u, lstart)[0:2], u0, opt)

    ds = (lstart+lend)/50
    X, lam, mE = ns.CONTINUESOLS(func, unf.x, lstart, lend, ds, opt, adapt=0)  #, ALfn=ns.ARCORTHOGFN)

    ######################################################################################
    # DYNAMIC STABILITY EXPONENTS
    ######################################################################################
    dE = []
    for k in range(len(X)):
        Jlin = nd.LINEARIZEDJAC(Xnds, X[k], X[k]*0, No, props, btrans, lambda u: func(u, lam[k]))
        dE.append(np.sort(np.linalg.eigvals(Jlin.todense())))
        print('%d/%d DONE.' % (k+1, len(X)))
    c1 = np.where(np.real(np.array(dE)[:, -1]) > 1e-5)[0][0]
    c2 = np.where(np.real(np.array(dE)[:, -2]) > 1e-5)[0][0]
    c3 = np.where(np.real(np.array(dE)[:, -3]) > 1e-5)[0][0]
    ####################################################################################
    # Bifurcation 1 - Singularity through first Eigenvalue
    ####################################################################################
    du1, al1, du2, al2, ots = ns.SINGTANGENTS(lambda u, lam, d3=0:
                                              func(u, lam, d3=d3, smeasure=smeasure),
                                              X, lam, mE, opt, ei=0)
    du2 = np.sign(du2[-1])*du2

    us1 = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du1*al1, al1)), ds, lam[ots.cpi-1], opt)
    us2a = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du2*al2, al2)), ds, lam[ots.cpi-1], opt)
    us2b = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((-du2*al2, al2)), ds, lam[ots.cpi-1], opt)
    print('%d, %d, %d' % (us1.status, us2a.status, us2b.status))
    print('%e, %e, %e' % (us1.lam, us2a.lam, us2b.lam))
    Xb1a, lamb1a, mEb1a = ns.CONTINUESOLS(func, us2a.x, us2a.lam, lend,
                                          0.25, opt, ALfn=ns.ARCORTHOGFN)
    Xb1b, lamb1b, mEb1b = ns.CONTINUESOLS(func, us2b.x, us2b.lam, lend,
                                          0.25, opt, ALfn=ns.ARCORTHOGFN)
    # pdb.set_trace()
    ####################################################################################
    # Bifurcation 2 - Singularity through Second Eigenvalue
    ####################################################################################
    du1, al1, du2, al2, ots = ns.SINGTANGENTS(lambda u, lam, d3=0:
                                              func(u, lam, d3=d3, smeasure=smeasure),
                                              X, lam, mE, opt, ei=1)
    du2 = np.sign(du2[-1])*du2
    
    us1 = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du1*al1, al1)), ds, lam[ots.cpi-1], opt)
    us2a = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du2*al2, al2)), ds, lam[ots.cpi-1], opt)
    us2b = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((-du2*al2, al2)), ds, lam[ots.cpi-1], opt)
    print('%d, %d, %d' % (us1.status, us2a.status, us2b.status))
    print('%e, %e, %e' % (us1.lam, us2a.lam, us2b.lam))
    Xb2a, lamb2a, mEb2a = ns.CONTINUESOLS(func, us2a.x, us2a.lam, lend,
                                          0.25, opt, ALfn=ns.ARCORTHOGFN)
    Xb2b, lamb2b, mEb2b = ns.CONTINUESOLS(func, us2b.x, us2b.lam, lend,
                                          0.25, opt, ALfn=ns.ARCORTHOGFN)
    ####################################################################################
    # Bifurcation 3 - Singularity through Third Eigenvalue
    ####################################################################################
    du1, al1, du2, al2, ots = ns.SINGTANGENTS(lambda u, lam, d3=0:
                                              func(u, lam, d3=d3, smeasure=smeasure),
                                              X, lam, mE, opt, ei=2)
    du2 = np.sign(du2[-1])*du2
    
    us1 = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du1*al1, al1)), ds, lam[ots.cpi-1], opt)
    us2a = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du2*al2, al2)), ds, lam[ots.cpi-1], opt)
    us2b = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((-du2*al2, al2)), ds, lam[ots.cpi-1], opt)
    print('%d, %d, %d' % (us1.status, us2a.status, us2b.status))
    print('%e, %e, %e' % (us1.lam, us2a.lam, us2b.lam))
    Xb3a, lamb3a, mEb3a = ns.CONTINUESOLS(func, us2a.x, us2a.lam, lend,
                                          0.25, opt, ALfn=ns.ARCORTHOGFN)
    Xb3b, lamb3b, mEb3b = ns.CONTINUESOLS(func, us2b.x, us2b.lam, lend,
                                          0.25, opt, ALfn=ns.ARCORTHOGFN)
    ####################################################################################
    
    ###########################################################################################
    # SAVE RESULT
    ###########################################################################################
    fil = open('./DATS/SSBUCK_%.1f.pkl' % (smeasure), 'w')
    pickle.dump({'smeasure': smeasure, 'Follower_Forcing': Follower_Forcing,
                 'Xnds': Xnds, 'lstart': lstart, 'lend': lend,
                 'X': X, 'lam': lam, 'mE': mE, 'c1': c1, 'c2': c2, 'c3': c3,
                 'Xb1a': Xb1a, 'lamb1a': lamb1a, 'mEb1a': mEb1a,
                 'Xb1b': Xb1b, 'lamb1b': lamb1b, 'mEb1b': mEb1b,
                 'Xb2a': Xb2a, 'lamb2a': lamb2a, 'mEb2a': mEb2a,
                 'Xb2b': Xb2b, 'lamb2b': lamb2b, 'mEb2b': mEb2b,
                 'Xb3a': Xb3a, 'lamb3a': lamb3a, 'mEb3a': mEb3a,
                 'Xb3b': Xb3b, 'lamb3b': lamb3b, 'mEb3b': mEb3b,
                 'Pcrits': Pcrits},
                fil)
    fil.close()
    ###########################################################################################
else:
    fil = open('./DATS/SSBUCK_%.1f.pkl' % (smeasure), 'r')
    data = pickle.load(fil)
    fil.close()

    Pcrits = pi**2*E*I2/L**2*(np.arange(1, 4))**2

    smeasure = data['smeasure']
    Follower_Forcing = data['Follower_Forcing']
    Xnds = data['Xnds']
    lstart = data['lstart']
    lend = data['lend']
    X = data['X']
    lam = data['lam']
    mE = data['mE']
    c1 = data['c1']
    c2 = data['c2']
    c3 = data['c3']
    Xb1a = data['Xb1a']
    lamb1a = data['lamb1a']
    mEb1a = data['mEb1a']
    Xb1b = data['Xb1b']
    lamb1b = data['lamb1b']
    mEb1b = data['mEb1b']
    Xb2a = data['Xb2a']
    lamb2a = data['lamb2a']
    mEb2a = data['mEb2a']
    Xb2b = data['Xb2b']
    lamb2b = data['lamb2b']
    mEb2b = data['mEb2b']
    Xb3a = data['Xb3a']
    lamb3a = data['lamb3a']
    mEb3a = data['mEb3a']
    Xb3b = data['Xb3b']
    lamb3b = data['lamb3b']
    mEb3b = data['mEb3b']


# CRITICAL LOAD
if smeasure in STRAINMEASURES.keys():
    lt = STRAINMEASURES[smeasure]
else:
    lt = 'Seth-Hill: %.2f' % (smeasure)
print('%s; Critical Loads:' % (lt))
print(np.array([lamb1a[0], lamb2a[0], lamb3a[0]]))
print('Euler Buckling Critical Loads:')
print(Pcrits)

####################################################################################
# PLOTS                                                                            #
####################################################################################
if PLOT:
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, clear=True, num=smeasure+101, squeeze=True)
    axes[0].plot(lam, (np.array(X).dot(btrans.T.todense()))[:, -2], 'k.-', label='Main branch')
    axes[0].plot(lamb1a, (np.array(Xb1a))[:, -2], 'b-', label='Branch 1a')
    axes[0].plot(lamb1b, (np.array(Xb1b))[:, -2], 'r-', label='Branch 1b')
    axes[0].plot(lamb2a, (np.array(Xb2a))[:, -2], 'b.-', label='Branch 2a')
    axes[0].plot(lamb2b, (np.array(Xb2b))[:, -2], 'r.-', label='Branch 2b')
    axes[0].plot(lamb3a, (np.array(Xb3a))[:, -2], 'b+-', label='Branch 3a')
    axes[0].plot(lamb3b, (np.array(Xb3b))[:, -2], 'r+-', label='Branch 3b')
    axes[0].set(xlabel='Forcing (N)', ylabel='X Amplitude (m)', ylim=(-2.0, 2.0))
    axes[0].grid()
    axes[0].legend(loc='upper center', ncol=3)
    axes[1].plot(lam, (np.array(X))[:, -1], 'k.-')
    axes[1].plot(lamb1a, (np.array(Xb1a))[:, -1], 'b-')
    axes[1].plot(lamb1b, (np.array(Xb1b))[:, -1], 'r-')
    axes[1].plot(lamb2a, (np.array(Xb2a))[:, -1], 'b.-')
    axes[1].plot(lamb2b, (np.array(Xb2b))[:, -1], 'r.-')
    axes[1].plot(lamb3a, (np.array(Xb3a))[:, -1], 'b+-')
    axes[1].plot(lamb3b, (np.array(Xb3b))[:, -1], 'r+-')
    axes[1].set(xlabel='Forcing (N)', ylabel=r'End Angle ($\theta$)')
    axes[1].grid()
    for k in range(3):
        axes[0].axvline(x=Pcrits[k], color='k', linestyle='--')
        axes[1].axvline(x=Pcrits[k], color='k', linestyle='--')
        axes[0].arrow(lam[c1], X[c1][-3]-1.4, 0.0, 1.0, head_width=5000.0, head_length=0.2, fc='g', ec='g', linewidth=4)
        axes[0].arrow(lam[c2], X[c2][-3]-1.4, 0.0, 1.0, head_width=5000.0, head_length=0.2, fc='g', ec='g', linewidth=4)
        axes[0].arrow(lam[c3], X[c3][-3]-1.4, 0.0, 1.0, head_width=5000.0, head_length=0.2, fc='g', ec='g', linewidth=4)
        axes[1].arrow(lam[c1], X[c1][-3]-1.75, 0.0, 1.25, head_width=5000.0, head_length=0.25, fc='g', ec='g', linewidth=4)
        axes[1].arrow(lam[c2], X[c2][-3]-1.75, 0.0, 1.25, head_width=5000.0, head_length=0.25, fc='g', ec='g', linewidth=4)
        axes[1].arrow(lam[c3], X[c3][-3]-1.75, 0.0, 1.25, head_width=5000.0, head_length=0.25, fc='g', ec='g', linewidth=4)
    # fig.savefig('./FIGS/NONFOLLOWER_BUCKLING_RESP_SIMPSUPP.pdf', dpi=100)
    # fig.savefig('./FIGS/FOLLOWER_BUCKLING_RESP_SIMPSUPP.pdf', dpi=100)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, clear=True, num=2, squeeze=True)
    k1s = (np.array([1.0, 25.0, 50.0, 75.0, 100.0])/100*(len(lamb1a)-1)).astype(int)
    k2s = (np.array([1.0, 25.0, 50.0, 75.0, 100.0])/100*(len(lamb1b)-1)).astype(int)
    axes[0].plot(Xnds, Xnds*0, 'k--')
    for k in k1s:
        axes[0].plot(Xnds+btrans[0::3, :].dot(Xb1a[k]), btrans[1::3, :].dot(Xb1a[k]), '.-', label='Force = %.0f' % (lamb1a[k]))
    axes[0].grid()
    axes[0].legend(loc='upper center', ncol=3)
    axes[0].set(ylabel='Y Coordinate (m)', title='Branch 1a')
    axes[1].plot(Xnds, Xnds*0, 'k--')
    for k in k2s:
        axes[1].plot(Xnds+btrans[0::3, :].dot(Xb1b[k]), btrans[1::3, :].dot(Xb1b[k]), '.-', label='_nolegend_')
    axes[1].grid()
    axes[1].set(ylabel='Y Coordinate (m)', title='Branch 1b')
    # fig.savefig('./FIGS/NONFOLLOWER_BUCKLING_1_SIMPSUPP.pdf', dpi=100)
    # fig.savefig('./FIGS/FOLLOWER_BUCKLING_1_SIMPSUPP.pdf', dpi=100)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, clear=True, num=3, squeeze=True)
    k1s = (np.array([1.0, 25.0, 50.0, 75.0, 100.0])/100*(len(lamb2a)-1)).astype(int)
    k2s = (np.array([1.0, 25.0, 50.0, 75.0, 100.0])/100*(len(lamb2b)-1)).astype(int)
    axes[0].plot(Xnds, Xnds*0, 'k--')
    for k in k1s:
        axes[0].plot(Xnds+btrans[0::3, :].dot(Xb2a[k]), btrans[1::3, :].dot(Xb2a[k]), '.-', label='Force = %.0f' % (lamb2a[k]))
    axes[0].grid()
    axes[0].legend(loc='upper center', ncol=3)
    axes[0].set(ylabel='Y Coordinate (m)', title='Branch 2a')
    axes[1].plot(Xnds, Xnds*0, 'k--')
    for k in k2s:
        axes[1].plot(Xnds+btrans[0::3, :].dot(Xb2b[k]), btrans[1::3, :].dot(Xb2b[k]), '.-', label='_nolegend_')
    axes[1].grid()
    axes[1].set(ylabel='Y Coordinate (m)', title='Branch 2b')
    # fig.savefig('./FIGS/NONFOLLOWER_BUCKLING_2_SIMPSUPP.pdf', dpi=100)
    # fig.savefig('./FIGS/FOLLOWER_BUCKLING_2_SIMPSUPP.pdf', dpi=100)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, clear=True, num=4, squeeze=True)
    k1s = (np.array([1.0, 25.0, 50.0, 75.0, 100.0])/100*(len(lamb3a)-1)).astype(int)
    k2s = (np.array([1.0, 25.0, 50.0, 75.0, 100.0])/100*(len(lamb3b)-1)).astype(int)
    axes[0].plot(Xnds, Xnds*0, 'k--')
    for k in k1s:
        axes[0].plot(Xnds+btrans[0::3, :].dot(Xb3a[k]), btrans[1::3, :].dot(Xb3a[k]), '.-', label='Force = %.0f' % (lamb3a[k]))
    axes[0].grid()
    axes[0].legend(loc='upper center', ncol=3)
    axes[0].set(ylabel='Y Coordinate (m)', title='Branch 3a')
    axes[1].plot(Xnds, Xnds*0, 'k--')
    for k in k2s:
        axes[1].plot(Xnds+btrans[0::3, :].dot(Xb3b[k]), btrans[1::3, :].dot(Xb3b[k]), '.-', label='_nolegend_')
    axes[1].grid()
    axes[1].set(ylabel='Y Coordinate (m)', title='Branch 3b')
    # fig.savefig('./FIGS/NONFOLLOWER_BUCKLING_3_SIMPSUPP.pdf', dpi=100
    # fig.savefig('./FIGS/FOLLOWER_BUCKLING_3_SIMPSUPP.pdf', dpi=100)
