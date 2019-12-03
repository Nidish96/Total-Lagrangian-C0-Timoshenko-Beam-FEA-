#!/usr/bin/python
##########################################################################
# FILE: simpsupp_snapthrough.py                                          #
# Written by Nidish Narayanaa Balaji                                     #
# Last modified on 1 Dec 2018 by N. N. Balaji                            #
#                                                                        #
# Considers first the buckling of a beam under tip-displacement boundary #
# condition, and then considers its snap-through response.               #
##########################################################################
import numpy as np
from numpy import pi
# import time
import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as ss
import scipy.optimize as so
import scipy.linalg as sn
import sparse as sp
import STATICS as nr
import SOLVERS as ns
import DYNAMICS as nd
reload(ns)
reload(nr)

# Physical properties
E = 210e9
nu = 0.33
G = E/(2.0*(1.0+nu))
R = 1e-2
L = 1.0
A = pi*R**2
I2 = pi*R**4/4.0
I4 = pi*R**6/8.0
rho = 7800.0
props = type('', (), {})()
props.EA = E*A
props.GA = G*A
props.EI2 = E*I2
props.EI4 = E*I4
props.RA = rho*A
props.RI2 = rho*I2

Nn = 11
Nd = Nn*3
No = 5
Xnds = np.linspace(0.0, L, Nn)

# Non-follower force vector
fshape = np.zeros(Nd, dtype=float)

# Boundary Conditions (homogeneous)
bcs = (0, 1, (Nn-1)*3+1)
btrans = ss.csr_matrix(ss.eye(Nd))
btrans = btrans[:, np.setdiff1d(range(Nd), bcs)]

u0 = np.zeros(btrans.shape[1])

# Inhomogeneous Boundary Condition
ihbcs = nr.IHBCS()
ihbcs.push(Nn, 0, lambda famp: (-famp, -1.0))

# Continuation formulation
opt = type('', (), {})()
opt.ITMAX = 20
opt.relethr = 1e-16
opt.absethr = 1e-16
opt.absrthr = 1e-5
opt.dsmin = 1e-10
opt.dsmax = 5e-3
opt.b = 0.5
opt.maxsteps = 100


def func(u, l, d3=0): return nr.STATICRESFN(Xnds, np.hstack((u, l)), fshape*0, btrans, No, props, d3=d3, IhBcs=ihbcs)


lstart = 1e-6
lend = 8.0e-3
opt.minl = lstart
opt.maxl = lend
unf = ns.SPNEWTONSOLVER(lambda u: func(u, lstart)[0:2], u0, opt)
ds = 1e-4
X, lam, mE = ns.CONTINUESOLS(func, unf.x, lstart, lend, ds, opt, adapt=0)
####################################################################################
# Bifurcation 1 - Singularity through first Eigenvalue
####################################################################################
du1, al1, du2, al2, ots = ns.SINGTANGENTS(func, X, lam, mE, opt, ei=0)
du2 = np.sign(du2[-2])*du2

us1 = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du1*al1, al1)), ds, lam[ots.cpi-1], opt)
us2 = ns.CONTSTEP(func, X[ots.cpi-1], np.hstack((du2*al2, al2)), ds, lam[ots.cpi-1], opt)
Xb1, lamb1, mEb1 = ns.CONTINUESOLS(func, us2.x, us2.lam, lend, 1e-6, opt, ALfn=ns.ARCORTHOGFN)
####################################################################################

#######################################
# New Setup for Snap-Through Analysis #
#######################################
opt_st = opt
opt_st.maxsteps = 100
# kst = 28  # Simple Case - No bifurcations (only 1 Turning Points)
# kst = 39  # Beyond Second Critical - Symmetric Bifurcation
# kst = 50  # Beyond Third Critical - Symmetric Bifurcation and 2 Turning points)
kst = 65  # Beyond Fourth Critical - 2 Symmetric Bifurcations and "Fold"

# Inhomogeneous Boundary Condition
ihbcs_st = nr.IHBCS()
ihbcs_st.push(Nn, 0, lambda famp: (-lamb1[kst], 0.0), cnum=1e6)

# Non-follower force vector
fshape_st = np.zeros(Nd, dtype=float)
fcond = 1e6
fshape_st[5*3+1] = -fcond


def func_st(u, l, d3=0): return nr.STATICRESFN(Xnds, np.hstack((u, l)), fshape_st,
                                               btrans, No, props, d3=d3, IhBcs=ihbcs_st)


opt_st.dsmax = 10000.0/fcond
opt_st.dsmin = 10.0/fcond
opt.maxsteps = 600
# lstart_st = -20000.0/fcond
# lend_st = 20000.0/fcond
lstart_st = -60000.0/fcond
lend_st = 60000.0/fcond
opt_st.minl = lstart_st
opt_st.maxl = lend_st
u0 = Xb1[kst]
unf = ns.SPNEWTONSOLVER(lambda u: func_st(u, lstart_st)[0:2], u0, opt_st)
ds = 1000.0/fcond
opt_st.b = 0.5
X_s, lam_s, mE_s = ns.CONTINUESOLS(func_st, unf.x, lstart_st, lend_st, ds, opt_st,
                                   adapt=1.0, ALfn=ns.ARCORTHOGFN)
plt.ion()
plt.figure(1)
plt.clf()
plt.plot(np.array(lam_s)*fcond, (np.array(X_s).dot(btrans.T.todense()))[:, 5*3+1], 'k.-')
####################################################################################
# Bifurcation 1 - Singularity through first Eigenvalue
####################################################################################
opt_st.maxsteps = 100
du1, al1, du2, al2, ots = ns.SINGTANGENTS(func_st, X_s, lam_s, mE_s, opt_st, ei=0)

us1 = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((du1*al1, al1)), ds*5,
                  lam_s[ots.cpi-1], opt_st, ALfn=ns.ARCORTHOGFN)
us2a = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((du2*al2, al2)), ds*5,
                   lam_s[ots.cpi-1], opt_st, ALfn=ns.ARCORTHOGFN)
us2b = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((-du2*al2, al2)), ds*5,
                   lam_s[ots.cpi-1], opt_st, ALfn=ns.ARCORTHOGFN)
print((us1.status, us2a.status, us2b.status))
Xb11_s, lamb11_s, mEb11_s = ns.CONTINUESOLS(func_st, us1.x, us1.lam, lend_st,
                                            1e-6, opt_st, ALfn=ns.ARCORTHOGFN)
Xb21a_s, lamb21a_s, mEb21a_s = ns.CONTINUESOLS(func_st, us2a.x, us2a.lam, lend_st,
                                               1e-6, opt_st, ALfn=ns.ARCORTHOGFN)
Xb21b_s, lamb21b_s, mEb21b_s = ns.CONTINUESOLS(func_st, us2b.x, us2b.lam, lend_st,
                                               1e-6, opt_st, ALfn=ns.ARCORTHOGFN)
tm = (np.where(np.diff(lamb21a_s)[:-1]*np.diff(lamb21a_s)[1:] < 0)[0]).tolist()
if len(tm) > 1:
    st, en = tm[0:2]
else:
    st = 0
    en = tm[0]
Nl = len(lamb21a_s)
Xb21a_s = Xb21a_s[max(0, st-1):min(Nl, en+2)]
lamb21a_s = lamb21a_s[max(0, st-1):min(Nl, en+2)]
mEb21a_s = mEb21a_s[max(0, st-1):min(Nl, en+2)]
tm = (np.where(np.diff(lamb21b_s)[:-1]*np.diff(lamb21b_s)[1:] < 0)[0]).tolist()
if len(tm) > 1:
    st, en = tm[0:2]
else:
    st = 0
    en = tm[0]
Nl = len(lamb21b_s)
Xb21b_s = Xb21b_s[max(0, st-1):min(Nl, en+2)]
lamb21b_s = lamb21b_s[max(0, st-1):min(Nl, en+2)]
mEb21b_s = mEb21b_s[max(0, st-1):min(Nl, en+2)]

# ####################################################################################
# # Bifurcation 2 - Singularity through Second Eigenvalue: ASYMMETRIC BIFURCATION POINT!
# ####################################################################################
# du1, al1, du2, al2, ots = ns.SINGTANGENTS(func_st, X_s, lam_s, mE_s, opt_st, ei=1)
#
# us1 = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((du1*al1, al1)), ds, ots.biflam,
#                   opt_st, ALfn=ns.ARCORTHOGFN)
# us2a = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((du2*al2, al2)), ds, ots.biflam,
#                    opt_st, ALfn=ns.ARCORTHOGFN)
# us2b = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((-du2*al2, al2)), ds, ots.biflam,
#                    opt_st, ALfn=ns.ARCORTHOGFN)
# print((us1.status, us2a.status, us2b.status))
# opt_st.dsmax = 50000.0/fcond
# Xb12_s, lamb12_s, mEb12_s = ns.CONTINUESOLS(func_st, us1.x, us1.lam, lend_st,
#                                             opt.dsmax, opt_st, ALfn=ns.ARCORTHOGFN)
# Xb22a_s, lamb22a_s, mEb22a_s = ns.CONTINUESOLS(func_st, us2a.x, us2a.lam, lend_st,
#                                                opt.dsmax, opt_st, ALfn=ns.ARCORTHOGFN)
# Xb22b_s, lamb22b_s, mEb22b_s = ns.CONTINUESOLS(func_st, us2b.x, us2b.lam, lend_st,
#                                                opt.dsmax, opt_st, ALfn=ns.ARCORTHOGFN)
####################################################################################
# Bifurcation 3 - Singularity through Third Eigenvalue
####################################################################################
du1, al1, du2, al2, ots = ns.SINGTANGENTS(func_st, X_s, lam_s, mE_s, opt_st, ei=2)

us1 = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((du1*al1, al1)), ds*5,
                  lam_s[ots.cpi-1], opt_st, ALfn=ns.ARCORTHOGFN)
us2a = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((du2*al2, al2)), ds*5,
                   lam_s[ots.cpi-1], opt_st, ALfn=ns.ARCORTHOGFN)
us2b = ns.CONTSTEP(func_st, X_s[ots.cpi-1], np.hstack((-du2*al2, al2)), ds*5,
                   lam_s[ots.cpi-1], opt_st, ALfn=ns.ARCORTHOGFN)
print((us1.status, us2a.status, us2b.status))
Xb13_s, lamb13_s, mEb13_s = ns.CONTINUESOLS(func_st, us1.x, us1.lam, lend_st,
                                            1e-6, opt_st, ALfn=ns.ARCORTHOGFN)
Xb23a_s, lamb23a_s, mEb23a_s = ns.CONTINUESOLS(func_st, us2a.x, us2a.lam, lend_st,
                                               1e-6, opt_st, ALfn=ns.ARCORTHOGFN)
Xb23b_s, lamb23b_s, mEb23b_s = ns.CONTINUESOLS(func_st, us2b.x, us2b.lam, lend_st,
                                               1e-6, opt_st, ALfn=ns.ARCORTHOGFN)
tm = (np.where(np.diff(lamb23a_s)[:-1]*np.diff(lamb23a_s)[1:] < 0)[0]).tolist()
if len(tm) > 1:
    st, en = tm[0:2]
else:
    st = 0
    en = tm[0]
Nl = len(lamb23a_s)
Xb23a_s = Xb23a_s[max(0, st-1):min(Nl, en+2)]
lamb23a_s = lamb23a_s[max(0, st-1):min(Nl, en+2)]
mEb23a_s = mEb23a_s[max(0, st-1):min(Nl, en+2)]
tm = (np.where(np.diff(lamb23b_s)[:-1]*np.diff(lamb23b_s)[1:] < 0)[0]).tolist()
if len(tm) > 1:
    st, en = tm[0:2]
else:
    st = 0
    en = tm[0]
Nl = len(lamb23b_s)
Xb23b_s = Xb23b_s[max(0, st-1):min(Nl, en+2)]
lamb23b_s = lamb23b_s[max(0, st-1):min(Nl, en+2)]
mEb23b_s = mEb23b_s[max(0, st-1):min(Nl, en+2)]
####################################################################################

plt.ion()
plt.figure(1)
plt.clf()
plt.xlabel('Forcing (N)')
plt.ylabel('Peak Displacement (m)')
plt.plot(np.array(lam_s)*fcond, (np.array(X_s).dot(btrans.T.todense()))[:, 5*3+1], 'k.-')

plt.plot(np.array(lamb11_s)*fcond, (np.array(Xb11_s).dot(btrans.T.todense()))[:, 5*3+1], 'k.-')
plt.plot(np.array(lamb21a_s)*fcond, (np.array(Xb21a_s).dot(btrans.T.todense()))[:, 5*3+1], 'b.-')
plt.plot(np.array(lamb21b_s)*fcond, (np.array(Xb21b_s).dot(btrans.T.todense()))[:, 5*3+1], 'r.-')

plt.plot(np.array(lamb13_s)*fcond, (np.array(Xb13_s).dot(btrans.T.todense()))[:, 5*3+1], 'k.-')
plt.plot(np.array(lamb23a_s)*fcond, (np.array(Xb23a_s).dot(btrans.T.todense()))[:, 5*3+1], 'g.-')
plt.plot(np.array(lamb23b_s)*fcond, (np.array(Xb23b_s).dot(btrans.T.todense()))[:, 5*3+1], 'y.-')

plt.savefig('./FIGS/SNAPTHROUGH_DISP_%d.pdf' % (lamb1[kst]*fcond), dpi=100)

plt.ion()
plt.figure(10)
plt.clf()
plt.plot(np.array(lam_s)*fcond, (np.array(X_s).dot(btrans.T.todense()))[:, 5*3+2], 'k.-')
plt.plot(np.array(lamb11_s)*fcond, (np.array(Xb11_s).dot(btrans.T.todense()))[:, 5*3+2], 'm.-')
plt.plot(np.array(lamb21a_s)*fcond, (np.array(Xb21a_s).dot(btrans.T.todense()))[:, 5*3+2], 'b.-')
plt.plot(np.array(lamb21b_s)*fcond, (np.array(Xb21b_s).dot(btrans.T.todense()))[:, 5*3+2], 'r.-')

plt.plot(np.array(lamb13_s)*fcond, (np.array(Xb13_s).dot(btrans.T.todense()))[:, 5*3+2], 'k.-')
plt.plot(np.array(lamb23a_s)*fcond, (np.array(Xb23a_s).dot(btrans.T.todense()))[:, 5*3+2], 'g.-')
plt.plot(np.array(lamb23b_s)*fcond, (np.array(Xb23b_s).dot(btrans.T.todense()))[:, 5*3+2], 'y.-')

fig = plt.figure(100)
fig.clf()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=22, azim=-122)
ax.set(xlabel='Mid-Point Deflection (m)', ylabel='Mid-Point Orientation (rad)', zlabel='Forcing (N)')
ax.set(zlim=(lstart_st*fcond, lend_st*fcond))

ax.plot(np.squeeze(np.asarray(np.array(Xb13_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(Xb13_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lamb13_s)*fcond, linestyle='-.', color='k')
ax.plot(np.squeeze(np.asarray(np.array(Xb23a_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(Xb23a_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lamb23a_s)*fcond, linestyle='-.', color='g', linewidth=3)
ax.plot(np.squeeze(np.asarray(np.array(Xb23b_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(Xb23b_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lamb23b_s)*fcond, linestyle='-.', color='y', linewidth=3)

ax.plot(np.squeeze(np.asarray(np.array(Xb11_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(Xb11_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lamb11_s)*fcond, linestyle='--', color='k', linewidth=3)
ax.plot(np.squeeze(np.asarray(np.array(Xb21a_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(Xb21a_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lamb21a_s)*fcond, linestyle='--', color='b', linewidth=3)
ax.plot(np.squeeze(np.asarray(np.array(Xb21b_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(Xb21b_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lamb21b_s)*fcond, linestyle='--', color='r', linewidth=3)

ax.plot(np.squeeze(np.asarray(np.array(X_s).dot(btrans.T.todense()))[:, 5*3+1]),
        np.squeeze(np.asarray(np.array(X_s).dot(btrans.T.todense()))[:, 5*3+2]),
        np.array(lam_s)*fcond, color='k', linewidth=2)

fig.savefig('./FIGS/SNAPTHROUGH_3D_%d.pdf' % (lamb1[kst]*fcond), dpi=100)

plt.figure(2)
plt.clf()
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.plot(Xnds, Xnds*0, 'k--')
plt.plot(Xnds+btrans[0::3, :].dot(Xb1[-1]), btrans[1::3, :].dot(Xb1[-1]), 'k.-')
plt.plot(Xnds+btrans[0::3, :].dot(X_s[len(X_s)/3]),
         btrans[1::3, :].dot(X_s[len(X_s)/3]), 'k.--')
plt.plot(Xnds+btrans[0::3, :].dot(Xb21a_s[len(Xb21a_s)/3]),
         btrans[1::3, :].dot(Xb21a_s[len(Xb21a_s)/3]), 'b.-')
plt.plot(Xnds+btrans[0::3, :].dot(Xb21b_s[len(Xb21b_s)/3]),
         btrans[1::3, :].dot(Xb21b_s[len(Xb21b_s)/3]), 'r.-')

plt.plot(Xnds+btrans[0::3, :].dot(Xb23a_s[len(Xb23a_s)/3]),
         btrans[1::3, :].dot(Xb23a_s[len(Xb23a_s)/3]), 'g.-')
plt.plot(Xnds+btrans[0::3, :].dot(Xb23b_s[len(Xb23b_s)/3]),
         btrans[1::3, :].dot(Xb23b_s[len(Xb23b_s)/3]), 'y.-')

plt.savefig('./FIGS/SNAPTHROUGH_DEFN_%d.pdf' % (lamb1[kst]*fcond), dpi=100)
