#!/usr/bin/python
#############################################################################################
# FILE: DYNAMICS.py                                                                         #
# Written by Nidish Narayanaa Balaji on 17 Nov 2018                                         #
# Last modified by N. N. Balaji on 5 Dec 2019                                               #
#                                                                                           #
# This file contains definitions for routines necessary for dynamic analysis of the system. #
#############################################################################################
import numpy as np
from numpy import sqrt, kron, dot, diff, squeeze, sin, cos
import scipy.sparse as ss
import scipy.sparse.linalg as sl
import sparse as sp
import pdb
import STATICS as nr
from numpy import einsum #####
# from opt_einsum import contract
# einsum = contract
################################


def DMAT(xi, props, sla=0):
    """DMAT   : Returns the "D matrix" at given quadrature locations
    USAGE   :
        D = DMAT(xi, props, sla=0)
    INPUTS  :
    xi, props, sla=1
    OUTPUTS :
    D
    """
    Np = np.size(xi)
    Ns = nr.SF(xi)
    Nst = Ns
    ################################
    # CORRECTION FOR SHEAR-LOCKING #
    if sla == 1:
        Nst = np.ones_like(Ns)*0.5
    ################################
    D = np.zeros((6, Np*6))
    D[0, 0::6] = props.RA*Ns[:, 0]**2
    D[0, 3::6] = props.RA*Ns[:, 0]*Ns[:, 1]

    D[1, 1::6] = props.RA*Ns[:, 0]**2
    D[1, 4::6] = props.RA*Ns[:, 0]*Ns[:, 1]

    D[2, 2::6] = props.RI2*Nst[:, 0]**2
    D[2, 5::6] = props.RI2*Nst[:, 0]*Nst[:, 1]

    D[3, 0::6] = props.RA*Ns[:, 0]*Ns[:, 1]
    D[3, 3::6] = props.RA*Ns[:, 0]**2

    D[4, 1::6] = props.RA*Ns[:, 0]*Ns[:, 1]
    D[4, 4::6] = props.RA*Ns[:, 0]**2

    D[5, 2::6] = props.RI2*Nst[:, 0]*Nst[:, 1]
    D[5, 5::6] = props.RI2*Nst[:, 0]**2

    return D


def DYNMATS_E(X, d, ddot, No, props, sla=0, dd=0):
    """DYNMATS_E   : Returns the inertia and Christoffel Symbols matrices for the element
    USAGE   :
        De, Ce = DYNMATS_E(X, d, ddot, props, sla=0, dd=0)
    INPUTS  :
    X, d, ddot, props, sla=1
    OUTPUTS :
    De, Ce
    """
    xi, wi = np.polynomial.legendre.leggauss(No)  # Gauss-Legendre Quadrature Points & Weights

    Ds = DMAT(xi, props, sla)
    Dsi = Ds.dot(kron(wi, np.eye(6)).T)
    Le, dLe, d2Le = nr.LE(X, d)[0:3]
    Ledot = np.squeeze(dLe).dot(ddot)  # Time derivative of element length

    De = 0.5*Dsi*Le
    Ce = 0.25*Dsi*Ledot
    if dd == 1:
        dDedd = einsum('ij,k', Dsi, 0.5*dLe)
        dCedd = einsum('ij,lk,l', Dsi, d2Le, 0.25*ddot)
        dCedddot = dDedd/2.0
        return De, Ce, dDedd, dCedd, dCedddot
    return De, Ce


def DYNMATS(Xnds, u, udot, No, props, sla=0, spret=1, dd=0):
    """DYNMATS   : Returns the stitched Inertia & Christoffel symbols matrices for the FE model
    USAGE   :
        D, C = DYNMATS(Xnds, u, udot, No, props, sla=0, spret=1, dd=0)
    INPUTS  :
    Xnds, u, udot, No, props, sla=1, spret=1
    OUTPUTS :
    D, C
    """
    Nn = len(Xnds)
    Ne = Nn-1
    Nd = Nn*3
    D = ss.lil_matrix((Nd, Nd), dtype=float)
    C = ss.lil_matrix((Nd, Nd), dtype=float)
    for e in range(Ne):
        nstart = e
        nend = nstart+2
        istart = nstart*3
        iend = istart+6
        Dtmp, Ctmp = DYNMATS_E(Xnds[nstart:nend], u[istart:iend], udot[istart:iend], No, props, sla)
        D[istart:iend, istart:iend] += Dtmp
        C[istart:iend, istart:iend] += Ctmp
    if spret != 1:
        D = D.todense()
        C = C.todense()
    if dd == 1:
        dDdd = np.zeros((Nd, Nd, Nd))
        dCdd = np.zeros((Nd, Nd, Nd))
        dCdddot = np.zeros((Nd, Nd, Nd))
        for e in range(Ne):
            nstart = e
            nend = nstart+2
            istart = nstart*3
            iend = istart+6
            _, _, dDtmp, dCtmpd, dCtmpdot = DYNMATS_E(Xnds[nstart:nend], u[istart:iend], udot[istart:iend], No, props, sla, dd=dd)
            dDdd[istart:iend, istart:iend, istart:iend] += dDtmp
            dCdd[istart:iend, istart:iend, istart:iend] += dCtmpd
            dCdddot[istart:iend, istart:iend, istart:iend] += dCtmpdot
        if spret == 1:
            dDdd = sp.COO.from_numpy(dDdd)
            dCdd = sp.COO.from_numpy(dCdd)
            dCdddot = sp.COO.from_numpy(dCdddot)
        return D, C, dDdd, dCdd, dCdddot
    return D, C


def LINEARIZEDJAC(Xnds, u, udot, No, props, btrans, fres):
    """LINEARIZEDJAC   : Returns the linearized Jacobian
    USAGE   :
        Jlin = LINEARIZEDJAC(Xnds, u, udot, No, props, btrans, fres)
    INPUTS  :
    Xnds, u, udot, No, props, fres
    OUTPUTS :
    Jlin
    """
    R, dRdd = fres(u)[0:2]
    dRdd = dRdd.todense()
    uph = btrans.dot(u)
    udotph = btrans.dot(udot)

    D, C, dDdd, dCdd, dCdddot = DYNMATS(Xnds, uph, udotph, No, props, dd=1, spret=0)
    # Apply Homogeneous Boundary Conditions
    D = btrans.T.dot(D.dot(btrans.todense()))
    C = btrans.T.dot(C.dot(btrans.todense()))
    dDdd = einsum('lk,ijl->ijk', btrans.todense(), dDdd)
    dDdd = einsum('lj,ilk->ijk', btrans.todense(), dDdd)
    dDdd = einsum('li,ljk->ijk', btrans.todense(), dDdd)
    dCdd = einsum('lk,ijl->ijk', btrans.todense(), dCdd)
    dCdd = einsum('lj,ilk->ijk', btrans.todense(), dCdd)
    dCdd = einsum('li,ljk->ijk', btrans.todense(), dCdd)
    dCdddot = einsum('li,ljk->ijk', btrans.todense(), dCdddot)
    dCdddot = einsum('lj,ilk->ijk', btrans.todense(), dCdddot)
    dCdddot = einsum('lk,ijl->ijk', btrans.todense(), dCdddot)
    #
    Di = np.linalg.inv(D)
    dDidd = -einsum('im,mnk,nj', Di, dDdd, Di)

    dfdd = -(einsum('ijk,jl,l', dDidd, C, udot) +
             einsum('ij,jlk,l', Di, dCdd, udot) +
             einsum('ijk,j', dDidd, R) + einsum('ij,jk', Di, dRdd))
    dfdddot = -(einsum('ij,jlk,l', Di, dCdddot, udot) + einsum('ij,jk', Di, C))

    Nd = btrans.shape[1]
    Jlin = ss.lil_matrix((2*Nd, 2*Nd), dtype=float)
    Jlin[:Nd, Nd:] = ss.eye(Nd)
    Jlin[Nd:, :Nd] = dfdd
    Jlin[Nd:, Nd:] = dfdddot
    return Jlin
