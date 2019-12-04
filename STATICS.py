#!/usr/bin/python
###################################################################################################
# FILE: STATICS.py                                                                                #
# Written by Nidish Narayanaa Balaji on 16 Nov 2018                                               #
# Last modified by N. N. Balaji on 1 Dec 2018                                                     #
#                                                                                                 #
# The current file contains the definitions of routines required for solving static problems      #
# with the formulation. It also contains important routines for the finite element approximation. #
# There is also a sparse Newton solver implemented in the end since scipy's root method does not  #
# contain a full Jacobian solver implemented for systems with sparse matrices.                    #
###################################################################################################
import numpy as np
from numpy import sqrt, kron, dot, diff, squeeze, sin, cos, einsum, ones, ones_like, eye
import scipy.sparse as ss
import scipy.linalg as sn
import sparse as sp
import pdb

STRAINMEASURES = {
    -1: 'Integrated',
    0: 'Log',
    1: 'GreenLagrange'}


def SF(xi):
    """SF   : Returns the shape functions evaluated at set of points
    USAGE   :
        Ns = SF(xi)
    INPUTS  :
    xi	: Npx1 query points (iso CS)
    OUTPUTS :
    Ns	: Npx2 shape functions evaluations [N1, N2]
    """
    xi = np.atleast_1d(xi)
    xi = np.reshape(np.atleast_2d(xi), (np.size(xi), 1))
    Ns = np.hstack((1.0-xi, 1.0+xi))/2
    return Ns


def SD(xi):
    """SD   : Returns the shape function derivatives evaluated at set of points
    USAGE   :
        Nd = SD(xi)
    INPUTS  :
    xi	: Npx1 query points (iso CS)
    OUTPUTS :
    Ns	: Npx2 shape functions derivatives [[N1', N2']; ...]
    """
    xi = np.atleast_1d(xi)
    xi = np.reshape(np.atleast_2d(xi), (np.size(xi), 1))
    Nd = kron(ones((np.size(xi), 1)), [-1, 1])/2
    return Nd


def DOFEVAL(xi, X, d, sla=1):
    """DOFEVAL   : Evaluates the Degrees-of-freedom (and their first derivatives) at specific quadrature Locations
    USAGE   :
        u, du = DOFEVAL(xi, X, d)
    INPUTS  :
    xi	: Npx1 query points (iso CS)
    X	: 2x1 nodal X locations
    d	: 6x1 Nodal DOFs
    sla : int [1] flag for adjusting for shear locking
    OUTPUTS :
    u	: 3xNp DOFs
    du	: 3xNp DOF derivatives
    """
    d = squeeze(d)
    X = squeeze(X)

    Np = np.size(xi)
    u = np.zeros((3, Np))
    Ns = SF(xi)
    Nst = Ns
    ################################
    # CORRECTION FOR SHEAR-LOCKING #
    if sla == 1:
        Nst = ones_like(Ns)*0.5
    ################################
    u[0, :] = dot(Ns, d[[0, 3]])
    u[1, :] = dot(Ns, d[[1, 4]])
    u[2, :] = dot(Nst, d[[2, 5]])

    Nd = SD(xi)
    du = np.zeros((3, Np))
    du[0, :] = dot(Nd, d[[0, 3]])
    du[1, :] = dot(Nd, d[[1, 4]])
    du[2, :] = dot(Nd, d[[2, 5]])
    du = du*2.0/diff(X)
    return u, du


def LE(X, d):
    """LE   : Returns element length, its Jacobian vector, Hessian matrix, & cubic derivative tensor
    USAGE   :
        Le, Je, He, Te = LE(X, d)
    INPUTS  :
    X	: 2x1 nodal X locations
    d	: 6x1 nodal DOFs
    OUTPUTS :
    Le	: 1x1 Element Length
    Je	: 6x1 Length Jacobian
    He	: 6x6 Hessian
    Te  : 6x6x6 cubic derivatives
    """
    delx = (X[1]+d[3]) - (X[0]+d[0])
    dely = (d[4]-d[1])
    Le = sqrt(delx**2 + dely**2)

    Je = np.squeeze(np.vstack((-delx, -dely, 0, delx, dely, 0))/Le)

    He = kron([[1, -1], [-1, 1]], np.diag([1, 1, 0]))
    He = (He - einsum('i,j', Je, Je))/Le

    Te = -(einsum('ij,k', He, Je) + einsum('i,jk', Je, He) + einsum('ik,j', He, Je))/Le
    return Le, Je, He, Te  # CHECKED: OK


def EP0(xi, X, d, sla=1):
    """EP0   : Returns the epsilon_0 term with its derivatives
    USAGE   :
        ep0, dep0, d2ep0, d3ep0 = EP0(xi, X, d)
    INPUTS  :
    xi, X, d, sla
    OUTPUTS :
    ep0, dep0, d2ep0, d3ep0
    """
    X = squeeze(X)
    d = squeeze(d)
    xi = np.atleast_1d(xi)
    xi = np.reshape(np.atleast_2d(xi), (np.size(xi), 1))
    Np = np.size(xi)

    u, du = DOFEVAL(xi, X, d, sla)

    ep0 = (du[0, :]**2 + du[1, :]**2 + 2*du[0, :])/2.0

    dep0 = np.zeros((6, Np))
    dep0[0, :] = -(1.0+du[0, :])
    dep0[1, :] = -du[1, :]
    dep0[3, :] = -dep0[0, :]
    dep0[4, :] = -dep0[1, :]
    dep0 = dep0/diff(X)

    d2ep0 = kron(ones(Np), kron([[1.0, -1.0], [-1.0, 1.0]], np.diag([1.0, 1.0, 0.0])))/diff(X)**2

    d3ep0 = np.zeros((6, Np*6, 6))
    return ep0, dep0, d2ep0, d3ep0  # CHECKED: OK


def GM0(xi, X, d, sla=1):
    """GM0   : Returns the gamma_0 term with its derivatives
    USAGE   :
        gm0, dgm0, d2gm0, d3gm0 = GM0(xi, X, d, sla=1)
    INPUTS  :
    xi, X, d, sla
    OUTPUTS :
    gm0, dgm0, d2gm0, d3gm0
    """
    X = squeeze(X)
    d = squeeze(d)
    xi = np.atleast_1d(xi)
    xi = np.reshape(np.atleast_2d(xi), (np.size(xi), 1))
    Np = np.size(xi)

    u, du = DOFEVAL(xi, X, d, sla)
    Ns = SF(xi)
    ################################
    # CORRECTION FOR SHEAR-LOCKING #
    if sla == 1:
        Ns = ones_like(Ns)*0.5    #
    ################################
    gm0 = -(1.0+du[0, :])*sin(u[2, :]) + du[1, :]*cos(u[2, :])

    # First Derivatives
    dgm0 = np.zeros((6, Np))
    dX = diff(X)
    dgm0[0, :] = sin(u[2, :])/dX
    dgm0[1, :] = -cos(u[2, :])/dX
    dgm0[2, :] = -((1.0+du[0, :])*cos(u[2, :]) + du[1, :]*sin(u[2, :]))*Ns[:, 0]
    dgm0[3, :] = -sin(u[2, :])/dX
    dgm0[4, :] = cos(u[2, :])/dX
    dgm0[5, :] = -((1.0+du[0, :])*cos(u[2, :]) + du[1, :]*sin(u[2, :]))*Ns[:, 1]

    # Second Derivatives
    d2gm0 = np.zeros((6, Np*6))
    d2gm0[0, 2::6] = cos(u[2, :])*Ns[:, 0]/dX
    d2gm0[0, 5::6] = cos(u[2, :])*Ns[:, 1]/dX

    d2gm0[1, 2::6] = sin(u[2, :])*Ns[:, 0]/dX
    d2gm0[1, 5::6] = sin(u[2, :])*Ns[:, 1]/dX

    d2gm0[2, 0::6] = cos(u[2, :])*Ns[:, 0]/dX
    d2gm0[2, 1::6] = sin(u[2, :])*Ns[:, 0]/dX
    d2gm0[2, 2::6] = ((1.0+du[0, :])*sin(u[2, :]) - du[1, :]*cos(u[2, :]))*Ns[:, 0]**2
    d2gm0[2, 3::6] = -d2gm0[2, 0:Np*6:6]
    d2gm0[2, 4::6] = -d2gm0[2, 1:Np*6:6]
    d2gm0[2, 5::6] = ((1.0+du[0, :])*sin(u[2, :]) - du[1, :]*cos(u[2, :]))*Ns[:, 0]*Ns[:, 1]

    d2gm0[3, 2::6] = -d2gm0[0, 2:Np*6:6]
    d2gm0[3, 5::6] = -d2gm0[0, 5:Np*6:6]

    d2gm0[4, 2::6] = -d2gm0[1, 2:Np*6:6]
    d2gm0[4, 5::6] = -d2gm0[1, 5:Np*6:6]

    d2gm0[5, 0::6] = cos(u[2, :])*Ns[:, 1]/dX
    d2gm0[5, 1::6] = sin(u[2, :])*Ns[:, 1]/dX
    d2gm0[5, 2::6] = ((1.0+du[0, :])*sin(u[2, :]) - du[1, :]*cos(u[2, :]))*Ns[:, 0]*Ns[:, 1]
    d2gm0[5, 3::6] = -d2gm0[5, 0:Np*6:6]
    d2gm0[5, 4::6] = -d2gm0[5, 1:Np*6:6]
    d2gm0[5, 5::6] = ((1.0+du[0, :])*sin(u[2, :]) - du[1, :]*cos(u[2, :]))*Ns[:, 1]**2

    # Third Derivatives
    d3gm0 = np.zeros((6, Np*6, 6))
    d3gm0[2, 2::6, 0] = -Ns[:, 0]**2*sin(u[2, :])/dX
    d3gm0[2, 5::6, 0] = -Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[5, 2::6, 0] = -Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[5, 5::6, 0] = -Ns[:, 1]**2*sin(u[2, :])/dX

    d3gm0[2, 2::6, 1] = Ns[:, 0]**2*cos(u[2, :])/dX
    d3gm0[2, 5::6, 1] = Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 2::6, 1] = Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 5::6, 1] = Ns[:, 1]**2*cos(u[2, :])/dX

    d3gm0[0, 2::6, 2] = -Ns[:, 0]**2*sin(u[2, :])/dX
    d3gm0[0, 5::6, 2] = -Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[1, 2::6, 2] = Ns[:, 0]**2*cos(u[2, :])/dX
    d3gm0[1, 5::6, 2] = Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[2, 0::6, 2] = -Ns[:, 0]**2*sin(u[2, :])/dX
    d3gm0[2, 1::6, 2] = Ns[:, 0]**2*cos(u[2, :])/dX
    d3gm0[2, 2::6, 2] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**3
    d3gm0[2, 3::6, 2] = Ns[:, 0]**2*sin(u[2, :])/dX
    d3gm0[2, 4::6, 2] = -Ns[:, 0]**2*cos(u[2, :])/dX
    d3gm0[2, 5::6, 2] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**2*Ns[:, 1]
    d3gm0[3, 2::6, 2] = Ns[:, 0]**2*sin(u[2, :])/dX
    d3gm0[3, 5::6, 2] = Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[4, 2::6, 2] = -Ns[:, 0]**2*cos(u[2, :])/dX
    d3gm0[4, 5::6, 2] = -Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 0::6, 2] = -Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[5, 1::6, 2] = Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 2::6, 2] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**2*Ns[:, 1]
    d3gm0[5, 3::6, 2] = Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[5, 4::6, 2] = -Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 5::6, 2] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]*Ns[:, 1]**2

    d3gm0[2, 2::6, 3] = Ns[:, 0]**2*sin(u[2, :])/dX
    d3gm0[2, 5::6, 3] = Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[5, 2::6, 3] = Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[5, 5::6, 3] = Ns[:, 1]**2*sin(u[2, :])/dX

    d3gm0[2, 2::6, 4] = -Ns[:, 0]**2*cos(u[2, :])/dX
    d3gm0[2, 5::6, 4] = -Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 2::6, 4] = -Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[5, 5::6, 4] = -Ns[:, 1]**2*cos(u[2, :])/dX

    d3gm0[0, 2::6, 5] = -Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[0, 5::6, 5] = -Ns[:, 1]**2*sin(u[2, :])/dX
    d3gm0[1, 2::6, 5] = Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[1, 5::6, 5] = Ns[:, 1]**2*cos(u[2, :])/dX
    d3gm0[2, 0::6, 5] = -Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[2, 1::6, 5] = Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[2, 2::6, 5] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**2*Ns[:, 1]
    d3gm0[2, 3::6, 5] = Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[2, 4::6, 5] = -Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[2, 5::6, 5] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]*Ns[:, 1]**2
    d3gm0[3, 2::6, 5] = Ns[:, 0]*Ns[:, 1]*sin(u[2, :])/dX
    d3gm0[3, 5::6, 5] = Ns[:, 1]**2*sin(u[2, :])/dX
    d3gm0[4, 2::6, 5] = -Ns[:, 0]*Ns[:, 1]*cos(u[2, :])/dX
    d3gm0[4, 5::6, 5] = -Ns[:, 1]**2*cos(u[2, :])/dX
    d3gm0[5, 0::6, 5] = -Ns[:, 1]**2*sin(u[2, :])/dX
    d3gm0[5, 1::6, 5] = Ns[:, 1]**2*cos(u[2, :])/dX
    d3gm0[5, 2::6, 5] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]*Ns[:, 1]**2
    d3gm0[5, 3::6, 5] = Ns[:, 1]**2*sin(u[2, :])/dX
    d3gm0[5, 4::6, 5] = -Ns[:, 1]**2*cos(u[2, :])/dX
    d3gm0[5, 5::6, 5] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 1]**3
    return gm0, dgm0, d2gm0, d3gm0  # CHECKED: OK


def EP1(xi, X, d, sla=1):
    """EP1   : Returns the epsilon_1 term with its Jacobian and Hessian
    USAGE   :
        ep1, dep1, d2ep1, d3ep1 = EP1(xi, X, d)
    INPUTS  :
    xi, X, d, sla
    OUTPUTS :
    ep1, dep1, d2ep1, d3ep1
    """
    X = squeeze(X)
    d = squeeze(d)
    xi = np.atleast_1d(xi)
    xi = np.reshape(np.atleast_2d(xi), (np.size(xi), 1))
    Np = np.size(xi)

    u, du = DOFEVAL(xi, X, d, sla)
    Ns = SF(xi)
    ################################
    # CORRECTION FOR SHEAR-LOCKING #
    if sla == 1:
        Ns = ones_like(Ns)*0.5    #
    ################################
    ep1 = -du[2, :]*((1.0+du[0, :])*cos(u[2, :]) + du[1, :]*sin(u[2, :]))

    # Relevant Functions
    dX = diff(X)
    g1 = -cos(u[2, :])/dX**2 - du[2, :]*sin(u[2, :])*Ns[:, 0]/dX
    dg1 = np.zeros((6, Np))
    dg1[2, :] = 2.0*sin(u[2, :])*Ns[:, 0]/dX**2-du[2, :]*cos(u[2, :])*Ns[:, 0]**2/dX
    dg1[5, :] = sin(u[2, :])*(Ns[:, 1]-Ns[:, 0])/dX**2 - du[2, :]*cos(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX

    g2 = cos(u[2, :])/dX**2-du[2, :]*sin(u[2, :])*Ns[:, 1]/dX
    dg2 = np.zeros((6, Np))
    dg2[2, :] = -sin(u[2, :])*Ns[:, 0]/dX**2 + sin(u[2, :])*Ns[:, 1]/dX**2 - du[2, :]*cos(u[2, :])*Ns[:, 1]**2/dX
    dg2[5, :] = -sin(u[2, :])*Ns[:, 1]/dX**2 - sin(u[2, :])*Ns[:, 1]/dX**2 - du[2, :]*cos(u[2, :])*Ns[:, 1]**2/dX

    g3 = -sin(u[2, :])/dX**2 + du[2, :]*cos(u[2, :])*Ns[:, 0]/dX
    dg3 = np.zeros((6, Np))
    dg3[2, :] = -2.0*cos(u[2, :])*Ns[:, 0]/dX**2 - du[2, :]*sin(u[2, :])*Ns[:, 0]**2/dX
    dg3[5, :] = cos(u[2, :])*(Ns[:, 0] - Ns[:, 1])/dX**2 - du[2, :]*sin(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX

    g4 = sin(u[2, :])/dX**2 + du[2, :]*cos(u[2, :])*Ns[:, 1]/dX
    dg4 = np.zeros((6, Np))
    dg4[2, :] = cos(u[2, :])*Ns[:, 0]/dX**2 - cos(u[2, :])*Ns[:, 1]/dX**2 - du[2, :]*sin(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX
    dg4[5, :] = cos(u[2, :])*Ns[:, 1]/dX**2 + cos(u[2, :])*Ns[:, 1]/dX**2 - du[2, :]*sin(u[2, :])*Ns[:, 1]**2/dX

    g5 = du[2, :]*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**2 - 2.0*Ns[:, 0]*((1.0+du[0, :])*sin(u[2, :])-du[1, :]*cos(u[2, :]))/dX
    dg5 = np.zeros((6, Np))
    dg5[0, :] = -du[2, :]*cos(u[2, :])*Ns[:, 0]**2/dX + 2.0*Ns[:, 0]*sin(u[2, :])/dX**2
    dg5[1, :] = -du[2, :]*sin(u[2, :])*Ns[:, 0]**2/dX - 2.0*Ns[:, 0]*cos(u[2, :])/dX**2
    dg5[2, :] = -((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**2/dX + \
        du[2, :]*(-(1.0+du[0, :])*sin(u[2, :])+du[1, :]*cos(u[2, :]))*Ns[:, 0]**3 - \
        2.0*Ns[:, 0]**2*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX
    dg5[3, :] = du[2, :]*cos(u[2, :])*Ns[:, 0]**2/dX - 2.0*Ns[:, 0]*sin(u[2, :])/dX**2
    dg5[4, :] = du[2, :]*sin(u[2, :])*Ns[:, 0]**2/dX + 2.0*Ns[:, 0]*cos(u[2, :])/dX**2
    dg5[5, :] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]**2/dX + \
        du[2, :]*(-(1.0+du[0, :])*sin(u[2, :])+du[1, :]*cos(u[2, :]))*Ns[:, 0]**2*Ns[:, 1] - \
        2.0*Ns[:, 0]*Ns[:, 1]*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX

    g6 = du[2, :]*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]*Ns[:, 1] - \
        (Ns[:, 0]-Ns[:, 1])*((1.0+du[0, :])*sin(u[2, :])-du[1, :]*cos(u[2, :]))/dX
    dg6 = np.zeros((6, Np))
    dg6[0, :] = -du[2, :]*cos(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX + (Ns[:, 0]-Ns[:, 1])*sin(u[2, :])/dX**2
    dg6[1, :] = -du[2, :]*sin(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX - (Ns[:, 0]-Ns[:, 1])*cos(u[2, :])/dX**2
    dg6[2, :] = -((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]*Ns[:, 1]/dX + \
        du[2, :]*(-(1.0+du[0, :])*sin(u[2, :])+du[1, :]*cos(u[2, :]))*Ns[:, 0]**2*Ns[:, 1] - \
        Ns[:, 0]*(Ns[:, 0]-Ns[:, 1])*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX
    dg6[3, :] = du[2, :]*cos(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX - (Ns[:, 0]-Ns[:, 1])*sin(u[2, :])/dX**2
    dg6[4, :] = du[2, :]*sin(u[2, :])*Ns[:, 0]*Ns[:, 1]/dX + (Ns[:, 0]-Ns[:, 1])*cos(u[2, :])/dX**2
    dg6[5, :] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 0]*Ns[:, 1]/dX + \
        du[2, :]*(-(1.0+du[0, :])*sin(u[2, :])+du[1, :]*cos(u[2, :]))*Ns[:, 0]*Ns[:, 1]**2 - \
        Ns[:, 1]*(Ns[:, 0]-Ns[:, 1])*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX

    g7 = du[2, :]*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 1]**2 + 2.0*Ns[:, 1]*((1.0+du[0, :])*sin(u[2, :])-du[1, :]*cos(u[2, :]))/dX
    dg7 = np.zeros((6, Np))
    dg7[0, :] = -du[2, :]*cos(u[2, :])*Ns[:, 1]**2/dX - 2.0*Ns[:, 1]*sin(u[2, :])/dX**2
    dg7[1, :] = -du[2, :]*sin(u[2, :])*Ns[:, 1]**2/dX + 2.0*Ns[:, 1]*cos(u[2, :])/dX**2
    dg7[2, :] = -((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 1]**2/dX + \
        du[2, :]*(-(1.0+du[0, :])*sin(u[2, :])+du[1, :]*cos(u[2, :]))*Ns[:, 0]*Ns[:, 1]**2 + \
        2.0*Ns[:, 0]*Ns[:, 1]*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX
    dg7[3, :] = du[2, :]*cos(u[2, :])*Ns[:, 1]**2/dX + 2.0*Ns[:, 1]*sin(u[2, :])/dX**2
    dg7[4, :] = du[2, :]*sin(u[2, :])*Ns[:, 1]**2/dX - 2.0*Ns[:, 1]*cos(u[2, :])/dX**2
    dg7[5, :] = ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))*Ns[:, 1]**2/dX + \
        du[2, :]*(-(1.0+du[0, :])*sin(u[2, :])+du[1, :]*cos(u[2, :]))*Ns[:, 0]*Ns[:, 1]**2 + \
        2.0*Ns[:, 1]**2*((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX

    # First Derivatives
    dep1 = np.zeros((6, Np))
    dep1[0, :] = du[2, :]*cos(u[2, :])/dX
    dep1[1, :] = du[2, :]*sin(u[2, :])/dX
    dep1[3, :] = -dep1[0, :]
    dep1[4, :] = -dep1[1, :]
    dep1[2, :] = du[2, :]*((1.0+du[0, :])*sin(u[2, :]) - du[1, :]*cos(u[2, :]))*Ns[:, 0] + \
        ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX
    dep1[5, :] = du[2, :]*((1.0+du[0, :])*sin(u[2, :]) - du[1, :]*cos(u[2, :]))*Ns[:, 1] - \
        ((1.0+du[0, :])*cos(u[2, :])+du[1, :]*sin(u[2, :]))/dX

    # Second Derivatives
    d2ep1 = np.zeros((6, Np*6))
    d2ep1[0, 2::6] = g1
    d2ep1[0, 5::6] = g2
    d2ep1[1, 2::6] = g3
    d2ep1[1, 5::6] = g4
    d2ep1[2, 0::6] = g1
    d2ep1[2, 1::6] = g3
    d2ep1[2, 2::6] = g5
    d2ep1[2, 3::6] = -g1
    d2ep1[2, 4::6] = -g3
    d2ep1[2, 5::6] = g6
    d2ep1[3, 2::6] = -g1
    d2ep1[3, 5::6] = -g2
    d2ep1[4, 2::6] = -g3
    d2ep1[4, 5::6] = -g4
    d2ep1[5, 0::6] = g2
    d2ep1[5, 1::6] = g4
    d2ep1[5, 2::6] = g6
    d2ep1[5, 3::6] = -g2
    d2ep1[5, 4::6] = -g4
    d2ep1[5, 5::6] = g7

    # Third derivatives
    d3ep1 = np.zeros((6, Np*6, 6))
    d3ep1[0, 2::6, :] = dg1.T
    d3ep1[0, 5::6, :] = dg2.T
    d3ep1[1, 2::6, :] = dg3.T
    d3ep1[1, 5::6, :] = dg4.T
    d3ep1[2, 0::6, :] = dg1.T
    d3ep1[2, 1::6, :] = dg3.T
    d3ep1[2, 2::6, :] = dg5.T
    d3ep1[2, 3::6, :] = -dg1.T
    d3ep1[2, 4::6, :] = -dg3.T
    d3ep1[2, 5::6, :] = dg6.T
    d3ep1[3, 2::6, :] = -dg1.T
    d3ep1[3, 5::6, :] = -dg2.T
    d3ep1[4, 2::6, :] = -dg3.T
    d3ep1[4, 5::6, :] = -dg4.T
    d3ep1[5, 0::6, :] = dg2.T
    d3ep1[5, 1::6, :] = dg4.T
    d3ep1[5, 2::6, :] = dg6.T
    d3ep1[5, 3::6, :] = -dg2.T
    d3ep1[5, 4::6, :] = -dg4.T
    d3ep1[5, 5::6, :] = dg7.T

    return ep1, dep1, d2ep1, d3ep1  # CHECKED : OK


def EP2(xi, X, d, sla=1):
    """EP2   : Returns the epsilon_2 term with its Jacobian and Hessian
    USAGE   :
        ep2, dep2, d2ep2, d3ep2 = EP2(xi, X, d)
    INPUTS  :
    xi, X, d, sla
    OUTPUTS :
    ep2, dep2, d2ep2, d3ep2
    """
    X = squeeze(X)
    d = squeeze(d)
    xi = np.atleast_1d(xi)
    xi = np.reshape(np.atleast_2d(xi), (np.size(xi), 1))
    Np = np.size(xi)

    u, du = DOFEVAL(xi, X, d, sla)
    ep2 = du[2, :]**2/2

    dep2 = np.zeros((6, Np))
    dX = diff(X)
    dep2[2, :] = -du[2, :]/dX
    dep2[5, :] = du[2, :]/dX

    d2ep2 = np.zeros((6, 6*Np))
    d2ep2[2, 2:Np*6:6] = 1.0/dX**2
    d2ep2[2, 5:Np*6:6] = -1.0/dX**2

    d2ep2[5, 2:Np*6:6] = -1.0/dX**2
    d2ep2[5, 5:Np*6:6] = 1.0/dX**2

    d3ep2 = np.zeros((6, Np*6, 6))
    return ep2, dep2, d2ep2, d3ep2  # CHECKED : OK


def STRAIN(xi, y, X, d, props, sla=1, d3=0, smeasure=0):
    """STRAIN   : Returns the Strain and its derivatives
    USAGE   :
        strain, dstrain, d2strain = STRAIN(xi, y, X, d, props, sla, d3, smeasure)
    INPUTS  :
        xi	 : Points on the element
        y	 : Points across section
        X	 : 2x1 Nodal locations in initial configuration
        d	 : 6x1 Nodal DoF vector
        props    : Properties class with members,
        		EA, GA, EI2, EI4
        sla	 : (=1) Shear locking adjustment flag
        d3	 : (=0) 3rd derivatives required?
        smeasure : (=0) Choice of strain measure (See STRAINMEASURES dictionary)
    OUTPUTS :
        strain   : strain_(ijkl) with e_(ij) evaluated at point xi_k, section point y_l
        dstrain  : dstrain_(ijklm) with de_(ij)/dd_(m) evaluated at point xi_k, section point y_l
        d2strain  : dstrain_(ijklmn) with d2e_(ij)/dd2_(mn) evaluated at point xi_k, section point y_l
        d2strain  : d2strain_(ijklmn) with d2e_(ij)/dd2_(mn) evaluated at point xi_k, section point y_l
        d3strain  : (if d3!=0) d3strain_(ijklmno) with d3e_(ij)/dd3_(mno) evaluated at point xi_k, section point y_l
    """
    Np_xi = len(xi)
    Np_y = len(y)
    Nd = len(d)

    e0, de0, d2e0, d3e0 = EP0(xi, X, d, sla)
    g0, dg0, d2g0, d3g0 = GM0(xi, X, d, sla)
    e1, de1, d2e1, d3e1 = EP1(xi, X, d, sla)
    e2, de2, d2e2, d3e2 = EP2(xi, X, d, sla)

    d2e0 = d2e0.reshape((Nd, Np_xi, Nd))
    d2g0 = d2g0.reshape((Nd, Np_xi, Nd))
    d2e1 = d2e1.reshape((Nd, Np_xi, Nd))
    d2e2 = d2e2.reshape((Nd, Np_xi, Nd))

    # Right Cauchy-Green Deformation Tensor F^T.F ################################################
    FTF = np.zeros((2, 2, Np_xi, Np_y))
    FTF[0, 0, :, :] = 1.0+2*(einsum("i,j->ij", e0, ones_like(y)) +
                             einsum("i,j->ij", e1, y) +
                             einsum("i,j->ij", e2, y**2))
    FTF[0, 1, :, :] = 2*einsum("i,j->ij", g0, ones_like(y))
    FTF[1, 0, :, :] = FTF[0, 1, :, :]
    FTF[1, 1, :, :] = 1.0

    dFTF = np.zeros((2, 2, Np_xi, Np_y, Nd))
    dFTF[0, 0, :, :, :] = 2*(einsum("ki,j->ijk", de0, ones_like(y)) +
                             einsum("ki,j->ijk", de1, y) +
                             einsum("ki,j->ijk", de2, y**2))
    dFTF[0, 1, :, :, :] = 2*einsum("ki,j->ijk", dg0, ones_like(y))
    dFTF[1, 0, :, :, :] = dFTF[0, 1, :, :, :]

    d2FTF = np.zeros((2, 2, Np_xi, Np_y, Nd, Nd))
    d2FTF[0, 0, :, :, :, :] = 2*(einsum("kil,j->ijkl", d2e0, ones_like(y)) +
                                 einsum("kil,j->ijkl", d2e1, y) +
                                 einsum("kil,j->ijkl", d2e2, y**2))
    d2FTF[0, 1, :, :, :, :] = 2*einsum("kil,j->ijkl", d2g0, ones_like(y))
    d2FTF[1, 0, :, :, :, :] = d2FTF[0, 1, :, :, :, :]

    if d3 == 1:
        d3FTF = np.zeros((2, 2, Np_xi, Np_y, Nd, Nd, Nd))
        d3FTF[0, 0, :, :, :, :, :] = 2*(einsum("kilm,j->ijklm", d3e0, ones_like(y)) +
                                        einsum("kilm,j->ijklm", d3e1, y) +
                                        einsum("kilm,j->ijklm", d3e2, y**2))
        d3FTF[0, 1, :, :, :, :, :] = 2*einsum("kilm,j->ijklm", d3g0, ones_like(y))
        d3FTF[1, 0, :, :, :, :, :] = d3FTF[0, 1, :, :, :, :, :]

    # Strain Measure Estimation ##################################################################
    if smeasure == 1:   # Green-Lagrange Strain Measure - Explicit Formulae Employed
        EYE = einsum("ij,kl->ijkl", eye(2), ones((Np_xi, Np_y)))
        strain = (FTF-EYE)/2
        dstrain = dFTF/2
        d2strain = d2FTF/2
        if d3 == 1:
            d3strain = d3FTF/2
    else:  # Seth-Hill Generalized Strains (case 0 (log-strain) will be dealt with specially below)
        strain = np.zeros_like(FTF)
        dstrain = np.zeros_like(dFTF)
        d2strain = np.zeros_like(d2FTF)
        d3strain = np.zeros((2, 2, Np_xi, Np_y, Nd, Nd, Nd*d3))

        # Variables for Eigendecomposition and derivatives
        lam = np.zeros(2)  # Eigenvalues
        dlam = np.zeros((2, Nd))
        d2lam = np.zeros((2, Nd, Nd))
        d3lam = np.zeros((2, Nd, Nd, Nd*d3))

        phi = np.zeros((2, 2))  # Eigenvectors
        dphi = np.zeros((2, 2, Nd))
        d2phi = np.zeros((2, 2, Nd, Nd))
        d3phi = np.zeros((2, 2, Nd, Nd, Nd*d3))
        for i in range(Np_xi):
            for j in range(Np_y):
                # Eigendecomposition of FTF
                try:
                    lam, phi = sn.eigh(FTF[:, :, i, j])
                except Exception as inst:
                    pdb.set_trace()
                    raise inst
                    raise Exception('dnan')

                if lam[0] != lam[1]:  # Nonrepeated eigenvalues
                    # Eigenvalue 1st Derivatives
                    dlam[0, :] = einsum("i,ijk,j->k", phi[:, 0], dFTF[:, :, i, j, :], phi[:, 0])
                    dlam[1, :] = einsum("i,ijk,j->k", phi[:, 1], dFTF[:, :, i, j, :], phi[:, 1])
                    # Eigenvector 1st Derivatives
                    dphi[:, 0, :] = einsum(",p,pjk,j,i->ik", (lam[0]-lam[1])**(-1), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1])
                    dphi[:, 1, :] = einsum(",p,pjk,j,i->ik", (lam[1]-lam[0])**(-1), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0])
                    # Eigenvalue 2nd Derivatives
                    d2lam[0, :, :] = einsum("il,ijk,j->kl", dphi[:, 0, :], dFTF[:, :, i, j, :], phi[:, 0]) + \
                        einsum("i,ijkl,j->kl", phi[:, 0], d2FTF[:, :, i, j, :, :], phi[:, 0]) + \
                        einsum("i,ijk,jl->kl", phi[:, 0], dFTF[:, :, i, j, :], dphi[:, 0, :])

                    d2lam[1, :, :] = einsum("il,ijk,j->kl", dphi[:, 1, :], dFTF[:, :, i, j, :], phi[:, 1]) + \
                        einsum("i,ijkl,j->kl", phi[:, 1], d2FTF[:, :, i, j, :, :], phi[:, 1]) + \
                        einsum("i,ijk,jl->kl", phi[:, 1], dFTF[:, :, i, j, :], dphi[:, 1, :])
                    # Eigenvectors 2nd Derivatives
                    d2phi[:, 0, :, :] = einsum(",l,p,pjk,j,i->ikl", -(lam[0]-lam[1])**(-2), (dlam[0, :]-dlam[1, :]), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                        einsum(",pl,pjk,j,i->ikl", 1.0/(lam[0]-lam[1]), dphi[:, 1, :], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                        einsum(",p,pjkl,j,i->ikl", 1.0/(lam[0]-lam[1]), phi[:, 1], d2FTF[:, :, i, j, :, :], phi[:, 0], phi[:, 1]) + \
                        einsum(",p,pjk,jl,i->ikl", 1.0/(lam[0]-lam[1]), phi[:, 1], dFTF[:, :, i, j, :], dphi[:, 0, :], phi[:, 1]) + \
                        einsum(",p,pjk,j,il->ikl", 1.0/(lam[0]-lam[1]), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], dphi[:, 1, :])
                    
                    d2phi[:, 1, :, :] = einsum(",l,p,pjk,j,i->ikl", -(lam[1]-lam[0])**(-2), (dlam[1, :] - dlam[0, :]), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                        einsum(",pl,pjk,j,i->ikl", 1.0/(lam[1]-lam[0]), dphi[:, 0, :], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                        einsum(",p,pjkl,j,i->ikl", 1.0/(lam[1]-lam[0]), phi[:, 0], d2FTF[:, :, i, j, :, :], phi[:, 1], phi[:, 0]) + \
                        einsum(",p,pjk,jl,i->ikl", 1.0/(lam[1]-lam[0]), phi[:, 0], dFTF[:, :, i, j, :], dphi[:, 1, :], phi[:, 0]) + \
                        einsum(",p,pjk,j,il->ikl", 1.0/(lam[1]-lam[0]), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], dphi[:, 0, :])

                    if d3 == 1:  # 3rd Derivatives
                        # Eigenvalues 3rd Derivatives
                        d3lam[0, :, :, :] = einsum("ilm,ijk,j->klm", d2phi[:, 0, :, :], dFTF[:, :, i, j, :], phi[:, 0]) + \
                            einsum("il,ijkm,j->klm", dphi[:, 0, :], d2FTF[:, :, i, j, :, :], phi[:, 0]) + \
                            einsum("il,ijk,jm->klm", dphi[:, 0, :], dFTF[:, :, i, j, :], dphi[:, 0, :]) + \
                            einsum("im,ijkl,j->klm", dphi[:, 0, :], d2FTF[:, :, i, j, :, :], phi[:, 0]) + \
                            einsum("i,ijklm,j->klm", phi[:, 0], d3FTF[:, :, i, j, :, :, :], phi[:, 0]) + \
                            einsum("i,ijkl,jm->klm", phi[:, 0], d2FTF[:, :, i, j, :, :], dphi[:, 0, :]) + \
                            einsum("im,ijk,jl->klm", dphi[:, 0, :], dFTF[:, :, i, j, :], dphi[:, 0, :]) + \
                            einsum("i,ijkm,jl->klm", phi[:, 0], d2FTF[:, :, i, j, :, :], dphi[:, 0, :]) + \
                            einsum("i,ijk,jlm->klm", phi[:, 0], dFTF[:, :, i, j, :], d2phi[:, 0, :, :])
                        d3lam[1, :, :, :] = einsum("ilm,ijk,j->klm", d2phi[:, 1, :, :], dFTF[:, :, i, j, :], phi[:, 1]) + \
                            einsum("il,ijkm,j->klm", dphi[:, 1, :], d2FTF[:, :, i, j, :, :], phi[:, 1]) + \
                            einsum("il,ijk,jm->klm", dphi[:, 1, :], dFTF[:, :, i, j, :], dphi[:, 1, :]) + \
                            einsum("im,ijkl,j->klm", dphi[:, 1, :], d2FTF[:, :, i, j, :, :], phi[:, 1]) + \
                            einsum("i,ijklm,j->klm", phi[:, 1], d3FTF[:, :, i, j, :, :, :], phi[:, 1]) + \
                            einsum("i,ijkl,jm->klm", phi[:, 1], d2FTF[:, :, i, j, :, :], dphi[:, 1, :]) + \
                            einsum("im,ijk,jl->klm", dphi[:, 1, :], dFTF[:, :, i, j, :], dphi[:, 1, :]) + \
                            einsum("i,ijkm,jl->klm", phi[:, 1], d2FTF[:, :, i, j, :, :], dphi[:, 1, :]) + \
                            einsum("i,ijk,jlm->klm", phi[:, 1], dFTF[:, :, i, j, :], d2phi[:, 1, :, :])

                        # Eigenvectors 3rd Derivatives
                        d3phi[:, 0, :, :, :] = einsum("m,l,p,pjk,j,i->iklm", 2*(lam[0]-lam[1])**(-3)*(dlam[0, :]-dlam[1, :]), (dlam[0, :]-dlam[1, :]), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",lm,p,pjk,j,i->iklm", -(lam[0]-lam[1])**(-2), (d2lam[0, :, :]-d2lam[1, :, :]), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",l,pm,pjk,j,i->iklm", -(lam[0]-lam[1])**(-2), (dlam[0, :]-dlam[1, :]), dphi[:, 1, :], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",l,p,pjkm,j,i->iklm", -(lam[0]-lam[1])**(-2), (dlam[0, :]-dlam[1, :]), phi[:, 1], d2FTF[:, :, i, j, :, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",l,p,pjk,jm,i->iklm", -(lam[0]-lam[1])**(-2), (dlam[0, :]-dlam[1, :]), phi[:, 1], dFTF[:, :, i, j, :], dphi[:, 0, :], phi[:, 1]) + \
                            einsum(",l,p,pjk,j,im->iklm", -(lam[0]-lam[1])**(-2), (dlam[0, :]-dlam[1, :]), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], dphi[:, 1, :]) + \
                            einsum("m,pl,pjk,j,i->iklm", (dlam[0, :]-dlam[1, :])/(lam[0]-lam[1])**2, dphi[:, 1, :], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",plm,pjk,j,i->iklm", (lam[0]-lam[1])**(-1), d2phi[:, 1, :, :], dFTF[:, :, i, j, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",pl,pjkm,j,i->iklm", (lam[0]-lam[1])**(-1), dphi[:, 1, :], d2FTF[:, :, i, j, :, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",pl,pjk,jm,i->iklm", (lam[0]-lam[1])**(-1), dphi[:, 1, :], dFTF[:, :, i, j, :], dphi[:, 0, :], phi[:, 1]) + \
                            einsum(",pl,pjk,j,im->iklm", (lam[0]-lam[1])**(-1), dphi[:, 1, :], dFTF[:, :, i, j, :], phi[:, 0], dphi[:, 1, :]) + \
                            einsum("m,p,pjkl,j,i->iklm", (dlam[0, :]-dlam[1, :])/(lam[0]-lam[1])**2, phi[:, 1], d2FTF[:, :, i, j, :, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",pm,pjkl,j,i->iklm", (lam[0]-lam[1])**(-1), dphi[:, 1, :], d2FTF[:, :, i, j, :, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",p,pjklm,j,i->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], d3FTF[:, :, i, j, :, :, :], phi[:, 0], phi[:, 1]) + \
                            einsum(",p,pjkl,jm,i->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], d2FTF[:, :, i, j, :, :], dphi[:, 0, :], phi[:, 1]) + \
                            einsum(",p,pjkl,j,im->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], d2FTF[:, :, i, j, :, :], phi[:, 0], dphi[:, 1, :]) + \
                            einsum("m,p,pjk,jl,i->iklm", (dlam[0, :]-dlam[1, :])/(lam[0]-lam[1])**2, phi[:, 1], dFTF[:, :, i, j, :], dphi[:, 0, :], phi[:, 1]) + \
                            einsum(",pm,pjk,jl,i->iklm", (lam[0]-lam[1])**(-1), dphi[:, 1, :], dFTF[:, :, i, j, :], dphi[:, 0, :], phi[:, 1]) + \
                            einsum(",p,pjkm,jl,i->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], d2FTF[:, :, i, j, :, :], dphi[:, 0, :], phi[:, 1]) + \
                            einsum(",p,pjk,jlm,i->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], dFTF[:, :, i, j, :], d2phi[:, 0, :, :], phi[:, 1]) + \
                            einsum(",p,pjk,jl,im->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], dFTF[:, :, i, j, :], dphi[:, 0, :], dphi[:, 1, :]) + \
                            einsum("m,p,pjk,j,il->iklm", (dlam[0, :]-dlam[1, :])/(lam[0]-lam[1])**2, phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], dphi[:, 1, :]) + \
                            einsum(",pm,pjk,j,il->iklm", (lam[0]-lam[1])**(-1), dphi[:, 1, :], dFTF[:, :, i, j, :], phi[:, 0], dphi[:, 1, :]) + \
                            einsum(",p,pjkm,j,il->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], d2FTF[:, :, i, j, :, :], phi[:, 0], dphi[:, 1, :]) + \
                            einsum(",p,pjk,jm,il->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], dFTF[:, :, i, j, :], dphi[:, 0, :], dphi[:, 1, :]) + \
                            einsum(",p,pjk,j,ilm->iklm", (lam[0]-lam[1])**(-1), phi[:, 1], dFTF[:, :, i, j, :], phi[:, 0], d2phi[:, 1, :, :])
                        d3phi[:, 1, :, :, :] = einsum("m,l,p,pjk,j,i->iklm", 2*(lam[1]-lam[0])**(-3)*(dlam[1, :]-dlam[0, :]), (dlam[1, :] - dlam[0, :]), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",lm,p,pjk,j,i->iklm", -(lam[1]-lam[0])**(-2), (d2lam[1, :, :] - d2lam[0, :, :]), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",l,pm,pjk,j,i->iklm", -(lam[1]-lam[0])**(-2), (dlam[1, :] - dlam[0, :]), dphi[:, 0, :], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",l,p,pjkm,j,i->iklm", -(lam[1]-lam[0])**(-2), (dlam[1, :] - dlam[0, :]), phi[:, 0], d2FTF[:, :, i, j, :, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",l,p,pjk,jm,i->iklm", -(lam[1]-lam[0])**(-2), (dlam[1, :] - dlam[0, :]), phi[:, 0], dFTF[:, :, i, j, :], dphi[:, 1, :], phi[:, 0]) + \
                            einsum(",l,p,pjk,j,im->iklm", -(lam[1]-lam[0])**(-2), (dlam[1, :] - dlam[0, :]), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], dphi[:, 0, :]) + \
                            einsum("m,pl,pjk,j,i->iklm", (dlam[1, :]-dlam[0, :])/(lam[1]-lam[0])**2, dphi[:, 0, :], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",plm,pjk,j,i->iklm", (lam[1]-lam[0])**(-1), d2phi[:, 0, :, :], dFTF[:, :, i, j, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",pl,pjkm,j,i->iklm", (lam[1]-lam[0])**(-1), dphi[:, 0, :], d2FTF[:, :, i, j, :, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",pl,pjk,jm,i->iklm", (lam[1]-lam[0])**(-1), dphi[:, 0, :], dFTF[:, :, i, j, :], dphi[:, 1, :], phi[:, 0]) + \
                            einsum(",pl,pjk,j,im->iklm", (lam[1]-lam[0])**(-1), dphi[:, 0, :], dFTF[:, :, i, j, :], phi[:, 1], dphi[:, 0, :]) + \
                            einsum("m,p,pjkl,j,i->iklm", (dlam[1, :]-dlam[0, :])/(lam[1]-lam[0])**2, phi[:, 0], d2FTF[:, :, i, j, :, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",pm,pjkl,j,i->iklm", (lam[1]-lam[0])**(-1), dphi[:, 0, :], d2FTF[:, :, i, j, :, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",p,pjklm,j,i->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], d3FTF[:, :, i, j, :, :, :], phi[:, 1], phi[:, 0]) + \
                            einsum(",p,pjkl,jm,i->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], d2FTF[:, :, i, j, :, :], dphi[:, 1, :], phi[:, 0]) + \
                            einsum(",p,pjkl,j,im->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], d2FTF[:, :, i, j, :, :], phi[:, 1], dphi[:, 0, :]) + \
                            einsum("m,p,pjk,jl,i->iklm", (dlam[1, :]-dlam[0, :])/(lam[1]-lam[0])**2, phi[:, 0], dFTF[:, :, i, j, :], dphi[:, 1, :], phi[:, 0]) + \
                            einsum(",pm,pjk,jl,i->iklm", (lam[1]-lam[0])**(-1), dphi[:, 0, :], dFTF[:, :, i, j, :], dphi[:, 1, :], phi[:, 0]) + \
                            einsum(",p,pjkm,jl,i->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], d2FTF[:, :, i, j, :, :], dphi[:, 1, :], phi[:, 0]) + \
                            einsum(",p,pjk,jlm,i->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], dFTF[:, :, i, j, :], d2phi[:, 1, :, :], phi[:, 0]) + \
                            einsum(",p,pjk,jl,im->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], dFTF[:, :, i, j, :], dphi[:, 1, :], dphi[:, 0, :]) + \
                            einsum("m,p,pjk,j,il->iklm", (dlam[1, :]-dlam[0, :])/(lam[1]-lam[0])**2, phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], dphi[:, 0, :]) + \
                            einsum(",pm,pjk,j,il->iklm", (lam[1]-lam[0])**(-1), dphi[:, 0, :], dFTF[:, :, i, j, :], phi[:, 1], dphi[:, 0, :]) + \
                            einsum(",p,pjkm,j,il->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], d2FTF[:, :, i, j, :, :], phi[:, 1], dphi[:, 0, :]) + \
                            einsum(",p,pjk,jm,il->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], dFTF[:, :, i, j, :], dphi[:, 1, :], dphi[:, 0, :]) + \
                            einsum(",p,pjk,j,ilm->iklm", (lam[1]-lam[0])**(-1), phi[:, 0], dFTF[:, :, i, j, :], phi[:, 1], d2phi[:, 0, :, :])
                    # Strain derivatives
                    if smeasure == 0:  # Log-Strain
                        tmp0 = np.log(lam)/2
                        tmp1 = einsum("m,mk->mk", 1.0/lam, dlam)/2
                        tmp2 = (einsum("m,mkl->mkl", 1.0/lam, d2lam) -
                                einsum("m,mk,ml->mkl", 1.0/lam**2, dlam, dlam))/2
                        if d3 == 1:
                            tmp3 = (einsum("m,n,mkl->mkln", 1.0/lam**2, dlam, d2lam) +
                                    einsum("m,mkln->mkln", 1.0/lam, d3lam) -
                                    einsum("m,n,mk,ml->mkln", -2*lam**(-2), dlam, dlam, dlam) +
                                    einsum("m,mkn,ml->mkln", 1.0/lam**2, d2lam, dlam) +
                                    einsum("m,mk,mln->mkln", 1.0/lam**2, dlam, d2lam))/2
                    elif smeasure == 1:  # Can't happen but left here for debugging purposes
                        tmp0 = (lam-1)/2
                        tmp1 = dlam/2
                        tmp2 = d2lam/2
                        if d3 == 1:
                            tmp3 = d3lam/2
                    else:  # Generalized Seth-Hill
                        tmp0 = (lam**(smeasure)-1.0)/(2*smeasure)
                        tmp1 = einsum("m,mk->mk", lam**(smeasure-1), dlam)/2
                        tmp2 = (einsum("m,ml,mk->mkl", (smeasure-1)*lam**(smeasure-2), dlam, dlam) +
                                einsum("m,mkl->mkl", lam**(smeasure-1), d2lam))/2
                        if d3 == 1:
                            tmp3 = (einsum("m,mn,ml,mk->mkln", (smeasure-1)*(smeasure-2)*lam**(smeasure-3), dlam, dlam, dlam) +
                                    einsum("m,ln,mk->mkln", (smeasure-1)*lam**(smeasure-2), d2lam, dlam) +
                                    einsum("m,ml,mkn->mkln", (smeasure-1)*lam**(smeasure-2), dlam, d2lam) +
                                    einsum("m,mn,mkl->mkln", (smeasure-1)*lam**(smeasure-2), dlam, d2lam) +
                                    einsum("m,mkln->mkln", lam**(smeasure-1), d3lam))/2

                    # Strain and Gradients
                    strain[:, :, i, j] = einsum("im,m,jm->ij", phi, tmp0, phi)
                    dstrain[:, :, i, j, :] = einsum("imk,m,jm->ijk", dphi, tmp0, phi) + \
                        einsum("im,mk,jm->ijk", phi, tmp1, phi) + \
                        einsum("im,m,jmk->ijk", phi, tmp0, dphi)
                    d2strain[:, :, i, j, :, :] = einsum("imkl,m,jm->ijkl", d2phi, tmp0, phi) + \
                        einsum("imk,ml,jm->ijkl", dphi, tmp1, phi) + \
                        einsum("imk,m,jml->ijkl", dphi, tmp0, dphi) + \
                        einsum("iml,mk,jm->ijkl", dphi, tmp1, phi) + \
                        einsum("im,mkl,jm->ijkl", phi, tmp2, phi) + \
                        einsum("im,mk,jml->ijkl", phi, tmp1, dphi) + \
                        einsum("iml,m,jmk->ijkl", dphi, tmp0, dphi) + \
                        einsum("im,ml,jmk->ijkl", phi, tmp1, dphi) + \
                        einsum("im,m,jmkl->ijkl", phi, tmp0, d2phi)

                    if d3 == 1:
                        d3strain[:, :, i, j, :, :, :] = einsum("imkln,m,jm->ijkln", d3phi, tmp0, phi) + \
                            einsum("imkl,mn,jm->ijkln", d2phi, tmp1, phi) + \
                            einsum("imkl,m,jmn->ijkln", d2phi, tmp0, dphi) + \
                            einsum("imkn,ml,jm->ijkln", d2phi, tmp1, phi) + \
                            einsum("imk,mln,jm->ijkln", dphi, tmp2, phi) + \
                            einsum("imk,ml,jmn->ijkln", dphi, tmp1, dphi) + \
                            einsum("imkn,m,jml->ijkln", d2phi, tmp0, dphi) + \
                            einsum("imk,mn,jml->ijkln", dphi, tmp1, dphi) + \
                            einsum("imk,m,jmln->ijkln", dphi, tmp0, d2phi) + \
                            einsum("imln,mk,jm->ijkln", d2phi, tmp1, phi) + \
                            einsum("iml,mkn,jm->ijkln", dphi, tmp2, phi) + \
                            einsum("iml,mk,jmn->ijkln", dphi, tmp1, dphi) + \
                            einsum("imn,mkl,jm->ijkln", dphi, tmp2, phi) + \
                            einsum("im,mkln,jm->ijkln", phi, tmp3, phi) + \
                            einsum("im,mkl,jmn->ijkln", phi, tmp2, dphi) + \
                            einsum("imn,mk,jml->ijkln", dphi, tmp1, dphi) + \
                            einsum("im,mkn,jml->ijkln", phi, tmp2, dphi) + \
                            einsum("im,mk,jmln->ijkln", phi, tmp1, d2phi) + \
                            einsum("imln,m,jmk->ijkln", d2phi, tmp0, dphi) + \
                            einsum("iml,mn,jmk->ijkln", dphi, tmp1, dphi) + \
                            einsum("iml,m,jmkn->ijkln", dphi, tmp0, d2phi) + \
                            einsum("imn,ml,jmk->ijkln", dphi, tmp1, dphi) + \
                            einsum("im,mln,jmk->ijkln", phi, tmp2, dphi) + \
                            einsum("im,ml,jmkn->ijkln", phi, tmp1, d2phi) + \
                            einsum("imn,m,jmkl->ijkln", dphi, tmp0, d2phi) + \
                            einsum("im,mn,jmkl->ijkln", phi, tmp1, d2phi) + \
                            einsum("im,m,jmkln->ijkln", phi, tmp0, d3phi)
                else:  # Repeated eigenvalues - use Green-Lagrange gradients - Happens only when stretch is unity (no strain) <hopefully :P>
                    dstrain[:, :, i, j, :] = dFTF[:, :, i, j, :]/2
                    d2strain[:, :, i, j, :, :] = d2FTF[:, :, i, j, :, :]/2
                    if d3 == 1:
                        d3strain[:, :, i, j, :, :, :] = d3FTF[:, :, i, j, :, :, :]/2
    if d3 == 1:
        return strain, dstrain, d2strain, d3strain
    return strain, dstrain, d2strain


def GENUFUNC_E(xi, X, d, props, sla=1, d3=0, smeasure=0):
    """GENUFUNC_E   : Returns the Hooke's Law potential energy and its derivatives
    USAGE   :
        U, dU, d2U = GENUFUNC_E(xi, X, d, props, sla, d3, smeasure)
    INPUTS  :
        xi	:
        X	:
        d	:
        props   : Properties class with members:
                   E, G, Npy,
                      yrange: [ymin ymax]
                      bfunc: bfunc(y) should return section breadth at location y
        sla	:
        d3	:
        smeasure:
    OUTPUTS :
    U, dU, d2U, d3U
    """
    yi, wyi = np.polynomial.legendre.leggauss(props.Npy)  # Quadrature for section
    jacy = np.diff(props.yrange)/2
    ys = props.yrange[0] + 2*jacy*yi
    ys = (1-yi)/2*props.yrange[0] + (1+yi)/2*props.yrange[1]

    Le, dLe, d2Le, d3Le = LE(X, d)
    dLe = np.reshape(dLe, (dLe.shape[0], 1))
    if d3 == 0:
        strain, dstrain, d2strain = STRAIN(xi, ys, X, d, props, sla, 0, smeasure)
    elif d3 == 1:
        strain, dstrain, d2strain, d3strain = STRAIN(xi, ys, X, d, props, sla, 0, smeasure)

    # Strain Energy
    U = 0.5*einsum("ij,j->i", props.E*(strain[0, 0, :, :]**2+strain[1, 1, :, :]**2) +
                   props.G*strain[0, 1, :, :]**2, wyi*props.bfunc(ys)*jacy)
    # First Derivative
    dU = einsum("ijk,j->ki", props.E*(einsum("ij,ijk->ijk", strain[0, 0, :, :],
                                             dstrain[0, 0, :, :, :]) +
                                      einsum("ij,ijk->ijk", strain[1, 1, :, :],
                                             dstrain[1, 1, :, :, :])) +
                props.G*einsum("ij,ijk->ijk", strain[0, 1, :, :], dstrain[0, 1, :, :, :]),
                wyi*props.bfunc(ys)*jacy)
    # Second Derivative
    d2U = einsum("ijkl,j->kil", props.E*(einsum("ijl,ijk->ijkl", dstrain[0, 0, :, :, :],
                                                dstrain[0, 0, :, :, :]) +
                                         einsum("ij,ijkl->ijkl", strain[0, 0, :, :],
                                                d2strain[0, 0, :, :, :, :]) +
                                         einsum("ijl,ijk->ijkl", dstrain[1, 1, :, :, :],
                                                dstrain[1, 1, :, :, :]) +
                                         einsum("ij,ijkl->ijkl", strain[1, 1, :, :],
                                                d2strain[1, 1, :, :, :, :])) +
                 props.G*(einsum("ijl,ijk->ijkl", dstrain[0, 1, :, :, :],
                                 dstrain[0, 1, :, :, :]) + einsum("ij,ijkl->ijkl",
                                                                  strain[0, 1, :, :],
                                                                  d2strain[0, 1, :, :, :, :])),
                 wyi*props.bfunc(ys)*jacy)
    d2U = d2U.reshape(6, 30)  # Compliant with FINT_E
    if d3 == 1:
        d3U = einsum("ijklm,j->kilm", props.E*(einsum("ijkl,ijm->klm", d2strain[0, 0, :, :, :, :],
                                                      dstrain[0, 0, :, :, :]) +
                                               einsum("ijk,ijlm->klm", dstrain[0, 0, :, :, :],
                                                      d2strain[0, 0, :, :, :, :]) +
                                               einsum("ijkm,ijl->klm", d2strain[0, 0, :, :, :, :],
                                                      dstrain[0, 0, :, :, :]) +
                                               einsum("ij,ijklm->klm", strain[0, 0, :, :],
                                                      d3strain[0, 0, :, :, :, :, :]) +
                                               einsum("ijkl,ijm->klm", d2strain[1, 1, :, :, :, :],
                                                      dstrain[1, 1, :, :, :]) +
                                               einsum("ijk,ijlm->klm", dstrain[1, 1, :, :, :],
                                                      d2strain[1, 1, :, :, :, :]) +
                                               einsum("ijkm,ijl->klm", d2strain[1, 1, :, :, :, :],
                                                      dstrain[1, 1, :, :, :]) +
                                               einsum("ij,ijklm->klm", strain[1, 1, :, :],
                                                      d3strain[1, 1, :, :, :, :, :])) +
                     props.G*(einsum("ijkl,ijm->klm", d2strain[0, 1, :, :, :, :],
                                     dstrain[0, 1, :, :, :]) +
                              einsum("ijk,ijlm->klm", dstrain[0, 1, :, :, :],
                                     d2strain[0, 1, :, :, :, :]) +
                              einsum("ijkm,ijl->klm", d2strain[0, 1, :, :, :, :],
                                     dstrain[0, 1, :, :, :]) +
                              einsum("ij,ijklm->klm", strain[0, 1, :, :],
                                     d3strain[0, 1, :, :, :, :, :])),
                     wyi*props.bfunc(ys)*jacy)
        return U, dU, d2U, d3U
    return U, dU, d2U


def UFUNC_E(xi, X, d, props, sla=1, d3=0):
    """UFUNC_E   : Returns the Hooke's Law potential energy and its derivatives
    USAGE   :
        U, dU, d2U = UFUNC_E(xi, X, d, props, sla, d3)
    INPUTS  :
    xi, X, d, props, sla, d3=0
    OUTPUTS :
    U, dU, d2U, d3U
    """
    Np = len(xi)

    e0, de0, d2e0, d3e0 = EP0(xi, X, d, sla)
    g0, dg0, d2g0, d3g0 = GM0(xi, X, d, sla)
    e1, de1, d2e1, d3e1 = EP1(xi, X, d, sla)
    e2, de2, d2e2, d3e2 = EP2(xi, X, d, sla)
    Le, dLe, d2Le, d3Le = LE(X, d)
    dLe = np.reshape(dLe, (dLe.shape[0], 1))

    EA = props.EA
    GA = props.GA
    EI2 = props.EI2
    EI4 = props.EI4

    na = np.newaxis
    U = (EA*e0**2+GA*g0**2+EI2*e1**2+2.0*EI2*e0*e2+EI4*e2**2)/2.0
    dU = ((EA*e0+EI2*e2)[na, :]*de0 + (GA*g0)[na, :]*dg0 + (EI2*e1)[na, :]*de1 + (EI2*e0+EI4*e2)[na, :]*de2)
    d2U = (kron(EA*e0+EI2*e2, ones(6))[na, :]*d2e0 + kron(GA*g0, ones(6))[na, :]*d2g0 +
           kron(EI2*e1, ones(6))[na, :]*d2e1 + kron(EI2*e0+EI4*e2, ones(6))[na, :]*d2e2) + \
        np.reshape(einsum('ij,kj->ikj', EA*de0+EI2*de2, de0) + einsum('ij,kj->ikj', GA*dg0, dg0) +
                   einsum('ij,kj->ikj', EI2*de1, de1) +
                   einsum('ij,kj->ikj', EI2*de0+EI4*de2, de2), (6, Np*6), order='F')
    if d3 == 1:
        tmo1 = ones(6)
        tmo2 = ones((6, 6))
        d2e0r = np.reshape(d2e0, (6, 6, Np), order='F')
        d2g0r = np.reshape(d2g0, (6, 6, Np), order='F')
        d2e1r = np.reshape(d2e1, (6, 6, Np), order='F')
        d2e2r = np.reshape(d2e2, (6, 6, Np), order='F')
        d2e0r = einsum('ikj', d2e0r)
        d2g0r = einsum('ikj', d2g0r)
        d2e1r = einsum('ikj', d2e1r)
        d2e2r = einsum('ikj', d2e2r)
        d3U = einsum('ij,ijk->ijk', kron(EA*e0+EI2*e2, tmo2), d3e0) + \
            einsum('ij,ijk->ijk', kron(GA*g0, tmo2), d3g0) + \
            einsum('ij,ijk->ijk', kron(EI2*e1, tmo2), d3e1) + \
            einsum('ij,ijk->ijk', kron(EI2*e0+EI4*e2, tmo2), d3e2) + \
            einsum('kj,ij->ijk', kron(EA*de0+EI2*de2, tmo1), d2e0) + \
            einsum('kj,ij->ijk', kron(GA*dg0, tmo1), d2g0) + \
            einsum('kj,ij->ijk', kron(EI2*de1, tmo1), d2e1) + \
            einsum('kj,ij->ijk', kron(EI2*de0+EI4*de2, tmo1), d2e2)
        eterms = einsum('jl,ilk->ijkl', EA*de0+EI2*de2, d2e0r) + \
            einsum('jl,ilk->ijkl', GA*dg0, d2g0r) + \
            einsum('jl,ilk->ijkl', EI2*de1, d2e1r) + \
            einsum('jl,ilk->ijkl', EI2*de0+EI4*de2, d2e2r) + \
            einsum('jlk,il->ijkl', EA*d2e0r+EI2*d2e2r, de0) + \
            einsum('jlk,il->ijkl', GA*d2g0r, dg0) + \
            einsum('jlk,il->ijkl', EI2*d2e1r, de1) + \
            einsum('jlk,il->ijkl', EI2*d2e0r+EI4*d2e2r, de2)
        d3U += np.rollaxis(eterms, 3, 1).reshape(6, -1, 6)
        return U, dU, d2U, d3U  # Checked: OK
    return U, dU, d2U  # Checked: OK


def FINT_E(X, d, No, props, sla=1, d3=0, smeasure=-1):
    """FINT_E   : Returns the static internal force developed in the system along with its Jacobian calclulated with Gaussian Quadrature
    USAGE   :
    Finte, Jinte = FINT_E(X, d, No, props, sla=1, d3=0)
    INPUTS  :
    X		: 2x1 nodal X coordinates
    d		: 6x1 nodal degrees of freedom
    No		: 1x1 Number of quadrature points per element for integration
    props	: Properties class with members,
        EA, GA, EI2, EI4
    sla		: int [1] Flag for adjusting for shear locking
    d3 		: int [0] Flag for returning third derivative tensor
    smeasure    : int [-1] Specify which strain measure to use
    OUTPUTS :
    Finte	: 6x1 nodal forces
    Jinte	: 6x6 nodal Jacobian
    Hinte	: 6x6x6 (optional) Hessian tensor
    """
    xi, wi = np.polynomial.legendre.leggauss(No)  # Quadrature Points & Weights

    # e0, de0, d2e0, d3e0 = EP0(xi, X, d, sla)
    # g0, dg0, d2g0, d3g0 = GM0(xi, X, d, sla)
    # e1, de1, d2e1, d3e1 = EP1(xi, X, d, sla)
    # e2, de2, d2e2, d3e2 = EP2(xi, X, d, sla)
    Le, dLe, d2Le, d3Le = LE(X, d)
    dLe = np.reshape(dLe, (dLe.shape[0], 1))

    # Relevant function Call
    if smeasure == -1:
        U, dU, d2U = UFUNC_E(xi, X, d, props, sla, 0)
    else:
        U, dU, d2U = GENUFUNC_E(xi, X, d, props, sla, 0, smeasure)

    Fins1 = kron(2.0*U, dLe)/4
    Fins2 = dU*Le/2
    Finte = dot((Fins1 + Fins2), wi)

    Jins1 = kron(2.0*U, d2Le)/4.0
    Jins2 = kron(dU, dLe.T)
    Jins3 = d2U*Le/2.0
    Jins13i = dot(Jins1+Jins3, kron(wi, np.eye(6)).T)
    Jins2i = dot(Jins2, kron(wi, np.eye(6)).T)
    Jins2i = 0.5*(Jins2i+Jins2i.T)
    Jinte = Jins13i + Jins2i  # + Jins4

    if d3 == 1:  # Calculate third derivatives
        pdb.set_trace()
        if smeasure == -1:
            U, dU, d2U, d3U = UFUNC_E(xi, X, d, props, sla, d3)
        else:
            U, dU, d2U, d3U = GENUFUNC_E(xi, X, d, props, sla, d3, smeasure=smeasure)
        Ui = U.dot(wi)
        dUi = dU.dot(wi)
        d2Ui = d2U.dot(kron(wi, eye(6)).T)
        d3Ui = einsum('ijk,jl->ilk', d3U, kron(wi, eye(6)).T)
        dLe = squeeze(dLe)
        Hinte = (einsum('k,ij', dUi, d2Le) + Ui*d3Le + einsum('ik,j', d2Ui, dLe) +
                 einsum('i,jk', dUi, d2Le) + einsum('ik,j', d2Le, dUi) +
                 einsum('i,jk', dLe, d2Ui) + einsum('ij,k', d2Ui, dLe) + Le*d3Ui)/2.0
        return Finte, Jinte, Hinte
    return Finte, Jinte


def FLWFRC_N(X, d, F):
    """FLWFRC_N   : Returns the follower forcing if applied in a nodal sense
    USAGE   :
    F, J = FLWFRC_N(X, d, F)
    INPUTS  :
    X		: 1x1 nodal location
    d		: 3x1 nodal dofs
    F		: 2x1 Force Amplitudes
    OUTPUTS :
    Ff		: 3x1 Nodal forcing
    Jf		: 3x3 Nodal force Jacobian
    """
    Ff = np.zeros(3)
    Ff[0] = F[0]*cos(d[2]) - F[1]*sin(d[2])
    Ff[1] = F[0]*sin(d[2]) + F[1]*cos(d[2])
    Ff[2] = F[0]*(-d[0]*sin(d[2])+d[1]*cos(d[2])) - F[1]*(d[0]*cos(d[2])+d[1]*sin(d[2]))

    J1 = np.array([[0, 0, -sin(d[2])],
                   [0, 0, cos(d[2])],
                   [-sin(d[2]), cos(d[2]), -(d[0]*cos(d[2])+d[1]*sin(d[2]))]])
    J2 = np.array([[0, 0, -cos(d[2])],
                   [0, 0, -sin(d[2])],
                   [-cos(d[2]), -sin(d[2]), (d[0]*sin(d[2])-d[1]*cos(d[2]))]])
    Jf = F[0]*J1 + F[1]*J2
    return Ff, Jf


class FLWLDS:
    # Class helpful for setting follower loads
    def __init__(self):
        self.Nl = 0
        self.nd = []
        self.F = []


class IHBCS:
    # Class helpful for setting inhomogeneous boundary conditions
    def __init__(self):
        self.Nb = 0
        self.nd = []
        self.dof = []
        self.val = []
        self.cnum = []

    def push(self, nd, dof, valfun, cnum=1.0):
        self.nd.append(nd)
        self.dof.append(dof)
        self.val.append(valfun)
        self.cnum.append(cnum)
        self.Nb += 1


def STATICRESFN(Xnds, u, fshape, btrans, No, props, sla=1, spret=1, NDFls=FLWLDS(), IhBcs=IHBCS(), d3=0, smeasure=-1):
    """STATICRESFN   : Returns the static residue and Jacobian for the system under static conditions
    USAGE   :
        R, dRdX, dRdf = STATICRESFN(Xnds, u, fshape, btrans, No, props, sla=1, spret=1, NDFls=FLWLDS(), IhBcs=IHBCS(), d3=0, smeasure=-1)
    INPUTS  :
    Xnds, u, fshape, btrans, No, props, sla=1, spret=1, NDFls=FLWLDS(), IhBcs=IHBCS(), d3=0, smeasure=-1
    OUTPUTS :
    R, dRdX, dRdf, d2RdXdf
    """
    #    pdb.set_trace()
    Nn = np.int(btrans.shape[0]/3)  # Total number of nodes
    Ne = Nn-1
    Nd = Nn*3
    uph = btrans.dot(u[0:-1])

    # pdb.set_trace()
    # Stitching
    R = np.zeros(Nd)
    dRdX = ss.lil_matrix((Nd, Nd), dtype=float)
    Rtmp = np.zeros(6)
    dRdXtmp = np.zeros((6, 6), dtype=float)
    for e in range(0, Ne):
        nstart = e
        nend = nstart+2
        istart = nstart*3
        iend = istart+6
        Rtmp, dRdXtmp = FINT_E(Xnds[nstart:nend], uph[istart:iend], No, props, sla, 0, smeasure)
        R[istart:iend] += Rtmp
        dRdX[istart:iend, istart:iend] += dRdXtmp
    f = u[-1]  # Forcing amplitude

    # External Forces - Non-follower
    Fext = fshape*f
    # External Force - Follower
    Jext = ss.lil_matrix((Nd, Nd), dtype=float)
    for k in range(NDFls.Nl):
        nd = NDFls.nd[k]
        istart = nd*3
        iend = istart+3
        Ftmp, Jtmp = FLWFRC_N(Xnds[nd], uph[istart:iend], NDFls.F[k]*f)
        Fext[istart:iend] += Ftmp
        Jext[istart:iend, istart:iend] += Jtmp
    dFexdf = Fext/f
    d2Fexdf = Jext/f
    # Inhomogeneous Boundary Conditions
    for k in range(IhBcs.Nb):
        nds = IhBcs.nd[k]
        df = IhBcs.dof[k]
        R[(nds-1)*3+df] = uph[(nds-1)*3+df]*IhBcs.cnum[k]
        Fext[(nds-1)*3+df], dFexdf[(nds-1)*3+df] = IhBcs.val[k](f)
        Fext[(nds-1)*3+df] *= IhBcs.cnum[k]
        dFexdf[(nds-1)*3+df] *= IhBcs.cnum[k]
        Jext[(nds-1)*3+df, :], d2Fexdf[(nds-1)*3+df] = 0.0, 0.0
        dRdX[(nds-1)*3+df, :] = 0.0
        dRdX[(nds-1)*3+df, (nds-1)*3+df] = IhBcs.cnum[k]

    R = btrans.T.dot(R-Fext)
    dRdX = btrans.T.dot(dRdX-Jext).dot(btrans)
    if spret != 1:
        dRdX = dRdX.todense()
    dRdf = -btrans.T.dot(dFexdf)
    d2RdXdf = -btrans.T.dot(d2Fexdf).dot(btrans)
    if d3 == 1:
        # d2RdX2 = ss.lil_matrix((Nd, Nd, Nd), dtype=float)
        d2RdX2 = np.zeros((Nd, Nd, Nd), dtype=float)
        d2RdX2tmp = np.zeros((6, 6, 6), dtype=float)
        for e in range(0, Ne):
            nstart = e
            nend = nstart+2
            istart = nstart*3
            iend = istart+6
            _, _, d2RdX2tmp = FINT_E(Xnds[nstart:nend], uph[istart:iend], No, props, sla, d3=1)
            d2RdX2[istart:iend, istart:iend, istart:iend] += d2RdX2tmp
        # Inhomogeneous Boundary Conditions
        for k in range(IhBcs.Nb):
            nds = IhBcs.nd[k]
            df = IhBcs.dof[k]
            d2RdX2[(nds-1)*3+df, :, :] = 0.0
        d2RdX2 = np.einsum('lk,ijl->ijk', btrans.todense(), d2RdX2)
        d2RdX2 = np.einsum('lj,ilk->ijk', btrans.todense(), d2RdX2)
        d2RdX2 = np.einsum('li,ljk->ijk', btrans.todense(), d2RdX2)

        if spret == 1:
            d2RdX2 = sp.COO.from_numpy(d2RdX2)
        return R, dRdX, dRdf, d2RdXdf, d2RdX2
    return R, dRdX, dRdf, d2RdXdf
