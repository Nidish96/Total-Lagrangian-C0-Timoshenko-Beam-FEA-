#!/usr/bin/python
###################################################################################################
# FILE: SOLVERS.py                                                                                #
# Written by Nidish Narayanaa Balaji on 19 Nov 2018                                               #
# Last modified by N. N. Balaji on 1 Dec 2018                                                     #
#                                                                                                 #
# The current file includes routines required for the solution and path-tracing carrieed out. An  #
# exact Newton solver is implemented for the sparse FEM systems since SciPy's full Jacobian root  #
# routines are not adapted for sparse Jacobian matrices yet.                                      #
###################################################################################################
import numpy as np
from numpy import sqrt, kron, dot, diff, squeeze, sin, cos
import scipy.sparse as ss
import scipy.sparse.linalg as sl
import scipy.linalg as sn
import scipy.optimize as so
import sparse as sp
import pdb


def SPNEWTONSOLVER(func, u0, opt):
    """SPNEWTONSOLVER   : Solves the system using a sparse Newton-Solver. Expects the full Jacobian returned as a sparse matrix from the function handle.
    USAGE   :
        sol = SPNEWTONSOLVER(func, u0, opt)
    INPUTS  :
    func	: Function handle for force and Jacobian
    u0		: Initial Guess
    opt		: Options structure with members
        ITMAX, relethr, absethr, absrthr
    OUTPUTS :
    sol
    """
    R0, J0 = func(u0)[0:2]
    du0 = -sl.spsolve(J0, R0)
    e0 = np.abs(dot(R0, du0))
    it = 0
    u = u0
    du = du0

    e = [e0]
    r = [np.sqrt(np.mean(R0**2))]
    rele = [1.0]
    flag = 0
    while it < opt.ITMAX:
        u = u + du

        R, J = func(u)[0:2]
        du = -sl.spsolve(J, R)
        r.append(np.sqrt(np.mean(R**2)))
        e.append(np.abs(dot(R, du)))
        rele.append(e[-1]/e0)

        it += 1
        status = 0
        if rele[-1] <= opt.relethr:
            status += 1
        if e[-1] <= opt.absethr:
            status += 2
        if r[-1] <= opt.absrthr:
            status += 4

        if status > 1:
            flag = 1
            break
#        if rele[-1] <= opt.relethr and e[-1] <= opt.absethr and r[-1] <= opt.absrthr:
#            flag = 1  # Solved before ITMAX
#            break
#    status = 0
    if flag == 0:  # Status if not all criteria met at last iteration
        if rele[-1] <= opt.relethr:
            status += 1
        if e[-1] <= opt.absethr:
            status += 2
        if r[-1] <= opt.absrthr:
            status += 4
    else:
        status = 7
    sol = type('', (), {'fjac': J, 'fun': R, 'nfev': it+1, 'njac': it+1, 'success': flag,
                        'status': status, 'x': u, 'e': e, 'r': r, 'rele': rele, 'nit': it})()
    return sol


def ARCLENGTHFN(dd, dlam, K, opt):
    """ARCLENGTHFN   : Returns the standard arclength function and its first derivatives
    USAGE   :
        FAL, dFdd, dFdl = ARCLENGTHFN(d, lam, K, opt)
    INPUTS  :
    d, lam, K, opt (with .b & .c)
    OUTPUTS :
    FAL, dFdd, dFdl
    """
    F = np.sqrt(opt.c*dd.dot(K.dot(dd)) + opt.b*dlam**2)
    dFdd = opt.c*K.dot(dd)/F
    dFdl = opt.b*dlam/F
    return F, dFdd, dFdl


def ARCORTHOGFN(dd, dlam, K, opt):
    """ARCORTHOGFN   : Orthogonal continuation function
    USAGE   :
    F, dFdd, dFdl = ARCORTHOGFN(dd, dlam, K, opt)
    INPUTS  :
    dd, dlam, K, opt
    OUTPUTS :
    F, dFdd, dFdl
    """
    F = dd.dot(opt.du0[:-1])+dlam*opt.du0[-1]
    dFdd = opt.du0[:-1]
    dFdl = opt.du0[-1]
    return F, dFdd, dFdl


def CONTSTEP(func, u0, du0, ds, Lam, opt, ALfn=ARCLENGTHFN):
    """CONTSTEP   : Conducts arc-length continuation of the solution of the parameterized system.
    USAGE   :
    sol = CONTSTEP(func, u0, du0, ds, Lam, opt, ALfn=ARCLENGTHFN)
    INPUTS  :
    func, u0, du0, ds, Lam, opt, ALfn=ARCLENGTHFN
    OUTPUTS :
    sol
    """
    # pdb.set_trace()
    # Starting point
    R0, dRdu0, dRdl0 = func(u0, Lam)[0:3]
    dRdu0i = sl.inv(dRdu0)
    q = np.atleast_1d(dRdu0i.dot(dRdl0))
    if ALfn.func_name == 'ARCLENGTHFN':
        opt.c = (1.0-opt.b)/q.dot(dRdu0.dot(q))
        try:
            sc = 1.0/np.sqrt(opt.c*du0[0:-1].dot(dRdu0.dot(du0[0:-1])) + opt.b*du0[-1]**2)
        except Exception as inst:
            pdb.set_trace()
            print("hey")
    else:
        opt.c = 1.0
        sc = 1.0
        opt.du0 = du0
    up = u0 + sc*du0[0:-1]*ds  # Prediction
    lp = Lam + sc*du0[-1]*ds
    f0, dfdu0, dfdl0 = ALfn(up-u0, lp-Lam, dRdu0, opt)
    f0 = f0-ds
    dellam0 = (dfdu0.dot(dRdu0i.dot(R0))-f0)/(dfdl0-dfdu0.dot(dRdu0i.dot(dRdl0)))
    delu0 = -dRdu0i.dot(R0+dRdl0*dellam0)
    e0 = np.abs(R0.dot(delu0) + f0*dellam0)

    u = up
    lam = lp
    delu = delu0
    dellam = dellam0
    it = 0
    e = [e0]
    rele = [1.0]
    r = [np.linalg.norm(np.hstack((R0, f0)))]
    flag = 0
    while it < opt.ITMAX:
        u = u + delu
        lam = lam + dellam

        R, dRdu, dRdl = func(u, lam)[0:3]
        dRdui = sl.inv(dRdu)
        f, dfdu, dfdl = ALfn(u-u0, lam-Lam, dRdu0, opt)
        f = f-ds
        dellam = (dfdu.dot(dRdui.dot(R))-f)/(dfdl-dfdu.dot(dRdui.dot(dRdl)))
        delu = -dRdui.dot(R+dRdl*dellam)

        e.append(np.squeeze(np.abs(R.dot(delu) + f*dellam)))
        rele.append(e[-1]/e0)
        r.append(np.linalg.norm(np.hstack((R, f))))

        it += 1
        status = 0
        if rele[-1] <= opt.relethr:
            status += 1
        if e[-1] <= opt.absethr:
            status += 2
        if r[-1] <= opt.absrthr:
            status += 4
        if status > 1:
            flag = 1  # 2 Criteria met before ITMAX
            break
    sol = type('', (), {'fjac': dRdu, 'fun': R, 'nfev': it+1, 'success': flag, 'status': status, 'x': u, 'lam': lam, 'e': e, 'r': r, 'rele': rele, 'nit': it})()
    return sol


def CONTINUESOLS(func, X0, l0, le, ds, opt, ALfn=ARCLENGTHFN, adapt=1):
    """CONTINUESOLS   : Computes branch starting at given point using tangent predictors successively
    USAGE   :
        X, lam, mE = CONTINUESOLS(func, X0, l0, le, ds, opt, ALfn=ARCLENGTHFN, adapt=1)
    INPUTS  :
    func, X0, l0, le, ds, opt, ALfn=ARCLENGTHFN
    OUTPUTS :
    X, lam, mE
    """
    sgl = np.sign(le-l0)
    X = [X0]
    lam = [l0]
    _, dRdX, dRdl = func(X[-1], lam[-1])[0:3]
    mE = [np.sort(np.linalg.eigvals(dRdX.todense()))]
    z = -sl.spsolve(dRdX, dRdl)
    al = sgl*1.0/np.sqrt(1.0+z.dot(z))
    while (sgl*lam[-1] < sgl*le and len(lam) < opt.maxsteps and
           lam[-1] >= opt.minl and lam[-1] <= opt.maxl):
        try:
            # pdb.set_trace()
            sol = CONTSTEP(func, X[-1], np.hstack((z*al, al)), ds, lam[-1], opt, ALfn=ALfn)
        except Exception as inst:
            if ds == opt.dsmax:
                print('Singular Matrix Encountered - Quitting.')
                break
            else:
                print('Singular Matrix Encountered - Trying to step with dsmax.')
                if len(lam) > 1:
                    lam.pop()
                    X.pop()
                    mE.pop()
                try:
                    dstry = opt.dsmax
                    ds /= 2
                    sol = CONTSTEP(func, X[-1], np.hstack((z*al, al)), dstry, lam[-1], opt, ALfn=ALfn)
                except Exception as inst:
                    print('Singular Matrix Encountered - Quitting.')
                    break

        if sol.status != 0:
            X.append(sol.x)
            lam.append(sol.lam)
            _, dRdX, dRdl = func(X[-1], lam[-1])[0:3]
            mE.append(np.sort(np.linalg.eigvals(dRdX.todense())))

            zp = z
            alp = al
            z = -sl.spsolve(dRdX, dRdl)
            al = np.sign(alp*(zp.dot(z)+1.0))/np.sqrt(1.0+z.dot(z))
            if adapt > 0:
                if sol.nit <= 5 and ds < opt.dsmax:
                    ds *= 1.0+adapt
                if sol.nit > 20 and ds > opt.dsmin:
                    ds /= 1.0+adapt
        else:
            if ds <= opt.dsmin:
                print('Failure - Quitting Continuation.')
                break
            else:
                ds /= 1.0+adapt
        print('Step %d. Lam = %e. stat = %d. Its = %d. ds = %.2e. nE = %d. th1 = %.2e' %
              (len(lam)-1, lam[-1], sol.status, sol.nit, ds, np.sum(mE[-1] < 0), sol.x[15]))
    return X, lam, mE


def SINGTANGENTS(resfn, X, lam, mE, opt, ei=0):
    """SINGTANGENTS   : Returns the tangents from the two branches at singular point. Assumes simple bifurcation
    USAGE   :
    du1, du2, others = SINGTANGENTS(resfn, X, lam, mE, ei=0)
    INPUTS  :
    resfn, X, lam, mE, ei=0
    OUTPUTS :
    du1, du2, others
    """
    # pdb.set_trace()
    def mineigval(lam, u0, k=0):
        us = SPNEWTONSOLVER(lambda u: resfn(u, lam)[0:2], u0, opt)
        return np.sort(np.linalg.eigvals(us.fjac.todense()))[k]

    # 1. Find Critical Point
    # pdb.set_trace()
    cpi = np.where(np.array(mE)[:-1, ei]*np.array(mE)[1:, ei] < 0)[0][0]+1
    mc = np.argmin([mE[cpi][0], mE[cpi-1][0]])
    biflam = so.fsolve(lambda lmu: mineigval(lmu, X[cpi-mc], k=ei), lam[cpi-mc])
    # biflam = so.bisect(lambda lmu: mineigval(lmu, X[cpi-1], k=ei), lam[cpi-1], lam[cpi])
    us = SPNEWTONSOLVER(lambda u: resfn(u, biflam)[0:2], X[cpi-mc], opt)

    Rb, dRdXb, dRdlb, d2RdXlb, d2RdX2b = resfn(us.x, biflam, d3=1)
    evals, evecs = np.linalg.eig(dRdXb.todense())
    evecs = np.asarray(evecs[:, np.argsort(evals)])
    evals = evals[np.argsort(evals)]

    # pdb.set_trace()
    # 2. Branch-Switching
    zv = evecs[:, ei]
    Lm = sn.null_space(zv[np.newaxis, :])
    LdL = Lm.T.dot(dRdXb.todense()).dot(Lm)
    yv = -Lm.dot(np.linalg.solve(LdL, Lm.T.dot(dRdlb)))
    # yv = -ss.linalg.spsolve(dRdXb, dRdlb)

    aval = zv.dot(sp.tensordot(d2RdX2b, zv, axes=1).dot(zv))
    bval = zv.dot(sp.tensordot(d2RdX2b, zv, axes=1).dot(yv) +
                  sp.tensordot(d2RdX2b, yv, axes=1).dot(zv) +
                  2.0*d2RdXlb.dot(zv))
    cval = zv.dot(d2RdX2b.dot(yv).dot(yv) + 2.0*d2RdXlb.dot(yv) + 0.0)
    if np.abs(aval > 1e-10):
        sig1 = (-bval - np.sqrt(bval**2-4*aval*cval))/(2.0*aval)
        sig2 = (-bval + np.sqrt(bval**2-4*aval*cval))/(2.0*aval)
    else:
        sig1 = 0.0
        sig2 = 1e10  # Some large number, representative of infty
    sig1, sig2 = (sig1, sig2)[np.argmin((np.abs(sig1), np.abs(sig2)))], (sig1, sig2)[np.argmax((np.abs(sig1), np.abs(sig2)))]
    du1 = (sig1*zv+yv)  # Trivial branch
    if min(np.abs(sig1), np.abs(sig2)) == 0.0:
        du1 = du1/np.linalg.norm(du1)
    al1 = 1.0/np.sqrt(1.0+du1.dot(du1))
    du2 = (sig2*zv+yv)  # Bifurcated Branch
    if min(np.abs(sig1), np.abs(sig2)) == 0.0:
        du2 = du2/np.linalg.norm(du2)
    al2 = 1.0/np.sqrt(1.0+du2.dot(du2))

    others = type('', (), {'zv': zv, 'yv': yv, 'sig1': sig1, 'sig2': sig2, 'biflam': biflam, 'cpi': cpi})()
    return du1, al1, du2, al2, others
