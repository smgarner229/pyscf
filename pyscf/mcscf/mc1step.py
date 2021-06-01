#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import time
import copy
import os
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib, tools
from pyscf.lib import logger
from pyscf.mcscf import casci, addons
from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf import chkfile
from pyscf import ao2mo, scf
from pyscf import gto
from pyscf import fci
from pyscf.soscf import ciah
from pyscf.data import nist
from pyscf import __config__

import scipy
import scipy.optimize

WITH_MICRO_SCHEDULER = getattr(__config__, 'mcscf_mc1step_CASSCF_with_micro_scheduler', False)
WITH_STEPSIZE_SCHEDULER = getattr(__config__, 'mcscf_mc1step_CASSCF_with_stepsize_scheduler', True)

# ref. JCP, 82, 5053;  JCP, 73, 2342

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, u, casdm1, casdm2, eris, get_g=False):
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    dm1 = numpy.zeros((nmo,nmo))
    idx = numpy.arange(ncore)
    dm1[idx,idx] = 2
    dm1[ncore:nocc,ncore:nocc] = casdm1

    # part5
    jkcaa = numpy.empty((nocc,ncas))
    # part2, part3
    vhf_a = numpy.empty((nmo,nmo))
    # part1 ~ (J + 2K)
    dm2tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3)
    dm2tmp = dm2tmp.reshape(ncas**2,-1)
    hdm2 = numpy.empty((nmo,ncas,nmo,ncas))
    g_dm2 = numpy.empty((nmo,ncas))
    for i in range(nmo):
        jbuf = eris.ppaa[i]
        kbuf = eris.papa[i]
        if i < nocc:
            jkcaa[i] = numpy.einsum('ik,ik->i', 6*kbuf[:,i]-2*jbuf[i], casdm1)
        vhf_a[i] =(numpy.einsum('quv,uv->q', jbuf, casdm1)
                 - numpy.einsum('uqv,uv->q', kbuf, casdm1) * .5)
        jtmp = lib.dot(jbuf.reshape(nmo,-1), casdm2.reshape(ncas*ncas,-1))
        jtmp = jtmp.reshape(nmo,ncas,ncas)
        ktmp = lib.dot(kbuf.transpose(1,0,2).reshape(nmo,-1), dm2tmp)
        hdm2[i] = (ktmp.reshape(nmo,ncas,ncas)+jtmp).transpose(1,0,2)
        g_dm2[i] = numpy.einsum('uuv->v', jtmp[ncore:nocc])
    jbuf = kbuf = jtmp = ktmp = dm2tmp = None
    vhf_ca = eris.vhf_c + vhf_a
    h1e_mo = reduce(numpy.dot, (mo.T, casscf.get_hcore(), mo))

    ################# gradient #################
    g = numpy.zeros_like(h1e_mo)
    g[:,:ncore] = (h1e_mo[:,:ncore] + vhf_ca[:,:ncore]) * 2
    g[:,ncore:nocc] = numpy.dot(h1e_mo[:,ncore:nocc]+eris.vhf_c[:,ncore:nocc],casdm1)
    g[:,ncore:nocc] += g_dm2
    
    
    def gorb_update(u, fcivec):
        uc = u[:,:ncore].copy()
        ua = u[:,ncore:nocc].copy()
        rmat = u - numpy.eye(nmo)
        ra = rmat[:,ncore:nocc].copy()
        mo1 = numpy.dot(mo, u)
        mo_c = numpy.dot(mo, uc)
        mo_a = numpy.dot(mo, ua)
        dm_c = numpy.dot(mo_c, mo_c.T) * 2

        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, nelecas)
        dm_a = reduce(numpy.dot, (mo_a, casdm1, mo_a.T))
        vj, vk = casscf.get_jk(casscf.mol, (dm_c, dm_a))
        vhf_c = reduce(numpy.dot, (mo1.T, vj[0]-vk[0]*.5, mo1[:,:nocc]))
        vhf_a = reduce(numpy.dot, (mo1.T, vj[1]-vk[1]*.5, mo1[:,:nocc]))
        h1e_mo1 = reduce(numpy.dot, (u.T, h1e_mo, u[:,:nocc]))
        p1aa = numpy.empty((nmo,ncas,ncas*ncas))
        paa1 = numpy.empty((nmo,ncas*ncas,ncas))
        aaaa = numpy.empty([ncas]*4)
        for i in range(nmo):
            jbuf = eris.ppaa[i]
            kbuf = eris.papa[i]
            p1aa[i] = lib.dot(ua.T, jbuf.reshape(nmo,-1))
            paa1[i] = lib.dot(kbuf.transpose(0,2,1).reshape(-1,nmo), ra)
            if ncore <= i < nocc:
                aaaa[i-ncore] = jbuf[ncore:nocc]

        g = numpy.zeros_like(h1e_mo)
        g[:,:ncore] = (h1e_mo1[:,:ncore] + vhf_c[:,:ncore] + vhf_a[:,:ncore]) * 2
        g[:,ncore:nocc] = numpy.dot(h1e_mo1[:,ncore:nocc]+vhf_c[:,ncore:nocc], casdm1)
# 0000 + 1000 + 0100 + 0010 + 0001 + 1100 + 1010 + 1001  (missing 0110 + 0101 + 0011)
        p1aa = lib.dot(u.T, p1aa.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        paa1 = lib.dot(u.T, paa1.reshape(nmo,-1)).reshape(nmo,ncas,ncas,ncas)
        p1aa += paa1
        p1aa += paa1.transpose(0,1,3,2)
        g[:,ncore:nocc] += numpy.einsum('puwx,wxuv->pv', p1aa, casdm2)
        return casscf.pack_uniq_var(g-g.T)

    #Lan needs to get g only
    if get_g:
        return g, gorb_update

    ############## hessian, diagonal ###########

    # part7
    h_diag = numpy.einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    g_diag = g.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)
    idx = numpy.arange(nmo)
    h_diag[idx,idx] += g_diag * 2

    # part2, part3
    v_diag = vhf_ca.diagonal() # (pr|kl) * E(sq,lk)
    h_diag[:,:ncore] += v_diag.reshape(-1,1) * 2
    h_diag[:ncore] += v_diag * 2
    idx = numpy.arange(ncore)
    h_diag[idx,idx] -= v_diag[:ncore] * 4
    # V_{pr} E_{sq}
    tmp = numpy.einsum('ii,jj->ij', eris.vhf_c, casdm1)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -eris.vhf_c[ncore:nocc,ncore:nocc] * casdm1
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # part4
    # -2(pr|sq) + 4(pq|sr) + 4(pq|rs) - 2(ps|rq)
    tmp = 6 * eris.k_pc - 2 * eris.j_pc
    h_diag[ncore:,:ncore] += tmp[ncore:]
    h_diag[:ncore,ncore:] += tmp[ncore:].T

    # part5 and part6 diag
    # -(qr|kp) E_s^k  p in core, sk in active
    h_diag[:nocc,ncore:nocc] -= jkcaa
    h_diag[ncore:nocc,:nocc] -= jkcaa.T

    v_diag = numpy.einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag.T
    h_diag[:,ncore:nocc] += v_diag

# Does this term contribute to internal rotation?
#    h_diag[ncore:nocc,ncore:nocc] -= v_diag[:,ncore:nocc]*2


    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)
    #print "g shape", g.shape[0], g.shape[1], g_orb.shape[0]

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x)

        # part7
        # (-h_{sp} R_{rs} gamma_{rq} - h_{rq} R_{pq} gamma_{sp})/2 + (pr<->qs)
        x2 = reduce(lib.dot, (h1e_mo, x1, dm1))
        # part8
        # (g_{ps}\delta_{qr}R_rs + g_{qr}\delta_{ps}) * R_pq)/2 + (pr<->qs)
        x2 -= numpy.dot((g+g.T), x1) * .5
        # part2
        # (-2Vhf_{sp}\delta_{qr}R_pq - 2Vhf_{qr}\delta_{sp}R_rs)/2 + (pr<->qs)
        x2[:ncore] += reduce(numpy.dot, (x1[:ncore,ncore:], vhf_ca[ncore:])) * 2
        # part3
        # (-Vhf_{sp}gamma_{qr}R_{pq} - Vhf_{qr}gamma_{sp}R_{rs})/2 + (pr<->qs)
        x2[ncore:nocc] += reduce(numpy.dot, (casdm1, x1[ncore:nocc], eris.vhf_c))
        # part1
        x2[:,ncore:nocc] += numpy.einsum('purv,rv->pu', hdm2, x1[:,ncore:nocc])

        if ncore > 0:
            # part4, part5, part6
# Due to x1_rs [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
#    == -x1_sr [4(pq|sr) + 4(pq|rs) - 2(pr|sq) - 2(ps|rq)] for r>s p>q,
# x2[:,:ncore] += H * x1[:,:ncore] => (becuase x1=-x1.T) =>
# x2[:,:ncore] += -H' * x1[:ncore] => (becuase x2-x2.T) =>
# x2[:ncore] += H' * x1[:ncore]
            va, vc = casscf.update_jk_in_ah(mo, x1, casdm1, eris)
            x2[ncore:nocc] += va
            x2[:ncore,ncore:] += vc

        # (pr<->qs)
        x2 = x2 - x2.T
        if casscf.is_gmres_trust_region:
            x2 += 0.1 * x1
        return casscf.pack_uniq_var(x2)

    return g_orb, gorb_update, h_op, h_diag





# BGN Lan's SS-CASSCF

#GMRES solver
def genMinRes(casscf, bvec, xguess, linear_transform, inner_product=None, precondition=None, thresh=1.0e-6, maxiter=20):
  """
  Solves the linear system A x = b via the generalized minimal residual method, written by Eric Neuscamman.

  required inputs:

    bvec                the vector b
    xguess              an initial guess for the vector x
    linear_transform    function that performs the linear transformation A x

  optional inputs:

    inner_product       function of two inputs that computes their inner product, defaults to vector dot product
    precondition        function of one input (e.g. x) that returns an approximation to  A^(-1) x, defaults to the identity
    thresh              error threshold below which to stop the iterations, defaults to 1.0e-6
    maxiter             maximum number of iterations, defaults to 20

  Returns the solution vector x.
  """
  import numpy as np
  log = logger.new_logger(casscf, verbose=None)
  
  # default is to use no preconditioning
  if precondition is None:
    precondition = lambda x: x

  # default inner product is the simple vector dot product
  if inner_product is None:
    inner_product = lambda x, y: np.sum( x * y )

  # create a function to reshape vectors into the same shape as the input
  rsInput = lambda x: np.reshape(x, bvec.shape)

  # create a function to reshape vectors into the shape used internally
  rsInternal = lambda x: np.reshape(x, [bvec.size, 1] )

  # create a function to take inner products
  ip = lambda x, y: inner_product( rsInput(x), rsInput(y) )

  # create a function to evaluate a vector's norm
  norm = lambda x: np.sqrt( ip(x,x) )

  # create a function to apply the linear transformation and reshape
  lt = lambda x: rsInternal( linear_transform( rsInput(x) ) )

  # get initial c vector
  c = np.reshape( np.array( [ norm(xguess) ] ), [1,1] )

  # get first Krylov vector
  Y = np.reshape( xguess / c[0,0], [bvec.size, 1] )

  # get linear transformation of first Krylov vector
  AY = lt(Y[:,0:1])

  # iterate
  for iteration in range(maxiter+1):

    # get linear transform on current x vector
    Ax = np.dot(AY,c)

    # get residual
    r = rsInternal(bvec) - Ax

    # get residual norm and check for convergence
    res_norm = norm(r)
    #log.debug("iteration %4i       res_norm = %12.6e", iteration, res_norm )
    sys.stdout.flush()
    if res_norm < thresh:
      log.info( "genMinRes converged after %i iterations", iteration)
      return rsInput(np.dot(Y,c))

    # stop if the maximum number of iterations has been reached
    if iteration == maxiter:
      log.info("genMinRes reached the maximum number of iterations.")
      log.info("Returning the current best estimate to the solution.")
      return rsInput(np.dot(Y,c))

    # get next krylov vector
    q = rsInternal( precondition( rsInput(r) ) )

    # orthonormalize new krylov vector against existing krylov vectors
    for i in range(Y.shape[1]):
      q = q - ip(Y[:,i:i+1], q) * Y[:,i:i+1]
    q = q / norm(q)

    # save the new Krylov vector
    Y = np.concatenate( [ Y, q ], 1 )

    # save the linear transformation of the new Krylov vector
    AY = np.concatenate( [ AY, lt(q) ], 1 )

    # get the SVD of the matrix of linearly transformed Krylov vectors
    U, sigma, VT = np.linalg.svd(AY, full_matrices=False)

    # get the pseudo-inverse of the singular values
    inv_sigma = np.zeros([sigma.size, 1])
    for i in range(sigma.size):
      if np.abs( sigma[i] / sigma[0] ) > 1.0e-8:
        inv_sigma[i,0] = 1.0 / sigma[i]

    # get the new c vector
    c = np.dot( np.transpose(VT), inv_sigma * np.dot( np.transpose(U), rsInternal(bvec) ) )

#line search to minimize gradE w.r.t orb. rotation
def lineSearch_naive(casscf, dr, fcivec, u, gorb_update):
    log = logger.new_logger(casscf, verbose=None)
    u_new = casscf.update_rotate_matrix(dr, u)
    gorb = gorb_update(u_new, fcivec())    
    norm_gorb = numpy.linalg.norm(gorb)
    log.debug(' before line search  |g|=%5.3g', norm_gorb)
    if norm_gorb < 1e-04:
        return u_new, gorb
    
    alpha_min = 1
    norm_gorb_min = norm_gorb
    alpha = 0
    while alpha < 3:
        u_new = casscf.update_rotate_matrix(alpha*dr, u)
        gorb = gorb_update(u_new, fcivec())    
        norm_gorb = numpy.linalg.norm(gorb)
        #log.debug("alpha = %5.4f     norm_gorb = %12.6e", alpha, norm_gorb )
        if norm_gorb < norm_gorb_min:
            norm_gorb_min = norm_gorb
            alpha_min = alpha
        alpha += 0.1        

    alpha = alpha_min
    u_new = casscf.update_rotate_matrix(alpha*dr, u)
    gorb = gorb_update(u_new, fcivec())    
    norm_gorb = numpy.linalg.norm(gorb)
    log.info(" Best step length = %5.4f       norm_gorb = %5.6f", alpha, norm_gorb )
    
    return u_new, gorb

def lineSearch(casscf, dr, fcivec, u, gorb_update):
    log = logger.new_logger(casscf, verbose=None)
    ncore = casscf.ncore
    ncas  = casscf.ncas
    nocc  = ncore+ncas

    u_new = casscf.update_rotate_matrix(dr, u)
    gorb = gorb_update(u_new, fcivec())    
    norm_gorb = numpy.linalg.norm(gorb)
    log.debug(' before line search  |g|=%5.3g', norm_gorb)
    if norm_gorb < 1e-06:
        return u_new, gorb

    def dgorb_dalpha(alpha):
        dalpha = 1e-07
        u_new  = casscf.update_rotate_matrix((alpha+dalpha)*dr, u)
        gorb   = gorb_update(u_new, fcivec())    
        norm_gorb = numpy.linalg.norm(gorb)
        
        u_new  = casscf.update_rotate_matrix(alpha*dr, u)
        gorb   = gorb_update(u_new, fcivec())    
        norm_gorb -= numpy.linalg.norm(gorb)

        return norm_gorb/dalpha 
    
    alpha_upper = 0
    alpha_lower = 0

    alpha = 0.
    list_bound = []
    list_norm_min = []
    while alpha <= 2:
        dnorm  = dgorb_dalpha(alpha)
        u_new  = casscf.update_rotate_matrix(alpha*dr, u)
        gorb   = gorb_update(u_new, fcivec())    
        norm_gorb = numpy.linalg.norm(gorb)
        if alpha > 0.:
            if dnorm_old < 0 and dnorm > 0:
                list_bound.append([alpha_old, alpha])
                list_norm_min.append(norm_gorb)
            elif dnorm_old > 0 and dnorm < 0:
                list_bound.append([alpha, alpha_old])
                list_norm_min.append(norm_gorb)
        dnorm_old  = dnorm
        alpha_old = alpha
        alpha  += 0.02    

    
    if (len(list_bound) != 0):

        # find the global minimum of norm_gorb
        from operator import itemgetter
        idx = min(enumerate(list_norm_min), key=itemgetter(1))[0]
        alpha_lower = list_bound[idx][0]
        alpha_upper = list_bound[idx][1]
        log.info(" global minimum of norm_gorb is in [%5.4f, %5.4f]", alpha_lower, alpha_upper)
        log.info(" perform bisection to find the best step length")
        
        alpha = 0.5*(alpha_upper + alpha_lower)
        dnorm = dgorb_dalpha(alpha)
        while abs(dnorm) > 1e-04:
            if dnorm < 0:
                alpha_lower = alpha
            else:
                alpha_upper = alpha
            alpha = 0.5*(alpha_upper + alpha_lower)
            dnorm = dgorb_dalpha(alpha)
            u_new = casscf.update_rotate_matrix(alpha*dr, u)
            gorb = gorb_update(u_new, fcivec())    
            norm_gorb = numpy.linalg.norm(gorb)
            log.debug("alpha = %5.4f  dnorm_gorb = %12.6f  norm_gorb = %5.6f",  alpha, dnorm, norm_gorb )
    else:
        log.info('there are no bounds => alpha = 1')
        alpha = 1

    u_new = casscf.update_rotate_matrix(alpha*dr, u)
    gorb = gorb_update(u_new, fcivec())    
    norm_gorb = numpy.linalg.norm(gorb)
    log.info(" Best step length = %5.4f       norm_gorb = %5.6f", alpha, norm_gorb )
    
    return u_new, gorb

# orbital optimization using GMRES
def rotate_orb_gmres(casscf, mo, fcivec, fcasdm1, fcasdm2, eris, imacro, x0_guess=None,
                     conv_tol_grad=1e-4, max_stepsize=None, verbose=None):

    log = logger.new_logger(casscf, verbose)
    
    t3m = (time.clock(), time.time())
    u = 1
    #print "fcasdm1"
    #print fcasdm1()
    g_orb, gorb_update, h_op, h_diag = \
            casscf.gen_g_hop(mo, u, fcasdm1(), fcasdm2(), eris)
    norm_gorb = numpy.linalg.norm(g_orb)
    log.debug(' before gmres  |g|=%5.3g', norm_gorb)
    t3m = log.timer('gen h_op', *t3m)

    def precond(x):
        hdiagd = h_diag - casscf.gmres_hess_shift
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
        norm_x = numpy.linalg.norm(x)
        x *= 1/norm_x
        #if norm_x < 1e-2:
        #    x *= 1e-2/norm_x
        return x
    
    jkcount = 0
    x0_guess = g_orb
    dr = 0
    
    #call GMRES
    bvec = -g_orb
    norm_gorb  = numpy.linalg.norm(g_orb)

    tol = casscf.gmres_conv_tol
    if casscf.is_gmres_conv_dynm:
        tol = 0.25*norm_gorb
        log.info('genMinRes tol is dynamically changed: %5.3g', tol ) 
    if casscf.is_gmres_precond:
        dr = genMinRes(casscf, bvec, x0_guess, h_op, thresh=tol, maxiter=casscf.gmres_max_cycle, precondition=precond)
    else:
        dr = genMinRes(casscf, bvec, x0_guess, h_op, thresh=tol, maxiter=casscf.gmres_max_cycle)

    #call line search
    if casscf.is_line_search:
        u_new, g_orb = lineSearch_naive(casscf, dr, fcivec, u, gorb_update)
    else:
        #if numpy.amax(abs(dr)) > 0.05:
        #    dr *= 0.1/numpy.amax(abs(dr))
        u_new = casscf.update_rotate_matrix(dr, u)
        g_orb = gorb_update(u_new, fcivec())    

    norm_gorb = numpy.linalg.norm(g_orb)
    log.debug(' after gmres + linesearch  |g|=%5.3g', norm_gorb)
    
    return u_new, g_orb

#Target state
def rota_rdms(mo_coeff, rdm1_AO):
    from numpy.linalg import pinv
    mo_coeff_inv = pinv(mo_coeff)
    rdm1_MO = reduce(numpy.dot, (mo_coeff_inv, rdm1_AO, mo_coeff_inv.T))
    
    return rdm1_MO

def read_sCI_output(ncas):
    file1 = open('output.dat', 'r')
    Lines = file1.readlines()
    
    count = 0
    save_line = []
    state_list = []
    k = 0
    list_len_civec = []
    for line in Lines: 
        save_line.append(line.strip())
        if line.strip()[0:5] == 'State':
            if line.strip()[10] != "0": 
                list_len_civec.append(k-1)
            k = 0
    
        if  line.strip()[0:9]  == "Returning" or line.strip()[0:12]  == "PERTURBATION":
            list_len_civec.append(k-4)                        
        k += 1
    
    i = 0
    civec = []
    config = []
    while i < len(save_line):
        if save_line[i][0:5] == "State":
            istate = int(save_line[i][10])
            len_civec = list_len_civec[istate]
            civec_istate = []
            config_istate = []
            for j in xrange(len_civec):
                civec_istate.append(float(save_line[i+j+1][7:19]))
                tmp_ = []
                l = 0
                for k in xrange(ncas):
                    l = 22+2*k
                    tmp_.append(save_line[i+j+1][l])
                config_istate.append(tmp_)
            civec.append(civec_istate)
            config.append(config_istate)
        i += 1
        
    return civec, config

def extract_spin_state_sCI(istate, civec, config):
    spin_istate = 0.0
    tmp_ = config[istate][0]
    #print "for state", istate
    #print civec[istate]
    #print config[istate]
    count_alpha = tmp_.count('a')
    count_beta  = tmp_.count('b')
    #if count_alpha != count_beta:
    #    spin_istate = abs(count_alpha-count_beta)
    #    #print "we are here 111", count_alpha, count_beta, tmp_
    #elif abs(civec[istate][0]+civec[istate][1]) < 1e-03:
    #    spin_istate = 1.0
    #else:
    #print "we are here 222"
    #for iconfig in xrange(0,len(civec[istate])):
    #    if abs(civec[istate][iconfig]) > 0:
    #        tmp_ = config[istate][iconfig]
    #        count_alpha = tmp_.count('a')
    #        count_beta  = tmp_.count('b')            
    #        #print "we are here 333"
    #        if count_alpha != count_beta:
    spin_istate = abs(count_alpha-count_beta)
    #            break
    return 0.5*spin_istate*(0.5*spin_istate+1)

    
def select_target_state(casscf, mo_coeff, fcivec, e_tot, envs, target_state, nroots, eris):
    log = logger.new_logger(casscf, verbose=None)
    norb  = mo_coeff.shape[1]
    ncore = casscf.ncore
    ncas  = casscf.ncas
    nocc  = ncore+ncas
    omega = envs['omega']
    rdm1Target_AO = envs['rdm1Target_AO'] # in AOs
    d_target = scf.hf.dip_moment(mol=casscf.mol, dm=rdm1Target_AO)

    #reading sCI information from output.dat
    civec = [] #dump variable for sCI
    config = [] # dump variable for sCI
    if(casscf.is_use_sCI):
        civec, config = read_sCI_output(ncas)
        
    def eval_Hsqr(s):
        #wfn_norm = numpy.sqrt(numpy.sum(numpy.matmul(fcivec[s], fcivec[s].T)))
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec[s], ncas, casscf.nelecas)
        #casdm1 = casscf.fcisolver.make_rdm1(fcivec[s], ncas, casscf.nelecas)
        #casdm2 = casscf.fcisolver.make_rdm2(fcivec[s], ncas, casscf.nelecas)
        g, gorb_update = gen_g_hop(casscf, mo_coeff, 1, casdm1, casdm2, eris, get_g=True)
        gorb = casscf.pack_uniq_var(g-g.T)
        Hsqr = numpy.sum(numpy.square(gorb)) #/ wfn_norm
        gorbNorm = numpy.linalg.norm(gorb)
        
        return Hsqr, gorbNorm

    rdm1Target = rota_rdms(mo_coeff, rdm1Target_AO)

    W_list = []
    s_list = []
    d_list = []
    for s in range(nroots):
        if not casscf.is_use_sCI:
            ss = casscf.fcisolver.spin_square(fcivec[s], ncas, casscf.nelecas)[0]
        else:
            ss = extract_spin_state_sCI(s, civec, config)
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec[s], ncas, casscf.nelecas)
        rdm1_MO, rdm2_MO = addons._make_rdm12_on_mo(casdm1, casdm2, ncore, ncas, norb)

        
        ddmNorm = 1./ncas * numpy.linalg.norm(rdm1_MO - rdm1Target) 

        print()
        print("dipole for Root", s)
        rdm1_AO = addons.make_rdm1(casscf, mo_coeff, fcivec[s])
        d = scf.hf.dip_moment(mol=casscf.mol, dm=rdm1_AO, print_dip=False)
        dNorm = numpy.linalg.norm(d - d_target) / (5.*ncas)
        d_list.append(d)

        W = numpy.square(omega - e_tot[s]) + eval_Hsqr(s)[0] + ddmNorm
        if casscf.is_target_dipole:
            W += dNorm

        if casscf.is_only_ddm:
            W = ddmNorm
        elif casscf.is_only_W:
            W = numpy.square(omega - e_tot[s]) + eval_Hsqr(s)[0]
        elif casscf.is_only_E:
            W = numpy.square(omega - e_tot[s])
        elif casscf.is_only_dipole:
            dNorm = numpy.linalg.norm(d - d_target) / (5.*ncas)
            W = dNorm
        elif casscf.is_ddm_and_dipole:
            W = ddmNorm + dNorm

        #print
        log.info('Root %d : Total = %.6f  (omega-E)^2 = %.6f Hsqr = %.6f ddmNorm = %.6f dNorm = %.6f S^2 = %.6f',
                 s, W, numpy.square(omega - e_tot[s]), eval_Hsqr(s)[0], ddmNorm, dNorm, ss)
        #d *= nist.AU2DEBYE
        #log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *d)
        
        print()
        ss_target = 0.5*casscf.target_state_spin*(0.5*casscf.target_state_spin+1)
        if abs(ss - ss_target) < 1e-2:
            W_list.append(W)
            s_list.append(s)
        #print "Natural orbital analysis: "
        if not casscf.is_use_sCI:
            nat_orbs = casscf.cas_natorb(mo_coeff, fcivec[s])
        #print
        #print "CI vector"
        #print fcivec[s]
        #print 'wfn_norm = ', wfn_norm
        #print
        
    assert(len(W_list) == len(s_list))
    
    W_min = W_list[0]
    for i in range(len(W_list)):
        #print 'w_min', W_min, W_list[0], target_state
        if abs(W_list[i]) <= abs(W_min):
            target_state = s_list[i]
            W_min   = W_list[i]
                
    return target_state, d_list[target_state]

#not used#    def eval_energy(alpha, dalpha=0): # objective function
#not used#        alpha += dalpha
#not used#        u_new = casscf.update_rotate_matrix(alpha*dr, u)
#not used#        mo_coeff_new = casscf.rotate_mo(mo_coeff, u_new)
#not used#        energy_core  = casscf.energy_nuc()
#not used#        h1e_ao  = casscf.get_hcore()
#not used#        if(ncore > 0):
#not used#            mo_core = mo_coeff_new[:,:ncore]
#not used#            core_dm = numpy.dot(mo_core, mo_core.T) * 2
#not used#            energy_core += numpy.einsum('ij,ji', core_dm, casscf.get_hcore())
#not used#            energy_core += numpy.einsum('ij,ji', core_dm, casscf.get_veff(casscf.mol)) * .5
#not used#            h1e_ao      += casscf.get_veff(casscf.mol, core_dm)
#not used#    
#not used#        h1e = reduce(numpy.dot, (mo_coeff_new[:,ncore:nocc].T, h1e_ao, mo_coeff_new[:,ncore:nocc]))
#not used#        eri = ao2mo.kernel(casscf.mol, mo_coeff_new[:,ncore:nocc], compact=False)
#not used#        eri = numpy.reshape(eri, (ncas, ncas, ncas, ncas))
#not used#        
#not used#        #rdm1, rdm2 = make_rdm12(casscf, s)
#not used#        casdm1 = casscf.fcisolver.make_rdm1(fcivec(), ncas, casscf.nelecas)
#not used#        casdm2 = casscf.fcisolver.make_rdm2(fcivec(), ncas, casscf.nelecas)
#not used#        H_1e = numpy.einsum('ik,ik->',     casdm1, h1e)  #[:ncas,:ncas]
#not used#        H_2e = numpy.einsum('ijkl,ijkl->', casdm2, eri) # 2. W_2e [:ncas,:ncas,:ncas,:ncas]
#not used#        
#not used#        return energy_core + H_1e + 0.5 * H_2e 
#not used#

def eval_energy(mol, h1e_ao, enuc, mo_coeff, ncas, casdm1, casdm2): # objective function
    #mo_coeff = casscf.mo_coeff
    #energy_core  = mol.energy_nuc()
    #h1e_ao  = casscf.get_hcore()
    #if(ncore > 0):
    #    mo_core = mo_coeff[:,:ncore]
    #    core_dm = numpy.dot(mo_core, mo_core.T) * 2
    #    energy_core += numpy.einsum('ij,ji', core_dm, casscf.get_hcore())
    #    energy_core += numpy.einsum('ij,ji', core_dm, casscf.get_veff(casscf.mol)) * .5
    #    h1e_ao      += casscf.get_veff(casscf.mol, core_dm)

    ncore = 0
    nocc = ncas
    h1e = reduce(numpy.dot, (mo_coeff[:,ncore:nocc].T, h1e_ao, mo_coeff[:,ncore:nocc]))
    eri = ao2mo.kernel(mol, mo_coeff[:,ncore:nocc], compact=False)
    eri = numpy.reshape(eri, (ncas, ncas, ncas, ncas))
    
    #rdm1, rdm2 = make_rdm12(casscf, s)
    #casdm1 = casscf.fcisolver.make_rdm1(fcivec(), ncas, casscf.nelecas)
    #casdm2 = casscf.fcisolver.make_rdm2(fcivec(), ncas, casscf.nelecas)
    H_1e = numpy.einsum('ik,ik->',     casdm1, h1e)  #[:ncas,:ncas]
    H_2e = numpy.einsum('ijkl,ijkl->', casdm2, eri) # 2. W_2e [:ncas,:ncas,:ncas,:ncas]
    print("H_1e", H_1e, H_2e)
    return enuc + H_1e + 0.5 * H_2e 


def genMOandCI(mc, mol, civec=None, mo_coeff=None):

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if civec is None:
        #if mc.fcisolver.nroots == 1:
        civec    = mc.ci
        #else:
        #    civec = mc.ci[mc.target_state]
    #print civec
    # print "xxxx ", civec.shape[0], civec.shape[1]
    ncas     = mc.ncas
    neleca, nelecb = mc.nelecas
    #mol = mc.mol
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    nvir = nmo - ncore - ncas
    nocc = ncas


    #normalize civec
    wfn_norm = numpy.sqrt(numpy.sum(numpy.matmul(civec, civec.T)))
    #civec = civec/wfn_norm

    print('||civec|| = ', numpy.sqrt(numpy.sum(numpy.matmul(civec, civec.T))))
    
    #print ""
    #print "NEW MO COEFFICIENTS"
    #print ""
    #print ""
    #print ""
    #
    #tools.dump_mat.dump_mo(mol,  mo_coeff, label=mol.ao_labels(), ncol=5, digits=6)

    def gen_string():
        # for alpha
        a = [bin(x) for x in fci.cistring.make_strings(range(ncas),neleca)]        
        stringa = []
        for k in range(len(a)):
            b = list(a[k])[len(list(a[k]))-1]
            for i in range(0,len(list(a[k]))-1):
                j = len(list(a[k]))-2-i
                if j > 1:
                    b += list(a[k])[j]
            if len(list(b)) < ncas:
                for i in range(ncas-len(list(b))):
                    b += '0'
        
            #print b
            c = ''
            for i in range(ncore):
                c += '1'
            d = ''
            for i in range(nvir):
                d += '0'
            stringa.append(b)

        # for beta
        a = [bin(x) for x in fci.cistring.make_strings(range(ncas),nelecb)]        
        stringb = []
        for k in range(len(a)):
            b = list(a[k])[len(list(a[k]))-1]
            for i in range(0,len(list(a[k]))-1):
                j = len(list(a[k]))-2-i
                if j > 1:
                    b += list(a[k])[j]
            if len(list(b)) < ncas:
                for i in range(ncas-len(list(b))):
                    b += '0'
        
            #print b
            c = ''
            for i in range(ncore):
                c += '1'
            d = ''
            for i in range(nvir):
                d += '0'
            stringb.append(b)

        combo = []
        #print "xxxx ", len(stringa), len(stringb), civec.shape[0], civec.shape[1]
        for i in range(len(stringb)):
            for j in range(len(stringa)):
                #print i, j, stringa[j], stringb[i], civec[j,i]
                combo.append([stringa[j], stringb[i], civec[j,i]])

        #for i in range(len(combo)):
        #    print combo[i][0], combo[i][1], combo[i][2]

        #print(fci.cistring.gen_linkstr_index(range(4),2))
        return combo
    
    string = gen_string()

    print()
    print()
    print(" ALPHA   |  BETA    | COEFFICIENT")
    for i in range(nocc):
        sys.stdout.write("-")
    print("-|-",)
    for i in range(nocc):
        sys.stdout.write("-")
        #print "-",
    print("-|-",)
    for i in range(20):
        sys.stdout.write("-")
        #print "-",
    print
    #print "   ", stringa
    #print "   ", stringb
    for i in range(len(string)):
            if abs(string[i][2]) > 0.1:
                sys.stdout.write(str(string[i][0]))
                print(" | ",)
                sys.stdout.write(str(string[i][1]))
                print(" | ",)
                sys.stdout.write(str(string[i][2]))
                print()
    print("..... DONE WITH GENERAL CI COMPUTATION ..... ")



# END Lan's SS-CASSCF

def rotate_orb_cc(casscf, mo, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                  conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    log = logger.new_logger(casscf, verbose)
    if max_stepsize is None:
        max_stepsize = casscf.max_stepsize

    t3m = (time.clock(), time.time())
    u = 1
    g_orb, gorb_update, h_op, h_diag = \
            casscf.gen_g_hop(mo, u, fcasdm1(), fcasdm2(), eris)
    ngorb = g_orb.size
    g_kf = g_orb
    norm_gkf = norm_gorb = numpy.linalg.norm(g_orb)
    log.debug('    |g|=%5.3g', norm_gorb)
    t3m = log.timer('gen h_op', *t3m)

    if norm_gorb < conv_tol_grad*.3:
        u = casscf.update_rotate_matrix(g_orb*0)
        yield u, g_orb, 1, x0_guess
        return

    def precond(x, e):
        hdiagd = h_diag-(e-casscf.ah_level_shift)
        hdiagd[abs(hdiagd)<1e-8] = 1e-8
        x = x/hdiagd
        norm_x = numpy.linalg.norm(x)
        x *= 1/norm_x
        #if norm_x < 1e-2:
        #    x *= 1e-2/norm_x
        return x

    jkcount = 0
    if x0_guess is None:
        x0_guess = g_orb
    imic = 0
    dr = 0
    ikf = 0
    g_op = lambda: g_orb

    for ah_end, ihop, w, dxi, hdxi, residual, seig \
            in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                                tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                                lindep=casscf.ah_lindep, verbose=log):
        # residual = v[0] * (g+(h-e)x) ~ v[0] * grad
        norm_residual = numpy.linalg.norm(residual)
        if (ah_end or ihop == casscf.ah_max_cycle or # make sure to use the last step
            ((norm_residual < casscf.ah_start_tol) and (ihop >= casscf.ah_start_cycle)) or
            (seig < casscf.ah_lindep)):
            imic += 1
            dxmax = numpy.max(abs(dxi))
            if dxmax > max_stepsize:
                scale = max_stepsize / dxmax
                log.debug1('... scale rotation size %g', scale)
                dxi *= scale
                hdxi *= scale
            else:
                scale = None

            g_orb = g_orb + hdxi
            dr = dr + dxi
            norm_gorb = numpy.linalg.norm(g_orb)
            norm_dxi = numpy.linalg.norm(dxi)
            norm_dr = numpy.linalg.norm(dr)
            log.debug('    imic %d(%d)  |g[o]|=%5.3g  |dxi|=%5.3g  '
                      'max(|x|)=%5.3g  |dr|=%5.3g  eig=%5.3g  seig=%5.3g',
                      imic, ihop, norm_gorb, norm_dxi,
                      dxmax, norm_dr, w, seig)

            ikf += 1
            if ikf > 1 and norm_gorb > norm_gkf*casscf.ah_grad_trust_region:
                g_orb = g_orb - hdxi
                dr -= dxi
                #norm_gorb = numpy.linalg.norm(g_orb)
                log.debug('|g| >> keyframe, Restore previouse step')
                break

            elif (norm_gorb < conv_tol_grad*.3):
                break

            elif (ikf >= max(casscf.kf_interval, -numpy.log(norm_dr+1e-7)) or
# Insert keyframe if the keyframe and the esitimated grad are too different
                   norm_gorb < norm_gkf/casscf.kf_trust_region):
                ikf = 0
                u = casscf.update_rotate_matrix(dr, u)
                t3m = log.timer('aug_hess in %d inner iters' % imic, *t3m)
                yield u, g_kf, ihop+jkcount, dxi

                t3m = (time.clock(), time.time())
# TODO: test whether to update h_op, h_diag to change the orbital hessian.
# It leads to the different hessian operations in the same davidson
# diagonalization procedure.  This is generally a bad approximation because it
# results in ill-defined hessian eigenvalue in the davidson algorithm.  But in
# certain cases, it is a small perturbation that help the mcscf optimization
# algorithm move out of local minimum
#                h_op, h_diag = \
#                        casscf.gen_g_hop(mo, u, fcasdm1(), fcasdm2(), eris)[2:4]
                g_kf1 = gorb_update(u, fcivec())
                jkcount += 1

                norm_gkf1 = numpy.linalg.norm(g_kf1)
                norm_dg = numpy.linalg.norm(g_kf1-g_orb)
                log.debug('    |g|=%5.3g (keyframe), |g-correction|=%5.3g',
                          norm_gkf1, norm_dg)
#
# Special treatment if out of trust region
#
                if (norm_dg > norm_gorb*casscf.ah_grad_trust_region and
                    norm_gkf1 > norm_gkf and
                    norm_gkf1 > norm_gkf*casscf.ah_grad_trust_region):
                    log.debug('    Keyframe |g|=%5.3g  |g_last| =%5.3g out of trust region',
                              norm_gkf1, norm_gorb)
# Slightly moving forward, not completely restoring last step.
# In some cases, the optimization moves out of trust region in the first micro
# iteration.  The small forward step can ensure the orbital changes in the
# current iteration.
                    dr = -dxi * .5
                    g_kf = g_kf1
                    break
                t3m = log.timer('gen h_op', *t3m)
                g_orb = g_kf = g_kf1
                norm_gorb = norm_gkf = norm_gkf1
                dr[:] = 0

    u = casscf.update_rotate_matrix(dr, u)
    yield u, g_kf, ihop+jkcount, dxi


def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=1e-03,
           ci0=None, callback=None, verbose=logger.NOTE, dump_chk=True):
    '''quasi-newton CASSCF optimization driver
    '''
    log = logger.new_logger(casscf, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 1-step CASSCF')
    if callback is None:
        callback = casscf.callback

    mo = mo_coeff
    nmo = mo_coeff.shape[1]
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.ao2mo(mo)
    if os.environ.get("cycle") is not None:
        #if (int(os.environ["cycle"])+1) == 1:
        #    conv_tol_grad = 5e-03
        #    tol = 5e-04
        #    logger.info(casscf, 'Lan sets conv_tol to %g', tol)
        #    logger.info(casscf, 'Lan sets conv_tol_grad to %g', conv_tol_grad)
        if (int(os.environ["cycle"])+1) > 1:
            if(casscf.is_select_state):
                rdm1_pregeom_AO = addons.make_rdm12(casscf, mo, ci0) #, rdm2Target_AO
            #print "rdm1_pregeom_AO"
            #print rdm1_pregeom_AO
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    if casscf.ncas == nmo and not casscf.internal_rotation:
        if casscf.canonicalization:
            log.debug('CASSCF canonicalization')
            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
                                                        casscf.sorting_mo_energy,
                                                        casscf.natorb, verbose=log)
        else:
            mo_energy = None
        return True, e_tot, e_cas, fcivec, mo, mo_energy

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 5
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot
    r0 = None

    #Lan generates targeted rdms
    if(casscf.is_select_state):
        rdm1Target_AO = addons.make_rdm12(casscf, mo, fcivec) #, rdm2Target_AO
    #casdm1Target, casdm2Target = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas, casscf.nelecas)
    
    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas, casscf.nelecas)
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)
    imacro = 0
    dr0 = None
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        omega = e_tot
        casdm1_old = casdm1
        #casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas, casscf.nelecas)
        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())
        imicro = 0
        if(casscf.is_use_gmres):
            print("Using GMRES")
            for imicro in range(max_cycle_micro):
                imicro += 1
                print('imicro', imicro)
                u, g_orb = casscf.rotate_orb_gmres(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                                   eris, imacro, r0, conv_tol_grad*.3, max_stepsize, log)
                
                norm_gorb = numpy.linalg.norm(g_orb)
                log.debug(' |g|=%5.3g', norm_gorb)
                if imicro == 1:
                    norm_gorb0 = norm_gorb
                norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
                t3m = log.timer('orbital rotation', *t3m)
                if imicro >= max_cycle_micro:
                    log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g',
                              imicro, norm_t, norm_gorb)
                    break
                casdm1, casdm2, gci, fcivec = \
                        casscf.update_casdm(mo, u, fcivec, e_cas, eris, locals())
                norm_ddm = numpy.linalg.norm(casdm1 - casdm1_last)
                norm_ddm_micro = numpy.linalg.norm(casdm1 - casdm1_prev)
                casdm1_prev = casdm1
                t3m = log.timer('update CAS DM', *t3m)
                if isinstance(gci, numpy.ndarray):
                    norm_gci = numpy.linalg.norm(gci)
                    log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%5.3g  |ddm|=%5.3g',
                              imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
                else:
                    norm_gci = None
                    log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%s  |ddm|=%5.3g',
                              imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
    
                if callable(callback):
                    callback(locals())
    
                t3m = log.timer('micro iter %d'%imicro, *t3m)
                #if (norm_t < conv_tol_grad or
                #    (norm_gorb < conv_tol_grad*.5 and
                #     (norm_ddm < conv_tol_ddm*.4 or norm_ddm_micro < conv_tol_ddm*.4))):
                #    break

                #eris = None
                #u = u.copy()
                #g_orb = g_orb.copy()
                #mo = casscf.rotate_mo(mo, u, log=None)                
                #eris = casscf.ao2mo(mo)
                #t3m = log.timer('update eri', *t3m)
                ##norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
                ##de = numpy.dot(casscf.pack_uniq_var(u), g_orb)
                ##save current imicro
                #norm_gorb_old =  norm_gorb
                #u_old = u
                #g_orb_old = g_orb

                
                #if norm_gorb < 1e-04: #norm_t < 1e-4 or abs(de) < tol*.4 or 
                #    break
            #print "new molecular orbitals"
            #from pyscf import tools
            #tools.dump_mat.dump_mo(casscf.mol, mo, label=casscf.mol.ao_labels(), digits=6)
            #if imacro%5 == 0:
            #    fname='mo_iter_'+str(imacro)+'.txt'
            #    with open(fname, 'w') as f:
            #        for i in range(mo.shape[0]):
            #            for j in range(mo.shape[1]):
            #                f.write(" %20.10f" % mo[i,j])
            #        f.write("\n")
            
        else:
            rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                        eris, r0, conv_tol_grad*.3, max_stepsize, log)
            for u, g_orb, njk, r0 in rota:
                imicro += 1
                norm_gorb = numpy.linalg.norm(g_orb)
                if imicro == 1:
                    norm_gorb0 = norm_gorb
                norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
                t3m = log.timer('orbital rotation', *t3m)
                if imicro >= max_cycle_micro:
                    log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g',
                              imicro, norm_t, norm_gorb)
                    break
                casdm1, casdm2, gci, fcivec = \
                        casscf.update_casdm(mo, u, fcivec, e_cas, eris, locals())
                norm_ddm = numpy.linalg.norm(casdm1 - casdm1_last)
                norm_ddm_micro = numpy.linalg.norm(casdm1 - casdm1_prev)
                casdm1_prev = casdm1
                t3m = log.timer('update CAS DM', *t3m)
                if isinstance(gci, numpy.ndarray):
                    norm_gci = numpy.linalg.norm(gci)
                    log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%5.3g  |ddm|=%5.3g',
                              imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
                else:
                    norm_gci = None
                    log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%s  |ddm|=%5.3g',
                              imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
    
                if callable(callback):
                    callback(locals())
    
                t3m = log.timer('micro iter %d'%imicro, *t3m)
                if (norm_t < conv_tol_grad or
                    (norm_gorb < conv_tol_grad*.5 and
                     (norm_ddm < conv_tol_ddm*.4 or norm_ddm_micro < conv_tol_ddm*.4))):
                    break
    
            rota.close()
            rota = None

        eris = None
        # keep u, g_orb in locals() so that they can be accessed by callback
        u = u.copy()
        g_orb = g_orb.copy()
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        totmicro += imicro
        totinner += njk

        if casscf.is_use_sCI and casscf.is_save_sCIout:
            if os.path.exists("output.dat"):
                import shutil
                if os.environ.get("cycle") is not None:
                    icycle = int(os.environ["cycle"])+1
                    shutil.copyfile("output.dat","output_"+str(icycle)+"_"+str(imacro)+".dat")
                else:
                    shutil.copyfile("output.dat","output_"+str(imacro)+".dat")
                
        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas, casscf.nelecas)
        norm_ddm = numpy.linalg.norm(casdm1 - casdm1_last)
        casdm1_prev = casdm1_last = casdm1
        log.timer('CASCI solver', *t2m)
        t3m = t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        #print "AAAAAAAAAAAA ", tol, conv_tol_grad, conv_tol_ddm
        #print "BBBBBBBBBBBB ", abs(de), norm_gorb0, norm_ddm
        if (abs(de) < tol
            and (norm_gorb0 < conv_tol_grad and norm_ddm < conv_tol_ddm)):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb and dump_chk: # dump_chk may save casdm1
            nocc = casscf.ncore + casscf.ncas
            occ, ucas = casscf._eig(-casdm1, casscf.ncore, nocc)
            casdm1 = numpy.diag(-occ)
    else:
        mo_energy = None

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy


def as_scanner(mc, envs=None): #, mo_fname=None):
    '''Generating a scanner for CASSCF PES.

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CASSCF energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters of MCSCF object
    (conv_tol, max_memory etc) are automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1.2', verbose=0)
    >>> mc_scanner = mcscf.CASSCF(scf.RHF(mol), 4, 4).as_scanner()
    >>> e = mc_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> e = mc_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    from pyscf.mcscf.addons import project_init_guess
    if isinstance(mc, lib.SinglePointScanner):
        return mc

    logger.info(mc, 'Create scanner for %s', mc.__class__)

    def write_mat(y, fname, ut=False):
      x = y
      if type(y) == type(1) or type(y) == type(1.0) or type(y) == type(numpy.array([1.0])[0]) or type(y) == type(numpy.array([1])[0]):
        x = numpy.array([[ 1.0 * y ]])
      elif len(y.shape) == 1:
        x = numpy.reshape(y, [y.size,1])
      with open(fname, 'w') as f:
        for i in range(x.shape[0]):
          for j in range(x.shape[1]):
            if j >= i or not ut:
              f.write(" %20.10f" % x[i,j])
            else:
              f.write(" %20s" % " ")
          f.write("\n")
    
    class CASSCF_Scanner(mc.__class__, lib.SinglePointScanner):
        def __init__(self, mc):
            self.__dict__.update(mc.__dict__)
            self._scf = mc._scf.as_scanner()
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mf_scanner = self._scf
            mf_scanner(mol)
            

            self.mol = mol
            
            if (self.mo_coeff is None):
                print("we are using HF orbitals")
                mo = mf_scanner.mo_coeff
            else:
                print("we are using predefined orbitals")
                mo = self.mo_coeff
                
            #geom_cycle = int(os.environ["cycle"])+1
            #if geom_cycle == 1:
            #    mo_coeff_fix = mo
            #else:
            #    print "we are using the fixed MOs"
            #    mo = mo_coeff_fix
                
            mo = project_init_guess(self, mo)
            #print "envs[cycle]", envs['cycle']
            #Lan: optimizing excited state
            if(self.is_select_state):
                print("You are optimizing your selected state")
                #if os.environ.get("cycle") is not None:
                #    geom_cycle = int(os.environ["cycle"])+1
                #    if geom_cycle > 1:
                #        print "Using SA as initial guess of SS"
                #        nroots = 4 #self.fcisolver.nroots
                #        w = 1./nroots
                #        #w = 1./7
                #        
                #        mc_sa = CASSCF(mol, self.ncas, self.nelecas, frozen=None)
                #        mc_sa.fcisolver = fci.solver(mol, singlet=True)
                #        #mc_sa.fcisolver.wfnsym = 1 #self.fcisolver.wfnsym 
                #        #mc_sa = addons.state_average_(mc_sa, (w,w,0,0,0,w,w,w,w,w)) #nroots*(w,)) #
                #        mc_sa = addons.state_average_(mc_sa, nroots*(w,)) #
                #        mc_sa.is_use_gmres = True
                #        mc_sa.gmres_hess_shift=0.2
                #        mc_sa.is_select_state = False
                #        mc_sa.mo_coeff = mf_scanner.mo_coeff
                #        mc_sa.mc1step()
                #        mo = mc_sa.mo_coeff
                #        #e_tot = self.mc2step(mo, self.ci)[0]
                e_tot = self.mc1step(mo, self.ci)[0]
                #from pyscf import molden
                #with open('geomopt_'+str(geom_cycle)+'.molden', 'w') as f:
                #    molden.header(mol,f)
                #    molden.orbital_coeff(self.mol, f, self.mo_coeff)
            elif(self.sa_geom_opt):
                print()
                print("WARN: you are optimizing geometry using SA-CASSCF orbitals w/o solving Z-vector equation!!!")
                nroots = self.fcisolver.nroots
                w = 1./nroots
                mc_sa = CASSCF(mol, self.ncas, self.nelecas, frozen=None)
                #cas_list = [25,27,28,29,30,31,32,33,34,37,43]
                #geom_cycle = int(os.environ["cycle"])+1
                #mc_sa.mo_coeff = addons.sort_mo(mc_sa, mo, cas_list)
                mc_sa.fcisolver = fci.solver(mol, singlet=True)
                mc_sa = addons.state_average_(mc_sa, nroots*(w,))
                mc_sa.is_select_state = False
                #mc_sa.is_use_gmres = True
                #mc_sa.gmres_hess_shift=0.2
                mc_sa.kernel(mo)
                mo = mc_sa.mo_coeff
                e_tot = self.mc2step(mo, self.ci)[0]
            else:
                e_tot = self.kernel(mo, self.ci)[0]
            return e_tot
    return CASSCF_Scanner(mc)

# To extend CASSCF for certain CAS space solver, it can be done by assign an
# object or a module to CASSCF.fcisolver.  The fcisolver object or module
# should at least have three member functions "kernel" (wfn for given
# hamiltonain), "make_rdm12" (1- and 2-pdm), "absorb_h1e" (effective
# 2e-hamiltonain) in 1-step CASSCF solver, and two member functions "kernel"
# and "make_rdm12" in 2-step CASSCF solver
class CASSCF(casci.CASCI):
    __doc__ = casci.CASCI.__doc__ + '''CASSCF

    Extra attributes for CASSCF:

        conv_tol : float
            Converge threshold.  Default is 1e-7
        conv_tol_grad : float
            Converge threshold for CI gradients and orbital rotation gradients.
            Default is 1e-4
        max_stepsize : float
            The step size for orbital rotation.  Small step (0.005 - 0.05) is prefered.
            Default is 0.03.
        max_cycle_macro : int
            Max number of macro iterations.  Default is 50.
        max_cycle_micro : int
            Max number of micro iterations in each macro iteration.  Depending on
            systems, increasing this value might reduce the total macro
            iterations.  Generally, 2 - 5 steps should be enough.  Default is 3.
        ah_level_shift : float, for AH solver.
            Level shift for the Davidson diagonalization in AH solver.  Default is 1e-8.
        ah_conv_tol : float, for AH solver.
            converge threshold for AH solver.  Default is 1e-12.
        ah_max_cycle : float, for AH solver.
            Max number of iterations allowd in AH solver.  Default is 30.
        ah_lindep : float, for AH solver.
            Linear dependence threshold for AH solver.  Default is 1e-14.
        ah_start_tol : flat, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is 0.2.
        ah_start_cycle : int, for AH solver.
            In AH solver, the orbital rotation is started without completely solving the AH problem.
            This value is to control the start point. Default is 2.

            ``ah_conv_tol``, ``ah_max_cycle``, ``ah_lindep``, ``ah_start_tol`` and ``ah_start_cycle``
            can affect the accuracy and performance of CASSCF solver.  Lower
            ``ah_conv_tol`` and ``ah_lindep`` might improve the accuracy of CASSCF
            optimization, but decrease the performance.
            
            >>> from pyscf import gto, scf, mcscf
            >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
            >>> mf = scf.UHF(mol)
            >>> mf.scf()
            >>> mc = mcscf.CASSCF(mf, 6, 6)
            >>> mc.conv_tol = 1e-10
            >>> mc.ah_conv_tol = 1e-5
            >>> mc.kernel()[0]
            -109.044401898486001
            >>> mc.ah_conv_tol = 1e-10
            >>> mc.kernel()[0]
            -109.044401887945668

        chkfile : str
            Checkpoint file to save the intermediate orbitals during the CASSCF optimization.
            Default is the checkpoint file of mean field object.
        ci_response_space : int
            subspace size to solve the CI vector response.  Default is 3.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.

    Saved results

        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        e_cas : float
            CAS space FCI energy
        ci : ndarray
            CAS space FCI coefficients
        mo_coeff : ndarray
            Optimized CASSCF orbitals coefficients. When canonicalization is
            specified, the returned orbitals make the general Fock matrix
            (Fock operator on top of MCSCF 1-particle density matrix)
            diagonalized within each subspace (core, active, external).
            If natorb (natural orbitals in active space) is specified,
            the active segment of the mo_coeff is natural orbitls.
        mo_energy : ndarray
            Diagonal elements of general Fock matrix (in mo_coeff
            representation).

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    '''

# the max orbital rotation and CI increment, prefer small step size
    max_stepsize = getattr(__config__, 'mcscf_mc1step_CASSCF_max_stepsize', .02)
    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_micro', 1)
    conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_conv_tol', 5e-6)
    conv_tol_grad = getattr(__config__, 'mcscf_mc1step_CASSCF_conv_tol_grad', 5e-04)
    # for augmented hessian
    ah_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_lindep', 1e-14)
# * ah_start_tol and ah_start_cycle control the start point to use AH step.
#   In function rotate_orb_cc, the orbital rotation is carried out with the
#   approximate aug_hessian step after a few davidson updates of the AH eigen
#   problem.  Reducing ah_start_tol or increasing ah_start_cycle will delay
#   the start point of orbital rotation.
# * We can do early ah_start since it only affect the first few iterations.
#   The start tol will be reduced when approach the convergence point.
# * Be careful with the SYMMETRY BROKEN caused by ah_start_tol/ah_start_cycle.
#   ah_start_tol/ah_start_cycle actually approximates the hessian to reduce
#   the J/K evaluation required by AH.  When the system symmetry is higher
#   than the one given by mol.symmetry/mol.groupname,  symmetry broken might
#   occur due to this approximation,  e.g.  with the default ah_start_tol,
#   C2 (16o, 8e) under D2h symmetry might break the degeneracy between
#   pi_x, pi_y orbitals since pi_x, pi_y belong to different irreps.  It can
#   be fixed by increasing the accuracy of AH solver, e.g.
#               ah_start_tol = 1e-8;  ah_conv_tol = 1e-10
# * Classic AH can be simulated by setting eg
#               ah_start_tol = 1e-7
#               max_stepsize = 1.5
#               ah_grad_trust_region = 1e6
# ah_grad_trust_region allow gradients being increased in AH optimization
    ah_start_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_grad_trust_region', 3.0)

    internal_rotation = getattr(__config__, 'mcscf_mc1step_CASSCF_internal_rotation', False)
    ci_response_space = getattr(__config__, 'mcscf_mc1step_CASSCF_ci_response_space', 4)
    ci_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ci_grad_trust_region', 3)
    with_dep4 = getattr(__config__, 'mcscf_mc1step_CASSCF_with_dep4', False)
    chk_ci = getattr(__config__, 'mcscf_mc1step_CASSCF_chk_ci', False)
    kf_interval = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_trust_region', 3.0)

    ao2mo_level = getattr(__config__, 'mcscf_mc1step_CASSCF_ao2mo_level', 2)
    natorb = getattr(__config__, 'mcscf_mc1step_CASSCF_natorb', False)
    canonicalization = getattr(__config__, 'mcscf_mc1step_CASSCF_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'mcscf_mc1step_CASSCF_sorting_mo_energy', False)

    # Lan's options are below:
    #for selecting excited state 
    target_state = getattr(__config__, 'mcscf_mc1step_CASSCF_target_state', 0)
    target_state_spin  = getattr(__config__, 'mcscf_mc1step_CASSCF_target_state_spin', 0)
    is_select_state = getattr(__config__, 'mcscf_mc1step_CASSCF_is_select_state', True)
    is_target_dipole = getattr(__config__, 'mcscf_mc1step_CASSCF_is_target_dipole', False)
    sa_geom_opt = getattr(__config__, 'state-averaging geometry optimization w/o Z-vector', False)
    #for GMRES and line search
    is_use_gmres     = getattr(__config__, 'mcscf_mc1step_CASSCF_is_use_gmres', False)
    gmres_conv_tol   = getattr(__config__, 'mcscf_mc1step_CASSCF_gmres_conv_tol', 1e-06)
    gmres_max_cycle  = getattr(__config__, 'mcscf_mc1step_CASSCF_gmres_max_cycle', 100)
    gmres_hess_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_gmres_hess_shift', 0.)
    is_gmres_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_is_gmres_precond', True)
    is_gmres_precond = getattr(__config__, 'mcscf_mc1step_CASSCF_is_gmres_precond', True)
    is_gmres_conv_dynm = getattr(__config__, 'mcscf_mc1step_CASSCF_is_gmres_conv_dynm', False)
    is_line_search   = getattr(__config__, 'mcscf_mc1step_CASSCF_is_line_search', False)
    is_only_ddm      = getattr(__config__, 'mcscf_mc1step_CASSCF_is_only_ddm', False)
    is_only_W      = getattr(__config__, 'mcscf_mc1step_CASSCF_is_only_W', False)
    is_only_E      = getattr(__config__, 'mcscf_mc1step_CASSCF_is_only_E', False)
    is_only_dipole      = getattr(__config__, 'mcscf_mc1step_CASSCF_is_only_dipole', False)
    is_ddm_and_dipole = getattr(__config__, 'mcscf_mc1step_CASSCF_is_ddm_and_dipole', False)
    #for sCI
    is_use_sCI     = getattr(__config__, 'mcscf_mc1step_CASSCF_is_use_sCI', False)
    is_save_sCIout = getattr(__config__, 'mcscf_mc1step_CASSCF_is_save_sCIout', False)


    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        casci.CASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen

        self.callback = None
        self.chkfile = self._scf.chkfile

        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_mc1step_CASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_mc1step_CASSCF_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

        keys = set(('max_stepsize', 'max_cycle_macro', 'max_cycle_micro',
                    'conv_tol', 'conv_tol_grad', 'ah_level_shift',
                    'ah_conv_tol', 'ah_max_cycle', 'ah_lindep',
                    'ah_start_tol', 'ah_start_cycle', 'ah_grad_trust_region',
                    'internal_rotation', 'ci_response_space',
                    'ci_grad_trust_region', 'with_dep4', 'chk_ci',
                    'kf_interval', 'kf_trust_region', 'fcisolver_max_cycle',
                    'fcisolver_conv_tol', 'natorb', 'canonicalization',
                    'sorting_mo_energy', 'is_use_gmres', 'gmres_conv_tol',
                    'gmres_max_cycle', 'gmres_hess_shift', 'is_gmres_trust_region', 
                    'is_gmres_precond', 'is_gmres_conv_dynm', 
                    'is_line_search', 'is_only_ddm', 'is_only_W', 'is_only_E', 'is_only_dipole', 'is_ddm_and_dipole',
                    'target_state', 'is_select_state', 'is_target_dipole',  'sa_geom_opt',
                    'target_state_spin', 'is_use_sCI', 'is_save_sCIout'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        nvir = self.mo_coeff.shape[1] - self.ncore - self.ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], self.ncas, self.ncore, nvir)
        assert(self.ncas > 0)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle_macro = %d', self.max_cycle_macro)
        log.info('max_cycle_micro = %d', self.max_cycle_micro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('ci_response_space = %d', self.ci_response_space)
        log.info('ci_grad_trust_region = %d', self.ci_grad_trust_region)
        log.info('kf_trust_region = %g', self.kf_trust_region)
        log.info('kf_interval = %d', self.kf_interval)

        if not self.is_use_gmres:
            log.info('orbital rotation max_stepsize = %g', self.max_stepsize)
            log.info('augmented hessian ah_max_cycle = %d', self.ah_max_cycle)
            log.info('augmented hessian ah_conv_tol = %g', self.ah_conv_tol)
            log.info('augmented hessian ah_linear dependence = %g', self.ah_lindep)
            log.info('augmented hessian ah_level shift = %d', self.ah_level_shift)
            log.info('augmented hessian ah_start_tol = %g', self.ah_start_tol)
            log.info('augmented hessian ah_start_cycle = %d', self.ah_start_cycle)
            log.info('augmented hessian ah_grad_trust_region = %g', self.ah_grad_trust_region)
        else:
            log.info('is_select_state = %s', self.is_select_state)            
            if self.is_select_state:
                log.info('is_target_dipole = %s', self.is_select_state)
                log.info('is_only_ddm = %s', self.is_only_ddm)
                log.info('is_only_W = %s', self.is_only_W)
                log.info('is_only_E = %s', self.is_only_E)
                log.info('is_only_dipole = %s', self.is_only_dipole)
                log.info('is_ddm_and_dipole = %s', self.is_ddm_and_dipole)
            log.info('gmres_max_cycle = %d', self.gmres_max_cycle)
            if not self.is_gmres_conv_dynm:
                log.info('gmres_conv_tol = %g', self.gmres_conv_tol)
            log.info('gmres_hess_shift = %g', self.gmres_hess_shift)
            log.info('is_gmres_precond = %s', self.is_gmres_precond)
            log.info('is_gmres_trust_region = %s', self.is_gmres_trust_region)
            log.info('is_gmres_conv_dynm = %s', self.is_gmres_conv_dynm)
            log.info('is_line_search = %s', self.is_line_search)
        log.info('with_dep4 %d', self.with_dep4)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('ao2mo_level = %d', self.ao2mo_level)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(self.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            log.warn('''Solvent model %s was found in SCF object.
It is not applied to the CASSCF object. The CASSCF result is not affected by the SCF solvent model.
To enable the solvent model for CASSCF, a decoration to CASSCF object as below needs be called
        from pyscf import solvent
        mc = mcscf.CASSCF(...)
        mc = solvent.ddCOSMO(mc)
''',
                     self._scf.with_solvent.__class__)
        return self

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else: # overwrite self.mo_coeff because it is needed in many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        logger.note(self, 'CASSCF energy = %.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from pyscf.mcscf import mc2step
        return self.kernel(mo_coeff, ci0, callback, mc2step.kernel)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)

        e_tot, e_cas, fcivec = casci.kernel(fcasci, mo_coeff, ci0, log)
        
        #if not isinstance(e_cas, (float, numpy.number)):
        #    raise RuntimeError('Multiple roots are detected in fcisolver.  '
        #                       'CASSCF does not know which state to optimize.\n'
        #                       'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')
        
        if isinstance(e_tot, (float, numpy.number)):
            if envs is not None and log.verbose >= logger.INFO:
                log.debug('CAS space CI energy = %.15g', e_cas)

                if hasattr(self.fcisolver,'spin_square'):
                    ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
                else:
                    ss = None
    
                if 'imicro' in envs:  # Within CASSCF iteration
                    if ss is None:
                        log.info('macro iter %d (%d JK  %d micro), '
                                 'CASSCF E = %.15g  dE = %.8g',
                                 envs['imacro'], envs['njk'], envs['imicro'],
                                 e_tot, e_tot-envs['elast'])
                    else:
                        log.info('macro iter %d (%d JK  %d micro), '
                                 'CASSCF E = %.15g  dE = %.8g  S^2 = %.7f',
                                 envs['imacro'], envs['njk'], envs['imicro'],
                                 e_tot, e_tot-envs['elast'], ss[0])
                    if 'norm_gci' in envs:
                        log.info('               |grad[o]|=%5.3g  '
                                 '|grad[c]|= %s  |ddm|=%5.3g',
                                 envs['norm_gorb0'],
                                 envs['norm_gci'], envs['norm_ddm'])
                    else:
                        log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                                 envs['norm_gorb0'], envs['norm_ddm'])
                else:  # Initialization step
                    if ss is None:
                        log.info('CASCI E = %.15g', e_tot)
                    else:
                        log.info('CASCI E = %.15g  S^2 = %.7f', e_tot, ss[0])
        else:
            nroots = len(e_tot)
            if envs is not None and log.verbose >= logger.INFO:
                #log.debug('CAS space CI energy = %.15g', e_cas)
                
                if 'imicro' in envs:  # Within CASSCF iteration
                    #if self.target_state == 0: # ground state
                    #    #if (numpy.ndim(e_cas) > 0): 
                    #    # This is a workaround for external CI solver compatibility.
                    #    e_cas  = e_cas[0]
                    #    fcivec = fcivec[0]
                    #    e_tot  = e_tot[0]
    
                    #Lan: select target state
                    if(self.is_select_state):
                        if envs['imacro'] > 0:
                            log.info('CASCI information')
                            if(self.is_use_sCI):
                                civec, config = read_sCI_output(self.ncas)

                            for i in range(nroots):
                                if not self.is_use_sCI:
                                    ss = self.fcisolver.spin_square(fcivec[i], self.ncas, self.nelecas)[0]
                                else:
                                    ss = extract_spin_state_sCI(i, civec, config)
                                log.info('CASCI state %d  E = %.15g S^2 = %.7f',
                                         i, e_tot[i], ss)
                            
                            # target the desired state
                            print()
                            log.info('Selecting the CI vector ...')
                            
                            self.target_state, dipole = select_target_state(self, mo_coeff, fcivec, e_tot, envs, self.target_state, nroots, eris)

                            log.info('Targeted root %d, dipole: %8.5f, %8.5f, %8.5f', self.target_state, *dipole)
                            if not self.is_use_sCI:
                                nat_orbs = self.cas_natorb(mo_coeff, fcivec[self.target_state])
                    #else:
                    #    target_state = self.target_state
                                
                    e_tot = e_tot[self.target_state]
                    e_cas = e_cas[self.target_state]
                    fcivec = fcivec[self.target_state]
    
                    if getattr(self.fcisolver, 'spin_square', None):
                        ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
                    else:
                        ss = None
    
                            
                    if ss is None:
                        log.info('macro iter %d (%d JK  %d micro), '
                                 'CASSCF E = %.15g  dE = %.8g',
                                 envs['imacro'], envs['njk'], envs['imicro'],
                                 e_tot, e_tot-envs['elast'])
                    else:
                        log.info('macro iter %d (%d JK  %d micro), '
                                 'CASSCF E = %.15g  dE = %.8g  S^2 = %.7f',
                                 envs['imacro'], envs['njk'], envs['imicro'],
                                 e_tot, e_tot-envs['elast'], ss[0])
                    if 'norm_gci' in envs:
                        log.info('               |grad[o]|=%5.3g  '
                                 '|grad[c]|= %s  |ddm|=%5.3g',
                                 envs['norm_gorb0'],
                                 envs['norm_gci'], envs['norm_ddm'])
                    else:
                        log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                                 envs['norm_gorb0'], envs['norm_ddm'])
                else:  # Initialization step
                    geom_cycle = 0
                    if os.environ.get("cycle") is not None:
                        geom_cycle = int(os.environ["cycle"])+1
                        #print "geomopt cycle = ", geom_cycle
                    print()
                    log.info('CASCI information')
                    #reading sCI information from output.dat
                    civec = [] #dump variable for sCI
                    config = [] # dump variable for sCI
                    if(self.is_use_sCI):
                        civec, config = read_sCI_output(self.ncas)

                    for i in range(nroots):
                        if not self.is_use_sCI:
                            ss = self.fcisolver.spin_square(fcivec[i], self.ncas, self.nelecas)[0]
                        else:
                            ss = extract_spin_state_sCI(i, civec, config)
                        log.info('CASCI state %d  E = %.15g S^2 = %.7f',
                                 i, e_tot[i], ss)
                    print()
                    for i in range(nroots):
                        print("Root ", i)
                        rdm1_AO = addons.make_rdm1(self, mo_coeff, fcivec[i])
                        #print "rdm1_AO"
                        #print rdm1_AO
                        d = scf.hf.dip_moment(mol=self.mol, dm=rdm1_AO)
                        print()
                        if not self.is_use_sCI:
                            nat_orbs = self.cas_natorb(mo_coeff, fcivec[i])
                        print()
                        #print "CI vector"
                        #print fcivec[i]
                    #exit()
                    #select target state for new geometry opt cycle
                    if geom_cycle > 1 and self.is_select_state:
                        print("Selecting target state for new geometry opt cycle")
                        nmo = mo_coeff.shape[1]
                        rdm1_prevgeom_AO = envs['rdm1_pregeom_AO']
                        rdm1_prevgeom_MO = rota_rdms(mo_coeff, rdm1_prevgeom_AO) #rotate rdm1 to new MOs
                        
                        d_prevgeom = scf.hf.dip_moment(mol=self.mol, dm=rdm1_prevgeom_AO)
                        
                        k = 0
                        self.target_state = 0
                        for s in range(nroots):
                            if not self.is_use_sCI:
                                ss = self.fcisolver.spin_square(fcivec[s], self.ncas, self.nelecas)[0]
                            else:
                                ss = extract_spin_state_sCI(s, civec, config)

                            casdm1, casdm2 = self.fcisolver.make_rdm12(fcivec[s], self.ncas, self.nelecas)
                            #casdm1 = self.fcisolver.make_rdm1(fcivec[s], self.ncas, self.nelecas)
                            #casdm2 = self.fcisolver.make_rdm2(fcivec[s], self.ncas, self.nelecas)
                            rdm1_MO, rdm2_MO = addons._make_rdm12_on_mo(casdm1, casdm2, self.ncore, self.ncas, nmo)
                            dNorm = ddmNorm = 1./self.ncas * numpy.linalg.norm(rdm1_MO - rdm1_prevgeom_MO)
                            
                            if self.is_target_dipole:
                                rdm1_AO = addons.make_rdm12(self, self.mo_coeff, fcivec[s]) 
                                d = scf.hf.dip_moment(mol=self.mol, dm=rdm1_AO)
                                ddipoleNorm = numpy.linalg.norm(d - d_prevgeom)
                                dNorm += ddipoleNorm
                                
                            print("state ", s, "ddmNorm = ", abs(ddmNorm), "dNorm = ", abs(dNorm))
                            #print rdm1_MO
                            #print
                            #ddmNorm_min = 0.
                            #if s == 0:
                            ss_target = 0.5*self.target_state_spin*(0.5*self.target_state_spin+1)
                            if abs(ss - ss_target) < 1e-2:
                                if k == 0:
                                    dNorm_min = abs(dNorm)
                                    k += 1
                                else:
                                    if abs(dNorm) < dNorm_min:
                                        self.target_state = s
                                        dNorm_min = abs(dNorm)
                                        print("dNorm ", dNorm, dNorm_min)
                            else:
                                print("there is no desirable spin state")
                                exit()
                                    
                    print()
                    #self.target_state = 2
                    log.info('Initial targeted root %d', self.target_state)
                    print()
                    e_tot = e_tot[self.target_state]
                    e_cas = e_cas[self.target_state]
                    fcivec = fcivec[self.target_state]
                    #if getattr(self.fcisolver, 'spin_square', None):
                    #    ss = self.fcisolver.spin_square(fcivec, self .ncas, self.nelecas)
                    #else:
                    #    ss = None
                    #if ss is None:
                    #    log.info('CASCI E = %.15g', e_tot)
                    #else:
                    #    log.info('CASCI E = %.15g  S^2 = %.7f', e_tot, ss[0])
                    #    
        return e_tot, e_cas, fcivec

    as_scanner = as_scanner


    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        if self.internal_rotation:
            mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask

    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo))
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return numpy.dot(u0, expmat(dr))

    gen_g_hop = gen_g_hop
    rotate_orb_cc = rotate_orb_cc
    #genMinRes = genMinRes
    rotate_orb_gmres = rotate_orb_gmres

    def update_ao2mo(self, mo):
        raise DeprecationWarning('update_ao2mo was obseleted since pyscf v1.0.  '
                                 'Use .ao2mo method instead')

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
#        nmo = mo.shape[1]
#        ncore = self.ncore
#        ncas = self.ncas
#        nocc = ncore + ncas
#        eri = pyscf.ao2mo.incore.full(self._scf._eri, mo)
#        eri = pyscf.ao2mo.restore(1, eri, nmo)
#        eris = lambda:None
#        eris.j_cp = numpy.einsum('iipp->ip', eri[:ncore,:ncore,:,:])
#        eris.k_cp = numpy.einsum('ippi->ip', eri[:ncore,:,:,:ncore])
#        eris.vhf_c =(numpy.einsum('iipq->pq', eri[:ncore,:ncore,:,:])*2
#                    -numpy.einsum('ipqi->pq', eri[:ncore,:,:,:ncore]))
#        eris.ppaa = numpy.asarray(eri[:,:,ncore:nocc,ncore:nocc], order='C')
#        eris.papa = numpy.asarray(eri[:,ncore:nocc,:,ncore:nocc], order='C')
#        return eris

        return mc_ao2mo._ERIS(self, mo_coeff, method='incore',
                              level=self.ao2mo_level)

    # Don't remove the two functions.  They are used in df.approx_hessian code
    def get_h2eff(self, mo_coeff=None):
        '''Computing active space two-particle Hamiltonian.

        Note It is different to get_h2cas when df.approx_hessian is applied,
        in which get_h2eff function returns the DF integrals while get_h2cas
        returns the regular 2-electron integrals.
        '''
        return self.get_h2cas(mo_coeff)
    def get_h2cas(self, mo_coeff=None):
        '''Computing active space two-particle Hamiltonian.

        Note It is different to get_h2eff when df.approx_hessian is applied,
        in which get_h2eff function returns the DF integrals while get_h2cas
        returns the regular 2-electron integrals.
        '''
        return casci.CASCI.ao2mo(self, mo_coeff)

    def update_jk_in_ah(self, mo, r, casdm1, eris):
# J3 = eri_popc * pc + eri_cppo * cp
# K3 = eri_ppco * pc + eri_pcpo * cp
# J4 = eri_pcpa * pa + eri_appc * ap
# K4 = eri_ppac * pa + eri_papc * ap
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas

        dm3 = reduce(numpy.dot, (mo[:,:ncore], r[:ncore,ncore:], mo[:,ncore:].T))
        dm3 = dm3 + dm3.T
        dm4 = reduce(numpy.dot, (mo[:,ncore:nocc], casdm1, r[ncore:nocc], mo.T))
        dm4 = dm4 + dm4.T
        vj, vk = self.get_jk(self.mol, (dm3,dm3*2+dm4))
        va = reduce(numpy.dot, (casdm1, mo[:,ncore:nocc].T, vj[0]*2-vk[0], mo))
        vc = reduce(numpy.dot, (mo[:,:ncore].T, vj[1]*2-vk[1], mo[:,ncore:]))
        return va, vc

# hessian_co exactly expands up to first order of H
# update_casdm exand to approx 2nd order of H
    def update_casdm(self, mo, u, fcivec, e_cas, eris, envs={}):
        nmo = mo.shape[1]
        rmat = u - numpy.eye(nmo)

        #g = hessian_co(self, mo, rmat, fcivec, e_cas, eris)
        ### hessian_co part start ###
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nocc = ncore + ncas
        uc = u[:,:ncore]
        ua = u[:,ncore:nocc].copy()
        ra = rmat[:,ncore:nocc].copy()
        h1e_mo = reduce(numpy.dot, (mo.T, self.get_hcore(), mo))
        ddm = numpy.dot(uc, uc.T) * 2
        ddm[numpy.diag_indices(ncore)] -= 2
        if self.with_dep4:
            mo1 = numpy.dot(mo, u)
            mo1_cas = mo1[:,ncore:nocc]
            dm_core  = numpy.dot(mo1[:,:ncore], mo1[:,:ncore].T) * 2
            vj, vk = self._scf.get_jk(self.mol, dm_core)
            h1 =(reduce(numpy.dot, (ua.T, h1e_mo, ua)) +
                 reduce(numpy.dot, (mo1_cas.T, vj-vk*.5, mo1_cas)))
            eris._paaa = self._exact_paaa(mo, u)
            h2 = eris._paaa[ncore:nocc]
            vj = vk = None
        else:
            p1aa = numpy.empty((nmo,ncas,ncas**2))
            paa1 = numpy.empty((nmo,ncas**2,ncas))
            jk = reduce(numpy.dot, (ua.T, eris.vhf_c, ua))
            for i in range(nmo):
                jbuf = eris.ppaa[i]
                kbuf = eris.papa[i]
                jk +=(numpy.einsum('quv,q->uv', jbuf, ddm[i])
                    - numpy.einsum('uqv,q->uv', kbuf, ddm[i]) * .5)
                p1aa[i] = lib.dot(ua.T, jbuf.reshape(nmo,-1))
                paa1[i] = lib.dot(kbuf.transpose(0,2,1).reshape(-1,nmo), ra)
            h1 = reduce(numpy.dot, (ua.T, h1e_mo, ua)) + jk
            aa11 = lib.dot(ua.T, p1aa.reshape(nmo,-1)).reshape((ncas,)*4)
            aaaa = eris.ppaa[ncore:nocc,ncore:nocc,:,:]
            aa11 = aa11 + aa11.transpose(2,3,0,1) - aaaa

            a11a = numpy.dot(ra.T, paa1.reshape(nmo,-1)).reshape((ncas,)*4)
            a11a = a11a + a11a.transpose(1,0,2,3)
            a11a = a11a + a11a.transpose(0,1,3,2)
            h2 = aa11 + a11a
            jbuf = kbuf = p1aa = paa1 = aaaa = aa11 = a11a = None

        # pure core response
        # response of (1/2 dm * vhf * dm) ~ ddm*vhf
# Should I consider core response as a part of CI gradients?
        ecore =(numpy.einsum('pq,pq->', h1e_mo, ddm)
              + numpy.einsum('pq,pq->', eris.vhf_c, ddm))
        ### hessian_co part end ###

        ci1, g = self.solve_approx_ci(h1, h2, fcivec, ecore, e_cas, envs)
        if g is not None:  # So state average CI, DMRG etc will not be applied
            ovlp = numpy.dot(fcivec.ravel(), ci1.ravel())
            #print "ci ovlp ", ovlp
            norm_g = numpy.linalg.norm(g)
            if 1-abs(ovlp) > norm_g * self.ci_grad_trust_region:
                logger.debug(self, '<ci1|ci0>=%5.3g |g|=%5.3g, ci1 out of trust region',
                             ovlp, norm_g)
                ci1 = fcivec.ravel() + g
                ci1 *= 1/numpy.linalg.norm(ci1)
        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, ncas, nelecas)

        return casdm1, casdm2, g, ci1

    def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs):
        ''' Solve CI eigenvalue/response problem approximately
        '''
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore
        nocc = ncore + ncas
        if 'norm_gorb' in envs:
            tol = max(self.conv_tol, envs['norm_gorb']**2*.1)
        else:
            tol = None
        if getattr(self.fcisolver, 'approx_kernel', None):
            fn = self.fcisolver.approx_kernel
            e, ci1 = fn(h1, h2, ncas, nelecas, ecore=ecore, ci0=ci0,
                        tol=tol, max_memory=self.max_memory)
            return ci1, None
        elif not (getattr(self.fcisolver, 'contract_2e', None) and
                  getattr(self.fcisolver, 'absorb_h1e', None)):
            fn = self.fcisolver.kernel
            e, ci1 = fn(h1, h2, ncas, nelecas, ecore=ecore, ci0=ci0,
                        tol=tol, max_memory=self.max_memory,
                        max_cycle=self.ci_response_space)
            return ci1, None

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)

        # Be careful with the symmetry adapted contract_2e function. When the
        # symmetry adapted FCI solver is used, the symmetry of ci0 may be
        # different to fcisolver.wfnsym. This function may output 0.
        if getattr(self.fcisolver, 'guess_wfnsym', None):
            wfnsym = self.fcisolver.guess_wfnsym(self.ncas, self.nelecas, ci0)
        else:
            wfnsym = None
        def contract_2e(c):
            if wfnsym is None:
                hc = self.fcisolver.contract_2e(h2eff, c, ncas, nelecas)
            else:
                with lib.temporary_env(self.fcisolver, wfnsym=wfnsym):
                    hc = self.fcisolver.contract_2e(h2eff, c, ncas, nelecas)
            return hc.ravel()

        hc = contract_2e(ci0)
        g = hc - (e_cas-ecore) * ci0.ravel()

        if self.ci_response_space > 7:
            logger.debug(self, 'CI step by full response')
            # full response
            max_memory = max(400, self.max_memory-lib.current_memory()[0])
            e, ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ecore=ecore,
                                           ci0=ci0, tol=tol, max_memory=max_memory)
        else:
            nd = min(max(self.ci_response_space, 2), ci0.size)
            logger.debug(self, 'CI step by %dD subspace response', nd)
            xs = [ci0.ravel()]
            ax = [hc]
            heff = numpy.empty((nd,nd))
            seff = numpy.empty((nd,nd))
            heff[0,0] = numpy.dot(xs[0], ax[0])
            seff[0,0] = 1
            for i in range(1, nd):
                xs.append(ax[i-1] - xs[i-1] * e_cas)
                ax.append(contract_2e(xs[i]))
                for j in range(i+1):
                    heff[i,j] = heff[j,i] = numpy.dot(xs[i], ax[j])
                    seff[i,j] = seff[j,i] = numpy.dot(xs[i], xs[j])
            e, v = lib.safe_eigh(heff, seff)[:2]
            ci1 = xs[0] * v[0,0]
            for i in range(1,nd):
                ci1 += xs[i] * v[i,0]
        return ci1[self.target_state], g

    def get_jk(self, mol, dm, hermi=1):
        return self._scf.get_jk(mol, dm, hermi=1)

    def get_grad(self, mo_coeff=None, casdm1_casdm2=None, eris=None):
        '''Orbital gradients'''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if eris is None: eris = self.ao2mo(mo_coeff)
        if casdm1_casdm2 is None:
            e_tot, e_cas, civec = self.casci(mo_coeff, self.ci, eris)
            casdm1, casdm2 = self.fcisolver.make_rdm12(civec, self.ncas, self.nelecas)
        else:
            casdm1, casdm2 = casdm1_casdm2
        return self.gen_g_hop(mo_coeff, 1, casdm1, casdm2, eris)[0]

    def _exact_paaa(self, mo, u, out=None):
        nmo = mo.shape[1]
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        mo1 = numpy.dot(mo, u)
        mo1_cas = mo1[:,ncore:nocc]
        mos = (mo1_cas, mo1_cas, mo1, mo1_cas)
        if self._scf._eri is None:
            aapa = ao2mo.general(self.mol, mos)
        else:
            aapa = ao2mo.general(self._scf._eri, mos)
        paaa = numpy.empty((nmo*ncas,ncas*ncas))
        buf = numpy.empty((ncas,ncas,nmo*ncas))
        for ij, (i, j) in enumerate(zip(*numpy.tril_indices(ncas))):
            buf[i,j] = buf[j,i] = aapa[ij]
        paaa = lib.transpose(buf.reshape(ncas*ncas,-1), out=out)
        return paaa.reshape(nmo,ncas,ncas,ncas)

    def dump_chk(self, envs):
        if not self.chkfile:
            return self

        if getattr(self.fcisolver, 'nevpt_intermediate', None):
            civec = None
        elif self.chk_ci:
            civec = envs['fcivec']
        else:
            civec = None
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        if 'mo' in envs:
            mo_coeff = envs['mo']
        else:
            mo_coeff = envs['mo_coeff']
        mo_occ = numpy.zeros(mo_coeff.shape[1])
        mo_occ[:ncore] = 2
        if self.natorb:
            occ = self._eig(-envs['casdm1'], ncore, nocc)[0]
            mo_occ[ncore:nocc] = -occ
        else:
            mo_occ[ncore:nocc] = envs['casdm1'].diagonal()
# Note: mo_energy in active space =/= F_{ii}  (F is general Fock)
        if 'mo_energy' in envs:
            mo_energy = envs['mo_energy']
        else:
            mo_energy = 'None'
        chkfile.dump_mcscf(self, self.chkfile, 'mcscf', envs['e_tot'],
                           mo_coeff, self.ncore, self.ncas, mo_occ,
                           mo_energy, envs['e_cas'], civec, envs['casdm1'],
                           overwrite_mol=False)
        return self

    def update_from_chk(self, chkfile=None):
        if chkfile is None: chkfile = self.chkfile
        self.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
        return self
    update = update_from_chk

    def rotate_mo(self, mo, u, log=None):
        '''Rotate orbitals with the given unitary matrix'''
        mo = numpy.dot(mo, u)
        if log is not None and log.verbose >= logger.DEBUG:
            ncore = self.ncore
            ncas = self.ncas
            nocc = ncore + ncas
            s = reduce(numpy.dot, (mo[:,ncore:nocc].T, self._scf.get_ovlp(),
                                   self.mo_coeff[:,ncore:nocc]))
            log.debug('Active space overlap to initial guess, SVD = %s',
                      numpy.linalg.svd(s)[1])
            log.debug('Active space overlap to last step, SVD = %s',
                      numpy.linalg.svd(u[ncore:nocc,ncore:nocc])[1])
        return mo

    def micro_cycle_scheduler(self, envs):
        if not WITH_MICRO_SCHEDULER:
            return self.max_cycle_micro

        log_norm_ddm = numpy.log(envs['norm_ddm'])
        return max(self.max_cycle_micro, int(self.max_cycle_micro-1-log_norm_ddm))

    def max_stepsize_scheduler(self, envs):
        if not WITH_STEPSIZE_SCHEDULER:
            return self.max_stepsize

        if self._max_stepsize is None:
            self._max_stepsize = self.max_stepsize
        if envs['de'] > -self.conv_tol:  # Avoid total energy increasing
            self._max_stepsize *= .3
            logger.debug(self, 'set max_stepsize to %g', self._max_stepsize)
        else:
            self._max_stepsize = (self.max_stepsize*self._max_stepsize)**.5
        return self._max_stepsize

    def ah_scheduler(self, envs):
        pass

    @property
    def max_orb_stepsize(self):  # pragma: no cover
        return self.max_stepsize
    @max_orb_stepsize.setter
    def max_orb_stepsize(self, x):  # pragma: no cover
        sys.stderr.write('WARN: Attribute "max_orb_stepsize" was replaced by "max_stepsize"\n')
        self.max_stepsize = x
    @property
    def ci_update_dep(self):  # pragma: no cover
        return self.with_dep4
    @ci_update_dep.setter
    def ci_update_dep(self, x):  # pragma: no cover
        sys.stderr.write('WARN: Attribute .ci_update_dep was replaced by .with_dep4 since PySCF v1.1.\n')
        self.with_dep4 = x == 4
    grad_update_dep = ci_update_dep

    @property
    def max_cycle(self):
        return self.max_cycle_macro
    @max_cycle.setter
    def max_cycle(self, x):
        self.max_cycle_macro = x

    def approx_hessian(self, auxbasis=None, with_df=None):
        from pyscf.mcscf import df
        return df.approx_hessian(self, auxbasis, with_df)

    def nuc_grad_method(self):
        from pyscf.grad import casscf
        return casscf.Gradients(self)

    def newton(self):
        from pyscf.mcscf import newton_casscf
        mc1 = newton_casscf.CASSCF(self._scf, self.ncas, self.nelecas)
        mc1.__dict__.update(self.__dict__)
        mc1.max_cycle_micro = 10
        return mc1


# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    mo_core = mo[:,:ncore]
    mo_cas = mo[:,ncore:nocc]
    core_dm = numpy.dot(mo_core, mo_core.T) * 2
    hcore = casscf.get_hcore()
    energy_core = casscf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore,:ncore].trace()
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore, mo_cas))
    h1eff += eris.vhf_c[ncore:nocc,ncore:nocc]
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    ncore = casscf.ncore
    nocc = ncore + casscf.ncas
    eri_cas = eris.ppaa[ncore:nocc,ncore:nocc,:,:].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc

def expmat(a):
    return scipy.linalg.expm(a)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import fci
    from pyscf.mcscf import addons

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    emc = kernel(CASSCF(m, 4, 4), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -3.22013929407)

    mc = CASSCF(m, 4, (3,1))
    mc.verbose = 4
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    emc = kernel(mc, m.mo_coeff, verbose=4)[1]
    print(emc - -15.950852049859-mol.energy_nuc())

    mol.atom = [
        ['H', ( 5.,-1.    , 1.   )],
        ['H', ( 0.,-5.    ,-2.   )],
        ['H', ( 4.,-0.5   ,-3.   )],
        ['H', ( 0.,-4.5   ,-1.   )],
        ['H', ( 3.,-0.5   ,-0.   )],
        ['H', ( 0.,-3.    ,-1.   )],
        ['H', ( 2.,-2.5   , 0.   )],
        ['H', ( 1., 1.    , 3.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    emc = kernel(CASSCF(m, 4, 4), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -3.62638367550087, emc - -3.6268060528596635)

    mc = CASSCF(m, 4, (3,1))
    mc.verbose = 4
    mc.natorb = 1
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    emc = kernel(mc, m.mo_coeff, verbose=4)[1]
    print(emc - -3.62638367550087)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = CASSCF(m, 6, 4)
    mc.fcisolver = fci.solver(mol)
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = CASSCF(m, 6, (3,1))
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    #mc.fcisolver = fci.direct_spin1
    mc.fcisolver = fci.solver(mol, False)
    mc.verbose = 4
    emc = mc.mc1step(mo)[0]
    #mc.analyze()
    print(emc - -75.7155632535814)

    mc.internal_rotation = True
    emc = mc.mc1step(mo)[0]
    print(emc - -75.7155632535814)
