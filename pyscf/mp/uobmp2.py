#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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


'''
OB-MP2
'''

import time
from functools import reduce
import copy
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.mp import obmp2
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
           verbose=logger.NOTE):
    if mp.mo_energy is None or mp.mo_coeff is None:
        #if mo_energy is None or mo_coeff is None:
        #    raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
        #                       'You may need to call mf.kernel() to generate them.')
        #moidx = mp.get_frozen_mask()
        #mo_coeff = None
        #mo_energy = (mp.mo_energy[0][moidx[0]], mp.mo_energy[1][moidx[1]])
        mo_coeff_init  = mp._scf.mo_coeff
        mo_coeff       = mp._scf.mo_coeff
        mo_energy = mp._scf.mo_energy
    else:
        #print("we are here")
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        #assert(mp.frozen is 0 or mp.frozen is None)
        mo_coeff_init  = mp.mo_coeff
        mo_coeff  = mp.mo_coeff
        mo_energy = mp.mo_energy
    #print("before")
    #print(mo_coeff[0])

    if mp.mom:
        mo_coeff_init = mp.mom_reorder(mo_coeff_init)
        mo_coeff = mp.mom_reorder(mo_coeff)
        mo_energy = mp.mo_energy

    #mo_coeff = mo_coeff_init

    #if eris is None: eris = mp.ao2mo(mo_coeff)

    #print("Initial MOs")
    #print(mo_coeff)
    nuc = mp._scf.energy_nuc()
    ene_hf = mp._scf.energy_tot()

    #initializing w/ HF
    #mo_coeff  = mp._scf.mo_coeff
    #mo_energy = mp._scf.mo_energy
    mo_occ    = mp._scf.mo_occ

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    shift = mp.shift


    niter = mp.niter
    ene_old = 0.
    #thres = 1e-8
    conv = False 
    eri_ao = mp.mol.intor('int2e_sph')

    print("shift = ", mp.shift)
    print ("thresh = ", mp.thresh)
    print ("niter = ", mp.niter)

    for it in range(niter):
        #if it == 5:
        #    print("mo_coeff_init")
        #    print(mo_coeff_init[0])
        #    print("mo_coeff")
        #    print(mo_coeff[0])

        #if it > 3:
        #    mo_coeff[0] = 0.5*mo_coeff[0] + 0.5*mo_coeff_init[0]

        h2mo_aa = int_transform_ss(eri_ao, mo_coeff[0])
        h2mo_bb = int_transform_ss(eri_ao, mo_coeff[1])
        h2mo_ab = int_transform_os(eri_ao, mo_coeff[0], mo_coeff[1])
        h2mo_ba = int_transform_os(eri_ao, mo_coeff[1], mo_coeff[0])
        
        h1ao = mp._scf.get_hcore(mp.mol)
        h1mo_a = numpy.matmul(mo_coeff[0].T,numpy.matmul(h1ao,mo_coeff[0]))
        h1mo_b = numpy.matmul(mo_coeff[1].T,numpy.matmul(h1ao,mo_coeff[1]))
        
        #####################
        ### Hartree-Fock

        fock_hfa = h1mo_a
        fock_hfb = h1mo_b

        for p in range(nmoa):
            for q in range(nmoa):
                for k in range(nocca):
                    fock_hfa[p,q] += (h2mo_aa[p,q,k,k]-h2mo_aa[p,k,k,q]) 
                for k in range(noccb):
                    fock_hfa[p,q] += h2mo_ab[p,q,k,k]
        for p in range(nmob):
            for q in range(nmob):
                for k in range(noccb):
                    fock_hfb[p,q] += (h2mo_bb[p,q,k,k]-h2mo_bb[p,k,k,q])
                for k in range(nocca):
                    fock_hfb[p,q] += h2mo_ba[p,q,k,k]

        if it > 0:
            fock_a_old = fock_a
            fock_b_old = fock_b
        fock_a = fock_hfa
        fock_b = fock_hfb

        c0 = 0.
        e_corr = 0.
        ene_hf = 0.
        for i in range(nocca):
            ene_hf += fock_a[i,i]
            for j in range(nocca):
                c0 -= (h2mo_aa[i,i,j,j]-h2mo_aa[i,j,j,i])
            for j in range(noccb):
                c0 -= h2mo_ab[i,i,j,j]
        for i in range(noccb):
            ene_hf += fock_b[i,i]
            for j in range(noccb):
                c0 -= (h2mo_bb[i,i,j,j]-h2mo_bb[i,j,j,i])
            for j in range(nocca):
                c0 -= h2mo_ba[i,i,j,j]
        c0 *= 0.5
        ene_hf += c0
        
        tmp1_aa = numpy.zeros((nocca,nvira,nocca,nvira)) #, dtype=fock_hfa.dtype)
        tmp1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb)) #, dtype=fock_hfb.dtype)
        tmp1_ab = numpy.zeros((nocca,nvira,noccb,nvirb)) #, dtype=fock_hf.dtype)
        tmp1_ba = numpy.zeros((noccb,nvirb,nocca,nvira)) #, dtype=fock_hf.dtype)

        for i in range(nocca):
            for a in range(nvira):
                for j in range(nocca):
                    for b in range(nvira):
                        A = a+nocca
                        B = b+nocca
                        x = mo_energy[0][i] + mo_energy[0][j] - mo_energy[0][A] - mo_energy[0][B] - shift
                        tmp1_aa[i,a,j,b] = 1. * h2mo_aa[i,A,j,B]/x

        for i in range(noccb):
            for a in range(nvirb):
                for j in range(noccb):
                    for b in range(nvirb):
                        A = a+noccb
                        B = b+noccb
                        x = mo_energy[1][i] + mo_energy[1][j] - mo_energy[1][A] - mo_energy[1][B] - shift
                        tmp1_bb[i,a,j,b] = 1. * h2mo_bb[i,A,j,B]/x

        for i in range(nocca):
            for a in range(nvira):
                for j in range(noccb):
                    for b in range(nvirb):
                        A = a+nocca
                        B = b+noccb
                        x = mo_energy[0][i] + mo_energy[1][j] - mo_energy[0][A] - mo_energy[1][B] - shift
                        tmp1_ab[i,a,j,b] = 1. * h2mo_ab[i,A,j,B]/x

        for i in range(noccb):
            for a in range(nvirb):
                for j in range(nocca):
                    for b in range(nvira):
                        A = a+noccb
                        B = b+nocca
                        x = mo_energy[1][i] + mo_energy[0][j] - mo_energy[1][A] - mo_energy[0][B] - shift
                        tmp1_ba[i,a,j,b] = 1. * h2mo_ba[i,A,j,B]/x

        tmp1_bar_aa = numpy.zeros((nocca,nvira,nocca,nvira)) #, dtype=tmp1.dtype)
        tmp1_bar_bb = numpy.zeros((noccb,nvirb,noccb,nvirb)) #, dtype=tmp1.dtype)
        tmp1_bar_ab = numpy.zeros((nocca,nvira,noccb,nvirb)) #, dtype=tmp1.dtype)
        tmp1_bar_ba = numpy.zeros((noccb,nvirb,nocca,nvira)) #, dtype=tmp1.dtype)
        for i in range(nocca):
            for a in range(nvira):
                for j in range(nocca):
                    for b in range(nvira):
                        tmp1_bar_aa[i,a,j,b] = tmp1_aa[i,a,j,b] -  tmp1_aa[i,b,j,a]
        for i in range(noccb):
            for a in range(nvirb):
                for j in range(noccb):
                    for b in range(nvirb):
                        tmp1_bar_bb[i,a,j,b] = tmp1_bb[i,a,j,b] - tmp1_bb[i,b,j,a]
        for i in range(nocca):
            for a in range(nvira):
                for j in range(noccb):
                    for b in range(nvirb):
                        tmp1_bar_ab[i,a,j,b] = tmp1_ab[i,a,j,b]
        for i in range(noccb):
            for a in range(nvirb):
                for j in range(nocca):
                    for b in range(nvira):
                        tmp1_bar_ba[i,a,j,b] = tmp1_ba[i,a,j,b]
                        
        if mp.second_order:
            mp.ampf = 1.0

        tmp1_bar_aa *= mp.ampf
        tmp1_bar_bb *= mp.ampf
        tmp1_bar_ab *= mp.ampf
        tmp1_bar_ba *= mp.ampf

        #####################
        ### BCH 1st order  

        c1_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
        c1_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)

        for j in range(nocca):
            for b in range(nvira):
                B = b+nocca
                for i in range(nocca):
                    for a in range(nvira):
                        A = a+nocca
                        c0 -= 1. * h2mo_aa[i,A,j,B] * tmp1_bar_aa[i,a,j,b]
                        c1_a[j,B] += 2. * fock_hfa[i,A] * tmp1_bar_aa[i,a,j,b]
                        for p in range(nmoa):
                            c1_a[p,j] += 2. * h2mo_aa[i,A,p,B] *  tmp1_bar_aa[i,a,j,b]
                            c1_a[p,B] -= 2. * h2mo_aa[i,A,j,p] *  tmp1_bar_aa[i,a,j,b]
                for i in range(noccb):
                    for a in range(nvirb):
                        A = a+noccb
                        c0 -= 1. * h2mo_ba[i,A,j,B] * tmp1_bar_ba[i,a,j,b]
                        c1_a[j,B] += 2. * fock_hfb[i,A] * tmp1_bar_ba[i,a,j,b]
                        for p in range(nmoa):
                            c1_a[p,j] += 2. * h2mo_ba[i,A,p,B] *  tmp1_bar_ba[i,a,j,b]
                            c1_a[p,B] -= 2. * h2mo_ba[i,A,j,p] *  tmp1_bar_ba[i,a,j,b]
                            
        for j in range(noccb):
            for b in range(nvirb):
                B = b+noccb
                for i in range(noccb):
                    for a in range(nvirb):
                        A = a+noccb
                        c0 -= 1. * h2mo_bb[i,A,j,B] * tmp1_bar_bb[i,a,j,b]
                        c1_b[j,B] += 2. * fock_hfb[i,A] * tmp1_bar_bb[i,a,j,b]
                        for p in range(nmob):
                            c1_b[p,j] += 2. * h2mo_bb[i,A,p,B] *  tmp1_bar_bb[i,a,j,b]
                            c1_b[p,B] -= 2. * h2mo_bb[i,A,j,p] *  tmp1_bar_bb[i,a,j,b]
                for i in range(nocca):
                    for a in range(nvira):
                        A = a+nocca
                        c0 -= 1. * h2mo_ab[i,A,j,B] * tmp1_bar_ab[i,a,j,b]
                        c1_b[j,B] += 2. * fock_hfa[i,A] * tmp1_bar_ab[i,a,j,b]
                        for p in range(nmoa):
                            c1_b[p,j] += 2. * h2mo_ab[i,A,p,B] *  tmp1_bar_ab[i,a,j,b]
                            c1_b[p,B] -= 2. * h2mo_ab[i,A,j,p] *  tmp1_bar_ab[i,a,j,b]
        # symmetrize c1
        for p in range(nmoa):
            for q in range(nmoa):
                fock_a[p,q] += 0.5 * (c1_a[p,q] + c1_a[q,p])
        for p in range(nmob):
            for q in range(nmob):
                fock_b[p,q] += 0.5 * (c1_b[p,q] + c1_b[q,p])

        #print("fock_a")
        #print(fock_a)
        #print("fock_b")
        #print(fock_b)
        #print("c0 ", c0)
        #
        #####################
        ### BCH 2nd order  

        if mp.second_order:
        
            c1_a = numpy.zeros((nmoa,nmoa), dtype=fock_hfa.dtype)
            c1_b = numpy.zeros((nmob,nmob), dtype=fock_hfb.dtype)
            
            #[1]
            y1_a = numpy.zeros((nocca,nvira), dtype=fock_a.dtype)
            y1_b = numpy.zeros((noccb,nvirb), dtype=fock_b.dtype)
            for j in range(nocca):
                for b in range(nvira):
                    for i in range(nocca):
                        for a in range(nvira):
                            A = a+nocca
                            y1_a[j,b] += fock_hfa[i,A] * tmp1_bar_aa[i,a,j,b]
                    for i in range(noccb):
                        for a in range(nvirb):
                            A = a+noccb
                            y1_a[j,b] += fock_hfb[i,A] * tmp1_bar_ba[i,a,j,b]
            for i in range(nocca):
                for a in range(nvira):
                    for j in range(nocca):
                        for b in range(nvira):
                            A = a+nocca
                            c1_a[i,A] += 1. * y1_a[j,b] * tmp1_bar_aa[i,a,j,b]
            for i in range(noccb):
                for a in range(nvirb):
                    for j in range(nocca):
                        for b in range(nvira):
                            A = a+noccb
                            c1_b[i,A] += 1. * y1_a[j,b] * tmp1_bar_ba[i,a,j,b]
    
            for j in range(noccb):
                for b in range(nvirb):
                    for i in range(nocca):
                        for a in range(nvira):
                            A = a+nocca
                            y1_b[j,b] += fock_hfa[i,A] * tmp1_bar_ab[i,a,j,b]
                    for i in range(noccb):
                        for a in range(nvirb):
                            A = a+noccb
                            y1_b[j,b] += fock_hfb[i,A] * tmp1_bar_bb[i,a,j,b]
            for i in range(nocca):
                for a in range(nvira):
                    for j in range(noccb):
                        for b in range(nvirb):
                            A = a+nocca
                            c1_a[i,A] += 1. * y1_b[j,b] * tmp1_bar_ab[i,a,j,b]
            for i in range(noccb):
                for a in range(nvirb):
                    for j in range(noccb):
                        for b in range(nvirb):
                            A = a+noccb
                            c1_b[i,A] += 1. * y1_b[j,b] * tmp1_bar_bb[i,a,j,b]
    
            #print("c1_a")
            #print(c1_a)
            #print("c1_b")
            #print(c1_b)
    
            #[2]
            y1_aa = numpy.zeros((nocca,nvira,nocca,nvira))
            y1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
            y1_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
            y1_ba = numpy.zeros((noccb,nvirb,nocca,nvira))
            for j in range(nocca):
                for b in range(nvira):
                    for i in range(nocca):
                        for a in range(nvira):
                            for c in range(nvira): 
                                A = a+nocca
                                C = c+nocca
                                y1_aa[i,a,j,b] += fock_hfa[A,C] * tmp1_bar_aa[i,c,j,b]
                    for i in range(noccb):
                        for a in range(nvirb):
                            for c in range(nvirb): 
                                A = a+noccb
                                C = c+noccb
                                y1_ba[i,a,j,b] += fock_hfb[A,C] * tmp1_bar_ba[i,c,j,b]
            for j in range(nocca):
                for k in range(nocca):
                    for b in range(nvira):
                        for i in range(nocca):
                            for a in range(nvira):
                                c1_a[j,k] += 1. * tmp1_aa[i,a,j,b] * y1_aa[i,a,k,b]
                        for i in range(noccb):
                            for a in range(nvirb):
                                c1_a[j,k] += 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,k,b]
    
            for j in range(noccb):
                for b in range(nvirb):
                    for i in range(noccb):
                        for a in range(nvirb):
                            for c in range(nvirb): 
                                A = a+noccb
                                C = c+noccb
                                y1_bb[i,a,j,b] += fock_hfb[A,C] * tmp1_bar_bb[i,c,j,b]
                    for i in range(nocca):
                        for a in range(nvira):
                            for c in range(nvira): 
                                A = a+nocca
                                C = c+nocca
                                y1_ab[i,a,j,b] += fock_hfa[A,C] * tmp1_bar_ab[i,c,j,b]
            for j in range(noccb):
                for k in range(noccb):
                    for b in range(nvirb):
                        for i in range(noccb):
                            for a in range(nvirb):
                                c1_b[j,k] += 1. * tmp1_bb[i,a,j,b] * y1_bb[i,a,k,b]
                        for i in range(nocca):
                            for a in range(nvira):
                                c1_b[j,k] += 1. * tmp1_ab[i,a,j,b] * y1_ab[i,a,k,b]
    
            for i in range(nocca):
                for a in range(nvira):
                    for j in range(nocca):
                        for b in range(nvira):
                            c0 -= 1. * tmp1_aa[i,a,j,b] * y1_aa[i,a,j,b]
                    for j in range(noccb):
                        for b in range(nvirb):
                            c0 -= 1. * tmp1_ab[i,a,j,b] * y1_ab[i,a,j,b]
            for i in range(noccb):
                for a in range(nvirb):
                    for j in range(noccb):
                        for b in range(nvirb):
                            c0 -= 1. * tmp1_bb[i,a,j,b] * y1_bb[i,a,j,b]
                    for j in range(nocca):
                        for b in range(nvira):
                            c0 -= 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,j,b]
    
            #[3]
            y1_aa = numpy.zeros((nocca,nvira,nocca,nvira))
            y1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
            y1_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
            y1_ba = numpy.zeros((noccb,nvirb,nocca,nvira))
            for i in range(nocca):
                for a in range(nvira):
                    for c in range(nvira): 
                        A = a+nocca
                        C = c+nocca
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_aa[i,a,j,b] += fock_hfa[A,C] * tmp1_bar_aa[i,c,j,b]
                        for j in range(noccb):
                            for b in range(nvirb): 
                                y1_ab[i,a,j,b] += fock_hfa[A,C] * tmp1_bar_ab[i,c,j,b]
            for k in range(nocca):
                for i in range(nocca):
                    for a in range(nvira):
                        for j in range(nocca):
                            for b in range(nvira):
                                c1_a[i,k] += 1. * tmp1_aa[i,a,j,b] * y1_aa[k,a,j,b]
                        for j in range(noccb):
                            for b in range(nvirb):
                                c1_a[i,k] += 1. * tmp1_ab[i,a,j,b] * y1_ab[k,a,j,b]
    
            for i in range(noccb):
                for a in range(nvirb):
                    for c in range(nvirb): 
                        A = a+noccb
                        C = c+noccb
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_bb[i,a,j,b] += fock_hfb[A,C] * tmp1_bar_bb[i,c,j,b]
                        for j in range(nocca):
                            for b in range(nvira): 
                                y1_ba[i,a,j,b] += fock_hfb[A,C] * tmp1_bar_ba[i,c,j,b]
            for k in range(noccb):
                for i in range(noccb):
                    for a in range(nvirb):
                        for j in range(noccb):
                            for b in range(nvira):
                                c1_b[i,k] += 1. * tmp1_bb[i,a,j,b] * y1_bb[k,a,j,b]
                        for j in range(nocca):
                            for b in range(nvira):
                                if mp.break_sym:
                                    c1_b[i,k] += 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,k,b]
                                else:
                                    c1_b[i,k] += 1. * tmp1_ba[i,a,j,b] * y1_ba[k,a,j,b]
                                    
    
            #[4]
            y1_aa = numpy.zeros((nocca,nvira,nocca,nvira))
            y1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
            y1_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
            y1_ba = numpy.zeros((noccb,nvirb,nocca,nvira))
            for b in range(nvira):
                for j in range(nocca):
                    for i in range(nocca):
                        for a in range(nvira):
                            for k in range(nocca):
                                y1_aa[i,a,j,b] += fock_hfa[i,k] * tmp1_bar_aa[k,a,j,b]
                    for i in range(noccb):
                        for a in range(nvirb):
                            for k in range(noccb):
                                y1_ba[i,a,j,b] += fock_hfb[i,k] * tmp1_bar_ba[k,a,j,b]
            for k in range(nocca):
                for j in range(nocca):
                    for b in range(nvira):
                        for i in range(nocca):
                            for a in range(nvira):
                                c1_a[j,k] -= 1. * tmp1_aa[i,a,j,b] * y1_aa[i,a,k,b]
                        for i in range(noccb):
                            for a in range(nvirb):
                                c1_a[j,k] -= 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,k,b]
    
            for b in range(nvirb):
                for j in range(noccb):
                    for i in range(noccb):
                        for a in range(nvirb):
                            for k in range(noccb):
                                y1_bb[i,a,j,b] += fock_hfb[i,k] * tmp1_bar_bb[k,a,j,b]
                    for i in range(nocca):
                        for a in range(nvira):
                            for k in range(nocca):
                                y1_ab[i,a,j,b] += fock_hfa[i,k] * tmp1_bar_ab[k,a,j,b]
            for k in range(noccb):
                for j in range(noccb):
                    for b in range(nvirb):
                        for i in range(noccb):
                            for a in range(nvirb):
                                c1_b[j,k] -= 1. * tmp1_bb[i,a,j,b] * y1_bb[i,a,k,b]
                        for i in range(nocca):
                            for a in range(nvira):
                                c1_b[j,k] -= 1. * tmp1_ab[i,a,j,b] * y1_ab[i,a,k,b]
    
            for i in range(nocca):
                for a in range(nvira):
                    for j in range(nocca):
                        for b in range(nvira):
                            c0 += 1. * tmp1_aa[i,a,j,b] * y1_aa[i,a,j,b]
                    for j in range(noccb):
                        for b in range(nvirb):
                            c0 += 1. * tmp1_ab[i,a,j,b] * y1_ab[i,a,j,b]
            for i in range(noccb):
                for a in range(nvirb):
                    for j in range(noccb):
                        for b in range(nvirb):
                            c0 += 1. * tmp1_bb[i,a,j,b] * y1_bb[i,a,j,b]
                    for j in range(nocca):
                        for b in range(nvira):
                            c0 += 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,j,b]
    
    
            #[5]
            y1_a = numpy.zeros((nocca,nocca))
            y1_b = numpy.zeros((noccb,noccb))
            for k in range(nocca):
                for i in range(nocca):
                    for a in range(nvira):
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_a[i,k] += tmp1_aa[i,a,j,b]  * tmp1_bar_aa[k,a,j,b]
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_a[i,k] += tmp1_ab[i,a,j,b]  * tmp1_bar_ab[k,a,j,b]
            for k in range(nocca):
                for i in range(nocca):
                    for p in range(nmoa):
                        c1_a[p,k] -= 1. * fock_hfa[i,p] * y1_a[i,k]
    
            for k in range(noccb):
                for i in range(noccb):
                    for a in range(nvirb):
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_b[i,k] += tmp1_bb[i,a,j,b]  * tmp1_bar_bb[k,a,j,b]
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_b[i,k] += tmp1_ba[i,a,j,b]  * tmp1_bar_ba[k,a,j,b]
            for k in range(noccb):
                for i in range(noccb):
                    for p in range(nmob):
                        c1_b[p,k] -= 1. * fock_hfb[i,p] * y1_b[i,k]
    
            #[6]
            y1_aa = numpy.zeros((nocca,nvira,nocca,nvira))
            y1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
            y1_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
            y1_ba = numpy.zeros((noccb,nvirb,nocca,nvira))
            for b in range(nvira):
                for j in range(nocca):
                    for i in range(nocca):
                        for a in range(nvira):
                            for k in range(nocca):
                                y1_aa[i,a,j,b] += fock_hfa[i,k] * tmp1_bar_aa[k,a,j,b]
                    for i in range(noccb):
                        for a in range(nvirb):
                            for k in range(noccb):
                                y1_ba[i,a,j,b] += fock_hfb[i,k] * tmp1_bar_ba[k,a,j,b]
            for j in range(nocca):
                for b in range(nvira):
                    for c in range(nvira): 
                        B = b+nocca
                        C = c+nocca
                        for i in range(nocca):
                            for a in range(nvira):
                                A = a+nocca
                                c1_a[B,C] += 1. * tmp1_aa[i,a,j,b] * y1_aa[i,a,j,c]
                        for i in range(noccb):
                            for a in range(nvirb):
                                A = a+noccb
                                c1_a[B,C] += 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,j,c]
    
            for b in range(nvirb):
                for j in range(noccb):
                    for i in range(noccb):
                        for a in range(nvirb):
                            for k in range(noccb):
                                y1_bb[i,a,j,b] += fock_hfb[i,k] * tmp1_bar_bb[k,a,j,b]
                    for i in range(nocca):
                        for a in range(nvira):
                            for k in range(nocca):
                                y1_ab[i,a,j,b] += fock_hfa[i,k] * tmp1_bar_ab[k,a,j,b]
            for j in range(noccb):
                for b in range(nvirb):
                    for c in range(nvirb): 
                        B = b+noccb
                        C = c+noccb
                        for i in range(noccb):
                            for a in range(nvirb):
                                A = a+noccb
                                c1_b[B,C] += 1. * tmp1_bb[i,a,j,b] * y1_bb[i,a,j,c]
                        for i in range(nocca):
                            for a in range(nvira):
                                A = a+nocca
                                c1_b[B,C] += 1. * tmp1_ab[i,a,j,b] * y1_ab[i,a,j,c]
                                
            #[7]
            y1_aa = numpy.zeros((nocca,nvira,nocca,nvira))
            y1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
            y1_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
            y1_ba = numpy.zeros((noccb,nvirb,nocca,nvira))
            for i in range(nocca):
                for a in range(nvira):
                    for k in range(nocca):
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_aa[i,a,j,b] += fock_hfa[i,k] * tmp1_bar_aa[k,a,j,b]
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_ab[i,a,j,b] += fock_hfa[i,k] * tmp1_bar_ab[k,a,j,b]
            for c in range(nvira): 
                for i in range(nocca):
                    for a in range(nvira):
                        A = a+nocca
                        C = c+nocca
                        for j in range(nocca):
                            for b in range(nvira):
                                B = b+nocca
                                c1_a[A,C] += 1. * tmp1_aa[i,a,j,b] * y1_aa[i,c,j,b]
                        for j in range(noccb):
                            for b in range(nvirb):
                                B = b+noccb
                                c1_a[A,C] += 1. * tmp1_ab[i,a,j,b] * y1_ab[i,c,j,b]
    
            for i in range(noccb):
                for a in range(nvirb):
                    for k in range(noccb):
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_bb[i,a,j,b] += fock_hfb[i,k] * tmp1_bar_bb[k,a,j,b]
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_ba[i,a,j,b] += fock_hfb[i,k] * tmp1_bar_ba[k,a,j,b]
            for c in range(nvirb): 
                for i in range(noccb):
                    for a in range(nvirb):
                        A = a+noccb
                        C = c+noccb
                        for j in range(noccb):
                            for b in range(nvirb):
                                B = b+noccb
                                c1_b[A,C] += 1. * tmp1_bb[i,a,j,b] * y1_bb[i,c,j,b]
                        for j in range(nocca):
                            for b in range(nvira):
                                B = b+nocca
                                c1_b[A,C] += 1. * tmp1_ba[i,a,j,b] * y1_ba[i,c,j,b]
                    
            #print("c1_a")
            #print(c1_a)
            #print("c1_b")
            #print(c1_b)
            #print("================================")
            #[8]
            y1_aa = numpy.zeros((nocca,nvira,nocca,nvira))
            y1_bb = numpy.zeros((noccb,nvirb,noccb,nvirb))
            y1_ab = numpy.zeros((nocca,nvira,noccb,nvirb))
            y1_ba = numpy.zeros((noccb,nvirb,nocca,nvira))
            for j in range(nocca):
                for b in range(nvira):
                    for c in range(nvira): 
                        for i in range(nocca):
                            for a in range(nvira):
                                A = a+nocca
                                C = c+nocca
                                y1_aa[i,a,j,b] += fock_hfa[A,C] * tmp1_bar_aa[i,c,j,b]
                    for c in range(nvirb): 
                        for i in range(noccb):
                            for a in range(nvirb):
                                A = a+noccb
                                C = c+noccb
                                y1_ba[i,a,j,b] += fock_hfb[A,C] * tmp1_bar_ba[i,c,j,b]
            for j in range(nocca):
                for b in range(nvira):
                    for c in range(nvira): 
                        B = b+nocca
                        C = c+nocca
                        for i in range(nocca):
                            for a in range(nvira):
                                c1_a[B,C] -= 1. * tmp1_aa[i,a,j,b] * y1_aa[i,a,j,c]
                        for i in range(noccb):
                            for a in range(nvirb):
                                c1_a[B,C] -= 1. * tmp1_ba[i,a,j,b] * y1_ba[i,a,j,c]
    
            for j in range(noccb):
                for b in range(nvirb):
                    for c in range(nvirb): 
                        for i in range(noccb):
                            for a in range(nvirb):
                                A = a+noccb
                                C = c+noccb
                                y1_bb[i,a,j,b] += fock_hfb[A,C] * tmp1_bar_bb[i,c,j,b]
                    for c in range(nvira): 
                        for i in range(nocca):
                            for a in range(nvira):
                                A = a+nocca
                                C = c+nocca
                                y1_ab[i,a,j,b] += fock_hfa[A,C] * tmp1_bar_ab[i,c,j,b]
            for j in range(noccb):
                for b in range(nvirb):
                    for c in range(nvirb): 
                        B = b+noccb
                        C = c+noccb
                        for i in range(noccb):
                            for a in range(nvirb):
                                c1_b[B,C] -= 1. * tmp1_bb[i,a,j,b] * y1_bb[i,a,j,c]
                        for i in range(nocca):
                            for a in range(nvira):
                                c1_b[B,C] -= 1. * tmp1_ab[i,a,j,b] * y1_ab[i,a,j,c]
    
            y1_a = numpy.zeros((nvira,nvira))
            y1_b = numpy.zeros((nvirb,nvirb))
            for c in range(nvira): 
                for i in range(nocca):
                    for a in range(nvira):
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_a[a,c] += tmp1_aa[i,a,j,b] * tmp1_bar_aa[i,c,j,b]
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_a[a,c] += tmp1_ab[i,a,j,b] * tmp1_bar_ab[i,c,j,b]
            for c in range(nvira): 
                for a in range(nvira):
                    for p in range(nmoa):
                        A = a+nocca
                        C = c+nocca
                        c1_a[p,C] -= 1. * fock_hfa[A,p] * y1_a[a,c]
    
            for c in range(nvirb): 
                for i in range(noccb):
                    for a in range(nvirb):
                        for j in range(noccb):
                            for b in range(nvirb):
                                y1_b[a,c] += tmp1_bb[i,a,j,b] * tmp1_bar_bb[i,c,j,b]
                        for j in range(nocca):
                            for b in range(nvira):
                                y1_b[a,c] += tmp1_ba[i,a,j,b] * tmp1_bar_ba[i,c,j,b]
            for c in range(nvirb): 
                for a in range(nvirb):
                    for p in range(nmob):
                        A = a+noccb
                        C = c+noccb
                        c1_b[p,C] -= 1. * fock_hfb[A,p] * y1_b[a,c]
    
           #print("c1_a")
           #print(c1_a)
           #print("c1_b")
           #print(c1_b)
    
            # symmetrize c1
            for p in range(nmoa):
                for q in range(nmoa):
                    fock_a[p,q] += 0.5 * (c1_a[p,q] + c1_a[q,p])
            for p in range(nmob):
                for q in range(nmob):
                    fock_b[p,q] += 0.5 * (c1_b[p,q] + c1_b[q,p])

        ene = c0
        for i in range(nocca):
            ene += 1. * fock_a[i,i]
        for i in range(noccb):
            ene += 1. * fock_b[i,i]

        ene_tot = ene + nuc
        de = abs(ene_tot - ene_old)
        ene_old = ene_tot
        print()
        print("========================")
        print('iter = %d'%it, ' energy = %8.6f'%ene_tot, ' energy diff = %8.6f'%de, flush=True)
        print()

        if de < mp.thresh:
            conv = True
            break

        if mp.eval_fc:
            print("Fermi contact using HF-like density")
            rdm1 = mp.make_rdm1()
            #R_reslv = None #[-1,4.0] #so primitive
            mp.make_fc(rdm1) #, it, R_reslv)

            #mp.make_fc(rdm1, it, R_reslv=None)
            #print("Spin occupation numbers:")
            #
            #print("Fermi contact using correlated density")
            ##rdm1 = mp.make_rdm1(use_t2=True)
            #rdm1 = mp.make_rdm1(use_t2=True,use_ao=False)
            #rdm1_ao =  (reduce(numpy.dot, (mo_coeff[0], rdm1[0], mo_coeff[0].T)), 
            #            reduce(numpy.dot, (mo_coeff[1], rdm1[1], mo_coeff[1].T)))
            #spinrdm1 = rdm1[0] - rdm1[1]
            ##spinocc = numpy.sort(spinocc)[::-1]    
            #print(spinocc[0:nocca])

            
        #if it > 0:
        #    fock_a = 0.01*fock_a + 0.99*fock_a_old
        ### diagonalizing correlated Fock 
        mo_energy[0], U = scipy.linalg.eigh(fock_a)
        mo_coeff[0] = numpy.matmul(mo_coeff[0], U)
        mo_energy[1], U = scipy.linalg.eigh(fock_b)
        mo_coeff[1] = numpy.matmul(mo_coeff[1], U)

        if mp.mom:
            #aa, ab = mp.vir_exc
            #if aa < nocca: 
            mp.mom_select(mo_coeff_init, mo_coeff)
            #print("mo_coeff here")
            #print(mo_coeff[0])
            aa, ab = mp.vir_exc
            if aa > nocca-1:
                mo_coeff = mp.mom_reorder(mo_coeff)
            #else:
            #   print("not need to reorder")
    #if not mp.mom:
    mp.mo_energy = mo_energy
    mp.mo_coeff  = mo_coeff
    e_corr = ene_tot - ene_hf
    ss, s = mp._scf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                 mo_coeff[1][:,mo_occ[1]>0]), mp._scf.get_ovlp())
    print('multiplicity <S^2> = %.8g' %ss, '2S+1 = %.8g' %s)

    #exit()
    #print("final mo_coeff")
    #print(mo_coeff[0][:,:4])

    print()
    if conv:
        print("UOB-MP2 has converged")
    else:
        print("UOB-MP2 has not converged")

    print("UOB-MP2 energy = ", ene_tot)


def int_transform_ss(eri_ao, mo_coeff):
    nao = mo_coeff.shape[0]
    nmo = mo_coeff.shape[1]
    eri_mo = numpy.dot(mo_coeff.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    return eri_mo

def int_transform_os(eri_ao, mo_coeff_s1, mo_coeff_s2):
    nao = mo_coeff_s1.shape[0]
    nmo = mo_coeff_s1.shape[1]
    eri_mo = numpy.dot(mo_coeff_s1.T, eri_ao.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff_s2)
    eri_mo = eri_mo.reshape(nmo,nao,nao,nmo).transpose(1,0,3,2)
    eri_mo = numpy.dot(mo_coeff_s1.T, eri_mo.reshape(nao,-1))
    eri_mo = numpy.dot(eri_mo.reshape(-1,nao), mo_coeff_s2)
    eri_mo = eri_mo.reshape(nmo,nmo,nmo,nmo)
    return eri_mo

def get_nocc(mp):
    frozen = mp.frozen
    if mp._nocc is not None:
        return mp._nocc
    elif frozen is None:
        nocca = numpy.count_nonzero(mp.mo_occ[0] > 0)
        noccb = numpy.count_nonzero(mp.mo_occ[1] > 0)
    elif isinstance(frozen, (int, numpy.integer)):
        nocca = numpy.count_nonzero(mp.mo_occ[0] > 0) - frozen
        noccb = numpy.count_nonzero(mp.mo_occ[1] > 0) - frozen
        #assert(nocca > 0 and noccb > 0)
    elif isinstance(frozen[0], (int, numpy.integer, list, numpy.ndarray)):
        if len(frozen) > 0 and isinstance(frozen[0], (int, numpy.integer)):
            # The same frozen orbital indices for alpha and beta orbitals
            frozen = [frozen, frozen]
        occidxa = mp.mo_occ[0] > 0
        occidxa[list(frozen[0])] = False
        occidxb = mp.mo_occ[1] > 0
        occidxb[list(frozen[1])] = False
        nocca = numpy.count_nonzero(occidxa)
        noccb = numpy.count_nonzero(occidxb)
    else:
        raise NotImplementedError
    return nocca, noccb

def get_nmo(mp):
    frozen = mp.frozen
    if mp._nmo is not None:
        return mp._nmo
    elif frozen is None:
        nmoa = mp.mo_occ[0].size
        nmob = mp.mo_occ[1].size
    elif isinstance(frozen, (int, numpy.integer)):
        nmoa = mp.mo_occ[0].size - frozen
        nmob = mp.mo_occ[1].size - frozen
    elif isinstance(frozen[0], (int, numpy.integer, list, numpy.ndarray)):
        if isinstance(frozen[0], (int, numpy.integer)):
            frozen = (frozen, frozen)
        nmoa = len(mp.mo_occ[0]) - len(set(frozen[0]))
        nmob = len(mp.mo_occ[1]) - len(set(frozen[1]))
    else:
        raise NotImplementedError
    return nmoa, nmob


def get_frozen_mask(mp):
    '''Get boolean mask for the unrestricted reference orbitals.

    In the returned boolean (mask) array of frozen orbital indices, the
    element is False if it corresonds to the frozen orbital.
    '''
    moidxa = numpy.ones(mp.mo_occ[0].size, dtype=bool)
    moidxb = numpy.ones(mp.mo_occ[1].size, dtype=bool)

    frozen = mp.frozen
    if mp._nmo is not None:
        moidxa[mp._nmo[0]:] = False
        moidxb[mp._nmo[1]:] = False
    elif frozen is None:
        pass
    elif isinstance(frozen, (int, numpy.integer)):
        moidxa[:frozen] = False
        moidxb[:frozen] = False
    elif isinstance(frozen[0], (int, numpy.integer, list, numpy.ndarray)):
        if isinstance(frozen[0], (int, numpy.integer)):
            frozen = (frozen, frozen)
        moidxa[list(frozen[0])] = False
        moidxb[list(frozen[1])] = False
    else:
        raise NotImplementedError
    return moidxa,moidxb

def mom_reorder(mp, mo_coeff):
    import copy
    mo_coeff_save = copy.copy(mo_coeff)
    #mo_energy_save = copy.copy(mo_energy)
    #mo_energy = mp.mo_energy
    mo_coeff = copy.copy(mo_coeff_save)
    print("before")
    print(mo_coeff[0][:,:4])
    ia, ib = mp.occ_exc
    aa, ab = mp.vir_exc
    #print("ia ", ia)
    #print(mo_coeff_save[0][:,ia])
    mo_coeff[0][:,ia] = mo_coeff_save[0][:,aa]
    #mo_energy[0][ia]  = mo_energy_save[0][aa]
    #print("test")
    #print(mo_coeff_save[0][:,ia])
    mo_coeff[0][:,aa] = mo_coeff_save[0][:,ia]
    #mo_energy[0][aa]  = mo_energy_save[0][ia]
    if (ib is not None) and (ab is not None):
        mo_coeff[1][:,ib] = mo_coeff_save[1][:,ab]
        mo_coeff[1][:,ab] = mo_coeff_save[1][:,ib]
    print("after")
    print(mo_coeff[0][:,:4])
    return mo_coeff #, mo_energy
        
def mom_select(mp, mo_coeff_init, mo_coeff_new):
    #print("old")
    #print(mo_coeff_init[0][:,:4])
    #print("new")
    #print(mo_coeff_new[0][:,:4])
    ovi = mp._scf.get_ovlp()
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    ia, ib = mp.occ_exc
    aa, ab = mp.vir_exc
    Oa = numpy.matmul(mo_coeff_init[0][:,0:nocca].T,
                      numpy.matmul(ovi,mo_coeff_new[0][:,:]))
    #print("Oa")
    #print(Oa)
    Pa = []
    for j in range(nmoa):
        tmp = 0.
        for i in range(nocca):
            tmp += Oa[i,j]
        Pa.append(abs(tmp))
        #print("Paj = ", Pa[j])
    max_el = max(Pa)
    indxa = 0
    for j in range(nmoa):
        if Pa[j] == max_el:
            indxa = j
    if (ib is not None) and (ab is not None):
        Ob = numpy.matmul(mo_coeff_init[1][:,0:noccb].T,
                          numpy.matmul(ovi,mo_coeff_new[1][:,:]))
        Pb = []
        for j in range(nmob):
            tmp = 0.
            for i in range(noccb):
                tmp += Ob[i,j]
            Pb.append(abs(tmp))
        max_el = max(Pa)
        indxb = 0
        for j in range(nmob):
            if Pa[j] == max_el:
                indxb = j
    else:
        indxb = None

    print("indxa = %d"%indxa, "Pa = %8.6f"%Pa[indxa])
    if indxb is not None:
        print("indxb = %d"%indxb, "Pb = %8.6f"%Pb[indxb])
    mp.vir_exc = [indxa, indxb]
    #mp.ib = indxb
    #return indxa, indxb

def make_rdm1(mp, use_t2=False, use_ao=True, **kwargs):
    '''One-particle density matrix

    Returns:
        A list of 2D ndarrays for alpha and beta spins
    '''
    mo_coeff = mp.mo_coeff
    mo_occ   = mp._scf.mo_occ
    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    mo_ea, mo_eb = mp.mo_energy
    eia_a = mo_ea[:nocca,None] - mo_ea[None,nocca:]
    eia_b = mo_eb[:noccb,None] - mo_eb[None,noccb:]

    if not use_t2:
        mo_a = mo_coeff[0]
        mo_b = mo_coeff[1]
        dm_a = numpy.dot(mo_a*mo_occ[0], mo_a.conj().T)
        dm_b = numpy.dot(mo_b*mo_occ[1], mo_b.conj().T)
        return numpy.array((dm_a,dm_b))
    else:
        from pyscf.cc import uccsd_rdm

        eris = mp.ao2mo(mo_coeff)

        dtype = eris.ovov.dtype
        t2aa = numpy.empty((nocca,nocca,nvira,nvira), dtype=dtype)
        t2ab = numpy.empty((nocca,noccb,nvira,nvirb), dtype=dtype)
        t2bb = numpy.empty((noccb,noccb,nvirb,nvirb), dtype=dtype)
        t2 = (t2aa,t2ab,t2bb)
        
        for i in range(nocca):
            if isinstance(eris.ovov, numpy.ndarray) and eris.ovov.ndim == 4:
                # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                eris_ovov = eris.ovov[i]
            else:
                eris_ovov = numpy.asarray(eris.ovov[i*nvira:(i+1)*nvira])

            eris_ovov = eris_ovov.reshape(nvira,nocca,nvira).transpose(1,0,2)
            t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_a)
            t2aa[i] = t2i - t2i.transpose(0,2,1)
            #print("t2aa")
            #print(t2aa[i])


            if isinstance(eris.ovOV, numpy.ndarray) and eris.ovOV.ndim == 4:
                # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                eris_ovov = eris.ovOV[i]
            else:
                eris_ovov = numpy.asarray(eris.ovOV[i*nvira:(i+1)*nvira])
            eris_ovov = eris_ovov.reshape(nvira,noccb,nvirb).transpose(1,0,2)
            t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_a[i], eia_b)
            t2ab[i] = t2i
            

        for i in range(noccb):
            if isinstance(eris.OVOV, numpy.ndarray) and eris.OVOV.ndim == 4:
                # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                eris_ovov = eris.OVOV[i]
            else:
                eris_ovov = numpy.asarray(eris.OVOV[i*nvirb:(i+1)*nvirb])
            eris_ovov = eris_ovov.reshape(nvirb,noccb,nvirb).transpose(1,0,2)
            t2i = eris_ovov.conj()/lib.direct_sum('a+jb->jab', eia_b[i], eia_b)
            t2bb[i] = t2i - t2i.transpose(0,2,1)
            
        doo, dvv = _gamma1_intermediates(mp, t2)
        nocca, noccb, nvira, nvirb = t2[1].shape
        dov = numpy.zeros((nocca,nvira))
        dOV = numpy.zeros((noccb,nvirb))
        d1 = (doo, (dov, dOV), (dov.T, dOV.T), dvv)
        rdm1 = uccsd_rdm._make_rdm1(mp, d1, with_frozen=True, ao_repr=False)
        if use_ao:
            rdm1_ao =  (reduce(numpy.dot, (mo_coeff[0], rdm1[0], mo_coeff[0].T)), 
                        reduce(numpy.dot, (mo_coeff[1], rdm1[1], mo_coeff[1].T)))
            return rdm1_ao
        else:
            return rdm1
# DO NOT make tag_array for DM here because the DM arrays may be modified and
# passed to functions like get_jk, get_vxc.  These functions may take the tags
# (mo_coeff, mo_occ) to compute the potential if tags were found in the DM
# arrays and modifications to DM arrays may be ignored.

def _gamma1_intermediates(mp, t2):
    t2aa, t2ab, t2bb = t2
    dooa  = lib.einsum('imef,jmef->ij', t2aa.conj(), t2aa) *-.5
    dooa -= lib.einsum('imef,jmef->ij', t2ab.conj(), t2ab)
    doob  = lib.einsum('imef,jmef->ij', t2bb.conj(), t2bb) *-.5
    doob -= lib.einsum('mief,mjef->ij', t2ab.conj(), t2ab)

    dvva  = lib.einsum('mnae,mnbe->ba', t2aa.conj(), t2aa) * .5
    dvva += lib.einsum('mnae,mnbe->ba', t2ab.conj(), t2ab)
    dvvb  = lib.einsum('mnae,mnbe->ba', t2bb.conj(), t2bb) * .5
    dvvb += lib.einsum('mnea,mneb->ba', t2ab.conj(), t2ab)
    return ((dooa, doob), (dvva, dvvb))

def make_fc(mp, dm0, it=None, R_reslv=None, hfc_nuc=None, verbose=None):
    '''The contribution of Fermi-contact term and dipole-dipole interactions'''
    #log = logger.new_logger(hfcobj, verbose)
    mol = mp.mol
    if hfc_nuc is None:
        hfc_nuc = range(mol.natm)
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5

    #if R_reslv is not None:
    #    mo_coeff = mp.mo_coeff
    #    nocca, noccb = mp.get_nocc()
    #    dma_mo, dmb_mo = mp.make_rdm1(use_t2=True,use_ao=False)
    #    spinnocca, U = scipy.linalg.eigh(dma_mo)
    #    spinmoa = numpy.matmul(mo_coeff[0], U)
    #    nao = mo_coeff[0].shape[0]
    #    tmp = numpy.zeros((nao,nao))
    #    for mu in range(nao):
    #        for nu in range(nao):
    #            tmp[mu,nu] = spinmoa[] * spinmoa[mu,nocca-1] * spinmoa[nu,nocca-1]
    #    np = 1000
    #    dz = (R_reslv[1] - R_reslv[0])/np
    #    fname = "spinden_somo"+str(it)+".dat"
    #    with open(fname, 'w') as f:
    #        for i in range(np):
    #            r = i*dz + R_reslv[0]
    #            coords = [[0,0,r]]
    #            h1fc = _get_integrals_fc_Rreslv(mol, coords)
    #            fc = numpy.einsum('ij,ji', h1fc, tmp)
    #            f.write(" %8.6f %8.6f \n"  %(r, fc))


    e_gyro = .5 * nist.G_ELECTRON
    nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2MHz

    hfc = []
    for i, atm_id in enumerate(hfc_nuc):
        nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atm_id)) * nuc_mag
        #h1 = _get_integrals_fcdip(mol, atm_id)
        #fcsd = numpy.einsum('xyij,ji->xy', h1, spindm)

        h1fc = _get_integrals_fc(mol, atm_id)
        fc = numpy.einsum('ij,ji', h1fc, spindm)

        #sd = fcsd + numpy.eye(3) * fc

        print('FC of atom %d :'%atm_id, '%8.6f (in MHz)' %(2*fac * nuc_gyro * fc))
        #if hfcobj.verbose >= logger.INFO:
        #    _write(hfcobj, align(fac*nuc_gyro*sd)[0], 'SD of atom %d (in MHz)' % atm_id)
        #hfc.append(fac * nuc_gyro * fcsd)
    #return numpy.asarray(hfc)

def _get_integrals_fcdip(mol, atm_id):
    '''AO integrals for FC + Dipole-dipole'''
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atm_id)):
        # Note the fermi-contact part is different to the fermi-contact
        # operator in SSC.  FC here is associated to the the integrals of
        # (\nabla \nabla 1/r), which includes the contribution of Poisson
        # equation, 4\pi rho.  Factor 4.\pi/3 is used in the Fermi contact
        # contribution.  In SSC, the factor of FC part is -8\pi/3.
        ipipv = mol.intor('int1e_ipiprinv', 9).reshape(3,3,nao,nao)
        ipvip = mol.intor('int1e_iprinvip', 9).reshape(3,3,nao,nao)
        h1ao = ipipv + ipvip  # (nabla i | r/r^3 | j)
        h1ao = h1ao + h1ao.transpose(0,1,3,2)
        trace = h1ao[0,0] + h1ao[1,1] + h1ao[2,2]
        idx = numpy.arange(3)
        h1ao[idx,idx] -= trace
    return h1ao

def _get_integrals_fc(mol, atm_id):
    '''AO integrals for Fermi contact term'''
    coords = mol.atom_coord(atm_id).reshape(1, 3)
    ao = mol.eval_gto('GTOval', coords)
    return 4*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)

def _get_integrals_fc_Rreslv(mol, coords):
    '''AO integrals for Fermi contact term'''
    ao = mol.eval_gto('GTOval', coords)
    return 4*numpy.pi/3 * numpy.einsum('ip,iq->pq', ao, ao)


class UOBMP2(obmp2.OBMP2):

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    int_transform_ss = int_transform_ss
    int_transform_os = int_transform_os
    mom_select = mom_select
    mom_reorder = mom_reorder
    break_sym = False
    #use_t2 = False

    @lib.with_doc(obmp2.OBMP2.kernel.__doc__)
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2, kernel)

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris(self, mo_coeff, verbose=self.verbose)

    make_rdm1 = make_rdm1
    #make_rdm2 = make_rdm2
    make_fc = make_fc
    eval_fc = False

    def nuc_grad_method(self):
        from pyscf.grad import ump2
        return ump2.Gradients(self)

OBMP2 = UOBMP2

#from pyscf import scf
#scf.uhf.UHF.MP2 = lib.class_as_method(MP2)


class _ChemistsERIs(obmp2._ChemistsERIs):
    def __init__(self, mp, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mp.mo_coeff
        moidx = mp.get_frozen_mask()
        self.mo_coeff = mo_coeff = \
                (mo_coeff[0][:,moidx[0]], mo_coeff[1][:,moidx[1]])

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.clock(), time.time())
    eris = _ChemistsERIs(mp, mo_coeff)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nao = eris.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (nao_pair**2 + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory-mem_now)

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    orboa = moa[:,:nocca]
    orbob = mob[:,:noccb]
    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]

    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore+mem_now < mp.max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            eris.ovov = ao2mofn((orboa,orbva,orboa,orbva)).reshape(nocca*nvira,nocca*nvira)
            eris.ovOV = ao2mofn((orboa,orbva,orbob,orbvb)).reshape(nocca*nvira,noccb*nvirb)
            eris.OVOV = ao2mofn((orbob,orbvb,orbob,orbvb)).reshape(noccb*nvirb,noccb*nvirb)
        else:
            eris.ovov = ao2mo.general(mp._scf._eri, (orboa,orbva,orboa,orbva))
            eris.ovOV = ao2mo.general(mp._scf._eri, (orboa,orbva,orbob,orbvb))
            eris.OVOV = ao2mo.general(mp._scf._eri, (orbob,orbvb,orbob,orbvb))

    elif getattr(mp._scf, 'with_df', None):
        logger.warn(mp, 'UMP2 detected DF being used in the HF object. '
                    'MO integrals are computed based on the DF 3-index tensors.\n'
                    'It\'s recommended to use DF-UMP2 module.')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((orboa,orbva,orboa,orbva))
        eris.ovOV = mp._scf.with_df.ao2mo((orboa,orbva,orbob,orbvb))
        eris.OVOV = mp._scf.with_df.ao2mo((orbob,orbvb,orbob,orbvb))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        _ao2mo_ovov(mp, (orboa,orbva,orbob,orbvb), eris.feri,
                    max(2000, max_memory), log)
        eris.ovov = eris.feri['ovov']
        eris.ovOV = eris.feri['ovOV']
        eris.OVOV = eris.feri['OVOV']

    time1 = log.timer('Integral transformation', *time0)
    return eris

def _ao2mo_ovov(mp, orbs, feri, max_memory=2000, verbose=None):
    time0 = (time.clock(), time.time())
    log = logger.new_logger(mp, verbose)
    orboa = numpy.asarray(orbs[0], order='F')
    orbva = numpy.asarray(orbs[1], order='F')
    orbob = numpy.asarray(orbs[2], order='F')
    orbvb = numpy.asarray(orbs[3], order='F')
    nao, nocca = orboa.shape
    noccb = orbob.shape[1]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]

    mol = mp.mol
    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nbas = mol.nbas
    assert(nvira <= nao)
    assert(nvirb <= nao)

    ao_loc = mol.ao_loc_nr()
    dmax = max(4, min(nao/3, numpy.sqrt(max_memory*.95e6/8/(nao+nocca)**2)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao))
    ftmp = lib.H5TmpFile()
    disk = (nocca**2*(nao*(nao+dmax)/2+nvira**2) +
            noccb**2*(nao*(nao+dmax)/2+nvirb**2) +
            nocca*noccb*(nao**2+nvira*nvirb))
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, disk*8/1e6)

    fint = gto.moleintor.getints4c
    aa_blk_slices = []
    ab_blk_slices = []
    count_ab = 0
    count_aa = 0
    time1 = time0
    with lib.call_in_background(ftmp.__setitem__) as save:
        for ish0, ish1, ni in sh_ranges:
            for jsh0, jsh1, nj in sh_ranges:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]

                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=(0,nbas,ish0,ish1, jsh0,jsh1,0,nbas),
                           aosym='s1', ao_loc=ao_loc, cintopt=ao2mopt._cintopt,
                           out=eribuf)
                tmp_i = lib.ddot(orboa.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao))
                tmp_li = lib.ddot(orbob.T, tmp_i.reshape(nocca*(i1-i0)*(j1-j0),nao).T)
                tmp_li = tmp_li.reshape(noccb,nocca,(i1-i0),(j1-j0))
                save('ab/%d'%count_ab, tmp_li.transpose(1,0,2,3))
                ab_blk_slices.append((i0,i1,j0,j1))
                count_ab += 1

                if ish0 >= jsh0:
                    tmp_li = lib.ddot(orboa.T, tmp_i.reshape(nocca*(i1-i0)*(j1-j0),nao).T)
                    tmp_li = tmp_li.reshape(nocca,nocca,(i1-i0),(j1-j0))
                    save('aa/%d'%count_aa, tmp_li.transpose(1,0,2,3))

                    tmp_i = lib.ddot(orbob.T, eri.reshape(nao,(i1-i0)*(j1-j0)*nao))
                    tmp_li = lib.ddot(orbob.T, tmp_i.reshape(noccb*(i1-i0)*(j1-j0),nao).T)
                    tmp_li = tmp_li.reshape(noccb,noccb,(i1-i0),(j1-j0))
                    save('bb/%d'%count_aa, tmp_li.transpose(1,0,2,3))
                    aa_blk_slices.append((i0,i1,j0,j1))
                    count_aa += 1

                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)
    eri = eribuf = tmp_i = tmp_li = None

    fovov = feri.create_dataset('ovov', (nocca*nvira,nocca*nvira), 'f8',
                                chunks=(nvira,nvira))
    fovOV = feri.create_dataset('ovOV', (nocca*nvira,noccb*nvirb), 'f8',
                                chunks=(nvira,nvirb))
    fOVOV = feri.create_dataset('OVOV', (noccb*nvirb,noccb*nvirb), 'f8',
                                chunks=(nvirb,nvirb))
    occblk = int(min(max(nocca,noccb),
                     max(4, 250/nocca, max_memory*.9e6/8/(nao**2*nocca)/5)))

    def load_aa(h5g, nocc, i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(aa_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = h5g[str(k)][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(h5g[str(k)][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def load_ab(h5g, nocca, i0, eri):
        if i0 < nocca:
            i1 = min(i0+occblk, nocca)
            for k, (p0,p1,q0,q1) in enumerate(ab_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = h5g[str(k)][i0:i1]

    def save(h5dat, nvir, i0, i1, dat):
        for i in range(i0, i1):
            h5dat[i*nvir:(i+1)*nvir] = dat[i-i0].reshape(nvir,-1)

    with lib.call_in_background(save) as bsave:
        with lib.call_in_background(load_aa) as prefetch:
            buf_prefecth = numpy.empty((occblk,nocca,nao,nao))
            buf = numpy.empty_like(buf_prefecth)
            load_aa(ftmp['aa'], nocca, 0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocca, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(ftmp['aa'], nocca, i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*nocca,nao,nao)
                dat = _ao2mo.nr_e2(eri, orbva, (0,nvira,0,nvira), 's1', 's1')
                bsave(fovov, nvira, i0, i1,
                      dat.reshape(i1-i0,nocca,nvira,nvira).transpose(0,2,1,3))
                time1 = log.timer_debug1('pass2 ao2mo for aa [%d:%d]' % (i0,i1), *time1)

            buf_prefecth = numpy.empty((occblk,noccb,nao,nao))
            buf = numpy.empty_like(buf_prefecth)
            load_aa(ftmp['bb'], noccb, 0, buf_prefecth)
            for i0, i1 in lib.prange(0, noccb, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(ftmp['bb'], noccb, i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*noccb,nao,nao)
                dat = _ao2mo.nr_e2(eri, orbvb, (0,nvirb,0,nvirb), 's1', 's1')
                bsave(fOVOV, nvirb, i0, i1,
                      dat.reshape(i1-i0,noccb,nvirb,nvirb).transpose(0,2,1,3))
                time1 = log.timer_debug1('pass2 ao2mo for bb [%d:%d]' % (i0,i1), *time1)

        orbvab = numpy.asarray(numpy.hstack((orbva, orbvb)), order='F')
        with lib.call_in_background(load_ab) as prefetch:
            load_ab(ftmp['ab'], nocca, 0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocca, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(ftmp['ab'], nocca, i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*noccb,nao,nao)
                dat = _ao2mo.nr_e2(eri, orbvab, (0,nvira,nvira,nvira+nvirb), 's1', 's1')
                bsave(fovOV, nvira, i0, i1,
                      dat.reshape(i1-i0,noccb,nvira,nvirb).transpose(0,2,1,3))
                time1 = log.timer_debug1('pass2 ao2mo for ab [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
del(WITH_T2)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    mp = OBMP2(mf)
    mp.verbose = 5

    #pt = OBMP2(mf)
    #emp2, t2 = pt.kernel()
    #print(emp2 - -0.204019967288338)
    #pt.max_memory = 1
    #emp2, t2 = pt.kernel()
    #print(emp2 - -0.204019967288338)
    #
    #pt = MP2(scf.density_fit(mf, 'weigend'))
    #print(pt.kernel()[0] - -0.204254500454)
