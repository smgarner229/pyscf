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

import os
import time
from functools import reduce
import numpy
import pyscf.lib.logger as logger
from pyscf.mcscf import mc1step, addons



def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=5e-04,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    if callback is None:
        callback = casscf.callback

    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 2-step CASSCF')

    mo = mo_coeff
    nmo = mo.shape[1]
    print("nmo", nmo)
    eris = casscf.ao2mo(mo)
    #print 'ci0', ci0
    if os.environ.get("cycle") is not None:
        #if (int(os.environ["cycle"])+1) == 1:
        #    conv_tol_grad = 5e-03
        #    tol = 5e-04
        #    logger.info(casscf, 'Lan sets conv_tol to %g', tol)
        #    logger.info(casscf, 'Lan sets conv_tol_grad to %g', conv_tol_grad)
        if (int(os.environ["cycle"])+1) > 1:
            rdm1_pregeom_AO = addons.make_rdm12(casscf, mo, ci0) #, rdm2Target_AO
            #print "rdm1_pregeom_AO"
            #print rdm1_pregeom_AO
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())    
    if (casscf.ncas == nmo and not casscf.internal_rotation) or casscf.sa_geom_opt:
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
    conv_tol_ddm = 3.*conv_tol_grad
    conv = False
    de, elast = e_tot, e_tot
    totmicro = totinner = 0
    casdm1 = 0
    r0 = None

    #Lan generates targeted rdms
    rdm1Target_AO = addons.make_rdm12(casscf, mo, fcivec) #, rdm2Target_AO
    #casdm1Target, casdm2Target = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas, casscf.nelecas)
    
    t2m = t1m = log.timer('Initializing 2-step CASSCF', *cput0)
    imacro = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        t3m = t2m
        omega = e_tot
        casdm1_old = casdm1
        #print "fcivec", fcivec
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas, casscf.nelecas)
        print("casdm1 casdm1")
        print(casdm1)
        norm_ddm = numpy.linalg.norm(casdm1 - casdm1_old)
        t3m = log.timer('update CAS DM', *t3m)
        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())

        if(casscf.is_use_gmres): 
            print("Using GMRES")
            for imicro in range(max_cycle_micro):
                print('imicro', imicro)
                u, g_orb = casscf.rotate_orb_gmres(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                                   eris, imacro, r0, conv_tol_grad*.3, max_stepsize, log)
                
                norm_gorb = numpy.linalg.norm(g_orb)
                log.debug(' |g|=%5.3g', norm_gorb)
                if imicro == 0:
                    norm_gorb0 = norm_gorb
                eris = None
                u = u.copy()
                g_orb = g_orb.copy()
                mo = casscf.rotate_mo(mo, u, log=None)                
                eris = casscf.ao2mo(mo)
                t3m = log.timer('update eri', *t3m)
                norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
                de = numpy.dot(casscf.pack_uniq_var(u), g_orb)
                #save current imicro
                norm_gorb_old =  norm_gorb
                u_old = u
                g_orb_old = g_orb

                if norm_gorb < 1e-04: #norm_t < 1e-4 or abs(de) < tol*.4 or 
                    break
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


        else: # using standard pyscf
        
            for imicro in range(max_cycle_micro):
                rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                            eris, r0, conv_tol_grad*.3, max_stepsize, log)
                #exit()
                u, g_orb, njk1, r0 = next(rota)
                rota.close()
                njk += njk1
                norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
                norm_gorb = numpy.linalg.norm(g_orb)
                if imicro == 0:
                    norm_gorb0 = norm_gorb
                de = numpy.dot(casscf.pack_uniq_var(u), g_orb)
                t3m = log.timer('orbital rotation', *t3m)
                
                eris = None
                u = u.copy()
                g_orb = g_orb.copy()
                mo = casscf.rotate_mo(mo, u, log)
                eris = casscf.ao2mo(mo)
                t3m = log.timer('update eri', *t3m)
                
                log.debug('micro %d  ~dE=%5.3g  |u-1|=%5.3g  |g[o]|=%5.3g  |dm1|=%5.3g',
                          imicro, de, norm_t, norm_gorb, norm_ddm)
                
                if callable(callback):
                    callback(locals())

                t2m = log.timer('micro iter %d'%imicro, *t2m)
                if norm_t < 1e-4 or abs(de) < tol*.4 or norm_gorb < conv_tol_grad*.2:
                    break

        totinner += njk
        totmicro += imicro + 1

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot

        if (abs(de) < tol and
            norm_gorb < conv_tol_grad and norm_ddm < conv_tol_ddm):
            conv = True
        else:
            elast = e_tot

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        #fname='mo_conv.txt'
        #with open(fname, 'w') as f:
        #    for i in range(mo.shape[0]):
        #        for j in range(mo.shape[1]):
        #            f.write(" %20.10f" % mo[i,j])
        #    f.write("\n")
        #    
        log.info('2-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('2-step CASSCF not converged, %d macro (%d JK %d micro) steps',
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

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('2-step CASSCF', *cput0)
    #print "fcivec", fcivec
    return conv, e_tot, e_cas, fcivec, mo, mo_energy



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

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
    emc = kernel(mc1step.CASSCF(m, 4, 4), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -3.22013929407)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = mc1step.CASSCF(m, 6, 4)
    mc.verbose = 4
    mo = m.mo_coeff.copy()
    mo[:,2:5] = m.mo_coeff[:,[4,2,3]]
    emc = mc.mc2step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

