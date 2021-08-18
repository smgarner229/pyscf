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
Moller-Plesset perturbation theory
'''

from pyscf import scf
from pyscf.mp import mp2
from pyscf.mp import dfmp2
from pyscf.mp import ump2
from pyscf.mp import gmp2
from pyscf.mp import obmp2
from pyscf.mp import obmp2_faster
from pyscf.mp import obmp2_active
from pyscf.mp import uobmp2
from pyscf.mp import uobmp2_faster
from pyscf.mp import uobmp2_scs

def MP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = mp2.MP2.__doc__
    if isinstance(mf, scf.uhf.UHF):
        return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif isinstance(mf, scf.ghf.GHF):
        return GMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)

def RMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = mp2.RMP2.__doc__
    from pyscf import lib
    from pyscf.soscf import newton_ah

    if isinstance(mf, scf.uhf.UHF):
        raise RuntimeError('RMP2 cannot be used with UHF method.')
    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RMP2 method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UMP2 method is called.')
        return UMP2(mf, frozen, mo_coeff, mo_occ)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    if getattr(mf, 'with_df', None):
        return dfmp2.DFMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)

def UMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = ump2.UMP2.__doc__
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.uhf.UHF):
        mf = scf.addons.convert_to_uhf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-UMP2')
    else:
        return ump2.UMP2(mf, frozen, mo_coeff, mo_occ)

def GMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = gmp2.GMP2.__doc__
    from pyscf.soscf import newton_ah

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.ghf.GHF):
        mf = scf.addons.convert_to_ghf(mf)

    if getattr(mf, 'with_df', None):
        raise NotImplementedError('DF-GMP2')
    else:
        return gmp2.GMP2(mf, frozen, mo_coeff, mo_occ)

def OBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = obmp2.OBMP2.__doc__
    if isinstance(mf, scf.rhf.RHF):
        return obmp2.OBMP2(mf, frozen, mo_coeff, mo_occ)
    #elif isinstance(mf, scf.ghf.GHF):
    #    return GMP2(mf, frozen, mo_coeff, mo_occ)
    #else:
    #    return RMP2(mf, frozen, mo_coeff, mo_occ)

def OBMP2_active(mf, nact, nocc_act, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = obmp2_active.OBMP2.__doc__
    if isinstance(mf, scf.rhf.RHF):
        return obmp2_active.OBMP2(mf, nact, nocc_act, frozen, mo_coeff, mo_occ)

def OBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = obmp2_faster.OBMP2.__doc__
    if isinstance(mf, scf.rhf.RHF):
        return obmp2_faster.OBMP2(mf, frozen, mo_coeff, mo_occ)
def UOBMP2(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = uobmp2.UOBMP2.__doc__
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2.UOBMP2(mf, frozen, mo_coeff, mo_occ)
def UOBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = uobmp2_faster.UOBMP2.__doc__
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_faster.UOBMP2(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_faster(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = uobmp2_faster.UOBMP2.__doc__
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_faster.UOBMP2(mf, frozen, mo_coeff, mo_occ)

def UOBMP2_SCS(mf, frozen=0, mo_coeff=None, mo_occ=None):
    __doc__ = uobmp2_scs.UOBMP2_SCS.__doc__
    if isinstance(mf, scf.uhf.UHF):
        return uobmp2_scs.UOBMP2_SCS(mf, frozen, mo_coeff, mo_occ)