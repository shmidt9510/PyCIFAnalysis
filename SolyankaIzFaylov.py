# coding=utf-8
import math
import copy
import sys
import random as ran
import numpy as np
import numpy.linalg as nl
import GSASIIpath
import GSASIImath as G2mth
import GSASIIspc as G2spc
import GSASIIElem as G2elem

sind = lambda x: np.sin(x*np.pi/180.)
asind = lambda x: 180.*np.arcsin(x)/np.pi
tand = lambda x: np.tan(x*np.pi/180.)
atand = lambda x: 180.*np.arctan(x)/np.pi
atan2d = lambda y,x: 180.*np.arctan2(y,x)/np.pi
cosd = lambda x: np.cos(x*np.pi/180.)
acosd = lambda x: 180.*np.arccos(x)/np.pi
rdsq2d = lambda x,p: round(1.0/np.sqrt(x),p)
rpd = np.pi/180.
RSQ2PI = 1./np.sqrt(2.*np.pi)
SQ2 = np.sqrt(2.)
RSQPI = 1./np.sqrt(np.pi)
R2pisq = 1./(2.*np.pi**2)
nxs = np.newaxis

def cell2AB(cell):
    """Computes orthogonalization matrix from unit cell constants

    :param tuple cell: a,b,c, alpha, beta, gamma (degrees)
    :returns: tuple of two 3x3 numpy arrays (A,B)
       A for crystal to Cartesian transformations A*x = np.inner(A,x) = X
       B (= inverse of A) for Cartesian to crystal transformation B*X = np.inner(B,X) = x
    """
    G,g = cell2Gmat(cell)
    cellstar = Gmat2cell(G)
    A = np.zeros(shape=(3,3))
    # from Giacovazzo (Fundamentals 2nd Ed.) p.75
    A[0][0] = cell[0]                # a
    A[0][1] = cell[1]*cosd(cell[5])  # b cos(gamma)
    A[0][2] = cell[2]*cosd(cell[4])  # c cos(beta)
    A[1][1] = cell[1]*sind(cell[5])  # b sin(gamma)
    A[1][2] = -cell[2]*cosd(cellstar[3])*sind(cell[4]) # - c cos(alpha*) sin(beta)
    A[2][2] = 1/cellstar[2]         # 1/c*
    B = nl.inv(A)
    return A,B

def calc_rDsq(H, A):
    'needs doc string'
    rdsq = H[0] * H[0] * A[0] + H[1] * H[1] * A[1] + H[2] * H[2] * A[2] + H[0] * H[1] * A[3] + H[0] * H[2] * A[4] + H[
                                                                                                                        1] * \
                                                                                                                    H[
                                                                                                                        2] * \
                                                                                                                    A[5]
    return rdsq

def calc_rDsq2(H, G):
    'needs doc string'
    return np.inner(H, np.inner(G, H))

def calc_rDsqSS(H, A, vec):
    'needs doc string'
    rdsq = calc_rDsq(H[:3] + (H[3] * vec).T, A)
    return rdsq

def calc_rDsqZ(H, A, Z, tth, lam):
    'needs doc string'
    rdsq = calc_rDsq(H, A) + Z * sind(tth) * 2.0 * rpd / lam ** 2
    return rdsq

def calc_rDsqZSS(H, A, vec, Z, tth, lam):
    'needs doc string'
    rdsq = calc_rDsq(H[:3] + (H[3][:, np.newaxis] * vec).T, A) + Z * sind(tth) * 2.0 * rpd / lam ** 2
    return rdsq

def SwapIndx(Axis, H):
    'needs doc string'
    if Axis in [1, -1]:
        return H
    elif Axis in [2, -3]:
        return [H[1], H[2], H[0]]
    else:
        return [H[2], H[0], H[1]]

def calc_rDsqT(H, A, Z, tof, difC):
    'needs doc string'
    rdsq = calc_rDsq(H, A) + Z / difC
    return rdsq

def calc_rDsqTSS(H, A, vec, Z, tof, difC):
    'needs doc string'
    rdsq = calc_rDsq(H[:3] + (H[3][:, np.newaxis] * vec).T, A) + Z / difC
    return rdsq

def sortHKLd(HKLd, ifreverse, ifdup, ifSS=False):
    '''sort reflection list on d-spacing; can sort in either order

    :param HKLd: a list of [h,k,l,d,...];
    :param ifreverse: True for largest d first
    :param ifdup: True if duplicate d-spacings allowed
    :return sorted reflection list
    '''
    T = []
    N = 3
    if ifSS:
        N = 4
    for i, H in enumerate(HKLd):
        if ifdup:
            T.append((H[N], i))
        else:
            T.append(H[N])
    D = dict(zip(T, HKLd))
    T.sort()
    if ifreverse:
        T.reverse()
    X = []
    okey = ''
    for key in T:
        if key != okey: X.append(D[key])  # remove duplicate d-spacings
        okey = key
    return X

def getHKLmax(dmin, SGData, A):
    'finds maximum allowed hkl for given A within dmin'
    SGLaue = SGData['SGLaue']
    if SGLaue in ['3R', '3mR']:  # Rhombohedral axes
        Hmax = [0, 0, 0]
        cell = A2cell(A)
        aHx = cell[0] * math.sqrt(2.0 * (1.0 - cosd(cell[3])))
        cHx = cell[0] * math.sqrt(3.0 * (1.0 + 2.0 * cosd(cell[3])))
        Hmax[0] = Hmax[1] = int(round(aHx / dmin))
        Hmax[2] = int(round(cHx / dmin))
        # print Hmax,aHx,cHx
    else:  # all others
        Hmax = MaxIndex(dmin, A)
    return Hmax

def MaxIndex(dmin,A):
    'needs doc string'
    Hmax = [0,0,0]
    try:
        cell = A2cell(A)
    except:
        cell = [1,1,1,90,90,90]
    for i in range(3):
        Hmax[i] = int(round(cell[i]/dmin))
    return Hmax

def CentCheck(Cent,H):
    #Проверяет на то принадлежит штуке или нет
    'needs doc string'
    h,k,l = H
    if Cent == 'A' and (k+l)%2:
        return False
    elif Cent == 'B' and (h+l)%2:
        return False
    elif Cent == 'C' and (h+k)%2:
        return False
    elif Cent == 'I' and (h+k+l)%2:
        return False
    elif Cent == 'F' and ((h+k)%2 or (h+l)%2 or (k+l)%2):
        return False
    elif Cent == 'R' and (-h+k+l)%3:
        return False
    else:
        return True
#Сюда запихивать штуки типа P 1/2 m3m чтобы получить center и system
def SpcGroup(SGSymbol):
        """
        Determines cell and symmetry information from a short H-M space group name

        :param SGSymbol: space group symbol (string) with spaces between axial fields
        :returns: (SGError,SGData)

           * SGError = 0 for no errors; >0 for errors (see SGErrors below for details)
           * SGData - is a dict (see :ref:`Space Group object<SGData_table>`) with entries:

                 * 'SpGrp': space group symbol, slightly cleaned up
                 * 'SGLaue':  one of '-1', '2/m', 'mmm', '4/m', '4/mmm', '3R',
                   '3mR', '3', '3m1', '31m', '6/m', '6/mmm', 'm3', 'm3m'
                 * 'SGInv': boolean; True if centrosymmetric, False if not
                 * 'SGLatt': one of 'P', 'A', 'B', 'C', 'I', 'F', 'R'
                 * 'SGUniq': one of 'a', 'b', 'c' if monoclinic, '' otherwise
                 * 'SGCen': cell centering vectors [0,0,0] at least
                 * 'SGOps': symmetry operations as [M,T] so that M*x+T = x'
                 * 'SGSys': one of 'triclinic', 'monoclinic', 'orthorhombic',
                   'tetragonal', 'rhombohedral', 'trigonal', 'hexagonal', 'cubic'
                 * 'SGPolax': one of ' ', 'x', 'y', 'x y', 'z', 'x z', 'y z',
                   'xyz', '111' for arbitrary axes
                 * 'SGPtGrp': one of 32 point group symbols (with some permutations), which
                    is filled by SGPtGroup, is external (KE) part of supersymmetry point group
                 * 'SSGKl': default internal (Kl) part of supersymmetry point group; modified
                    in supersymmetry stuff depending on chosen modulation vector for Mono & Ortho

        """
        LaueSym = ('-1', '2/m', 'mmm', '4/m', '4/mmm', '3R', '3mR', '3', '3m1', '31m', '6/m', '6/mmm', 'm3', 'm3m')
        LattSym = ('P', 'A', 'B', 'C', 'I', 'F', 'R')
        UniqSym = ('', '', 'a', 'b', 'c', '',)
        SysSym = (
        'triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'rhombohedral', 'trigonal', 'hexagonal', 'cubic')
        SGData = {}
        if ':R' in SGSymbol:
            SGSymbol = SGSymbol.replace(':', ' ')  # get rid of ':' in R space group symbols from some cif files
        SGSymbol = SGSymbol.split(':')[0]  # remove :1/2 setting symbol from some cif files
        SGInfo = pyspg.sgforpy(SGSymbol)
        SGData['SpGrp'] = SGSymbol.strip().lower().capitalize()
        SGData['SGLaue'] = LaueSym[SGInfo[0] - 1]
        SGData['SGInv'] = bool(SGInfo[1])
        SGData['SGLatt'] = LattSym[SGInfo[2] - 1]
        SGData['SGUniq'] = UniqSym[SGInfo[3] + 1]
        if SGData['SGLatt'] == 'P':
            SGData['SGCen'] = np.array(([0, 0, 0],))
        elif SGData['SGLatt'] == 'A':
            SGData['SGCen'] = np.array(([0, 0, 0], [0, .5, .5]))
        elif SGData['SGLatt'] == 'B':
            SGData['SGCen'] = np.array(([0, 0, 0], [.5, 0, .5]))
        elif SGData['SGLatt'] == 'C':
            SGData['SGCen'] = np.array(([0, 0, 0], [.5, .5, 0, ]))
        elif SGData['SGLatt'] == 'I':
            SGData['SGCen'] = np.array(([0, 0, 0], [.5, .5, .5]))
        elif SGData['SGLatt'] == 'F':
            SGData['SGCen'] = np.array(([0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0, ]))
        elif SGData['SGLatt'] == 'R':
            SGData['SGCen'] = np.array(([0, 0, 0], [1. / 3., 2. / 3., 2. / 3.], [2. / 3., 1. / 3., 1. / 3.]))
        SGData['SGOps'] = []
        SGData['SGGen'] = []
        SGData['SGSpin'] = []
        for i in range(SGInfo[5]):
            Mat = np.array(SGInfo[6][i])
            Trns = np.array(SGInfo[7][i])
            SGData['SGOps'].append([Mat, Trns])
            if 'array' in str(type(SGInfo[8])):  # patch for old fortran bin?
                SGData['SGGen'].append(int(SGInfo[8][i]))
            SGData['SGSpin'].append(1)
        if SGData['SGLaue'] == '2/m' and SGData['SGLatt'] != 'P' and '/' in SGData['SpGrp']:
            SGData['SGSpin'].append(1)  # fix bug in fortran
        if 'F' in SGData['SpGrp']:
            SGData['SGSpin'] += [1, 1, 1, 1]
        elif 'R' in SGData['SpGrp']:
            SGData['SGSpin'] += [1, 1, 1]
        elif SGData['SpGrp'][0] in ['A', 'B', 'C', 'I']:
            SGData['SGSpin'] += [1, ]
        if SGData['SGInv']:
            if SGData['SGLaue'] in ['-1', '2/m', 'mmm']:
                Ibar = 7
            elif SGData['SGLaue'] in ['4/m', '4/mmm']:
                Ibar = 1
            elif SGData['SGLaue'] in ['3R', '3mR', '3', '3m1', '31m', '6/m', '6/mmm']:
                Ibar = 15  # 8+4+2+1
            else:
                Ibar = 4
            Ibarx = Ibar & 14
        else:
            Ibarx = 8
            if SGData['SGLaue'] in ['-1', '2/m', 'mmm', 'm3', 'm3m']:
                Ibarx = 0
        moregen = []
        for i, gen in enumerate(SGData['SGGen']):
            if SGData['SGLaue'] in ['m3', 'm3m']:
                if gen in [1, 2, 4]:
                    SGData['SGGen'][i] = 4
                elif gen < 7:
                    SGData['SGGen'][i] = 0
            elif SGData['SGLaue'] in ['4/m', '4/mmm', '3R', '3mR', '3', '3m1', '31m', '6/m', '6/mmm']:
                if gen == 2:
                    SGData['SGGen'][i] = 4
                elif gen in [3, 5]:
                    SGData['SGGen'][i] = 3
                elif gen == 6:
                    if SGData['SGLaue'] in ['4/m', '4/mmm']:
                        SGData['SGGen'][i] = 128
                    else:
                        SGData['SGGen'][i] = 16
                elif not SGData['SGInv'] and gen == 12:
                    SGData['SGGen'][i] = 8
                elif (not SGData['SGInv']) and (SGData['SGLaue'] in ['3', '3m1', '31m', '6/m', '6/mmm']) and (gen == 1):
                    SGData['SGGen'][i] = 24
            gen = SGData['SGGen'][i]
            if gen == 99:
                gen = 8
                if SGData['SGLaue'] in ['3m1', '31m', '6/m', '6/mmm']:
                    gen = 3
                elif SGData['SGLaue'] == 'm3m':
                    gen = 12
                SGData['SGGen'][i] = gen
            elif gen == 98:
                gen = 8
                if SGData['SGLaue'] in ['3m1', '31m', '6/m', '6/mmm']:
                    gen = 4
                SGData['SGGen'][i] = gen
            elif not SGData['SGInv'] and gen in [23, ] and SGData['SGLaue'] in ['m3', 'm3m']:
                SGData['SGGen'][i] = 24
            elif gen >= 16 and gen != 128:
                if not SGData['SGInv']:
                    gen = 31
                else:
                    gen ^= Ibarx
                SGData['SGGen'][i] = gen
            if SGData['SGInv']:
                if gen < 128:
                    moregen.append(SGData['SGGen'][i] ^ Ibar)
                else:
                    moregen.append(1)
        SGData['SGGen'] += moregen
        #    GSASIIpath.IPyBreak()
        if SGData['SGLaue'] in '-1':
            SGData['SGSys'] = SysSym[0]
        elif SGData['SGLaue'] in '2/m':
            SGData['SGSys'] = SysSym[1]
        elif SGData['SGLaue'] in 'mmm':
            SGData['SGSys'] = SysSym[2]
        elif SGData['SGLaue'] in ['4/m', '4/mmm']:
            SGData['SGSys'] = SysSym[3]
        elif SGData['SGLaue'] in ['3R', '3mR']:
            SGData['SGSys'] = SysSym[4]
        elif SGData['SGLaue'] in ['3', '3m1', '31m']:
            SGData['SGSys'] = SysSym[5]
        elif SGData['SGLaue'] in ['6/m', '6/mmm']:
            SGData['SGSys'] = SysSym[6]
        elif SGData['SGLaue'] in ['m3', 'm3m']:
            SGData['SGSys'] = SysSym[7]
        SGData['SGPolax'] = SGpolar(SGData)
        SGData['SGPtGrp'], SGData['SSGKl'] = SGPtGroup(SGData)
        return SGInfo[-1], SGData

# Даст номер группы симметрии
def GetBraviasNum(center, system):
        """Determine the Bravais lattice number, as used in GenHBravais

        :param center: one of: 'P', 'C', 'I', 'F', 'R' (see SGLatt from GSASIIspc.SpcGroup)
        :param system: one of 'cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 'trigonal' (for R)
          'monoclinic', 'triclinic' (see SGSys from GSASIIspc.SpcGroup)
        :return: a number between 0 and 13
          or throws a ValueError exception if the combination of center, system is not found (i.e. non-standard)

        """
        if center.upper() == 'F' and system.lower() == 'cubic':
            return 0
        elif center.upper() == 'I' and system.lower() == 'cubic':
            return 1
        elif center.upper() == 'P' and system.lower() == 'cubic':
            return 2
        elif center.upper() == 'R' and system.lower() == 'trigonal':
            return 3
        elif center.upper() == 'P' and system.lower() == 'hexagonal':
            return 4
        elif center.upper() == 'I' and system.lower() == 'tetragonal':
            return 5
        elif center.upper() == 'P' and system.lower() == 'tetragonal':
            return 6
        elif center.upper() == 'F' and system.lower() == 'orthorhombic':
            return 7
        elif center.upper() == 'I' and system.lower() == 'orthorhombic':
            return 8
        elif center.upper() == 'C' and system.lower() == 'orthorhombic':
            return 9
        elif center.upper() == 'P' and system.lower() == 'orthorhombic':
            return 10
        elif center.upper() == 'C' and system.lower() == 'monoclinic':
            return 11
        elif center.upper() == 'P' and system.lower() == 'monoclinic':
            return 12
        elif center.upper() == 'P' and system.lower() == 'triclinic':
            return 13
        raise ValueError, 'non-standard Bravais lattice center=%s, cell=%s' % (center, system)

#Эта штука вернёт HKL и d если задать cell
def GenHBravais(dmin, Bravais, A):
    #Пишет параметры решётки
    """Generate the positionally unique powder diffraction reflections

    :param dmin: minimum d-spacing in A
    :param Bravais: lattice type (see GetBraviasNum). Bravais is one of::
             0 F cubic
             1 I cubic
             2 P cubic
             3 R hexagonal (trigonal not rhombohedral)
             4 P hexagonal
             5 I tetragonal
             6 P tetragonal
             7 F orthorhombic
             8 I orthorhombic
             9 C orthorhombic
             10 P orthorhombic
             11 C monoclinic
             12 P monoclinic
             13 P triclinic

    :param A: reciprocal metric tensor elements as [G11,G22,G33,2*G12,2*G13,2*G23]
    :return: HKL unique d list of [h,k,l,d,-1] sorted with largest d first

    """
    if Bravais in [9, 11]:
        Cent = 'C'
    elif Bravais in [1, 5, 8]:
        Cent = 'I'
    elif Bravais in [0, 7]:
        Cent = 'F'
    elif Bravais in [3]:
        Cent = 'R'
    else:
        Cent = 'P'
    Hmax = MaxIndex(dmin, A)
    dminsq = 1. / (dmin ** 2)
    HKL = []
    if Bravais == 13:  # triclinic
        for l in range(-Hmax[2], Hmax[2] + 1):
            for k in range(-Hmax[1], Hmax[1] + 1):
                hmin = 0
                if (k < 0): hmin = 1
                if (k == 0 and l < 0): hmin = 1
                for h in range(hmin, Hmax[0] + 1):
                    H = [h, k, l]
                    rdsq = calc_rDsq(H, A)
                    if 0 < rdsq <= dminsq:
                        HKL.append([h, k, l, rdsq2d(rdsq, 6), -1])
    elif Bravais in [11, 12]:  # monoclinic - b unique
        Hmax = SwapIndx(2, Hmax)
        for h in range(Hmax[0] + 1):
            for k in range(-Hmax[1], Hmax[1] + 1):
                lmin = 0
                if k < 0: lmin = 1
                for l in range(lmin, Hmax[2] + 1):
                    [h, k, l] = SwapIndx(-2, [h, k, l])
                    H = []
                    if CentCheck(Cent, [h, k, l]): H = [h, k, l]
                    if H:
                        rdsq = calc_rDsq(H, A)
                        if 0 < rdsq <= dminsq:
                            HKL.append([h, k, l, rdsq2d(rdsq, 6), -1])
                    [h, k, l] = SwapIndx(2, [h, k, l])
    elif Bravais in [7, 8, 9, 10]:  # orthorhombic
        for h in range(Hmax[0] + 1):
            for k in range(Hmax[1] + 1):
                for l in range(Hmax[2] + 1):
                    H = []
                    if CentCheck(Cent, [h, k, l]): H = [h, k, l]
                    if H:
                        rdsq = calc_rDsq(H, A)
                        if 0 < rdsq <= dminsq:
                            HKL.append([h, k, l, rdsq2d(rdsq, 6), -1])
    elif Bravais in [5, 6]:  # tetragonal
        for l in range(Hmax[2] + 1):
            for k in range(Hmax[1] + 1):
                for h in range(k, Hmax[0] + 1):
                    H = []
                    if CentCheck(Cent, [h, k, l]): H = [h, k, l]
                    if H:
                        rdsq = calc_rDsq(H, A)
                        if 0 < rdsq <= dminsq:
                            HKL.append([h, k, l, rdsq2d(rdsq, 6), -1])
    elif Bravais in [3, 4]:
        lmin = 0
        if Bravais == 3: lmin = -Hmax[2]  # hexagonal/trigonal
        for l in range(lmin, Hmax[2] + 1):
            for k in range(Hmax[1] + 1):
                hmin = k
                if l < 0: hmin += 1
                for h in range(hmin, Hmax[0] + 1):
                    H = []
                    if CentCheck(Cent, [h, k, l]): H = [h, k, l]
                    if H:
                        rdsq = calc_rDsq(H, A)
                        if 0 < rdsq <= dminsq:
                            HKL.append([h, k, l, rdsq2d(rdsq, 6), -1])
    else:  # cubic
        for l in range(Hmax[2] + 1):
            for k in range(l, Hmax[1] + 1):
                for h in range(k, Hmax[0] + 1):
                    H = []
                    if CentCheck(Cent, [h, k, l]):
                        H = [h, k, l]
                    if H:
                        rdsq = calc_rDsq(H, A)
                        if 0 < rdsq <= dminsq:
                            HKL.append([h, k, l, rdsq2d(rdsq, 6), -1])
    return sortHKLd(HKL, True, False)

#Чтобы получить

