#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraser, W. B. (1969). Dispersion of elastic waves in elliptical bars. 
*Journal of Sound and Vibration*, 10(2), 247‑260. 
https://doi.org/10.1016/0022-460X(69)90199-0



Created on Fri Sep  2 13:03:37 2022

@author: dina.zyani
@author: denis.brizard
"""

import numpy as np
from scipy.special import jv   #Bessel function (v,z) = (order,arg)
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

import warnings
import time
import pickle

import round_bar_pochhammer_chree as round_bar
import ellipticReferenceSolutions as el
import fraser_matrix

import figutils as fu

def char_func_elliptic_fortran(k, w, R, N, theta, gamma, c_1, c_2, mode='L'):
    """Characteristic function for elliptical bar, with underlying Fortran code
    
    :param float k: wavenumber
    :param float w: circular frequency
    :param array R: radius of collocation points
    :param int N: number of collocation points
    :param array theta: angle of collocation points
    :param array gamma: angle of normal at collocation points
    :param float c_1: velocity
    :param float c_2: velocity
    :param str mode: wave propagation mode ('L' or 'T')
    """
    if mode=='L':
        abc = True
        A = fraser_matrix.mat(k, w, R, N, gamma, theta, c_1, c_2, abc)
        detA = np.linalg.det(A[1:, 1:])
    elif mode=='T':
        abc = False
        A = fraser_matrix.mat(k, w, R, N, gamma, theta, c_1, c_2, abc)
        ind = list(range(3*N))
        ind.remove(N)   # XXX reason why indices N and 2*N ?
        ind.remove(2*N) # ditto
        detA = np.linalg.det(A[np.ix_(ind,ind)])
    return detA



def char_func_elliptic(k, w, R, N, theta, gamma, c_1, c_2):
    """Characteristic function for elliptical bar
    
    :param float k: wavenumber
    :param float w: circular frequency
    :param array R: radius of collocation points
    :param int N: number of collocation points
    :param array theta: angle of collocation points
    :param array gamma: angle of normal at collocation points
    :param float c_1: velocity
    :param float c_2: velocity
    """
    c = w/k
    cc2 = (c/c_2)**2
    cc1 = (c/c_1)**2 # XXX tester si plus rapide ou pas
    alpha =  k*np.lib.scimath.sqrt(cc1-1)
    beta = k*np.lib.scimath.sqrt(cc2-1)
    aR = alpha*R  #Fraser eq(7)
    bR = beta*R #Fraser eq(7)
    b2 = beta**2
    k2 = k**2
    K_ = (1/2)*(b2-k2)/alpha**2
  
# =============================================================================
#     Ces matrices sont issues de l'article de Fraser eq(2)
# =============================================================================
    A11_ = np.zeros((N, N), dtype = np.complex128)
    A12_ = np.zeros((N, N), dtype = np.complex128)
    A13_ = np.zeros((N, N), dtype = np.complex128)
    A21_ = np.zeros((N, N), dtype = np.complex128)
    A22_ = np.zeros((N, N), dtype = np.complex128)
    A23_ = np.zeros((N, N), dtype = np.complex128)
    A31_ = np.zeros((N, N), dtype = np.complex128)
    A32_ = np.zeros((N, N), dtype = np.complex128)
    A33_ = np.zeros((N, N), dtype = np.complex128)
    p =  np.arange(0, 2*N, 2)  # N premières valeurs paires
    for ii, nn in enumerate(p):
#bessel
        jb = jv(nn+2, bR)
        ja = jv(nn+2, aR)
        jb_ = jv(nn-2, bR)
        ja_ = jv(nn-2, aR)
        j1b = jv(nn+1, bR)
        j1b_ = jv(nn-1, bR)
#cos et sin
        cos1 = np.cos(nn*theta+2*gamma)
        cos2 = np.cos(nn*theta-2*gamma)
        sin1 = np.sin(nn*theta+2*gamma)
        sin2 = np.sin(nn*theta-2*gamma)
        cos_1 = np.cos(nn*theta+gamma)
        cos_2 = np.cos(nn*theta-gamma)
        cos = np.cos(nn*theta)
        jc1 = jb*cos1
        jc2 = jb_*cos2
        js1 = jb*sin1
        js2 = jb_*sin2
        j1c1 = j1b*cos_1
        j1c2 = j1b_*cos_2
    
        A11_[:, ii] = -jc1 + jc2
        A12_[:, ii] = jc1+jc2-2*jv(nn, bR)*cos
        A13_[:, ii] =  ja*cos1+ja_*cos2-2*(2*K_-1)*jv(nn,aR)*cos
        A21_[:, ii] = -js1-js2
        A22_[:, ii] = js1-js2
        A23_[:, ii] = ja*sin1-ja_*sin2
        A31_[:, ii] = np.array((j1c1+j1c2)*k/beta)*(1j)
        A32_[:, ii] = np.array((j1c1-j1c2)*(b2-k2)/(beta*k))*(1j)
        A33_[:, ii] = np.array((-jv(nn+1,aR)*cos_1+jv(nn-1,aR)*cos_2)*2*k/alpha)*(1j)
        
    # A = np.block([[A21_, A22_, A23_], [A11_, A12_, A13_], [A31_, A32_, A33_]])
    B = np.concatenate((A21_, A22_, A23_),axis=1)
    C = np.concatenate((A11_, A12_, A13_),axis=1)
    D = np.concatenate((A31_, A32_, A33_),axis=1)
    A = np.concatenate((B, C, D))
    detA = np.linalg.det(A[1:,1: ])
    return detA



class DispElliptic(round_bar.DetDispEquation):
    """Classe pour la barre elliptique."""
    
    def __init__(self, nu=0.3, E=210e9, rho=7800, a=0.05, e=0.4, N=4, 
                 mode='L', fortran=True):
        """
        Initialise variables.
    
        :param float nu: Poisson's ratio [-]
        :param float E: Young's modulus [Pa]
        :param float rho: density [kg/m3]
        :param float a: large radius of ellipse [m]
        :param float e: excentricity of elliptical cross section [-]
        :param int N: number of collocation points (on quarter cross-section)
        :param str mode: type of mode ('L':longitudinal, 'T': torsional)
        :param bool fortran: Fortran acceleration for characteristic matrix
        """
        mu = E / (2 * (1 + nu))  # coef de Lamé
        la = E * nu / ((1 + nu) * (1 - 2 * nu))  # coef de Lamé
        c_2 = np.sqrt(mu/rho) 
        # c_1 = np.sqrt((la+2*mu)/rho)
        c_1 = c_2*np.sqrt(2*(1-nu)/(1-2*nu)) #Fraser eq(7)
        self.mat = {'E':E, 'rho':rho, 'nu':nu, 'mu':mu, 'la':la}

        self.c = {'c_2':c_2, 'c_1':c_1, 'co': np.sqrt(E/rho)}
        b = a*np.sqrt(1-e**2) #Fraser eq(4) 
        self.a = a #b
        self.geo = {'e': e, 'b': b, 'N': N,'a':a}
        
        # Collocation points coordinates
        m = np.arange(1, N+1)
        theta = (m-1)*np.pi/2/N
        e2 = e**2
        cos__2 = np.cos(theta)**2
        gamma = -np.arctan((e2*np.sin(2*theta))/(2*(1-e2*cos__2))) #Frser eq(6)
        R = b*np.sqrt(1/(1 - e2*cos__2)) #Frser eq(5)
        self.ellipse = {'R':R, 'gamma':gamma, 'theta':theta}
        
        # Characteristic function
        if mode=='T' and fortran is False:
            fortran = True
            print('Forcing Fortran=True because equations for T mode are not written in Python')
        if fortran:
            def detfun(k, w):
                """Characteristic determinant function."""
                return char_func_elliptic_fortran(k, w, R, N, theta, gamma, 
                                                  c_1, c_2, mode=mode)
        else:
            def detfun(k, w):
                """Characteristic determinant function."""
                return char_func_elliptic(k, w, R, N, theta, gamma, c_1, c_2)
        self.detfun = detfun
        self.vectorized = False  # detfun is not vectorized
        self.dim = {'c':c_2, 'l':b}  # for dimensionless variables
        self.dimlab = {'c':'c_2', 'l':'b'}  # name of dimensionless variables for use un labels

    
    def plot_ellipse(self):
        """Plot elliptical cross section

        """
        plt.figure('ellipse')
        plt.polar(self.ellipse['theta'], self.ellipse['R'], '.-', label='ellipse')

        
        x = self.ellipse['R']*np.cos(self.ellipse['theta'])
        y = self.ellipse['R']*np.sin(self.ellipse['theta'])
        # r = np.sqrt(x**2+y**2)
        for ii in range(self.geo['N']):
            plt.quiver(0, 0, x[ii], y[ii], scale=0.09, color='g')
        x_g = self.ellipse['R']*np.cos(self.ellipse['theta']+self.ellipse['gamma'])
        y_g = self.ellipse['R']*np.sin(self.ellipse['theta']+self.ellipse['gamma'])
        for ii in range(self.geo['N']):
            plt.quiver(0, 0, x_g[ii], y_g[ii], scale=0.09, color='r')
     
        
    def computeKCmap(self, k, c, adim=True):
        """Compute the value of the characteristic equation on a (k,c) grid

        :param array k: wavenumbers
        :param array c: velocity
        :param bool adim: True if the given input arguments are dimensionless
        """
        if adim:
            K = k
            C = c
            c = c * self.c["c_2"]
            k = k / self.geo["b"]
        else: 
            K = k * self.geo["b"]
            C = c /self.c["c_2"]
        Z = np.empty((len(c), len(k)), dtype=np.complex128)
        start = time.clock()

        for ii, cc in enumerate(c):
            print('%i/%i'%(ii,len(c)))
            for jj, kk in enumerate(k):
                om = kk*cc
                Z[ii, jj] = self.detfun(kk, om)
                
        end = time.clock()
        temps = end - start
        npts = len(k)*len(c)
        print('#'*35)
        print('Took %g s to compute %i points'%(temps, npts))
        print('%g ms/1pt'%(temps/npts*1e3))
        print('#'*35)

        self.kc = {"c": c, "k": k, "det": Z, "K": K, "C": C}

        
        
    def plotDet_KC(self, xy="KC", typep="contour", nature="imag", level=[0], 
                   figname=None, adim=True, colors='b', lw=1):
        """
        Tracer les courbes de dispersion.

        Parameters
        ----------
        xy : string, optional
            Domaine. The default is "WC".
        typep : string, optional
            Type du graphiqDet.computeKCmap(k, c)ue. The default is "contour".
        nature : string, optional
            Partie réelle ou imaginaire du determinant. The default is "imag".
        figname : string, optional
            Nom de la figure. The default is None.

        Returns
        -------
        None.

        """
        if adim:
            x = self.kc["K"]
            y = self.kc["C"]
            det = self.kc["det"]
            xlabel = "K = k*b"
            ylabel = "C=c/c_2[-]"
        else:
            x = self.kc['k']
            y = self.kc['c']
            det = self.kc["det"]
            xlabel = "k [1/m]"
            ylabel = "c [m/s]"
            

        plt.figure(figname)
        if typep == "contour":
            # level = [0]
            if nature == "real":
                data = det.real
            elif nature == "imag":
                data = det.imag
            elif nature=='abs':
                data = abs(det)
            elif nature=='quotient_abs':
                data = abs(det.real)/abs(det.imag)
            CS = plt.contour(x, y, data, level, colors=colors, linewidths=lw)
            # CL = plt.clabel(CS, fmt='%g')
        elif typep == "sign":
            if nature == "real":
                data = np.sign(det.real)
            elif nature == "imag":
                data = np.sign(det.imag)
            plt.pcolormesh(x, y, data, shading="auto", cmap="cool", rasterized=True)
            # plt.colorbar()
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ymax=2.)
        plt.title(typep + "(" + nature + "(det))")
        
        if typep=='contour':
            return CS
        
if __name__ == "__main__":
    plt.close("all")
    # %% Resolution_numérique :: FOLLOW FIRST BRANCH
    if True:
        e = 0.4
        N = 4
        Det = DispElliptic(e=e, N=N, mode='L')
        omega = np.linspace(0, 8e5, 500)  # Ok pour k<1.208
        # omega = np.linspace(0, 1e6, 4000)  # trying very small step. Ok pour k<1.2182
        Det.followBranch0(omega, itermax=20)
        Det.plotFollow()

        plt.figure('sign_imag')
        plt.plot(Det.b0['k']*Det.geo['b'], Det.b0['c']/Det.c['c_2'], '+-', label='Regula Falsi algo')
        # plt.ylim(ymax=1.7, ymin=0.7)
        # plt.xlim(xmin=0., xmax=5)

        FR = el.Fraser()
        FR.plot(e=[e], figname='sign_imag')
        
        Det.computeKCmap(k=np.linspace(0, 5, 100), c=np.linspace(0.6, 2.2, 100), adim=True)
        Det.plotDet_KC('KC', typep='sign', nature='real')
        Det.plotDet_KC('KC', typep='sign', nature='imag')
        
    
    #%% Comparaison Fortran/Python + validation Fraser
    if False:
        e = 0.7
        N = 7
        ##### Calcul
        Detpy = DispElliptic(e=e, N=N, fortran=False)
        Detfo = DispElliptic(e=e, N=N, fortran=True)
        # Calcul Python
        k =  np.linspace(0.0001, 5/Detpy.geo["b"], 150)
        c = np.linspace(0.7*Detpy.c["c_2"], 1.7*Detpy.c["c_2"], 100)
        Detpy.computeKCmap(k, c, adim = False)
        # Calcul Fortran
        k =  np.linspace(0.0001, 5/Detfo.geo["b"], 150)
        c = np.linspace(0.7*Detfo.c["c_2"], 1.7*Detfo.c["c_2"], 100)
        Detfo.computeKCmap(k, c, adim = False)
        ### Affichage
        ## montrer que l'on obtient les mêmes courbes de dispersion
        #Affichage Python
        Detpy.plotDet_KC(figname='cont_imag', colors='tab:blue', lw=3)
        Detpy.plotDet_KC(typep="contour", nature='real', figname="cont_real", colors='tab:blue', lw=1)
        #Affichage Fortran
        Detfo.plotDet_KC(figname='cont_imag', colors='tab:orange', lw=1)
        Detfo.plotDet_KC(typep="contour", nature='real', figname="cont_real", colors='tab:orange')
        FR = el.Fraser()
        FR.plot(e=[e], figname='cont_imag')
        FR.plot(e=[e], figname='cont_imag', branch=1)
        FR.plot(e=[e], figname='cont_real')
        FR.plot(e=[e], figname='cont_real', branch=1)
        ## montrer les différences (soit graphique, soit valeurs)        
        levels = np.logspace(-99, 99, 67)
        Detpy.plotDet_KC('KC', 'contour', 'abs', level=levels)
        Detpy.plotDet_KC('KC', 'contour', 'real', level=levels)
        Detpy.plotDet_KC('KC', 'contour', 'imag', level=levels)
        Detpy.plotDet_KC('KC', 'contour', 'quotient_abs', level=levels)
    #%% Etude parité N
    if False:
        e = 0.7
        FR = el.Fraser()
        FR.plot(e=[e], figname='cont_imag')
        FR.plot(e=[e], figname='cont_imag', branch=1)
        FR.plot(e=[e], figname='cont_real')
        FR.plot(e=[e], figname='cont_real', branch=1)
        FR.plot()
        FR.plot(e=[e], figname='sign_imag')
        FR.plot(e=[e], figname='sign_imag', branch=1)
        N = [4, 6, 8, 10, 12]
        # N = [3, 5, 7, 9, 13]
        coul = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        DET = []
        # Calcul
        for nn in N:
            Det = DispElliptic(e=e, N=nn)
            k =  np.linspace(0.0001, 5/Det.geo["b"], 150)
            c = np.linspace(0.7*Det.c["c_2"], 1.7*Det.c["c_2"], 100)
            Det.computeKCmap(k, c, adim = False)
            DET.append(Det)
            
        # Affichage
        for nn, cc, Det in zip(N, coul, DET):
            Det.plotDet_KC(figname='cont_imag', colors=cc)
            Det.plotDet_KC(typep="contour", nature='real', figname="cont_real", colors=cc)
            Det.plotDet_KC(typep="sign", nature='real', figname="sign_real")
            Det.plotDet_KC(typep="sign", figname="sign_imag")
    #%% KC -- Convergence
    if False:
        e = 0.4                            
        FR = el.Fraser()
        FR.plot(e=[e], figname='cont_imag')
        FR.plot(e=[e], figname='cont_imag', branch=1)
        FR.plot(e=[e], figname='cont_real')
        FR.plot(e=[e], figname='cont_real', branch=1)
        N = [3, 4, 5, 6, 7, 8, 9, 10]
        # N = [3, 4]
        # coul = ['tab:blue', 'tab:orange']
        coul = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        DET = []
        # Calcul
        for nn in N:
            Det = DispElliptic(e=e, N=nn)
            k =  np.linspace(0.0001, 5/Det.geo["b"], 150)
            c = np.linspace(0.7*Det.c["c_2"], 1.7*Det.c["c_2"], 100)
            Det.computeKCmap(k, c, adim = False)
            DET.append(Det)
            
        # Affichage
        for nn, cc, Det in zip(N, coul, DET):
            Det.plotDet_KC(figname='cont_imag', colors=cc)
            Det.plotDet_KC(typep="contour", nature='real', figname="cont_real", colors=cc)
    #%% KC -- Convergence avec 2 N
    if False:
        e = 0.4                            
        FR = el.Fraser()
        FR.plot(e=[e], figname='cont')
        FR.plot(e=[e], figname='cont', branch=1)
        N = [3, 4]
        coul = ['tab:blue', 'tab:orange']
        DET = []
        # Calcul
        for nn in N:
            Det = DispElliptic(e=e, N=nn)
            k =  np.linspace(0.0001, 5/Det.geo["b"], 150)
            c = np.linspace(0.7*Det.c["c_2"], 1.7*Det.c["c_2"], 100)
            Det.computeKCmap(k, c, adim = False)
            DET.append(Det)
            
        # Affichage
        for nn, cc, Det in zip(N, coul, DET):
            Det.plotDet_KC(figname='cont', colors=cc)
            Det.plotDet_KC(typep="contour", nature='real', figname="cont", colors=cc)
    #%% KC  -- Comportement
    if False:
        e = 0.9
        N = [3, 4, 5, 6, 8, 9, 10, 11, 12]
        for nn in N:
            Det = DispElliptic(e=e, N=nn, fortran=False)
            k =  np.linspace(0.0001, 5/Det.geo["b"], 150)
            c = np.linspace(0.7*Det.c["c_2"], 2.*Det.c["c_2"], 100)
            Det.computeKCmap(k, c, adim = False)
            if nn%2==0:
                Det.plotDet_KC(typep="sign", nature='real', figname="sign_real%i"%nn)
            else:
                Det.plotDet_KC(typep="sign", figname="sign_imag%i"%nn)
    #%% KC

    if False:
        e = 0.4
        N = 7
        Det = DispElliptic(e=e, N=N, fortran=True)
        Det.plot_ellipse()
        FR = el.Fraser()
        FR.plot(e=[e], figname='cont_imag')
        FR.plot(e=[e], figname='cont_imag', branch=1)
        FR.plot(e=[e], figname='cont_real')
        FR.plot(e=[e], figname='cont_real', branch=1)
         
        FR.plot(e=[e], figname='sign_imag')
        FR.plot(e=[e], figname='sign_imag', branch=1)
        FR.plot(e=[e], figname='sign_real')
        FR.plot(e=[e], figname='sign_real', branch=1)
        
        k =  np.linspace(0.0001, 5/Det.geo["b"], 150)
        c = np.linspace(0.7*Det.c["c_2"], 2.*Det.c["c_2"], 100)
        Det.computeKCmap(k, c, adim = False)
        Det.plotDet_KC(figname='cont_imag')
        Det.plotDet_KC(typep="contour", nature='real', figname="cont_real")
        Det.plotDet_KC(typep="sign", nature='real', figname="sign_real")
        Det.plotDet_KC(typep="sign", figname="sign_imag")

    #%% Domaine KW
    if False:
        e = 0.4
        N = 7
        Det = DispElliptic(e=0.7, N=7)
        Det.plot_ellipse()
        FR = el.Fraser()
        k =  np.linspace(0.0001, 5/Det.geo["b"], 150)   
        w = np.linspace(0.1, 5e5, 100)
        Det.computeKWmap(k, w, adim=False)
        Det.plotDet( xy="KW", typep="contour", nature="imag", figname='cont_imag')
        Det.plotDet( xy="KW", typep="contour", nature="real", figname='cont_real')
        Det.plotDet( xy="KW", typep="sign", nature="imag", figname="sign_imag")
        Det.plotDet( xy="KW", typep="sign", nature="real", figname="sign_real")
        
        FR.plot(y='W', figname='cont_imag', e=[e])
        FR.plot(y='W', figname='cont_imag', branch=1, e=[e])
        
        FR.plot(y='W', figname='cont_real', e=[e])
        FR.plot(y='W', figname='cont_real', branch=1, e=[e])
        
        FR.plot(y='W', figname='sign_imag', e=[e])
        FR.plot(y='W', figname='sign_imag', branch=1, e=[e])
        
        FR.plot(y='W', figname='sign_real', e=[e])
        FR.plot(y='W', figname='sign_real', branch=1, e=[e])
        
    # %% Convergence --Résolution numérique
    if False:
        omega = np.linspace(0, 8e5, 200)  # ok pour e=0.5 (sauf N=7!)
        omega = np.linspace(0, 8e5, 300)  # ok pour e=0.2; 0.3 (sauf N=7!); 0.4 (sauf N=7!)
        omega = np.linspace(0, 8e5, 500)  # ok pour e=0.1 pour tous les N
        omega = np.linspace(0, 1e6, 500)   
        #omega = np.delete(omega, np.array([68, 69]))  # try to "jump" over difficulty!
        # omega = np.linspace(0, 5e6, 500)
        E = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        Nmax = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        # E = [0.9]
        # Nmax = [6]
        Nmin = 3
        
        DE = []
        njump =  {}
        # CALCUL
        for ee, nmax in zip(E, Nmax):
            print('='*10)
            print('e=%g'%ee)
            print('nmax=%i'%nmax)
            # Calcul de toutes les solutions jusqu'à N=Nmax
            DElist = []
            for nn in range(Nmin, nmax+1):
                print('N=%i'%nn)
                DEtemp = DispElliptic(e=ee, N=nn)
                # resolution numérique
                DEtemp.followBranch0(omega, itermax=20, jumpC2=0.004, interp='cubic')
                DElist.append(DEtemp)
            DE.append(DElist)
            njump[ee] = [de.b0['ignoredW'] for de in DElist]
        
        # AFFICHAGE
        error = True
        plt.close('all')
        for ee, nmax, DElist in zip(E, Nmax, DE):
            print('='*10)
            print('e=%g'%ee)
            print('nmax=%i'%nmax)
            
            # Plot dispersion curves
            plt.figure('dispcurveE%g'%ee)
            plt.title('$e=%g$'%ee)
            # plt.figure('gather')
            plt.axhline(y=1, color='0.8')
            for DEE in DElist:
                DEE.plotBranch0(x='W', y='C', figname='dispcurveE%g'%ee, label=DEE.geo['N'])
                # DEE.plotBranch0(x='W', y='C', figname='gather', label=DEE.geo['N'])
            plt.legend()
            plt.ylim(ymin=0.9, ymax=1.7)
            
            if error:
                # Take last dispersion curve (Nmax) as reference
                DEref = DElist[-1]
                wref, cref = DEref.getBranch0(x='W', y='C')
                
                # Plot relative error
                plt.figure('erreurE%g'%ee)
                plt.title('$e=%g$'%ee)
                plt.axhline(y=1e-3, color='0.8')
                plt.axhline(y=1e-6, color='0.8')
                plt.axhline(y=1e-9, color='0.8')
                for DEE in DElist[:-1]:
                    w, wlab, c, clab = DEE.getBranch0(x='W', y='C', label=True)
                    err = (c - cref)/cref
                    plt.semilogy(w, abs(err), '.', label=DEE.geo['N'])
                plt.legend()
                plt.ylabel('relative error on c/c2')
                plt.xlabel(wlab)
                # plt.xlim(xmax=4)
            
        plt.figure('gatherDispNmax')
        #fu.degrade(len(E))
        for ee, nmax, DElist in zip(E, Nmax, DE):
            DElist[-1].plotBranch0(x='W', y='C', ls='-', figname='gatherDispNmax', label=ee)
        plt.legend()
        plt.ylim(ymin=0.9, ymax=1.7)
                
        
        fu.savefigs(path='convergence')
