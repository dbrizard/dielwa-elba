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
from scipy.linalg import null_space
import matplotlib.pyplot as plt

import warnings
import time
import pickle

import round_bar_pochhammer_chree as round_bar
import ellipticReferenceSolutions as el
from fraser_elliptical import characteristic_matrix

import figutils as fu

def char_func_elliptic_fortran(k, w, R, theta, gamma, c_1, c_2, mode='L',
                               rEturn='det'):
    """Characteristic function for elliptical bar, with underlying Fortran code
    
    :param float k: wavenumber
    :param float w: circular frequency
    :param array R: radius of collocation points
    :param array theta: angle of collocation points
    :param array gamma: angle of normal at collocation points
    :param float c_1: velocity
    :param float c_2: velocity
    :param str mode: wave propagation mode ('L', 'T', 'Bx', 'By')
    :param str rEturn: 'det' or 'matrix'
    """
    N = len(theta)
    A = characteristic_matrix(k, w, N, R, theta, gamma, c_1, c_2, mode)
    if mode=='L':
        B = A[1:, 1:]
        detA = np.linalg.det(B)
    elif mode=='T':
        ind = list(range(3*N))
        ind.remove(N)   # XXX reason why indices N and 2*N ?
        ind.remove(2*N) # ditto
        B = A[np.ix_(ind,ind)]
        detA = np.linalg.det(B)
    elif mode in ('Bx', 'By'):
        B = A
        detA = np.linalg.det(A)
    
    if rEturn=='det':
        return detA
    elif rEturn=='matrix':
        return A, B



def char_func_elliptic(k, w, R, theta, gamma, c_1, c_2):
    """Characteristic function for elliptical bar
    
    :param float k: wavenumber
    :param float w: circular frequency
    :param array R: radius of collocation points
    :param array theta: angle of collocation points
    :param array gamma: angle of normal at collocation points
    :param float c_1: velocity
    :param float c_2: velocity
    """
    N = len(theta)
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
        c_1 = c_2*np.sqrt(2*(1-nu)/(1-2*nu))  # Fraser eq(7)
        self.mat = {'E':E, 'rho':rho, 'nu':nu, 'mu':mu, 'la':la}

        self.c = {'c_2':c_2, 'c_1':c_1, 'co': np.sqrt(E/rho)}
        b = a*np.sqrt(1-e**2)  # Fraser eq(4) 
        self.a = a #b
        self.geo = {'e': e, 'b': b, 'N': N,'a':a}
        
        # Collocation points coordinates and midway angles
        m = np.arange(1, N+1)
        if mode in ('L', 'T'):
            theta = (m-1)*np.pi/2/N
            theta_mid = (m-0.5)*np.pi/2/N
        elif mode in ('Bx', 'By'):
            theta = (m-0.5)*np.pi/2/N
            theta_mid = m*np.pi/2/N
            
        e2 = e**2
        
        def compute_gamma_R(theta, b=b):
            cos2 = np.cos(theta)**2
            gamma = -np.arctan((e2*np.sin(2*theta))/(2*(1-e2*cos2)))  # Fraser eq(6)
            R = b*np.sqrt(1/(1 - e2*cos2))  # Frser eq(5)
            return gamma, R
        
        gamma, R = compute_gamma_R(theta)
        self.ellipse = {'theta':theta, 'gamma':gamma, 'R':R}
        gamid, Rmid = compute_gamma_R(theta_mid)
        self.midpoints = {'theta':theta_mid, 'gamma':gamid, 'R':Rmid}
        
        # Characteristic function
        if mode in ('T', 'Bx', 'By') and fortran is False:
            fortran = True
            print('Forcing Fortran=True because equations for %s mode are not written in Python'%mode)
        if fortran:
            def detfun(k, w):
                """Characteristic determinant function."""
                return char_func_elliptic_fortran(k, w, R, theta, gamma, 
                                                  c_1, c_2, mode=mode, rEturn='det')
        else:
            def detfun(k, w):
                """Characteristic determinant function."""
                return char_func_elliptic(k, w, R, theta, gamma, c_1, c_2)
        self.detfun = detfun
        self.vectorized = False  # detfun is not vectorized
        self.dim = {'c':c_2, 'l':b}  # for dimensionless variables
        self.dimlab = {'c':'c_2', 'l':'b'}  # name of dimensionless variables for use un labels
        self.mode = mode
        
        self._nature = self._defineAutoReIm4map()
        
        # Also define some other useful functions
        if fortran:
            def matrix(k, w):
                """Characteristic matrix, on collocation points"""
                return char_func_elliptic_fortran(k, w, R, theta, gamma, 
                                                  c_1, c_2, mode=mode, rEturn='matrix')
            def matrix2(k, w, theta, b):
                """Characteristic matrix, for other points"""
                gamma, R = compute_gamma_R(theta, b)
                return char_func_elliptic_fortran(k, w, R, theta, gamma, 
                                                  c_1, c_2, mode=mode, rEturn='matrix')
            self._matrix = matrix
            self._matrix2 = matrix2
            
        
    def _defineAutoReIm4map(self):
        """Define if Re or Im part of characteristic equation should used for
        sign maps
        
        """
        if self.mode in ('L', 'T'):
            if self.geo['N']%2==1:
                if self.mode in ('L'):
                    nat = 'imag'
                elif self.mode in ('T'):
                    nat = 'real'
            elif self.geo['N']%2==0:
                if self.mode in ('L'):
                    nat = 'real'
                elif self.mode in ('T'):
                    nat = 'imag'
        elif self.mode in ('Bx', 'By'):
            nat = 'real'
        return nat
        
    
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
        

    def computeABCDEF(self, ind, rcond=1e-9, plot=True):
        """
        
        """
        k, w = self.getBranch0(x='k', y='w')
        
        Acp, Bcp = self._matrix(k[ind], w[ind])  # collocation points
        Amp, Bmp = self._matrix2(k[ind], w[ind], self.midpoints['theta'], self.geo['b'])
        Aro, Bro = self._matrix2(k[ind], w[ind], self.midpoints['theta'], 0)  # center point
         
        RES = []
        ZZ = []
        for ii, AA in enumerate((Aro, Acp, Amp)):
            Z = null_space(AA, rcond=rcond)
            temp = AA*Z[:,-2]  # last vector should be the right one
            
            N = self.geo['N']
            if self.mode in ('L'):
                tau_t = temp[:N, :]
                sig_n = temp[N:2*N, :]
                tau_z = temp[2*N:, :]
            
            if ii==0:
                # these are the reference values at center point of section
                residue = {'tau_t': tau_t.sum(axis=1), 
                           'tau_z': tau_z.sum(axis=1), 
                           'sig_n': sig_n.sum(axis=1)}
            else:
                # normalize wrt center point
                residue = {'tau_t': tau_t.sum(axis=1)/RES[0]['tau_t'], 
                           'tau_z': tau_z.sum(axis=1)/RES[0]['tau_z'], 
                           'sig_n': sig_n.sum(axis=1)/RES[0]['sig_n']}
                
            RES.append(residue)
            ZZ.append(Z)
        
        self.residue = RES
        labels = {'tau_t':'$\\tau_t$', 'tau_z':'$\\tau_z$', 'sig_n':'$\\sigma_n$'}
        colors = {'tau_t':'C1', 'tau_z':'C2', 'sig_n':'C0'}
        
        if plot:
            LS = ('+--', '.-')
            THETA = (self.ellipse['theta'], self.midpoints['theta'])
            for ls, res, theta in zip(LS, RES[1:], THETA):
                for kk in ('sig_n', 'tau_t', 'tau_z'):
                    plt.figure('abs')
                    plt.plot(theta, abs(res[kk]), ls, color=colors[kk], label=labels[kk])
                    plt.figure('REAL')
                    plt.plot(theta, np.real(res[kk]), ls, color=colors[kk], label=labels[kk])
                    plt.figure('imag')
                    plt.plot(theta, np.imag(res[kk]), ls, color=colors[kk], label=labels[kk])

            ticks = np.arange(0, 5, 1)*np.pi/8
            tickslabels = ['0', '$\\pi/8$', '$\\pi/4$', '$3\\pi/8$', '$\\pi/2$']
            for fig in ('abs', 'REAL', 'imag'):
                plt.figure(fig)
                plt.legend(title=fig)
                plt.xticks(ticks, tickslabels)

        
if __name__ == "__main__":
    plt.close("all")
    
    # %% Numerical solving: FOLLOW FIRST BRANCH
    if False:
        e = 0.8
        N = 4
        modes = ('L', 'T', 'Bx', 'By')
        # mode = 'By'
        for mode in modes:
            Det = DispElliptic(e=e, N=N, mode=mode)
            omega = np.linspace(0, 8e5, 500)  # Ok pour k<1.208
            # omega = np.linspace(0, 1e6, 4000)  # trying very small step. Ok pour k<1.2182
            
            follow = False
            FR = el.Fraser()
            if follow:
                Det.followBranch0(omega, itermax=20)
                Det.plotFollow()
        
                plt.figure('sign_imag')
                plt.plot(Det.b0['k']*Det.geo['b'], Det.b0['c']/Det.c['c_2'], '+-', label='Regula Falsi algo')
                # plt.ylim(ymax=1.7, ymin=0.7)
                # plt.xlim(xmin=0., xmax=5)
        
                FR.plot(e=[e], figname='sign_imag')
            
            Det.computeKCmap(k=np.linspace(0, 5, 100), c=np.linspace(0.6, 2.2, 100), adim=True)

            Det.plotDet('KC', typep='sign', nature='auto', figname=mode)
            FR.plot(e=[e], figname=mode, branch=mode+'1', x='K', y='C')
            FR.plot(e=[e], figname=mode, branch=mode+'2', x='K', y='C')
        
    
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
        
    #%% Case e=0, comparison with round bar
    if False:
        e=0
        

    #%% Compute maps for all 4 modes
    if False:
        modes = ['L']
        modes = ['L', 'T', 'Bx', 'By']
        N = [3, 4, 5]
        N = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        e = 0.9
        for mode in modes:
            print('\n\n'+'*'*10+' mode=%s '%mode+'*'*10)
            DET = []
            # COMPUTE
            for nn in N:
                print('='*10+'N=%i'%nn+'='*10)
                Det = DispElliptic(e=e, N=nn, fortran=True, mode=mode)
                k =  np.linspace(0.0001, 5/Det.geo["b"], 150)
                c = np.linspace(0.7*Det.c["c_2"], 2.*Det.c["c_2"], 100)
                Det.computeKCmap(k, c, adim=False, verbose=False)
                DET.append(Det)
            
            # PLOT
            for ddet in DET:
                title = '%s mode, e=%g, N=%i'%(mode, ddet.geo['e'], ddet.geo['N'])
                fn = '%s-e%02gN%02i'%(ddet.mode, ddet.geo['e']*10, ddet.geo['N'])
                ddet.plotDet(xy='KC', typep='sign', nature='real',
                             title=title+', real', figname=fn+'-r')
                ddet.plotDet(xy='KC', typep='sign', nature='imag', 
                             title=title+', imag', figname=fn+'-i')
            fu.savefigs(path='maps/mode%s'%mode, prefix='map', close=True, overw=True)
            # gather pdf figures with pdfjam: 
            # pdfjam modeT/*.pdf --nup 2x5 --outfile mapsT.pdf
        


    #%% Domaine KW
    if False:
        e = 0.4
        N = 7
        Det = DispElliptic(e=0.7, N=7)
        Det.plot_ellipse()
        FR = el.Fraser()
        k = np.linspace(0.0001, 5/Det.geo["b"], 150)   
        w = np.linspace(0.1, 5e5, 100)
        Det.computeKWmap(k, w, adim=False)

        Det.plotDet(xy="KW", typep="sign", nature="imag", figname="sign_imag")
        FR.plot(x='K', y='W', figname='sign_imag', e=[e])
        FR.plot(x='K', y='W', figname='sign_imag', branch='L2', e=[e])
        # XXX curves do not overlay... !

        
    # %% Convergence --Résolution numérique
    if False:
        omega = np.linspace(0, 8e5, 200)  # ok pour e=0.5 (sauf N=7!)
        omega = np.linspace(0, 8e5, 300)  # ok pour e=0.2; 0.3 (sauf N=7!); 0.4 (sauf N=7!)
        omega = np.linspace(0, 8e5, 500)  # ok pour e=0.1 pour tous les N
        omega = np.linspace(0, 1e6, 500)   
        #omega = np.delete(omega, np.array([68, 69]))  # try to "jump" over difficulty!
        # omega = np.linspace(0, 5e6, 500)
        E = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        E = [0., 0.1, 0.2, 0.3]
        Nmax = [7]*len(E)
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
                
        
        fu.savefigs(path='convergence', overw=False)
    
    #%% Compute residual stress between collocation points
    if True:
        Det = DispElliptic(e=0.5, N=6, mode='L')
        omega = np.linspace(0, 8e5, 500) 
        Det.followBranch0(omega, itermax=20, jumpC2=0.004, interp='cubic')
        Det.plotFollow()
        Det.computeABCDEF(10)
        
        
