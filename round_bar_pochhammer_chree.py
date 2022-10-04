#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to handle the dispersion of elastic waves in round bar according to
Pochhammer-Chree equation for the longitudinal mode of propagation.

Created on Fri Sep  2 15:03:21 2022

@author: dina.zyani
@author: denis.brizard
"""

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
import time
import warnings
from scipy.interpolate import interp1d


def f_Pochhammer_Chree(k, omega, lambda_, mu, rho, a):
    """Pochhammer-Chree characteristic function for round bars
    
    :param float k: wavenumber
    :param float omega: circular frequency
    :param float lamdba_: Lame constant $\lambda$
    :param float mu: Lame constant $\mu$
    """
    k_2 = k**2
    # alpha
    alpha_2 = (rho * omega**2) / (lambda_ + 2 * mu) - k_2
    alpha = np.lib.scimath.sqrt(alpha_2)
    alpha_a = alpha * a
    # beta
    beta_2 = rho * omega**2 / mu - k_2
    beta = np.lib.scimath.sqrt(beta_2)
    beta_a = beta * a

    A = (2 * alpha / a) * (beta_2 + k_2) * jv(1, alpha_a) * jv(1, beta_a)
    B = -((beta_2 - k_2) ** 2) * jv(0, alpha_a) * jv(1, beta_a)
    C = -4 * k_2 * alpha * beta * jv(1, alpha_a) * jv(0, beta_a)

    f = A + B + C
    return f


class DetDispEquation:
    """A class to handle the Pochhammer-Chree characteristic equation.
    
    """

    def __init__(self, nu=0.3317, E=210e9, rho=7800., a=0.05):
        """Instantiate bar with given parameters
        
        :param float nu: Poisson's ratio [-]
        :param float E: modulus of elasticity [Pa]
        :param float rho: density [kg/m3]
        :param float a: radius of the bar
        """
        la = E * nu / ((1 + nu) * (1 - 2 * nu))  # coef de Lamé
        mu = E / (2 * (1 + nu))  # coef de Lamé
        self.mat = {'E':E, 'rho':rho, 'nu':nu, 'la':la, 'mu':mu}
        self.c = {'co': np.sqrt(E/rho), 'c_2':np.sqrt(mu/rho)}
        self.a = a
        # detfun0 = lambda k,w:f_characteristic(k, w, lambda_=la, mu=mu, rho=rho, a=a)
        # self.detfun0 = detfun0

        def detfun(k, w):
            """Characteristic determinant function."""
            return f_Pochhammer_Chree(k, w, la, mu, rho, a)

        self.detfun = detfun
        self.vectorized = True  # detfun is vectorized
        self.dim = {'c':self.c['co'], 'l':a}
        self.dimlab = {'c':'c_0', 'l':'a'}
        
        self._nature = self._defineAutoReIm4map()


    def _defineAutoReIm4map(self):
        """Define if Re or Im part of characteristic equation should used for
        sign maps
        
        """
        return 'real'


    def computeKWmap(self, k, w, adim=True, verbose=True):
        """Compute the value of the characteristic equation on a (k,w) grid

        :param array k: wavenumbers
        :param array w: circular frequency
        :param bool adim: True if the given input arguments are dimensionless
        :param bool verbose: print progress and computation time
        """
        if adim:
            K = k
            W = w
            w = w * self.dim['c'] / self.dim['l']
            k = k / self.dim['l']
        else:
            K = k * self.dim['l']
            W = w * self.dim['l'] / self.dim['c']

        Z = np.empty((len(w), len(k)), dtype=np.complex128)
        start = time.clock()
        if self.vectorized:
            for ii, om in enumerate(w):
                Z[ii, :] = self.detfun(k, om)
        else:
            for ii, om in enumerate(w):
                if ii%10==0 and verbose:
                    print('%i/%i'%(ii, len(w)))
                for jj, kk in enumerate(k):
                    Z[ii, jj] = self.detfun(kk, om)

        end = time.clock()
        temps = end - start
        npts = len(w)*len(k)
        if verbose:
            print('#'*35)
            print('Took %g s to compute %i points'%(temps, npts))
            print('%g ms/pt'%(temps/npts*1e3))
            print('#'*35)
        self.kw = {"w": w, "k": k, "det": Z, "K": K, "W": W}


    def computeWCmap(self, w, c, adim=True, verbose=True):
        """Compute the value of the characteristic equation on a (w,c) grid

        :param array w: circular frequency
        :param array c: velocity
        :param bool adim: True if the given input arguments are dimensionless
        :param bool verbose: print progress and computation time
        """
        if adim:
            C = c
            W = w
            w = w * self.dim['c'] / self.dim['l']
            c = self.dim['c'] * c
        else:
            C = c / self.dim['c']
            W = w * self.dim['l'] / self.dim['c']
        Z = np.empty((len(c), len(w)), dtype=np.complex128)
        start = time.clock()
        
        if self.vectorized:
            for ii, om in enumerate(w):
                kk = om / c
                Z[:, ii] = self.detfun(kk, om)
        else:
            for ii, om in enumerate(w):
                if ii%10==0 and verbose:
                    print('%i/%i'%(ii,len(w)))
                for jj, kk in enumerate(om / c):
                    Z[jj, ii] = self.detfun(kk, om)
        
        end = time.clock()
        temps = end - start
        npts = len(w)*len(c)
        if verbose:
            print('#'*35)
            print('Took %g s to compute %i points'%(temps, npts))
            print('%g ms/pt'%(temps/npts*1e3))
            print('#'*35)
        self.wc = {"w": w, "c": c, "det": Z, "C": C, "W": W}


    def computeKCmap(self, k, c, adim=True, verbose=True):
        """Compute the value of the characteristic equation on a (k,c) grid

        :param array k: wavenumbers
        :param array c: velocity
        :param bool adim: True if the given input arguments are dimensionless
        :param bool verbose: print progress and computation time
        """
        if adim:
            K = k
            C = c
            c = c*self.dim['c']
            k = k/self.dim['l']
        else: 
            K = k*self.dim['l']
            C = c/self.dim['c']
        Z = np.empty((len(c), len(k)), dtype=np.complex128)
        start = time.clock()
        
        if self.vectorized:
            for ii, kk in enumerate(k):
                om = kk*c
                Z[:,ii] = self.detfun(kk, om)
            print('not sure')
        else:
            for ii, cc in enumerate(c):
                if ii%10==0 and verbose:
                    print('%i/%i'%(ii,len(c)))
                for jj, kk in enumerate(k):
                    om = kk*cc
                    Z[ii, jj] = self.detfun(kk, om)
                
        end = time.clock()
        temps = end - start
        npts = len(k)*len(c)
        if verbose:
            print('#'*35)
            print('Took %g s to compute %i points'%(temps, npts))
            print('%g ms/pt'%(temps/npts*1e3))
            print('#'*35)

        self.kc = {"c": c, "k": k, "det": Z, "K": K, "C": C}


    def plotDet(self, xy="WC", typep="contour", nature="imag", level=[0],
                figname=None, title=None, colors=None):
        """Plot value of characteristic function on a grid

        :param str xy: name of the grid ('WC', or 'KW')
        :param str typep: type of plot ('contour', 'sign', or 'log')
        :param str nature:  ('real', 'imag', or 'abs')
        :param list level: level(s) for the contour plot
        :param str figname: name for the figure
        :param str title: plot title
        :param coul color: color for the contour plot
        """
        if xy=="WC":
            x = self.wc["W"]
            y = self.wc["C"]
            det = self.wc["det"]
            xlabel = "$\\Omega=\\omega %s/%s$[-]"%(self.dimlab['l'], self.dimlab['c'])
            ylabel = "$C=c/%s [-]$"%self.dimlab['c']
        elif xy=="KW":
            x = self.kw["K"]
            y = self.kw["W"]
            det = self.kw["det"]
            xlabel = "K [-]"
            ylabel = "$\\Omega = \\omega %s/%s [-]$"%(self.dimlab['l'], self.dimlab['c'])
        elif xy=='KC':
            x = self.kc['K']
            y = self.kc['C']
            det = self.kc['det']
            xlabel = 'K [-]'
            ylabel = 'C [-]'
        
        if nature=='auto':
            nature = self._nature
            

        plt.figure(figname)
        if typep=="contour":
            if nature=="real":
                data = det.real
            elif nature=="imag":
                data = det.imag
            elif nature=="abs":
                data = abs(det)

            CS = plt.contour(x, y, data, level, 
                             colors=colors, linewidths=1)
            # CL = plt.clabel(CS, fmt='%g')

        elif typep=="sign":
            if nature=="real":
                data = np.sign(det.real)
            elif nature=="imag":
                data = np.sign(det.imag)
            else:
                return  # essaie de tracer qd même

            plt.pcolormesh(x, y, data, shading="auto", cmap="cool")

        elif typep=="log":
            if nature=="real":
                data = np.log10(det.real)
            elif nature=="imag":
                data = np.log10(det.imag)
            elif nature=="abs":
                data = np.log10(det)

            plt.pcolormesh(x, y, data, shading="auto", cmap="cool")
            plt.colorbar()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        else:
            plt.title(typep + "(" + nature + "(det))")
        #plt.title("%s(%s(det)), N=%i, e=%g"%(typep, nature, self.geo['N'], self.geo['e']))
        #plt.ylim(ymin=0)
        
        if typep=='contour':
            return CS
        else:
            plt.xlim((x.min(), x.max()))
            plt.ylim((y.min(), y.max()))


    def plotDet_KC(self, xy="KC", typep="contour", nature="imag", level=[0], 
                   figname=None, adim=True, colors='b', lw=1):
        """Tracer les courbes de dispersion.
        
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
        if typep=="contour":
            # level = [0]
            if nature=="real":
                data = det.real
            elif nature=="imag":
                data = det.imag
            elif nature=='abs':
                data = abs(det)
            elif nature=='quotient_abs':
                data = abs(det.real)/abs(det.imag)
            CS = plt.contour(x, y, data, level, colors=colors, linewidths=lw)
            # CL = plt.clabel(CS, fmt='%g')
        elif typep=="sign":
            if nature=="real":
                data = np.sign(det.real)
            elif nature=="imag":
                data = np.sign(det.imag)
            plt.pcolormesh(x, y, data, shading="auto", cmap="cool", rasterized=True)
            # plt.colorbar()
            
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ymax=2.)
        plt.title(typep + "(" + nature + "(det))")
        
        if typep=='contour':
            return CS


    def followBranch0(self, w, itermax=20, extrap='quadratic', jumpC2=None, interp=None):
        """Follow first branch of longitudinal mode.
        
        Numerical solving with *regula falsi* method.
        
        :param array w: circular frequencies
        :param int itermax: maximum number of iterations
        :param str extrap: kind of extrapolation (see :func:`prediction`)
        :param float jumpC2: dead zone where no points are computed (suggested value is 0.004)
        :param str interp: interpolate on ignored dead zone points 
        """
        if not w[0]==0.0:
            print('w[0] should be equal to 0')
            return
        
        ind_skp = []  # indices of skipped points
        ind_kpt = [0]  # indices of kept points
        W = [0]
        K = [0]
        RES = [0]
        NIT = [0]
        KPRED = []
        k_pred = w[1]/self.c['co'] + 0j  # is this line useful?
        KPRED.append(k_pred)
        
        for ii, ww in enumerate(w[1:]):
            # Prectiction step
            k_pred = prediction(W, K, ww, self.c['co'], 'quadratic', verbose=True)
            correction = True  # suppose we will do the correction step
            
            # Check if we enter the dangerous zone (around c/c2=1)
            if jumpC2 is not None:
                c_pred = ww/k_pred
                if c_pred<self.c['c_2']*(1+jumpC2) and c_pred>self.c['c_2']*(1-jumpC2):
                    print('w=%g. Prediction too close to c_2. Skipping point.'%ww)
                    ind_skp.append(ii+1)
                    correction = False
                else:
                    ind_kpt.append(ii+1)
            
            # Correction step
            if correction:
                func = lambda k: self.detfun(k, ww)
                def fun(kk):
                    return self.detfun(kk, ww)
                ksol, nit, res = regulaFalsi(fun, None, k_pred, verbose=True, 
                                             itermax=itermax, eps=1e-14)
                W.append(ww)
                K.append(ksol)
                KPRED.append(k_pred)
                RES.append(res)
                NIT.append(nit)
                # predict solution for next w = previous value of k
                #k_pred = ksol.real + 0j
        
        
        # Interpolate on dangerous zone points
        if jumpC2 is not None and interp is not None:
            print('Interpolation on %i skipped points'%len(ind_skp))
            interp_fun = interp1d(W, K, kind=interp, assume_sorted=True,
                                  fill_value='extrapolate')
            k = np.zeros(len(w))
            k[ind_kpt] = np.array(K)
            k[ind_skp] = interp_fun(w[ind_skp])  # interpolate ONLY on skipped points
            # k = interp_fun(w)
            w_ = w
        else:
            k = np.array(K)
            w_ = np.array(W)  # may be less points than initially asked   

        c = w_/k
        c[0] = self.c['co']
        self.b0 = {'w':w_, 'k':np.array(K), 'c':c, 'k':k,
                   'res':np.array(RES), 'nit':NIT, 'k_pred':np.array(KPRED),
                   'ind_skipped':ind_skp, 'ignoredW':len(ind_skp)}
    
    
    def getBranch0(self, x='w', y='c', label=False):
        """Get dispersion curve of first branch computed with :meth:`DetDispEquation.followBranch0`
        
        Upper case variables are dimensionless.
        Lower case variables are NOT dimensionless
        
        Returns either "x, y" or "x, xlab, y, ylab"
       
        :param str x: x variable ('w', 'W', 'k', '-')
        :param str y: y variable ('c', 'C')
        :param bool label: also return the corresponding labels
        """
        if x=='w':
            xlab = '$\\omega$ [rad/s]'
            x = self.b0['w']
        elif x=='W':
            xlab= '$\\Omega=\\omega %s/%s$ [-]'%(self.dimlab["l"], self.dimlab["c"])
            x = self.b0['w']*self.dim['l']/self.dim["c"]
        elif x=='k':
            xlab = '$k$ [1/m]'
            x = self.b0['k']
        elif x=='-':
            xlab = 'index'
            x = np.arange(len(self.b0['w']))
            
            
        if y=='c':
            ylab = '$c$ [m/s]'
            y = self.b0['c']
        elif y=='C':
            ylab = '$c/%s$ [-]'%self.dimlab['c']
            y = self.b0['c']/self.dim['c']
            
        
        if label:
            return x, xlab, y, ylab
        else:
            return x, y
        
        
    def plotBranch0(self, x='w', y='c', ls='.-', figname=None, label=None):
        """Plot the first branch computed by :meth:`DetDispEquation.followBranch0`
        
        :param str x: x variable ('w', 'W', 'k', '-')
        :param str y: y variable ('c', 'C')
        :param str ls: linestyle
        :param str figname: name for the figure 
        :param str label: label for the curve
        """
        x, xlab, y, ylab = self.getBranch0(x=x, y=y, label=True)
        
        plt.figure(figname)
        plt.plot(x, y, ls, label=label, lw=1)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
    
    
    def plotFollow(self, pred=True):
        """Plot dispersion curve, residue and number of iterations (Regula Falsi algo)
        
        :param bool pred: also plot predicted solutions        
        """
        plt.figure()
        ax = plt.subplot(311)
        plt.ylabel('c [m/s]')
        plt.plot(self.b0['w'], self.b0['c'], '.-')
        
        plt.subplot(312, sharex=ax)
        plt.ylabel('Residue')
        plt.semilogy(self.b0['w'], abs(self.b0['res'].real), 'b.-', label='real')
        plt.semilogy(self.b0['w'], abs(self.b0['res'].imag), 'g.-', label='imag')
        plt.semilogy(self.b0['w'], abs(self.b0['res']), 'r.-', label='abs')
        plt.legend()
        
        plt.subplot(313, sharex=ax)
        plt.ylabel('$N_{iter}$')
        plt.plot(self.b0['w'], self.b0['nit'], '.-')
        plt.xlabel('$\omega$ [rad/s]')
        
        if pred:
            plt.figure()
            plt.plot(self.b0['k'], self.b0['w'], '.-', label='sol')
            plt.plot(self.b0['k_pred'], self.b0['w'], '.-', label='pred')
            plt.legend()
            plt.xlabel('k')
            plt.ylabel('w')

        
    
        
def regulaFalsi(func, dfunc, xO, eps=1e-12, itermax=100, verbose=False, plot=False):
    """Regula falsi method, from Othman 2021.
    
    Seems to be more efficient on first branch of hollow bar than :func:`newton`.
    
    Same arguments as :func:`newton`, some arguments are therefore UNUSED here.
    
    :param func func: function of one variable whose roots are sought
    :param func dfunc: derivative of the function wrt the variable. UNUSED
    :param float xO: initial guess of root
    :param float eps: stop iteration if change in XI smaller than eps
    :param int itermax: maximum number of iteration (a warning is raised if reached)
    :param bool verbose: if True, also return number of iterations and value of function (=residue)
    :param bool plot: UNUSED.  
    """
    xi0 = xO
    xi1 = 1.0001*xO
    
    xip = xi1  # previous
    xipp = xi0  # previous previous
    fxip = func(xip)
    fxipp = func(xipp)
    
    cpt = 0
    xi = xi0  # in case we do not enter while loop
    while abs(xip - xipp)>eps and cpt<itermax:
        xi = xip - fxip*(xip - xipp)/(fxip - fxipp)
        cpt +=1
        xipp = xip
        xip = xi
        fxipp = fxip
        fxip = func(xip)
        
    if cpt==itermax:
        warnings.warn('Reached maximum number of iterations of %i'%itermax)
    
    if verbose:
        return xi, cpt, fxip
    else:
        return xi   


def prediction(WW, XI, wp, c_, extrap, verbose, speedPred=True):
    """Predict solution from previous steps (extrapolation).
    
    :param list W: increasing list of circular frequencies, up to previous step
    :param list XI: increasing list of solutions, up to previous step
    :param float wp: circular frquency at which extrapolate solution
    :param float c_: approximate value of wave velocity for the first frequency point
    :param str extrap: extrapolation type ('co', 'last', 'linear', 'quadratic')
    :param int verbose:
    :param bool speedPred: avoid using costly polyfit function (recommanded!)
    """
    # ---CHECK THERE ARE ENOUGH POINTS FOR THE REQUIRED EXTRAPOLATION METHOD---
    if len(WW)==0:
        # ---FIRST POINT, EXTRAPOLATION NOT POSSIBLE---
        guess = 'co'
    elif len(WW)==1:
        # ---SECOND POINT, ONLY ONE AVAILABLE FOR EXTRAP---
        guess = 'co'
        # 'last' # last works for bissec solving... Why???
    elif len(WW)==2:
        # ---THIRD POINT, ONLY 2 PREVIOUS PTS AVAILABLE---
        guess = 'linear'
    else:
        guess = extrap
    
    
    # ---EXTRAPOLATE FROM PRVIOUS POINTS---
    if guess=='co':  # or len(XI)<2: #ind in (0,1):
        # ---GUESS FROM GIVEN VALUE OF VLOCITY---
        # mainly usefull for starting point or near omega=0
        if False:
            xio = wp/c_
        else:
            xio = wp/c_ + 0j
        if verbose>1:
            print("co guess")
            
    elif guess=='last':  # or len(XI)==1: #and len(XI)>0: #ind>0:
        # ---LAST POINT GUESS---
        xio = XI[-1]
        if verbose>1:
            print("LAST guess")
            
    elif guess=='linear':  # or len(XI)==2: #ind==2:
        # ---LINEAR EXTRAPOLATION FROM LAST TWO POINTS---
        if speedPred:
            p = (XI[-1] - XI[-2])/(WW[-1] - WW[-2])
            xio = p*(wp - WW[-1]) + XI[-1]
        else:
            p = np.polyfit(WW[-2:], XI[-2:], 1)
            xio = np.polyval(p, wp)  # ??
        if verbose>1:
            print("linear guess")
            
    elif guess in ('poly2', 'quadratic'):
        # ---QUADRATIC EXTRAPOLATION FROM LAST THREE POINTS---
        if speedPred:
            # y = ax2 + bx + c
            a1 = (XI[-1] - XI[-3])/(WW[-1] - WW[-3])/(WW[-1] - WW[-2])
            a2 = (XI[-2] - XI[-3])/(WW[-2] - WW[-3])/(WW[-1] - WW[-2])
            a = a1 - a2
            b = (XI[-2] - XI[-3])/(WW[-2] - WW[-3]) - a*(WW[-2] + WW[-3])
            c = XI[-3] - a*WW[-3]*WW[-3] - b*WW[-3]
            # now prediction
            xio = a*wp*wp + b*wp + c
        else:
            p = np.polyfit(WW[-3:], XI[-3:], 2)
            xio = np.polyval(p, wp)  # ??
        if verbose>1:
            print("quadratic guess")

    return xio

if __name__=="__main__":
    plt.close("all")
    
    # %% TEST CLASS
    Det = DetDispEquation()
    
    # WC map
    if False:
        Det.computeWCmap(w=np.linspace(0.0001, 1245298.85199065, 500),
                         c=np.linspace(0.5 * Det.c["co"], 4 * Det.c["co"], 200),
                         adim=False)
        for typep in ("contour", "sign", "log"):
            for nature in ("real", "imag", "abs", 'auto'):
                    Det.plotDet(xy="WC", typep=typep, nature=nature, colors='g')
    
    # KW map
    if False:
        Det.computeKWmap(k=np.linspace(0, 200, 300, dtype=np.complex128),
                         w=np.linspace(0, 1e6, 500),
                         adim=False)
        for typep in ("contour", "sign", "log"):
            for nature in ("real", "imag", "abs", 'auto'):
                    Det.plotDet(xy="KW", typep=typep, nature=nature, colors='g')
    
    # KC map
    if True:
        Det.computeKCmap(k=np.linspace(0, 200, 300, dtype=np.complex128),
                         c=np.linspace(0.5 * Det.c["co"], 4 * Det.c["co"], 200),
                         adim=False)
        for typep in ("contour", "sign", "log"):
            for nature in ("real", "imag", "abs", 'auto'):
                    Det.plotDet(xy="KC", typep=typep, nature=nature, colors='g')        

    # %% Test follow Branch
    if True:
        omega = np.linspace(0, 8e5, 500)  # Ok pour k<1.208
        Det.followBranch0(omega, itermax=20)
        Det.plotFollow()

        plt.figure('sign_imag')
        plt.plot(Det.b0['k']*Det.dim['l'], Det.b0['c']/Det.dim['c'], '+-', label='Regula Falsi algo')
        Det.plotBranch0(y='C')
        Det.plotBranch0(y='C', x='W')
        plt.legend()

                
                
                
                
