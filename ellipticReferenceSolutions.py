#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference solutions for the wave propagation in bars with elliptical cross
section.

Fraser, W. B. (1969). Dispersion of elastic waves in elliptical bars. 
*Journal of Sound and Vibration*, 10(2), 247‑260. 
https://doi.org/10.1016/0022-460X(69)90199-0



Created on Thu Jul  7 08:19:50 2022

@author: dbrizard
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


class Fraser:
    """Fraser, W. B. (1969). Dispersion of elastic waves in elliptical bars. 
    *Journal of Sound and Vibration*, 10(2), 247‑260. 
    https://doi.org/10.1016/0022-460X(69)90199-0
    
    """
    
    def __init__(self):
        """Instantiation methode. Nothing to provide. Loading data from txt files.

        
        """
        self.nu = 0.3
        kb0 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 
               2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 
               4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
        # kb_L0 = np.arange(0.0, 5.2, 0.2)
        kb1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 
               1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 
               3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
        # kb_L1 = [0.1*ii for ii in range(5, 21, 1)] + [0.1*ii for ii in range(22, 52, 2)]
        kb_T1 = np.arange(0.2, 5.2, 0.2)
        kb_T2 = np.arange(0.8, 5.2, 0.2)
        kb_B1 = np.arange(0.0, 5.2, 0.2)
        kb_B2 = np.arange(1.0, 5.2, 0.2)
        
        self.kb = {'L1':kb0, 'L2':kb1, 'T1':kb_T1, 'T2':kb_T2,
                   'Bx1':kb_B1, 'Bx2':kb_B2, 'By1':kb_B1, 'By2':kb_B2}
        e_LT = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        e_B = [0.4, 0.6, 0.8]
        self.e = {'L':e_LT, 'T':e_LT, 'B':e_B}

        # c/c2, longitudinal mode, 1st and 2nd branch
        L1 = np.genfromtxt('fraserTable2.txt', delimiter=',')  
        L2 = np.genfromtxt('fraserTable3.txt', delimiter=',')
        # c/c2, torsional mode, 1st and 2nd branch
        T1 = np.genfromtxt('fraserTable4.txt', delimiter=',')
        T2 = np.genfromtxt('fraserTable5.txt', delimiter=',')
        # c/c2, x Bending mode, 1st and 2nd branch
        Bx1 = np.genfromtxt('fraserTableBx1.txt', delimiter=',')
        Bx2 = np.genfromtxt('fraserTableBx2.txt', delimiter=',')
        # c/c2, y Bending mode, 1st and 2nd branch
        By1 = np.genfromtxt('fraserTableBy1.txt', delimiter=',')
        By2 = np.genfromtxt('fraserTableBy2.txt', delimiter=',')
 
        self.branches = {'L1':L1, 'L2':L2, 'T1':T1, 'T2':T2,
                         'Bx1':Bx1, 'Bx2':Bx2, 'By1':By1, 'By2':By2}
        self.fit = {}


    def plot(self, y='C', e=[], branch='L1', ls='.-', figname=None):
        """Plot c/c_2 or Omega wrt kb for a given branch, longitudinal mode only.
        
        :param str y: choose y axis ('C' or 'W')
        :param list e: ellipticity values to plot (all the available values if empty)
        :param str branch: branch id
        :param str ls: linestyle
        :param str figname: name for the figure
        """
        mode = branch[0]
        if len(e)==0:
            e = self.e[mode]
        
        plt.figure(figname)
        
        for ee in e:
            ind = self.e[mode].index(ee)
            K = self.kb[branch]
            if y=='C':
                ydata = self.branches[branch][:,ind]
            elif y=='W':
                ydata = self.branches[branch][:,ind] * K  # XXX probablement à une constante près...
            plt.plot(K, ydata, ls, label="%g (%s)"%(ee,branch))
        
        plt.legend(title='e (b):')
        plt.xlabel('kb')
        if y=='C':
            plt.ylabel('$c/c_2$')
        elif y=='W':
            plt.ylabel('$\\Omega=\\omega b/c_2$')

    
    def getBranch(self, e, branch='L1', x='K', y='C', labels=True):
        """Get the dispersion curve for a given value of excentricity
        
        :param float e: excentricity
        :param str branch: which branch to get ('L1', 'L2', 'T1', or 'T2')
        :param str x: xdata ('K', or 'W')
        :param str y: ydata ('C', or 'W')
        :param bool labels: if True, also return accompanying labels (x, y, xl, yl)
        """
        mode = branch[0]
        ind = self.e[mode].index(e)
        K = self.kb[branch]
        
        if x=='K':
            xdata = K
            xlabel = '$K=kb$'
        elif x=='W':
            xdata = self.branches[branch][:,ind] * K
            xlabel = '$W=?$'
        
        if y=='C':
            ydata = self.branches[branch][:,ind]
            ylabel = '$C=c/c_2$'
        elif y=='W':
            ydata = self.branches[branch][:,ind] * K  # XXX probablement à une constante près...
            ylabel = '$W=?$'
    
        if labels:
            return xdata, ydata, xlabel, ylabel
        else:
            return xdata, ydata
    
    
    def curveFitting(self, e=0.4, plot=True, fun='poly', figname=None):
        """Apply curve fitting to dispersion curve
        
        :param float e: excentricity
        :param bool plot: plot curve fitting
        :param str fun: function for curve fitting ('poly', 'cos')
        :param str figname: name for the figure
        """
        
        if fun=='poly':
            def func(x, a0, a1, a2, a3, a4, a5):
                """Polynomial function used for fitting the dispersion curve"""
                den = a2 * x*x*x*x + a3 * x*x*x + a4 * x*x + a5 * x**1.5 + 1
                return a0 + a1 / den
            p0 = [1]*6
        elif fun=='cos':
            def func(x, a0, a1, a2, a3, a4):
                """Try another function"""
                return a0 + a1*np.cos(a2 + a3*x + a4*x*x)
            p0 = [1]*5
            
        xdata, ydata, xl, yl = self.getBranch(e, x='K', y='C')
        
        popt, pcov = opt.curve_fit(func, xdata, ydata, p0=p0)  # , jac=jac)
        
        if fun=='poly':
            yfit = func(np.array(xdata), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
        elif fun=='cos':
            yfit = func(np.array(xdata), popt[0], popt[1], popt[2], popt[3], popt[4])
        
        self.fit[e] = {'xdata':xdata, 'ydata':ydata, 'yfit':yfit,
                       'popt':popt, 'pcov':pcov}
        
        if plot:
            plt.figure(figname)
            plt.plot(xdata, ydata, '.-', label='data')
            plt.plot(xdata, yfit, '+-', label='fit')
            plt.legend()
            plt.title('e=%g'%e)
            plt.xlabel(xl)
            plt.ylabel(yl)


if __name__=='__main__':
    plt.close('all')
    
    # %% Test Fraser class
    FR = Fraser()
    FR.plot(figname='L1')
    FR.plot(branch='L2',figname='L2')
    
    # Plot a specific value of e
    FR.plot(e=[0.4], figname='0.4')
    FR.plot(e=[0.4], figname='0.4', branch='L2', ls='+-')

    # Longitudinal mode
    FR.plot(figname='longi')
    FR.plot(figname='longi', branch='L2', ls='+-')
    
    FR.plot(y='W', figname='KW')
    FR.plot(y='W', figname='KW', branch='L2')
    
    # Torsional mode
    FR.plot(figname='torsional', branch='T1')
    FR.plot(figname='torsional', branch='T2', ls='+-')
    
    # Bending mode
    FR.plot(branch='Bx1', figname='Bx')
    FR.plot(branch='Bx2', figname='Bx')
    FR.plot(branch='By1', figname='By')
    FR.plot(branch='By2', figname='By')
    # all bending modes on the same figure
    FR.plot(branch='Bx1', figname='bending')
    FR.plot(branch='By1', ls='+-', figname='bending')
    FR.plot(branch='Bx2', figname='bending')
    FR.plot(branch='By2', ls='+-', figname='bending')    
    
    # %% TRY CURVE FITTING
    # FR.curveFitting(e=0.4, plot=True, figname='CF0.4')  # error, missing values
    FR.curveFitting(e=0.5, plot=True, figname='CF0.5')
    FR.curveFitting(e=0.8, plot=True, figname='CF0.8')
