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
        self.e = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        kb0 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 
               2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 
               4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
        kb1 = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 
               1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 
               3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
        kb_T1 = np.arange(0.2, 5.2, 0.2)
        kb_T2 = np.arange(0.8, 5.2, 0.2)
        
        self.kb = {'L1':kb0, 'L2':kb1, 'T1':kb_T1, 'T2':kb_T2}

        L1 = np.genfromtxt('fraserTable2.txt', delimiter=',')  # c/c2, first branch, longitudinal mode
        L2 = np.genfromtxt('fraserTable3.txt', delimiter=',')  # c/c2, second branch, longitudinal mode
        T1 = np.genfromtxt('fraserTable4.txt', delimiter=',')  # c/c2, first branch, torsional mode
        T2 = np.genfromtxt('fraserTable5.txt', delimiter=',')  # c/c2, second branch, torsional mode
        self.modes = {'L1':L1, 'L2':L2, 'T1':T1, 'T2':T2}
        self.fit = {}


    def plot(self, y='C', e=[], branch='L1', figname=None):
        """Plot c/c_2 or Omega wrt kb for a given branch, longitudinal mode only.
        
        :param str y: choose y axis ('C' or 'W')
        :param list e: ellipticity values to plot (all the available values if empty)
        :param int branch: branch number
        :param str figname: name for the figure
        """
        if len(e)==0:
            e = self.e
        
        plt.figure(figname)
        
        for ee in e:
            ind = self.e.index(ee)
            K = self.kb[branch]
            if y=='C':
                ydata = self.modes[branch][:,ind]
            elif y=='W':
                ydata = self.modes[branch][:,ind] * K  # XXX probablement à une constante près...
            plt.plot(K, ydata, '.-', label="%g (%s)"%(ee,branch))
        
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
        ind = self.e.index(e)
        K = self.kb[branch]
        
        if x=='K':
            xdata = K
            xlabel = '$K=kb$'
        elif x=='W':
            xdata = self.modes[branch][:,ind] * K
            xlabel = '$W=?$'
        
        if y=='C':
            ydata = self.modes[branch][:,ind]
            ylabel = '$C=c/c_2$'
        elif y=='W':
            ydata = self.modes[branch][:,ind] * K  # XXX probablement à une constante près...
            ylabel = '$W=?$'
    
        if labels:
            return xdata, ydata, xlabel, ylabel
        else:
            return xdata, ydata
    
    
    def curveFitting(self, e=0.4, plot=True, fun='poly'):
        """Apply curve fitting to dispersion curve
        
        :param float e: excentricity
        :param bool plot: plot curve fitting
        :param str fun: function for curve fitting ('poly', 'cos')
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
            plt.figure()
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
    FR.plot()
    FR.plot(branch='L2')
    
    # Longitudinal mode
    FR.plot(figname='gather')
    FR.plot(figname='gather', branch='L2')
    
    FR.plot(y='W', figname='KW')
    FR.plot(y='W', figname='KW', branch='L2')
    
    # Torsional mode
    FR.plot(figname='torsional', branch='T1')
    FR.plot(figname='torsional', branch='T2')
    
    # Plot a specific value of e
    FR.plot(e=[0.4], figname='0.4')
    FR.plot(e=[0.4], figname='0.4', branch='L2')
    
    # %% TRY CURVE FITTING
    FR.curveFitting(e=0.5, plot=True)
    FR.curveFitting(e=0.8, plot=True)
