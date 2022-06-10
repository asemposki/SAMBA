##################################################
# Code originally written for Mathematica by 
# Masazumi Honda. The function fprset() below is 
# adapted from his code entirely. The rest is 
# written as a wrapper by Alexandra Semposki for 
# this BMM package. Notes below by Honda.
##################################################

#Explicit formula for interpolating functions
#in the paper "On Pertubation theory improved by Strong coupling expansion"
#by Masazumi Honda (Harish-Chandra Research Institute)

#All the interpolating functions are taking the form of Fractional Power of Rational function Subscript[F**(\[Alpha]), m,n](g).
#We denote this by FPR[\[Alpha],m,n,g]
#Zero-dimensional \[Phi]**4-theory

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (zoomed_inset_axes, mark_inset)
from cycler import cycler
from matplotlib.ticker import AutoMinorLocator
from mixing import Models 


__all__ = ['FPR']


class FPR(Models):

    def __init__(self, g, loworder, highorder):

        '''
        A class to calculate the FPR method curves for comparison
        to the mixed models in the three BMM methods of this package.
        
        :Example:
            FPR(g=np.linspace(1e-6,1.0,100), loworder=np.array([5]),
                highorder=np.array([5]))

        Parameters:
        -----------
        g : numpy.linspace
            The input space array over which the models are mixed.

        loworder : numpy.ndarray 
            The highest order considered in the small-g expansion.
        
        highorder : numpy.ndarray
            The highest order considered in the large-g expansion.

        Returns:
        --------
        None.
        '''

        self.g = g
        self.loworder = loworder
        self.highorder = highorder 

        #instantiate Models() class here
        self.m = Models(self.loworder, self.highorder)

        return None

    
    def fprset(self, key):

        '''
        Call the proper FPR function desired and obtain 
        an array of the results in the input space, g. 

        :Example: 
            FPR.fprset(key='(2,4)^(1/8)')

        Parameters:
        -----------
        key : str
            The preferred FPR function. Enter a key in the
            convention: '(m,n)^(\alpha)', where m,n are orders
            less than or equal to N_s and N_l (loworder, highorder
            in the other classes). \alpha is the value the FPR is 
            raised to in Eq. (2.7) (Honda 2014). 

        Returns:
        --------
        fpr : numpy.ndarray
            Results of the FPR function in an array. 
        '''

        #if statement for calling the proper FPR function
        fpr = np.zeros(len(self.g))

        self.keyvalue = key

        if key == '(0,0)^(1/2)':
                
            #FPR[1/(2*1),0,0,g_]
            fpr = np.sqrt(2 * np.pi * np.sqrt(1/(1 + (8 * self.g * np.pi)/sp.gamma(1/4)**2)))  

        elif key == '(1,1)^(1/2)':  

            #FPR[1/(2*1),1,1,g_] 
            fpr = np.sqrt(2 * np.pi * sp.gamma(1/4)) * np.sqrt((2 * np.pi * sp.gamma(-(1/4)) + 8 * \
                  self.g * np.pi * sp.gamma(1/4) + sp.gamma(1/4)**3) / (64 * self.g**2 * np.pi**2 + 2 * \
                  np.pi * sp.gamma(-(1/4)) * sp.gamma(1/4) + 8 * self.g * \
                  np.pi * sp.gamma(1/4)**2 + sp.gamma(1/4)**4))

        elif key == '(2,2)^(1/2)':
            #FPR[1/(2*1),2,2,g_] 
            fpr = 2.5066282746310002 * np.sqrt((1. + 10.1531607808241 * self.g + 37.91166947810798 * \
                  self.g**2) / (1. + 10.1531607808241 * self.g + 43.91166947810798 * \
                  self.g**2 + 72.48541321812907 * self.g**3))

        elif key == '(3,3)^(1/2)':
            #FPR[1/(2*1),3,3,g_] 
            fpr = 2.5066282746310002 * np.sqrt((1. + 16.030388229931486 * self.g + 110.26077245790317 * \
                  self.g**2 + 324.0187225963292 * self.g**3) / (1. + 16.030388229931486 * \
                  self.g + 116.26077245790317 * self.g**2 + 420.2010519759181 * \
                  self.g**3 + 619.509278307239 * self.g**4))

        elif key == '(4,4)^(1/2)':
            #FPR[1/(2*1),4,4,g_] 
            fpr = 2.5066282746310002 * np.sqrt(( 1. + 22.874503929271544 * self.g + 238.23905635876318 * \
                  self.g**2 + 1303.4929582331083 * self.g**3 + 3224.5631655188554 * self.g**4)/ \
                  ( 1. + 22.874503929271544 * self.g + 244.23905635876318 * self.g**2 + 1440.7399818087376 * \
                  self.g**3 + 4575.997503671434 * self.g**4 + 6165.220279617642 * self.g**5))

        elif key == '(1,1)^(1/6)':
            #FPR[1/(2*3),1,1,g_] 
            fpr = 2.5066282746310002 * (1/( 1. + 7.086913042848253 * self.g**2 + 6.989291097242859 * \
                  self.g**3))**(1/6)

        elif key == '(2,2)^(1/6)':
            #FPR[1/(2*3),2,2,g_] 
            fpr = 2.5066282746310002 * (( 1. + 3.7875802399388747 * self.g)/( 1. + 3.7875802399388747 * \
                  self.g + 18.000000000000007 * self.g**2 + 33.83154290049999 * self.g**3 + 26.47250085109775 * \
                  self.g**4))**(1/6)

        elif key == '(3,3)^(1/6)':
            #FPR[1/(2*3),3,3,g_] 
            fpr = 2.5066282746310002 * (( 1. + 8.099704591178746 * self.g + 28.252523713142793 * \
                  self.g**2)/( 1. + 8.099704591178746 * self.g + 46.25252371314279 * \
                  self.g**2 + 145.7946826412175 * self.g**3 + 256.83437198547387 * \
                  self.g**4 + 197.46511246291166 * self.g**5))**(1/6)

        elif key == '(4,4)^(1/6)':
            #FPR[1/(2*3),4,4,g_]
            fpr = 2.5066282746310002 * (( 1. + 13.470628417091639 * self.g + 87.53888949233587 * \
                  self.g**2 + 255.1535591483978 * self.g**3)/( 1. + 13.470628417091639 * \
                  self.g + 105.53888949233587 * self.g**2 + 497.6248706560473 * self.g**3 + 1449.7000108620534 * \
                  self.g**4 + 2420.0858672492423 * self.g**5 + 1783.3424993857257 * self.g**6))**(1/6)
        
        elif key == '(2,2)^(1/10)':
            #FPR[1/(2*5),2,2,g_] 
            fpr = 2.5066282746310002 * (1/( 1. + 30. * self.g**2 + 32.14821200212872 * \
                  self.g**3 + 43.17787580118569 * self.g**4 + 25.54986136647706 * self.g**5))**(1/10)

        elif key == '(4,4)^(1/10)':
            #FPR[1/(2*5),4,4,g_] 
            fpr = 2.5066282746310002 * (( 1. + 5.072505666220409 * self.g + 14.43685326985559 * \
                  self.g**2)/( 1. + 5.072505666220409 * self.g + 44.43685326985559 * self.g**2 + 152.17516998661225 * \
                  self.g**3 + 403.10559809566917 * self.g**4 + 708.6889005862954 * self.g**5 + 752.9544739983669 * \
                  self.g**6 + 368.85959961298136 * self.g**7))**(1/10)

        elif key == '(3,3)^(1/14)':
            #FPR[1/(2*7),3,3,g_] 
            fpr = 2.5066282746310002 * (1/( 1. + 42. * self.g**2+155.75843284764994 * \
                  self.g**4 + 239.21559267499774 * self.g**5 + 220.9758065100799 * \
                  self.g**6 + 93.39937438057375 * self.g**7))**(1/14)

        elif key == '(4,4)^(1/18)':
            #FPR[1/(2*9),4,4,g_] 
            fpr = 2.5066282746310002 * (1/( 1. + 54. * self.g**2+594. * self.g**4+780.7879756756589 * \
                  self.g**5 + 1294.340979801729 * self.g**6 + 1475.3510504866329 * self.g**7 + 1038.5911468627608 * \
                  self.g**8 + 341.42819835916043 * self.g**9))**(1/18)

        else:
            raise KeyError('The key provided does not match any in the FPR database.')

        return fpr

    #at the moment, this is very specific to the paper plot---disassemble later for package
    def fpr_plot(self, mean, intervals, fpr_keys=None, ci=68):

        '''
        A plotter for the overlay of the GP results and the FPR results
        from Honda (2014). 

        :Example:
            FPR.fpr_plot(mean=np.array(), intervals=np.array([,]), 
            fpr_keys=['(3,3)^(1/6)'], ci=95)

        Parameters:
        -----------
        mean : numpy.ndarray
            A PPD mean to be compared to the FPR results.
        
        intervals : numpy.ndarray
            A 2D array to plot a UQ band around the PPD. 

        fpr_keys : list
            A list of strings of fpr keys to be read in 
            by the function and calculated using the fprset()
            function above.

        ci : int
            The uncertainty calculated on the expansions. Can
            be either 68 or 95. 

        Returns:
        --------
        None.
        '''

        #set up plot configuration
        fig, ax1 = plt.subplots(figsize=(8,6), dpi=600)
        ax1.tick_params(axis='x', labelsize=18)
        ax1.tick_params(axis='y', labelsize=18)
        ax1.locator_params(nbins=8)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.xaxis.set_label_coords(.5, -.05)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_label_coords(-.05, .5)

        #set up x and y limits
        ax1.set_xlim(0,1)
        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_ylim(1.2,3.2)
        ax1.set_yticks([1.2, 1.6, 2.0, 2.4, 2.8, 3.2])
    
        #labels and true model
        ax1.set_xlabel('g', fontsize=22)
        ax1.set_ylabel('F(g)', fontsize=22)
        ax1.plot(self.g, Models.true_model(self, self.g), 'k', label='True model')

        #unpack ci
        self.ci = ci 

        #plot the small-g expansions and error bands
        ax1.plot(self.g, Models.low_g(self, self.g)[0,:], 'r--', label=r'$f_s$ ($N_s$ = {})'.format(self.loworder[0]))
        
        #plot the large-g expansions and error bands
        ax1.plot(self.g, Models.high_g(self, self.g)[0,:], 'b--', label=r'$f_l$ ($N_l$ = {})'.format(self.highorder[0]))
        
        #plot the GP results (mixed model)
        ax1.plot(self.g, mean, 'g', label='Mean')
        ax1.plot(self.g, intervals[:,0], 'g', linestyle='dotted', label=r'{}$\%$ CI'.format(int(self.ci)))
        ax1.plot(self.g, intervals[:,1], 'g', linestyle='dotted')
        ax1.fill_between(self.g, intervals[:,0], intervals[:,1], color='green', alpha=0.2)

        #FPR results
        if fpr_keys is not None:
            ax1.set_prop_cycle(cycler('color', ['darkviolet', 'deepskyblue', 'darkorange', 'gold']))
            for k in fpr_keys:
                mn = k[0:5]
                alpha = k[6:]
                fpr = self.fprset(k)
                ax1.plot(self.g, fpr, linestyle='dashed', label=r'$F_{{{}}}^{{{}}} (g)$'.format(mn, alpha))
                
        ax1.legend(fontsize=16, loc='upper right')

        #inset plot parameters
        x1 = 0.26
        x2 = 0.31
        y1 = 2.15
        y2 = 2.25
        axins = zoomed_inset_axes(ax1, 6, loc=9) 
        axins.plot(self.g, Models.true_model(self, self.g), 'k', label='True model')
        axins.plot(self.g, Models.low_g(self, self.g)[0,:], 'r--', label=r'$f_s$ ($N_s$ = {})'.format(self.loworder[0]))
        axins.plot(self.g, Models.high_g(self, self.g)[0,:], 'b--', label=r'$f_l$ ($N_l$ = {})'.format(self.highorder[0]))
        axins.plot(self.g, mean, 'g', label='Mean')
        axins.plot(self.g, intervals[:,0], 'g', linestyle='dotted', label=r'{}$\%$ interval'.format(int(self.ci)))
        axins.plot(self.g, intervals[:,1], 'g', linestyle='dotted')
        axins.fill_between(self.g, intervals[:,0], intervals[:,1], color='green', alpha=0.2)
        if fpr_keys is not None:
            axins.set_prop_cycle(cycler('color', ['darkviolet', 'deepskyblue', 'darkorange', 'gold']))
        for k in fpr_keys:
                mn = k[0:5]
                alpha = k[6:]
                fpr = self.fprset(k)
                axins.plot(self.g, fpr, linestyle='dashed', label=r'$F_{{{}}}^{{{}}} (g)$'.format(mn, alpha))
                
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        plt.draw()
        plt.show()

        #save figure option
        # response = input('Would you like to save this figure? (yes/no)')

        # if response == 'yes':
        #     name = input('Enter a file name (include .jpg, .png, etc.)')
        #     fig.savefig(name, bbox_inches='tight')

        return None