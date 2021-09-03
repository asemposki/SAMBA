import numpy as np
from scipy import special
import math
import matplotlib.pyplot as plt
from mixing import Models, Mixing

class Discrepancy(Mixing):

    def __init__(self):

        pass


    def variance_low(self, g, loworder):

        '''
        A function to calculate the variance corresponding to the small-g expansion model.

        :Example:
            Discrepancy.variance_low(g=np.linspace(1e-6, 0.5, 100), loworder=5)

        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.

        loworder : int
            The order to which we know our expansion model. This must be a single value. 

        Returns:
        --------
        var1 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #find coefficients
        c = np.empty([int(loworder) + 1])

        for k in range(int(2*int(loworder) + 1)):

            if k % 2 == 0:
                c[k//2] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))

        print(np.asarray(c))

        #rms value
        cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder + 1))

        print(cbar)

        #variance
        var1 = (cbar)**2.0 * (math.factorial(2.0*loworder + 2.0))**2.0 * g**(4.0*loworder + 4.0)

        return var1


    def variance_high(self, g, highorder):

        '''
        A function to calculate the variance corresponding to the large-g expansion model.

        :Example:
            Discrepancy.variance_low(g=np.linspace(1e-6, 0.5, 100), highorder=23)

        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.

        highorder : int
            The order to which we know our expansion model. This must be a single value. 
            
        Returns:
        --------
        var2 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #find coefficients
        d = np.empty([int(highorder) + 1])

        for k in range(int(highorder)+1):

            d[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k * (math.factorial(k)) / (2.0 * math.factorial(k))

        print(np.asarray(d))

        #rms value
        dbar = np.sqrt(np.sum((np.asarray(d))**2.0) / (highorder + 1.0))

        print(dbar)

        #variance
        var2 = (dbar)**2.0 * (g)**(-1.0) * (math.factorial(highorder + 1.0))**(-2.0) \
                * g**(-2.0*highorder - 2.0)

        return var2

    
    def validation(self, g, loworder, highorder):

        '''
        A function to calculate the validation variances needed by taking the last
        coefficient in the expansion that we know and using it in the variance for
        f_dagger. 

        :Example:
            Discrepancy.validation(g=np.linspace(1e-6, 0.5, 100), loworder=5, highorder=5)
        
        Parameters:
        -----------
        g : numpy.linspace
            The linspace used to calculate the model in the coupling constant, g.

        loworder : int
            The highest order the small-g expansion is calculated to.

        highorder : int
            The highest order the large-g expansion is calculated to.
        
        Returns:
        --------
        v1, v2 : float
            The two variances found for the two expansion models. 
        '''

        #find coefficients
        ctrue = np.zeros([int(2*loworder + 3)])
        dtrue = np.zeros([int(highorder + 2)])

        #loworder calculation
        for k in range(int(2*loworder + 3)):

            if k % 2 == 0:
               ctrue[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
            else:
               ctrue[k] = 0.0

        #highorder calculation
        for k in range(int(highorder)+2):

            dtrue[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

        #variance (with last coefficient == loworder, highorder + 1)
        v1 = (ctrue[-1])**2.0 * g**(4.0*(loworder) + 4.0)
        v2 = (dtrue[-1])**2.0 * (g)**(-1.0) * g**(-2.0*highorder - 2.0)

        return v1, v2


    def fdagger(self, g, loworder, highorder, plot_fdagger=True, next_order=False, validation=False):

        '''
        A function to determine the pdf of the mixed model.

        :Example:
            Discrepancy.fdagger(g=np.linspace(1e-6, 0.5, 100), loworder=np.array([5]), 
            highorder=np.array([23]))

        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.
        
        loworder : numpy.ndarray
            The chosen order to which this model is calculated regarding the 
            small-g expansions.

        highorder : numpy.ndarray
            The chosen order to which this model is calculated regarding the 
            large-g expansions. 

        plot_fdagger : bool
            If True, plot_fdagger instructs the function to plot the mean
            and intervals of the fdagger model. If False, this step is skipped.
            The default is set to True.

        next_order : bool
            If True, this will plot the next order in the expansion models for both
            low and high orders. If False, this step will be skipped. Default is set
            to False. 

        Returns:
        --------
        pdf : numpy.ndarray
            The log pdf result of this mixed model. 
        '''

        if loworder.ndim > 1 or highorder.ndim > 1:
            raise ValueError('Please specify only one order per model.')

        #which interval to use
        ci = float(input('Which interval do you want to use: 68 or 95?'))

        if ci == 68:
            val = 1.0
        elif ci == 95:
            val = 1.96 
        else:
            raise ValueError('Please enter either 68 or 95.')

        #variances
        if validation == True:
            v1, v2 = self.validation(g, loworder[0]//2, highorder[0])

        else:
            v1 = self.variance_low(g, loworder[0]//2)
            v2 = self.variance_high(g, highorder[0])

        #mean, variance, joint pdf
        mean = (v2 * Models.low_g(self, g, loworder) + v1 * Models.high_g(self, g, highorder)) / (v1 + v2)
        mean = mean[0]
        var = v1 * v2 / (v1 + v2)
        pdf = -np.log(np.sqrt(2.0 * np.pi * var)) - ((g - mean)**2.0/ (4.0 * var)) 

        #credible intervals -> 95% (confidence interval?)
        intervals = np.empty([len(g), 2])
        interval_f1 = np.empty([len(g), 2])
        interval_f2 = np.empty([len(g), 2])

        for i in range(len(g)):
            intervals[i, 0] = (mean[i] - val * np.sqrt(var[i]))
            intervals[i, 1] = (mean[i] + val * np.sqrt(var[i]))
            interval_f1[i, 0] = (Models.low_g(self, g[i], loworder) - val * np.sqrt(v1[i]))
            interval_f1[i, 1] = (Models.low_g(self, g[i], loworder) + val * np.sqrt(v1[i]))
            interval_f2[i, 0] = (Models.high_g(self, g[i], highorder) - val * np.sqrt(v2[i]))
            interval_f2[i, 1] = (Models.high_g(self, g[i], highorder) + val * np.sqrt(v2[i]))

        #plot the pdf, expansions, and true model
        fig = plt.figure(figsize=(8,6), dpi=100)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(-5.0, 5.0)
        ax.set_xlabel('g', fontsize=16)
        ax.set_ylabel('F(g)', fontsize=16)
        ax.set_title('F(g): discrepancy model', fontsize=16)
        ax.plot(g, Models.true_model(self, g), 'k', label='True model')

        #plot the small-g expansions and error bands
        ax.plot(g, Models.low_g(self, g, loworder)[0,:], 'r--', label=r'$f_s$ ({})'.format(loworder[0]))
        ax.plot(g, interval_f1[:, 0], 'r.', label=r'$f_s$ ({}) credible interval'.format(loworder[0]))
        ax.plot(g, interval_f1[:, 1], 'r.')

        #plot the large-g expansions and error bands
        ax.plot(g, Models.high_g(self, g, highorder)[0,:], 'b--', label=r'$f_l$ ({})'.format(highorder[0]))
        ax.plot(g, interval_f2[:, 0], 'b.', label=r'$f_l$ ({}) credible interval'.format(highorder[0]))
        ax.plot(g, interval_f2[:, 1], 'b.')

        if plot_fdagger == True:
            ax.plot(g, mean, 'g', label='Mean')
            ax.plot(g, intervals[:,0], 'g--', label=r'{}$\%$ credible interval'.format(ci))
            ax.plot(g, intervals[:,1], 'g--')

        if next_order == True:
            ax.plot(g, Models.low_g(self, g, loworder+1)[0,:], 'r', linestyle='dotted', \
                label=r'$f_s$ ({})'.format(loworder[0]+1))
            ax.plot(g, Models.high_g(self, g, highorder+1)[0,:], 'b', linestyle='dotted', \
                label=r'$f_l$ ({})'.format(highorder[0]+1))
        
        ax.legend(fontsize=12)
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)
        else:
            pass

        return pdf