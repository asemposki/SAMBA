import numpy as np
from scipy import special
import math
import matplotlib.pyplot as plt
from mixing import Models, Mixing

class Discrepancy(Mixing):

    def __init__(self):

        pass


    def variance_low(self, g, loworder, error_model):

        '''
        A function to calculate the variance corresponding to the small-g expansion model.

        :Example:
            Discrepancy.variance_low(g=np.linspace(1e-6, 0.5, 100), loworder=5)

        Parameters:
        -----------
        g : numpy.linspace
            The linspace over which this calculation is performed.

        loworder : int
            The order to which we know our expansion model. Must be passed one at a time if
            more than one model is to be calculated.

        error_model : int
            The model that the variance will be calculated from. 

        Returns:
        --------
        var1 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        #even order 
        if loworder % 2 == 0:
            
            #tell user error term used
            print(f'\nSmall-g expansion: error will be on the order of g^{loworder+2}.')

            #find coefficients
            c = np.empty([int(loworder + 2)])

            #model 1 for even orders
            if error_model == 1:

                for k in range(int(loworder + 2)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))
                    else:
                        c[k] = 0.0

                print(c)

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance 
                var1 = (cbar)**2.0 * (math.factorial(loworder + 2))**2.0 * g**(2.0*(loworder + 2))

            #model 2 for even orders
            elif error_model == 2:

                for k in range(int(loworder + 2)):

                    if k % 2 == 0:

                        #skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2) \
                                   * math.factorial(k//2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                print(c)

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial(loworder//2))**2.0 * (4.0 * g)**(2.0*(loworder + 2))

        #odd order
        else:

            #tell user error term used
            print(f'Small-g expansion: error will be on the order of g^{loworder+1}.')

            #find coefficients
            c = np.empty([int(loworder + 1)])

            #model 1 for odd orders
            if error_model == 1:

                for k in range(int(loworder + 1)):

                    if k % 2 == 0:
                        c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k) * math.factorial(k//2))
                    else:
                        c[k] = 0.0

                print(c)

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial(loworder + 1))**2.0 * g**(2.0*(loworder + 1))

            #model 2 for odd orders
            elif error_model == 2:

                for k in range(int(loworder + 1)):

                    if k % 2 == 0:

                        #skip first coefficient
                        if k == 0:
                            c[k] = 0.0
                        else:
                            c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2) \
                                    * math.factorial(k//2 - 1) * 4.0**(k))
                    else:
                        c[k] = 0.0

                print(c)

                #rms value
                cbar = np.sqrt(np.sum((np.asarray(c))**2.0) / (loworder//2 + 1))

                #variance
                var1 = (cbar)**2.0 * (math.factorial((loworder-1)//2))**2.0 * (4.0 * g)**(2.0*(loworder + 1))

        return var1


    def variance_high(self, g, highorder, error_model):

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

        error_model : int
            The error model that the variance will be calculated from.  
            
        Returns:
        --------
        var2 : numpy.ndarray
            The array of variance values corresponding to each value in the linspace of g. 
        '''

        print(f'Large-g expansion: error will be of the order g^{highorder+1}.')

        #find coefficients
        d = np.zeros([int(highorder) + 1])

        #model 1
        if error_model == 1:

            for k in range(int(highorder) + 1):

                d[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k * (math.factorial(k)) / (2.0 * math.factorial(k))

            print(d)

            #rms value (ignore first two coefficients in this model)
            dbar = np.sqrt(np.sum((np.asarray(d)[2:])**2.0) / (highorder-1))

            #variance
            var2 = (dbar)**2.0 * (g)**(-1.0) * (math.factorial(highorder + 1))**(-2.0) \
                    * g**(-2.0*highorder - 2)

        #model 2
        elif error_model == 2:

            for k in range(int(highorder) + 1):

                d[k] = special.gamma(k/2.0 + 0.25) * special.gamma(k/2.0 + 1.0) * 4.0**(k) \
                       * (-0.5)**k / (2.0 * math.factorial(k))

            print(d)

            #rms value
            dbar = np.sqrt(np.sum((np.asarray(d))**2.0) / (highorder + 1))

            #variance
            var2 = (dbar)**2.0 * g**(-1.0) * (special.gamma((highorder + 3)/2.0))**(-2.0) \
                    * (4.0 * g)**(-2.0*highorder - 2.0)

        return var2

    
    def validation(self, g, loworder, highorder): #this has been written for only two expansions; needs serious overhaul

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
        ctrue = np.zeros([int(loworder + 2)])
        dtrue = np.zeros([int(highorder + 2)])

        print('Orders: ', loworder+1, highorder+1)

        #loworder calculation
        for k in range(int(loworder + 2)):

            #print(k)

            if k % 2 == 0:
               ctrue[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
            else:
               ctrue[k] = 0.0

        #highorder calculation
        for k in range(int(highorder)+2):

            dtrue[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

        #variance (with last coefficient == loworder, highorder + 1)
        v1 = (ctrue[-1])**2.0 * g**(2.0*(loworder + 1.0))
        v2 = (dtrue[-1])**2.0 * (g)**(-1.0) * g**(-2.0*highorder - 2.0)

        return v1, v2


    def fdagger(self, g, loworder, highorder, validation=False): 

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

        validation : bool
            If True, this parameter will allow for fdagger to calculate the
            validation step of this mixed model.

        Returns:
        --------
        pdf : numpy.ndarray
            The log pdf result of this mixed model. 
        '''

        #check orders comply with function formatting
        if isinstance(loworder, np.ndarray) != True:
            loworder = np.array([loworder])
        if isinstance(highorder, np.ndarray) != True:
            highorder = np.array([highorder])

        #TODO: Fix the validation code to work for N models

        #select the proper variances -> need to fix the validation function later, so DO NOT USE NOW
        if validation == True:
            v1, v2 = self.validation(g, loworder[0], highorder[0])
        else:
            model = input('Which error model do you want to use, uninformative or informative?')

            if model == 'uninformative':
                v_low = np.asarray([self.variance_low(g, loworder[i], error_model=1) for i in range(len(loworder))])
                v_high = np.asarray([self.variance_high(g, highorder[i], error_model=1) for i in range(len(highorder))])

            elif model == 'informative':
                v_low = np.asarray([self.variance_low(g, loworder[i], error_model=2) for i in range(len(loworder))])
                v_high = np.asarray([self.variance_high(g, highorder[i], error_model=2) for i in range(len(highorder))])

            else:
                raise ValueError('Please select one of the options listed.')

        #calculating models
        f_low = [Models.low_g(self, g, i) for i in np.array([loworder])][0]
        f_high = [Models.high_g(self, g, i) for i in np.array([highorder])][0]

        #concatenate models and variances
        f = np.concatenate((f_low, f_high), axis=0) 
        v = np.concatenate((v_low, v_high), axis=0)

        #initialise arrays
        mean_n = np.zeros([len(f), len(g)])
        mean_d = np.zeros([len(f), len(g)])
        mean = np.zeros([len(g)])
        var = np.zeros([len(f), len(g)])
            
        #create fdagger for each value of g
        for i in range(len(f)):
            mean_n[i] = f[i]/v[i]
            mean_d[i] = 1.0/v[i]
            var[i] = 1.0/v[i]
        
        mean_n = np.sum(mean_n, axis=0)
        mean_d = np.sum(mean_d, axis=0)

        #mean, variance calculation
        mean = mean_n/mean_d
        var = 1.0/np.sum(var, axis=0)

        #which credibility interval to use
        self.ci = float(input('Which interval do you want to use: 68 or 95?'))
        if self.ci == 68:
            val = 1.0
        elif self.ci == 95:
            val = 1.96 
        else:
            raise ValueError('Please enter either 68 or 95.')

        #initialise credibility intervals
        intervals = np.zeros([len(g), 2])
        interval_low = np.zeros([len(loworder), len(g), 2])
        interval_high = np.zeros([len(highorder), len(g), 2])

        #calculate credibility intervals 
        intervals[:, 0] = (mean - val * np.sqrt(var))
        intervals[:, 1] = (mean + val * np.sqrt(var))

        #index
        i = 0
        print(np.shape(v_low))
        for j in loworder:
            interval_low[i,:,0] = (Models.low_g(self, g, j.item())[0,:] - val * np.sqrt(v_low[i,:]))
            interval_low[i,:,1] = (Models.low_g(self, g, j.item())[0,:] + val * np.sqrt(v_low[i,:]))
            i += 1

        #index
        i = 0
          
        for j in highorder:
            interval_high[i,:,0] = (Models.high_g(self, g, j.item())[0,:] - val * np.sqrt(v_high[i,:]))
            interval_high[i,:,1] = (Models.high_g(self, g, j.item())[0,:] + val * np.sqrt(v_high[i,:]))
            i += 1

        return mean, intervals, interval_low, interval_high

    
    def plot_mix(self, g, loworder, highorder, plot_fdagger=True, next_order=False, validation=False):

        '''
        An all-in-one plotting function that will plot the results of fdagger for N numbers
        of models, the next orders of the expansion models, and the validation step of the 
        model mixing in fdagger to test fdagger results.

        :Example:
            Discrepancy.plot_mix(g=np.linspace(1e-6, 0.5, 100), loworder=np.array([5, 8]), 
            highorder=5, plot_fdagger=True, next_order=False, validation=True)

        Parameters:
        -----------
        g : numpy.linspace
            The space over which the models are calculated.
        
        loworder : int, float, numpy.ndarray
            The highest orders to which the small-g expansion will be calculated.

        highorder : int, float, numpy.ndarray
            The highest orders to which the large-g expansion will be calculated.

        plot_fdagger : bool
            If True, this parameter will allow for the plotting of fdagger. 

        next_order : bool
            If True, the plotting function will show the next orders of each 
            expansion on the plot. 

        validation : bool
            If True, this parameter allows the plotting function to calculate the
            validation step and plot it. 

        Returns:
        --------
        None.
        '''

        #check orders comply with function formatting
        if isinstance(loworder, np.ndarray) != True:
            loworder = np.array([loworder])
        if isinstance(highorder, np.ndarray) != True:
            highorder = np.array([highorder])

        #set the dpi
        dpi = int(input('Set a dpi for the figure.'))

        #set up plot configuration
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        #set up x and y limits
        xlim = input('\nx-limits (enter "auto" if unknown): ')
        ylim = input('\ny-limits (enter "auto" if unknown): ')
        if xlim == "auto":
            ax.set_xlim(0.0,0.5)
        else:
            ax.set_xlim(tuple(map(float, xlim.split(','))))
        if ylim == "auto":
            ax.set_ylim(-4.0,4.0)
        else:
            ax.set_ylim(tuple(map(float, ylim.split(','))))

        #labels and true model
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): discrepancy model', fontsize=22)
        ax.plot(g, Models.true_model(self, g), 'k', label='True model')

        #TODO: Figure out a better labelling/colour/linestyle procedure

        #call fdagger to calculate results
        mean, intervals, interval_low, interval_high = self.fdagger(g, loworder, highorder)

        #plot the small-g expansions and error bands
        for j in loworder:
            ax.plot(g, Models.low_g(self, g, j.item())[0,:], 'r--', label=r'$f_s$ ({})'.format(j))
        
        for i in range(len(loworder)):
            ax.plot(g, interval_low[i, :, 0], 'r', linestyle='dotted', label=r'$f_s$ ({}) interval'.format(loworder[i]))
            ax.plot(g, interval_low[i, :, 1], 'r', linestyle='dotted')

        #plot the large-g expansions and error bands
        for j in highorder:
            ax.plot(g, Models.high_g(self, g, j.item())[0,:], 'b--', label=r'$f_l$ ({})'.format(j))
            
        for i in range(len(highorder)):
            ax.plot(g, interval_high[i, :, 0], 'b', linestyle='dotted', label=r'$f_l$ ({}) interval'.format(highorder[i]))
            ax.plot(g, interval_high[i, :, 1], 'b', linestyle='dotted')

        if plot_fdagger == True:
            ax.plot(g, mean, 'g', label='Mean')
            ax.plot(g, intervals[:,0], 'g--', label=r'{}$\%$ interval'.format(self.ci))
            ax.plot(g, intervals[:,1], 'g--')
            ax.fill_between(g, intervals[:,0], intervals[:,1], color='green', alpha=0.2)

        #TODO: Reinstate next order variable; however, this will be very confusing if I do not change labelling and colours first

        # if next_order == True:
        #     ax.plot(g, Models.low_g(self, g, loworder+1)[0,:], 'r', linestyle='dotted', \
        #         label=r'$f_s$ ({})'.format(loworder[0]+1))
        #     ax.plot(g, Models.high_g(self, g, highorder+1)[0,:], 'b', linestyle='dotted', \
        #         label=r'$f_l$ ({})'.format(highorder[0]+1))
        
        ax.legend(fontsize=14, loc='lower right')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)
        else:
            pass

        return None