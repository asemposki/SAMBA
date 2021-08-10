import numpy as np
from scipy import special, integrate
from cycler import cycler
import math
import time
import emcee 
import matplotlib.pyplot as plt


class Switching:
   
    
    def __init__(self):
        
        '''
        This class contains each of the current switching functions that the user can select from 
        when running the mixed model function in the class Mixing.
        
        :Example:
            Switching()
            
        Parameters:
        -----------
        None.
        
        Returns:
        --------
        None.
        '''
        
        pass
    
    
    def logistic(self, params, g):
        
        '''
        A basic logistic function often used in machine learning, implemented here with two free
        parameters to be determined via sampling.
        
        :Example:
            Switching.logistic(params=np.array(), g=0.5)
            
        Parameters:
        -----------
        params
            The array of parameters the sampler will determine (here labelled beta0 and beta1, where
            beta0 controls the location of the function and beta1 controls the slope). 

        Returns:
        --------
        mixing          
            The result of the logistic function given the value g.
        '''
        beta0, beta1 = params
    
        mixing = (1.0 + np.exp(-(beta0 + g*beta1)))**(-1.0)
    
        return mixing
    

    def cdf(self, params, g):
        
        '''
        The cumulative distribution function of a standard normal distribution, with two free parameters
        determined by sampling.
        
        :Example:
            Switching.cdf(params=np.array(), g=0.5)
        
        Parameters:
        -----------
        params
            The array of parameters the sampler will determine (here labelled beta0 and beta1, where
            beta0 controls the location of the function and beta1 controls the slope). 
                   
        Returns:
        --------
        function          
            The result of the cdf function at the value of g. 
        '''
        beta0, beta1 = params
    
        function = (1.0 + math.erf(beta0 + g*beta1)/np.sqrt(2.0)) / 2.0
    
        return function
    
    
    def switchcos(self, params, g):
        
        '''
        A piecewise function using two constants at either end, and two cosine functions in the centre,
        to be used as a switching function. One free parameter, g3, is found by sampling. 
        
        :Example:
            Switching.switchcos(params=np.array(), g=0.5)
            
        Parameters:
        -----------
        params
            The array of parameters to be determined by the sampler (here labelled g1, g2, and g3, where
            g1 is the separation point between the first constant function and the first cosine function, 
            g2 is the separation point between the second cosine function and the second constant function, 
            and g3 is the point between the two cosine functions). 

        Returns:
        --------
            The value of the function at a specific point in g. 
        '''

        #unpack the parameters
        g1, g2, g3 = params 
    
        if g <= g1:
            return 1.0
    
        elif g <= g3:
            return (1.0 + np.cos((np.pi/2.0) * ((g - g1)/(g3 - g1))))/2.0
    
        elif g < g2:
            return 0.5 + np.cos((np.pi/2.0) * (1.0 + ((g - g3)/(g2 - g3))))/2.0
    
        else:
            return 0.0
    

    def ppd(self, trace, g, loworder, highorder):
        
        '''
        A function to calculate the posterior predictive distribution (PPD) for any chosen switching function 
        defined in this class. 
        
        :Example:
            Switching.ppd(trace, g=np.linspace(0.0, 0.5, 100), loworder=5,
            highorder=23)
            
        Parameters:
        -----------
        trace
            The trace of each of the parameters from the sampler.
            
        g
            The linspace desired to calculate the PPD across.
            
        loworder
            The order of the small-g expansion to be calculated in the mixing model.
        
        highorder
            The order of the large-g expansion to be calculated in the mixing model. 
        
        Returns:
        --------
        result_array
            The array of results of the PPD for each of the points in the linspace g. 
        '''

        result_array = np.empty([len(g), len(trace[0].T)])
        gmax = max(g)

        #determine which switching function was used
        if self.choice == 'logistic' or self.choice == 'cdf':
    
            for i in range(len(g)):
                for j in range(len(trace[0].T)):
            
                    if (Mixing.low_g(self, g[i], loworder) - Mixing.high_g(self, g[i], highorder))\
                    > 0.1 and g[i] > (0.25*gmax):
                        result_array[i,j] = Mixing.high_g(self, g[i], highorder)
                    
                    elif (Mixing.low_g(self, g[i], loworder) - Mixing.high_g(self, g[i], highorder)) > 0.1:
                        result_array[i,j] = Mixing.low_g(self, g[i], loworder)
                    
                    else:
                        params = np.array([trace[0, j], trace[1, j]])

                        result_array[i,j] = self.f(params, g[i])*Mixing.low_g(self, g[i], loworder) \
                                        + (1.0 - self.f(params, g[i])) \
                                        *Mixing.high_g(self, g[i], highorder)
        
            return result_array    

        elif self.choice == 'cosine':

            for i in range(len(g)):
                for j in range(len(trace[0].T)):
                    
                    params = np.array([trace[0, j], trace[1, j], trace[2, j]])
                
                    result_array[i,j] = self.switchcos(params, g[i]) * Mixing.low_g(self, g[i], loworder) \
                                    + (1.0 - self.switchcos(params, g[i])) \
                                    * Mixing.high_g(self, g[i], highorder)
                
            return result_array
    
    
    def plot_ppd(self, g_data, g_true, g_ppd, data, ppd_results, ppd_intervals, percent, loworder, highorder):
        
        '''
        A plotting function that can be used to plot the posterior predictive distribution (PPD) results (mean and 
        credible interval) obtained from calling the functions above in the main code, as well as data generated, 
        the true model, and the small- and large-g expansions chosen for the mixed model calculation. 
        
        :Example:
            Switching.plot_ppd(g_data=np.linspace(0.0, 0.5, 20), g_true=np.linspace(0.0, 0.5, 100), 
            g_ppd=np.linspace(0.0, 0.5, 200), data=np.array(), ppd_results=np.array(), ppd_intervals=np.array(),
            loworder=5, highorder=23)
            
        Parameters:
        -----------
        g_data
            The linspace used to generate the data.
        
        g_true
            The linspace used to calculate the true model over the input space, as well as the expansions.
        
        g_ppd
            The linspace chosen to calculate the PPD over. 
        
        data
            An array of data either generated or supplied by the user.
            
        ppd_results
            An array of the mean of the PPD at each point in the g_ppd linspace.
        
        ppd_intervals
            A 2D array of the credibility interval calculated for the PPD (containing both bounds).
            
        percent
            The percent credibility interval calculated for the variable ppd_intervals (used in the plot
            legend). 
        
        loworder
            The order of the small-g expansion used in the mixed model.
        
        highorder
            The order of the large-g expansion used in the mixed model. 
        
        Returns:
        --------
        None.
        '''
    
        xmin = float(input('Enter the minimum g to plot.'))
        xmax = float(input('Enter the maximum g to plot.'))
        
        ymin = float(input('Enter the minimum for F(g).'))
        ymax = float(input('Enter the maximum for F(g).'))
        
        if ymin == None:
            ymin = 1.8
        if ymax == None:
            ymax = 2.6
        
        fig = plt.figure(figsize=(8,5), dpi=100)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('g', fontsize=16)
        ax.set_ylabel('F(g)', fontsize=16)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title('Mixed Prediction with Calibration Posteriors', fontsize=16)

        ax.plot(g_data, data, 'k.', label='Data set')  
        ax.plot(g_true, Mixing.true_model(self, g_true), 'k', label='Exact')

        ax.plot(g_true, Mixing.low_g(self, g_true, loworder)[0,:], 'r--', label=r'$f_s$ ({})'.format(loworder[0]))
        ax.plot(g_true, Mixing.high_g(self, g_true, highorder)[0,:], 'b--', label=r'$f_l$ ({})'.format(highorder[0]))

        ax.plot(g_ppd, ppd_results, 'g', label='Calibrated mixed model')
        ax.plot(g_ppd, ppd_intervals[:,0], 'g', linestyle='dotted', label='{}% credible interval (HPD)'.format(percent))
        ax.plot(g_ppd, ppd_intervals[:, 1], 'g', linestyle='dotted')

        ax.legend(fontsize=12)
        plt.show()

        answer = input('Would you like to save the plot to a file (yes/no)?')

        if answer == 'yes':
            name = input('Enter a file name to use (include file type as .png, .pdf, etc.).')
            fig.savefig(name)
        else:
            pass
        
        return None


class Mixing(Switching):
    
    
    def __init__(self):
        
        '''
        This class is designed with all of the necessary functions for creating a data set, plotting it 
        along with the true model, and calculating expansions of specific orders of the true model to mix.
        Dependent on the Switching class to run the mixed model functions. 
    
        :Example:            
            Mixing()
            
        Parameters:
        -----------
        None.
            
        Returns:
        --------
        None.
        
        '''
        
        print('Welcome to the BMM sandbox! Here you get to play!')
        
          
    def low_g(self, g, loworder):
        
        '''
        A function to calculate the small-g divergent asymptotic expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Mixing.low_g(g=np.linspace(0.0, 0.5, 20), loworder=np.array([5, 10, 20]))
            
        Parameters:
        -----------
        g  
            The linspace of the coupling constant for this calculation. 
            
        loworder     
            The array of different expansion orders to calculate. These indicate the highest power the expansions 
            are calculated up to. 
            
        Returns:
        --------
        output
            The array of values of the expansion in small-g at each point in g_true space, for each value of 
            loworder (highest power the expansion reaches).
        '''
        
        if loworder.ndim == 0:
            loworder = np.array([loworder])
        output = []
        
        for order in loworder:
            low_c = np.empty([int(order)+1])
            low_terms = np.empty([int(order) + 1])
            try:
                value = np.empty([len(g)])
       
                #loop over array in g
                for i in range(len(g)):      

                    #loop over orders
                    for k in range(int(order)+1):

                        if k % 2 == 0:
                            low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
                        else:
                            low_c[k] = 0

                        low_terms[k] = low_c[k] * g[i]**(k)

                    value[i] = np.sum(low_terms)

                output.append(value)
                data = np.array(output, dtype = np.float64)
            
            except:
                value = 0.0
                for k in range(int(order)+1):

                    if k % 2 == 0:
                        low_c[k] = np.sqrt(2.0) * special.gamma(k + 0.5) * (-4.0)**(k//2) / (math.factorial(k//2))
                    else:
                        low_c[k] = 0

                    low_terms[k] = low_c[k] * g**(k)

                value = np.sum(low_terms)
                data = value
        return data
        
    def high_g(self, g, highorder):
        
        '''
        A function to calculate the large-g convergent Taylor expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Mixing.high_g(highorder=np.array([5, 10, 20]))
            
        Parameters:
        -----------
        g         
            The linspace of the coupling constant for this calculation.
        
        highorder   
            The array of different expansions orders to calculate. These indicate the highest power the expansions
            are calculated up to. 
            
        Returns
        -------
        output        
            The array of values of the expansion at large-g at each point in g_true space, for each value of highorder
            (highest power the expansion reaches).
        '''
        if highorder.ndim == 0:
            highorder = np.array([highorder])
  
        output = []
        
        for order in highorder:
            high_c = np.empty([int(order) + 1])
            high_terms = np.empty([int(order) + 1])
            
            try:
                value = np.empty([len(g)])
        
                #loop over array in g
                for i in range(len(g)):

                    #loop over orders
                    for k in range(int(order)+1):

                        high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                        high_terms[k] = (high_c[k] * g[i]**(-k)) / np.sqrt(g[i])

                    #sum the terms for each value of g
                    value[i] = np.sum(high_terms)

                output.append(value)

                data = np.array(output, dtype = np.float64)
        
            except:
                value = 0.0

                #loop over orders
                for k in range(int(order)+1):

                    high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                    high_terms[k] = (high_c[k] * g**(-k)) / np.sqrt(g)

                #sum the terms for each value of g
                value = np.sum(high_terms)
                data = value
                
        return data 


    def true_model(self, g):
        
        '''
        The true model of the zero-dimensional phi^4 theory partition function using an input linspace.
        
        :Example:
            Mixing.true_model(g=np.linspace(0.0, 0.5, 100))
            
        Parameters:
        -----------
        g       
            The linspace for g desired to calculate the true model. This can be the g_true linspace, g_data
            linspace, or another linspace of the user's choosing. 
            
        Returns:
        -------
        model        
            The model calculated at each point in g space. 
        '''
    
        #define a function for the integrand
        def function(x,g):
            return np.exp(-(x**2.0)/2.0 - (g**2.0 * x**4.0))
    
        #initialization
        self.model = np.zeros([len(g)])
    
        #perform the integral for each g
        for i in range(len(g)):
            
            self.model[i], self.err = integrate.quad(function, -np.inf, np.inf, args=(g[i],))
        
        return self.model 
   

    def plot_models(self, g, loworder, highorder):
        
        '''
        A plotting function to produce a figure of the model expansions calculated in Mixing.low_g and Mixing.high_g, 
        and including the true model calculated using Mixing.true_model.
        
        :Example:
            Mixing.plot_models(g=np.linspace(0.0, 0.5, 100), lowk=np.array([5, 23]), highk=np.array([5, 23]))
            
        Parameters:
        -----------
        loworder       
            As in Mixing.low_g, the highest powers to calculate the series to for the asymptotic small-g expansion.
        
        highorder        
            As in Mixing.high_g, the highest powers to calculate the series to for the convergent large-g expansion.
            
        Returns
        -------
        None.
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=100)
        ax = plt.axes()
        ax.set_xlim(min(g), max(g))
        ax.set_ylim(1.8, 2.6)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('g', fontsize=16)
        ax.set_ylabel('F(g)', fontsize=16)
        ax.set_title('F(g): Expansions and true model', fontsize=16)
        
        #plot the true model 
        ax.plot(g, Mixing.true_model(self, g), 'k', label='True model')
        
        #add linestyle cycler
        linestyle_cycler = cycler(linestyle=['dashed', 'dotted', 'dashdot', 'dashed', 'dotted', 'dashdot'])
        ax.set_prop_cycle(linestyle_cycler)
                
        #for each small-g order, plot
        for i in np.array(loworder):
            ax.plot(g, Mixing.low_g(self, g, i)[0], color='r', label=r'$f_s$ ({})'.format(i))
        
        #for each large-g order, plot
        for i in np.array(highorder):
            ax.plot(g, Mixing.high_g(self, g, i)[0], color='b', label=r'$f_l$ ({})'.format(i))
            
        ax.legend(fontsize=12)
        plt.show()
        
         
    def residuals(self, loworder, highorder):
        
        '''
        A calculation and plot of the residuals of the model expansions vs the true model values at each point in g.
        g is set internally for this plot, as the plot must be shown in loglog format to see the power law of the
        residuals. 
        
        :Example:
            Mixing.residuals(loworder=np.array([5, 10, 20]), highorder=np.array([5, 10, 20]))
            
        Parameters:
        -----------
        loworder        
            The array of highest power series orders for the asymptotic, small-g expansion.
            
        highorder        
            The array of highest power series orders for the convergent, large-g expansion.
            
        Returns:
        --------
        None. 
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=100)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('g', fontsize=16)
        ax.set_ylabel('Residual', fontsize=16)
        ax.set_title('F(g): residuals', fontsize=16)
        ax.set_xlim(1e-2, 10.)
        ax.set_ylim(1e-6,1e17)

        #set range for g
        g_ext = np.logspace(-6., 6., 800)
        
        #set up marker cycler
        marker_cycler = cycler(marker=['.', '*', '+', '.', '*', '+'])
        ax.set_prop_cycle(marker_cycler)

        #calculate true model
        value_true = Mixing.true_model(self, g_ext)
        
        #for each small-g order, plot
        for i in list(loworder):
            valuelow = Mixing.low_g(self, g_ext, i)
            residlow = (valuelow - value_true)/value_true
            ax.loglog(g_ext, abs(residlow[0,:]), 'r', linestyle="None", label=r"$F_s({})$".format(i))

        #for each large-g order, plot
        for i in list(highorder):
            valuehi = Mixing.high_g(self, g_ext, i)
            residhi = (valuehi - value_true)/value_true
            ax.loglog(g_ext, abs(residhi[0,:]), 'b', linestyle="None", label=r"$F_l({})$".format(i))
        
        ax.legend(fontsize=12)
        plt.show()
        
        
    def add_data(self, g_true, g_data, data=None, sigma=None):
        
        '''
        A data generation function that generates data based on the g_data linspace provided (with the number of 
        points chosen by the user) and the error desired on each point (also input by the user), or accepts the
        user's input of an array of data and standard deviations of the data points. 
        
        :Example:
            Mixing.add_data(g_true=np.linspace(0.0, 0.5, 100), g_data=np.linspace(0.0, 0.5, 20))
        
        Parameters:
        -----------
        g_true         
            The linspace desired for the true model to be calculated.
        
        g_data    
            The linspace input for the data to be generated within. 

        data
            The data array entered by the user; if user wishes to generate data, this remains set to None.
        
        sigma
            The standard deviation array entered by the user; if user wishes to generate data, this will
            remain set to None. 
            
        Returns:
        --------
        data         
            The array of data (generated or entered by the user).
            
        sigma    
            The standard deviation at each data point (generated or entered by the user).
        '''
        
        #if user has an array of data, skip data generation
        if data is None:
           
            #ask for error desired
            error = float(input('Data will be generated. What percent error would you like on your data (please enter a decimal)?'))
        
            #generate fake data  
            data = self.true_model(g_data)
            rand = np.random.RandomState()
            var = error*rand.randn(len(g_data))
            data = data*(1 + var)
        
            #calculate standard deviation
            sigma = error*data
        
        #plot the data and true model
        self.plot_data(g_true, g_data, data)
    
        return data, sigma
    
    
    def plot_data(self, g_true, g_data, data):
        
        '''
        The plotting function to display the generated data and true model. 
        
        :Example:
            Mixing.plot_data(g_true=np.linspace(0.0, 0.5, 100), g_data=np.linspace(0.0, 0.5, 20), 
            data=np.array(1.8,...,2.6))
            
        Parameters:
        -----------
        g_true
            The linspace desired for the true model to be calculated.
            
        g_data
            The linspace over which the data was generated. 
            
        data
            The array of data generated using the Mixing.add_data function.
            
        Returns:
        --------
        None.
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=100)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim(min(g_true), max(g_true))
        ax.set_ylim(1.8, 2.6)
        ax.set_title('Generated data and true model', fontsize=16)
        ax.plot(g_data, data, 'k.', label='Data set')
        ax.plot(g_true, Mixing.true_model(self, g_true), 'k', label='True model')
        
        ax.legend(fontsize=12)
        plt.show()
        
        
    #credible interval calculation (highest posterior density method)
    def credible_intervals(self, trace, fraction):
        
        '''
        A function to calculate the Bayesian credible intervals of a posterior distribution. This function
        uses the HPD (highest posterior density) method.
        
        :Example:
            Mixing.credible_intervals(trace=emcee_trace, fraction=0.95)
            
        Parameters:
        -----------
        trace         
            The trace generated by a sampler when sampling a variable to obtain its posterior distribution.
        
        fraction    
            The percent (in decimal form) requested by the user to set the credibility interval. 
            
        Returns:
        --------
        interval         
            The credibility interval bounds in a numpy array (format: [min, max]).
        '''
    
        sort_list = np.sort(np.copy(trace))
        total_samples = len(trace)
    
        int_samples = np.floor(fraction * total_samples).astype(int)
        int_width = sort_list[int_samples:] - sort_list[:total_samples-int_samples]
    
        min_int = np.argmin(int_width)
        
        interval = np.array([sort_list[min_int], sort_list[min_int+int_samples]])
    
        return interval
    
    
    def likelihood_low(self, g_data, data, sigma, loworder):
        
        '''
        The likelihood function for the data using the small-g expansion as the model in the 
        chi-squared.
        
        :Example:
            Mixing.likelihood_low(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), 
            sigma=np.array(), loworder=5)
            
        Parameters:
        -----------
        g_data     
            A linspace used to generate data points. 
            
        data      
            An array of data points generated or supplied by the user.
            
        sigma          
            An array of standard deviations at each point in 'data'. 
           
        loworder          
            The specific small-g expansion order desired for calculating the mixed model. 
            
        Returns:
        --------
            An array of the likelihood calculated at each data point. 
            
        '''
    
        prelow = (np.sqrt(2.0 * np.pi) * sigma)**(-1.0)
        insidelow = -0.5 * ((data - Mixing.low_g(self, g_data, loworder))/(sigma))**2.0
    
        return prelow*np.exp(insidelow)

    
    def likelihood_high(self, g_data, data, sigma, highorder):
        
        '''
        The likelihood function for the data using the large-g expansion as the model in the 
        chi-squared.
        
        :Example:
            Mixing.likelihood_high(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), 
            sigma=np.array(), highorder=23)
            
        Parameters:
        -----------
        g_data       
            A linspace used to generate data points. 
            
        data       
            An array of data points generated or supplied by the user.
            
        sigma         
            An array of standard deviations at each point in 'data'. 
           
        highorder          
            The specific large-g expansion order desired for calculating the mixed model. 
            
        Returns:
        --------
            An array of the likelihood calculated at each data point. 
        '''
    
        prehigh = (np.sqrt(2.0 * np.pi) * sigma)**(-1.0)
        insidehigh = -0.5 * ((data - Mixing.high_g(self, g_data, highorder))/(sigma))**2.0
    
        return prehigh*np.exp(insidehigh)
    
    
    def prior_gauss(self, par, mean, sig):
        
        '''
        A Gaussian prior that can be implemented for any parameter desired.
        
        :Example:
            Mixing.prior_gauss(par, mean=0.15, sig=0.01)
            
        Parameters:
        -----------
        par         
            The parameter that this prior distribution will be applied to.
        
        mean
            The mean of the parameter (can be an educated guess).  
            
        sig
            The standard deviation for the parameter (can also be a guess).
            
        Returns:
        --------
            The value of the prior distribution given the value of the parameter 'par'. 
        '''
    
        insidep = -0.5 * ((par - mean)/sig)**2.0 
    
        prep = -np.log(np.sqrt(2.0 * np.pi) * sig)
    
        return prep + insidep 


    def sampler_mix(self, params, g_data, data, sigma, loworder, highorder, mu, sig):

        '''
        The model mixing function sent to the sampler to find the values of the parameters in the 
        selected switching function. 

        :Example:
            emcee.EnsembleSampler(nwalkers, ndim, self.sampler_mix,
                                  args=[g_data, data, sigma, loworder, highorder, mu, sig])

        Parameters:
        -----------
        params
            The parameters that are being determined by the sampler, in an array (not input).

        g_data
            The linspace used to generate the data.

        data
            An array of data either generated or supplied by the user. 

        sigma 
            An array of standard deviations for each data point.

        loworder
            The order of the small-g expansion desired for the mixing calculation.

        highorder
            The order of the large-g expansion desired for the mixing calculation.

        mu
            The array of means given to the priors on the switching parameters.

        sig 
            The array of standard deviations given to the priors on the switching parameters.

        Returns:
        --------
        mixed_results
            The results of the mixing function for the entire linspace in g, in an array format.
        '''

        #set up arrays
        mixed_likelihood = np.empty([len(g_data)])
        log_ml = np.empty([len(g_data)])

        #likelihood mixing
        for i in range(len(g_data)):

            mixed_likelihood[i] = self.f(params, g_data[i]) * \
                                Mixing.likelihood_low(self, g_data[i], data[i], sigma[i], loworder) \
                                + (1.0- self.f(params, g_data[i])) * \
                                Mixing.likelihood_high(self, g_data[i], data[i], sigma[i], highorder)

            log_ml[i] = np.log(mixed_likelihood[i])

        total_lml = np.sum(log_ml)

        #add the priors
        mixed_results = total_lml + np.sum([self.prior_gauss(params[i], mu[i], sig[i]) \
                                            for i in range(len(params))])

        return mixed_results

        
    def mixed_model(self, g_data, data, sigma, loworder, highorder):
        
        '''
        A function that will run the emcee ensemble sampler for a given mixed model to determine at least one
        unknown parameter in the switching function selected. The function asks the user to decide which switching
        function to use, and runs the subsequent code to use the correct one. Functions sent to the sampler are 
        found in the class 'Switching'.
        
        :Example:
            Mixing.mixed_model(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), sigma=np.array(),
            loworder=5, highorder=23)
            
        Parameters:
        -----------
        g_data      
            The linspace over which the data was generated.
      
        data          
            An array of data points, either generated or supplied by the user.
            
        sigma     
            An array of standard deviations at each data point.
            
        loworder          
            The order of the small-g expansion desired for the mixed model to be calculated at.
            
        highorder           
            The order of the large-g expansion desired for the mixed model to be calculated at.
            
        Returns:
        --------
            The sampler results, contained in a sampler object, from the determination of the
            unknown parameter. 
            
        '''

        #dictionary of switching functions
        self.function_mappings = {
            'logistic': self.logistic,
            'cdf': self.cdf,
            'cosine': self.switchcos
        }
        
        #ask user which switching function to use
        self.choice = input('What switching function would you like to use: logistic, cdf, or cosine?')
        
        #enter mu and sigma for both parameters
        mu = []
        sig = []

        if self.choice == 'logistic' or self.choice == 'cdf':
            for i in range(2):
                mu.append( float(input(f"Enter a guess for the mean of the parameter beta_{i}: ")))
                sig.append(float(input(f"Enter a guess for the standard deviation of the parameter beta_{i}: ")))
        elif self.choice == 'cosine':
            for i in range(3):
                mu.append(float(input(f"Enter a guess for the mean of the parameter g_{i+1}: ")))
                sig.append(float(input(f"Enter a guess for the standard deviation of the parameter g_{i+1}: ")))
        else:
            raise ValueError('Switching function requested is not found. Select one of the valid options.')
                
        mu = np.asarray(mu)
        sig = np.asarray(sig)

        #set up sampler
        ndim = len(mu)
        nwalkers = int(3*ndim + 1)
        nsteps = 1000

        total_samples = nwalkers * nsteps

        print('Using {} walkers with {} steps each, for a total of {} samples.'.format(nwalkers, nsteps, total_samples))

        starting_points = np.random.randn(nwalkers, ndim)

        #set the switching function
        self.f = self._select_function(self.choice)
        print(self.f, self.choice)
        
        #call emcee
        sampler_mixed = emcee.EnsembleSampler(nwalkers, ndim, self.sampler_mix, \
                                            args=[g_data, data, sigma, loworder, highorder, mu, sig])
        now = time.time()
        sampler_mixed.run_mcmc(starting_points, nsteps)
        stop = time.time()
        print('Calculation finished!')

        #time calculation
        elapsed = int(stop - now)
        if elapsed / 60 < 1.0:
            print(f"Duration = {elapsed} sec.")
        elif elapsed / 60 >= 1.0:
            minutes = int(elapsed / 60)
            seconds = int(elapsed - 60*minutes)
            print(f"Duration = {minutes} min, {seconds} sec.")

        #find the trace
        emcee_trace_mixed = self.burnin_trace(sampler_mixed, nwalkers, ndim)
        print(np.shape(emcee_trace_mixed))

        return emcee_trace_mixed


    def _select_function(self, x):

        '''
        An internal function that selects the proper switching function given the input from the user
        in the function 'Mixing.mixed_model'. 

        :Example:
            Mixing._select_function(x=self.choice)
        
        Parameters:
        -----------
        x
            The string that is input by the user to select one of the available switching functions. 
        
        Returns:
        --------
        self.function_mappings[x]
            The correct function label for the chosen switching function, converted from string to object.
        '''
        while True:
            try:
                return self.function_mappings[x]
            except KeyError:
                print('Invalid function, try again.')
     

    def burnin_trace(self, sampler_object, nwalkers, ndim):
        
        '''
        A small function to take the burn-in samples off of the sampler chain from the Mixing.mixed_model
        function, and to send back the trace of the sampler chain to Mixing.mixed_model.
        
        :Example:
            Mixing.burnin_trace(sampler_object=sampler_mixed, nwalkers=10, ndim=1)
            
        Parameters:
        -----------
        sampler_object         
            The chain sent back by the emcee sampler after it finishes running through the samples and walkers.
            
        nwalkers          
            The number of walkers for the sampler to use.
            
        ndim            
            The number of parameters the sampler is determining.
            
        Returns:
        ---------
        emcee_trace_mixed            
            The trace of the sampler chain with the user's desired number of burn-in samples removed.
        '''
        
        nburnin = int(input('How many burn-in samples per walker?'))

        total_burnin = nburnin * nwalkers

        print('Using {} samples as burn-in, for a total of {} burn-in samples.'.format(nburnin, total_burnin))

        #throw out the burn-in and reshape again
        emcee_trace_mixed = sampler_object.chain[:, nburnin:, :].reshape(-1, ndim).T
        
        return emcee_trace_mixed



if __name__=="__main__":
    pass
