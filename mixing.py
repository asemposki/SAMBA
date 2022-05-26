import numpy as np
from scipy import special, integrate
from cycler import cycler
import math
import time
import emcee
import corner
import warnings
import statistics
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from priors import Priors
from uncertainties import Uncertainties

__all__ = ['Models', 'Mixing']


class Models():

    def __init__(self, highorder):

        '''
        The class containing the expansion models from Honda's paper
        and the means to plot them. 

        :Example:
            Models()

        Parameters:
        -----------
        None.

        Returns:
        --------
        None.
        '''
        #check type and assign to class variable
        if isinstance(highorder, float) == True or isinstance(highorder, int) == True:
            highorder = np.array([highorder])

        self.highorder = highorder 

        return None 


    def low_g(self, g, loworder):
        
        '''
        A function to calculate the small-g divergent asymptotic expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Models.low_g(g=np.linspace(0.0, 0.5, 20), loworder=np.array([5, 10, 20]))
            
        Parameters:
        -----------
        g : linspace
            The linspace of the coupling constant for this calculation. 
            
        loworder : numpy.ndarray, int, float 
            The array of different expansion orders to calculate. These indicate the highest power the expansions 
            are calculated up to. 
            
        Returns:
        --------
        output : numpy.ndarray
            The array of values of the expansion in small-g at each point in g_true space, for each value of 
            loworder (highest power the expansion reaches).
        '''
        
        #converts float or int to an array
        if isinstance(loworder, float) == True or isinstance(loworder, int) == True:
            loworder = np.array([loworder])
        output = []
        
        for order in loworder:
            low_c = np.empty([int(order)+1])
            low_terms = np.empty([int(order) + 1])

            #if g is an array, execute here
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
            
            #if g is a single value, execute here
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

        
    def high_g(self, g):
        
        '''
        A function to calculate the large-g convergent Taylor expansion for a given range in the coupling 
        constant, g.
        
        :Example:
            Models.high_g(highorder=np.array([5, 10, 20]))
            
        Parameters:
        -----------
        g : linspace
            The linspace of the coupling constant for this calculation.
            
        Returns
        -------
        output : numpy.ndarray        
            The array of values of the expansion at large-g at each point in g_true space, for each value of highorder
            (highest power the expansion reaches).
        '''

        #converts a float or int into an array
        if isinstance(self.highorder, float) == True or isinstance(self.highorder, int) == True:
            self.highorder = np.array([self.highorder])
        output = []
        
        for order in self.highorder:
            high_c = np.empty([int(order) + 1])
            high_terms = np.empty([int(order) + 1])
            
            #if g is an array, execute here
            try:
                value = np.empty([len(g)])
        
                #loop over array in g
                for i in range(len(g)):

                    #loop over orders
                    for k in range(int(order)+1):

                        high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                        high_terms[k] = (high_c[k] * g[i]**(-k)) / np.sqrt(g[i]) #* np.sqrt(g[i])  #multiplying by sqrt(g)

                    #sum the terms for each value of g
                    value[i] = np.sum(high_terms)

                output.append(value)

                data = np.array(output, dtype = np.float64)
        
            #if g is a single value, execute here           
            except:
                value = 0.0

                #loop over orders
                for k in range(int(order)+1):

                    high_c[k] = special.gamma(k/2.0 + 0.25) * (-0.5)**k / (2.0 * math.factorial(k))

                    high_terms[k] = (high_c[k] * g**(-k)) / np.sqrt(g) #* np.sqrt(g)  #multiplying by sqrt(g)

                #sum the terms for each value of g
                value = np.sum(high_terms)
                data = value
                
        return data 


    def true_model(self, g):
        
        '''
        The true model of the zero-dimensional phi^4 theory partition function using an input linspace.
        
        :Example:
            Models.true_model(g=np.linspace(0.0, 0.5, 100))
            
        Parameters:
        -----------
        g : linspace
            The linspace for g desired to calculate the true model. This can be the g_true linspace, g_data
            linspace, or another linspace of the user's choosing. 
            
        Returns:
        -------
        model : numpy.ndarray        
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
   

    def plot_models(self, g, loworder):
        
        '''
        A plotting function to produce a figure of the model expansions calculated in Models.low_g and Models.high_g, 
        and including the true model calculated using Mixing.true_model.
        
        :Example:
            Mixing.plot_models(g=np.linspace(0.0, 0.5, 100), lowk=np.array([5, 23]), highk=np.array([5, 23]))
            
        Parameters:
        -----------
        g : linspace
            The linspace in on which the models will be plotted here. 

        loworder : numpy.ndarray, int, float   
            As in Models.low_g, the highest powers to calculate the series to for the asymptotic small-g expansion.
                 
        Returns
        -------
        None.
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)

        #x and y limits
        if min(g) == 1e-6:
            ax.set_xlim(0.0, max(g))
        else:
            ax.set_xlim(min(g), max(g))
        ax.set_ylim(1.2,3.2)
        ax.set_yticks([1.2, 1.6, 2.0, 2.4, 2.8, 3.2])
  
        #plot the true model 
        ax.plot(g, self.true_model(g), 'k', label='True model')
        
        #add linestyle cycler
        linestyle_cycler = cycler(linestyle=['dashed', 'dotted', 'dashdot', 'dashed', 'dotted', 'dashdot'])
        ax.set_prop_cycle(linestyle_cycler)
                
        #for each small-g order, plot
        for i in loworder:
            ax.plot(g, self.low_g(g, i.item())[0], color='r', label=r'$f_s$ ($N_s$ = {})'.format(i))

        #for each large-g order, calculate and plot
        for i,j in zip(range(len(self.highorder)), self.highorder):
            ax.plot(g, self.high_g(g)[i,:], color='b', label=r'$f_l$ ($N_l$ = {})'.format(j))
                    
        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name, bbox_inches='tight')
        else:
            pass

        return None
        
         
    def residuals(self, loworder):
        
        '''
        A calculation and plot of the residuals of the model expansions vs the true model values at each point in g.
        g is set internally for this plot, as the plot must be shown in loglog format to see the power law of the
        residuals. 
        
        :Example:
            Mixing.residuals(loworder=np.array([5, 10, 20]), highorder=np.array([5, 10, 20]))
            
        Parameters:
        -----------
        loworder : int, float        
            The array of highest power series orders for the asymptotic, small-g expansion.
                  
        Returns:
        --------
        None. 
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('Residual', fontsize=22)
        ax.set_title('F(g): residuals', fontsize=22)
        ax.set_xlim(1e-2, 10.)
        ax.set_ylim(1e-6, 1e17)

        #set range for g
        g_ext = np.logspace(-6., 6., 800)
        
        #set up marker cycler
        marker_cycler = cycler(marker=['.', '*', '+', '.', '*', '+'])
        ax.set_prop_cycle(marker_cycler)

        #calculate true model
        value_true = self.true_model(g_ext)
        
        #for each small-g order, plot
        for i in np.array([loworder]):
            valuelow = self.low_g(g_ext, i)
            residlow = (valuelow - value_true)/value_true
            ax.loglog(g_ext, abs(residlow[0,:]), 'r', linestyle="None", label=r"$F_s({})$".format(i))

        #for each large-g order, plot
        valuehi = np.zeros([len(self.highorder), len(g_ext)])
        residhi = np.zeros([len(self.highorder), len(g_ext)])

        for i,j in zip(range(len(self.highorder)), self.highorder):
            valuehi[i,:] = self.high_g(g_ext)[i]
            residhi[i,:] = (valuehi[i,:] - value_true)/value_true
            ax.loglog(g_ext, abs(residhi[i,:]), 'b', linestyle="None", label=r"$F_l({})$".format(j))
        
        ax.legend(fontsize=18)
        plt.show()


class Mixing(Models, Priors): 
    
    
    def __init__(self, highorder):
        
        '''
        This class is designed with all of the necessary functions for creating a data set, plotting it 
        along with the true model, and calculating expansions of specific orders of the true model to mix.
        Dependent on the Models class to run the expansion functions. 
    
        :Example:            
            Mixing()
            
        Parameters:
        -----------
        None.
            
        Returns:
        --------
        None.
        
        '''    

        print('Instantiating the linear mixture model method.')

        #check type and create class variables
        if isinstance(highorder, float) == True or isinstance(highorder, int) == True:
            highorder = np.array([highorder])

        self.highorder = highorder 

        return None
        
        
    def add_data(self, g_true, g_data, data=None, sigma=None):
        
        '''
        A data generation function that generates data based on the g_data linspace provided (with the number of 
        points chosen by the user) and the error desired on each point (also input by the user), or accepts the
        user's input of an array of data and standard deviations of the data points. 
        
        :Example:
            Mixing.add_data(g_true=np.linspace(0.0, 0.5, 100), g_data=np.linspace(0.0, 0.5, 20))
        
        Parameters:
        -----------
        g_true : linspace     
            The linspace desired for the true model to be calculated.
        
        g_data : linspace    
            The linspace input for the data to be generated within. 

        data : numpy.ndarray
            The data array entered by the user; if user wishes to generate data, this remains set to None.
        
        sigma : numpy.ndarray
            The standard deviation array entered by the user; if user wishes to generate data, this will
            remain set to None. 
            
        Returns:
        --------
        data : numpy.ndarray         
            The array of data (generated or entered by the user).
            
        sigma : numpy.ndarray    
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
            data=np.array([]))
            
        Parameters:
        -----------
        g_true : linspace
            The linspace desired for the true model to be calculated.
            
        g_data : linspace
            The linspace over which the data was generated. 
            
        data : numpy.ndarray
            The array of data generated using the Mixing.add_data function.
            
        Returns:
        --------
        None.
        
        '''
        
        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(min(g_true), max(g_true))

        #ylim settings
        ylim = input('\ny-limits (enter "auto" if unknown): ')
        if ylim == "auto":
            ax.set_ylim(1.8,2.6)
        else:
            ax.set_ylim(tuple(map(float, ylim.split(','))))

        ax.set_title('Generated data and true model', fontsize=22)
        ax.plot(g_data, data, 'k.', label='Data set')
        ax.plot(g_true, Mixing.true_model(self, g_true), 'k', label='True model')
        
        ax.legend(fontsize=18)
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name, bbox_inches='tight')
        else:
            pass

        return None
        
        
    def credible_intervals(self, trace, fraction):
        
        '''
        A function to calculate the Bayesian credible intervals of a posterior distribution. This function
        uses the HPD (highest posterior density) method.
        
        :Example:
            Mixing.credible_intervals(trace=emcee_trace, fraction=0.95)
            
        Parameters:
        -----------
        trace : numpy.ndarray         
            The trace generated by a sampler when sampling a variable to obtain its posterior distribution.
        
        fraction : float    
            The percent (in decimal form) requested by the user to set the credibility interval. 
            
        Returns:
        --------
        interval : numpy.ndarray         
            The credibility interval bounds in a numpy array (format: [min, max]).
        '''
    
        sort_list = np.sort(np.copy(trace))
        total_samples = len(trace)
    
        int_samples = np.floor(fraction * total_samples).astype(int)
        int_width = sort_list[int_samples:] - sort_list[:total_samples-int_samples]
    
        min_int = np.argmin(int_width)
        
        interval = np.array([sort_list[min_int], sort_list[min_int+int_samples]])
    
        return interval
    
    
    def likelihood_low(self, g_data, data, sigma, siglow, loworder):
        
        '''
        The likelihood function for the data using the small-g expansion as the model in the 
        chi-squared.
        
        :Example:
            Mixing.likelihood_low(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), 
            sigma=np.array(), loworder=5)
            
        Parameters:
        -----------
        g_data : linspace     
            A linspace used to generate data points. 
            
        data : numpy.ndarray      
            An array of data points generated or supplied by the user.
            
        sigma : numpy.ndarray          
            An array of standard deviations at each point in 'data'. 
           
        loworder : numpy.ndarray, int, float          
            The specific small-g expansion order desired for calculating the mixed model. 
            
        Returns:
        --------
            An array of the likelihood calculated at each data point. 
            
        '''

        #set up the uncertainties using experimental & theory errors
        sigma_t = np.sqrt(sigma**2.0 + siglow**2.0)
    
        prelow = (np.sqrt(2.0 * np.pi) * sigma_t)**(-1.0)
        insidelow = -0.5 * ((data - Models.low_g(self, g_data, loworder))/(sigma_t))**2.0
        
        return prelow*np.exp(insidelow)

    
    def likelihood_high(self, g_data, data, sigma, sighigh):
        
        '''
        The likelihood function for the data using the large-g expansion as the model in the 
        chi-squared.
        
        :Example:
            Mixing.likelihood_high(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), 
            sigma=np.array(), highorder=23)
            
        Parameters:
        -----------
        g_data : linspace       
            A linspace used to generate data points. 
            
        data : numpy.ndarray       
            An array of data points generated or supplied by the user.
            
        sigma : numpy.ndarray         
            An array of standard deviations at each point in 'data'. 
           
        Returns:
        --------
            An array of the likelihood calculated at each data point. 
        '''

        #set up the uncertainties using experimental & theory errors
        sigma_t = np.sqrt(sigma**2.0 + sighigh**2.0)
    
        prehigh = (np.sqrt(2.0 * np.pi) * sigma_t)**(-1.0)
        insidehigh = -0.5 * ((data - Models.high_g(self, g_data))/(sigma_t))**2.0
    
        return prehigh*np.exp(insidehigh)


    def sampler_mix(self, params, g_data, data, sigma, siglow, sighigh, loworder):

        '''
        The model mixing function sent to the sampler to find the values of the parameters in the 
        selected mixing function. 

        :Example:
            emcee.EnsembleSampler(nwalkers, ndim, self.sampler_mix,
                                  args=[g_data, data, sigma, loworder, mu, sig])

        Parameters:
        -----------
        params : numpy.ndarray
            The parameters that are being determined by the sampler, in an array (not input).

        g_data : linspace
            The linspace used to generate the data.

        data : numpy.ndarray
            An array of data either generated or supplied by the user. 

        sigma : numpy.ndarray
            An array of standard deviations for each data point.

        loworder : numpy.ndarray, int, float
            The order of the small-g expansion desired for the mixing calculation.

        Returns:
        --------
        mixed_results : numpy.ndarray
            The results of the mixing function for the entire linspace in g, in an array format.
        '''

        #set up arrays
        mixed_likelihood = np.empty([len(g_data)])
        log_ml = np.empty([len(g_data)])

        #test prior first
        logprior = Priors.lpdf(self, params)

        if math.isnan(logprior) == True or np.isinf(-logprior) == True:
            return -np.inf

        else:

            #likelihood mixing
            for i in range(len(g_data)):
                mixed_likelihood[i] = self.f(params, g_data[i]) * \
                                    Mixing.likelihood_low(self, g_data[i], data[i], sigma[i], siglow[i], loworder) \
                                    + (1.0- self.f(params, g_data[i])) * \
                                    Mixing.likelihood_high(self, g_data[i], data[i], sigma[i], sighigh[i])

                if mixed_likelihood[i] <= 0.0:
                    return -np.inf

                log_ml[i] = np.log(mixed_likelihood[i])

            total_lml = np.sum(log_ml)

            #add the priors
            mixed_results = total_lml + Priors.lpdf(self, params)

            return mixed_results

        
    def mixed_model(self, g_data, data, sigma, loworder):
        
        '''
        A function that will run the emcee ensemble sampler for a given mixed model to determine at least one
        unknown parameter in the mixing function selected. The function asks the user to decide which mixing
        function to use, and runs the subsequent code to use the correct one. Functions sent to the sampler are
        static methods defined at the end of this class.
        
        :Example:
            Mixing.mixed_model(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), sigma=np.array(),
            loworder=5)
            
        Parameters:
        -----------
        g_data : linspace      
            The linspace over which the data was generated.
      
        data : numpy.ndarray          
            An array of data points, either generated or supplied by the user.
            
        sigma : numpy.ndarray     
            An array of standard deviations at each data point.

        sigma_t : numpy.ndarray
            A 2D array of standard deviations at each data point. The first column is those
            from the small-g expansion; the second column is those from the large-g expansion.
            
        loworder : numpy.ndarray, int, float          
            The order of the small-g expansion desired for the mixed model to be calculated at.
            
        Returns:
        --------
            The sampler results, contained in a sampler object, from the determination of the
            unknown parameter. 
            
        '''

        #dictionary of mixing functions
        self.function_mappings = {
            'logistic': self.logistic,
            'cdf': self.cdf,
            'cosine': self.switchcos,
        }
        
        #ask user which mixing function to use
        self.choice = input('What mixing function would you like to use: logistic, cdf, or cosine?')
        
        if self.choice == 'logistic' or self.choice == 'cdf':
            ndim = 2 
        elif self.choice == 'cosine':
            ndim = 3
        else:
            raise ValueError('Mixing function requested is not found. Select one of the valid options.')

        #call Uncertainties class for the theory errors (variances)
        err = Uncertainties()
        siglow = np.sqrt(err.variance_low(g_data, loworder[0]))
        sighigh = np.sqrt(err.variance_high(g_data, self.highorder[0]))

        #set up sampler
        nwalkers = 2*int(3*ndim + 1)
        nsteps = int(input('Enter the number of steps per walker.'))

        total_samples = nwalkers * nsteps

        print('Using {} walkers with {} steps each, for a total of {} samples.'.format(nwalkers, nsteps, total_samples))

        #set starting points per parameter
        starting_points = np.zeros((nwalkers, ndim))
        starting_points[:,0] = np.random.uniform(0.12, 0.18, nwalkers)
        starting_points[:,2] = np.random.uniform(0.19, 0.24, nwalkers)
        starting_points[:,1] = np.random.uniform(0.25, 0.30, nwalkers)

        #set the mixing function
        self.f = self._select_function(self.choice)
        
        #call emcee
        sampler_mixed = emcee.EnsembleSampler(nwalkers, ndim, self.sampler_mix, \
                                            args=[g_data, data, sigma, siglow, sighigh, loworder])
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

        return sampler_mixed, emcee_trace_mixed


    def _select_function(self, x):

        '''
        An internal function that selects the proper mixing function given the input from the user
        in the function 'Mixing.mixed_model'. 

        :Example:
            Mixing._select_function(x=self.choice)
        
        Parameters:
        -----------
        x : string
            The string that is input by the user to select one of the available mixing functions. 
        
        Returns:
        --------
        self.function_mappings[x]
            The correct function label for the chosen mixing function, converted from string to object.
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
        sampler_object : emcee object         
            The chain sent back by the emcee sampler after it finishes running through the samples and walkers.
            
        nwalkers : int     
            The number of walkers for the sampler to use.
            
        ndim : int            
            The number of parameters the sampler is determining.
            
        Returns:
        ---------
        emcee_trace_mixed : numpy.ndarray            
            The trace of the sampler chain with the user's desired number of burn-in samples removed.
        '''
        
        nburnin = int(input('How many burn-in samples per walker?'))

        total_burnin = nburnin * nwalkers

        print('Using {} samples as burn-in, for a total of {} burn-in samples.'.format(nburnin, total_burnin))

        #throw out the burn-in and reshape again
        emcee_trace_mixed = sampler_object.chain[:, nburnin:, :].reshape(-1, ndim).T
        
        return emcee_trace_mixed

    
    def _autocorrelation(self, chain, max_lag=100):

        '''
        ***FINISH DOCUMENTATION***
        Copied from Christian Forssen's nuclear TALENT school work.
        Re-write this. Internal function, no outside access. 
        '''

        #determine the autocorrelation length
        dimension = len(chain)
        acors = np.empty(max_lag+1)
        if max_lag > len(chain)/5:
            warnings.warn('max_lag is more than one fifth the chain length')
        
        # Create a copy of the chain with average zero
        chain1d = chain - np.average(chain)
        for lag in range(max_lag+1):
            unshifted = None
            shifted = chain1d[lag:]
            if 0 == lag:
                unshifted = chain1d
            else:
                unshifted = chain1d[:-lag]
            normalization = np.sqrt(np.dot(unshifted, unshifted))
            normalization *= np.sqrt(np.dot(shifted, shifted))
            acors[lag] = np.dot(unshifted, shifted) / normalization 

        return acors


    def stats_chain(self, chain): #---> FURTHER BREAK THIS GIGUNDOUS FUNCTION UP (after reducing it)

        '''
        Calculates the autocorrelation time and thins the samples
        accordingly for a better estimate of the mean, median, and MAP values. 

        :Example: 
            Mixing.stats_chain(chain=emcee.object, params=np.array([]))

        Parameters:
        -----------
        chain : emcee object
            The object resulting from sampling the parameters
            using emcee. The chain of samples must be extracted
            from it. 

        params : numpy.ndarray
            The initial parameter array given to emcee. 

        Returns:
        --------
        median_results : numpy.ndarray
            Each of the median parameter values found from the 
            sampling.

        mean_results : numpy.ndarray
            Each of the mean parameter values found from the
            sampling. 
        '''

        #retrieve the chain
        chain_result = chain.chain[:,:,:]

        #set up arrays ---> FIX THIS ONCE WE CAN GENERALIZE
        chain1 = chain_result[:,:,0]
        chain2 = chain_result[:,:,1]
        chain3 = chain_result[:,:,2]
        
        #flatten each individual array
        flat1 = chain1.flatten()
        flat2 = chain2.flatten()
        flat3 = chain3.flatten()

        #call autocorrelation to find the lengths
        post_acors1 = self._autocorrelation(flat1, max_lag=200)
        post_acors2 = self._autocorrelation(flat2, max_lag=200)
        post_acors3 = self._autocorrelation(flat3, max_lag=200)

        # #plot the autocorrelation results
        # fig, ax = plt.subplots(3,1, figsize=(8,6), dpi=200)
        # ax[0].plot(post_acors1)
        # ax[1].plot(post_acors2)
        # ax[2].plot(post_acors3) 

        # for i in range(3):
        #     ax[i].set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))

        # ax[0].set_title(f'Autocorrelation function: before thinning: g1, g2, g3')

        #check the results and ask user for range to determine time
        response = input('What range of indices should be used for the autocorrelation time? (Enter in int:int format.)')

        #determine the autocorrelation time
        post_rho1 = post_acors1[tuple(map(int, response.split(':')))]
        post_rho2 = post_acors2[tuple(map(int, response.split(':')))]
        post_rho3 = post_acors3[tuple(map(int, response.split(':')))]

        # fig,ax = plt.subplots(3,1, figsize=(8,6), dpi=200)
        post_y = np.arange(10)
        post_x1 = -np.log(post_rho1)
        post_x2 = -np.log(post_rho2)
        post_x3 = -np.log(post_rho3)
                
        # ax[2].set_xlabel(r'ln($\rho(h)$)')
        # for i in range(3):
        #     ax[i].set_ylabel('h')
        # ax[0].set_title('Autocorrelation time')
        # ax[0].plot(post_x1, post_y)
        # ax[1].plot(post_x2, post_y)
        # ax[2].plot(post_x3, post_y)

        #linear fits
        p1, cov1 = np.polyfit(post_x1, post_y, 1, cov=True)
        p2, cov2 = np.polyfit(post_x2, post_y, 1, cov=True)
        p3, cov3 = np.polyfit(post_x3, post_y, 1, cov=True)

        print('The autocorrelation information is: p = {}; cov = {}'.format(p1, cov1))
        print('The autocorrelation information is: p = {}; cov = {}'.format(p2, cov2))
        print('The autocorrelation information is: p = {}; cov = {}'.format(p3, cov3))

        print('The autocorrelation time is: {}'.format(p1[0]))
        print('The autocorrelation time is: {}'.format(p2[0]))
        print('The autocorrelation time is: {}'.format(p3[0]))

        # ax[0].plot(post_x1, p1[0]*post_x1 + p1[1])
        # ax[1].plot(post_x2, p2[0]*post_x2 + p2[1])
        # ax[2].plot(post_x3, p3[0]*post_x3 + p3[1])

        #thin the samples given the determined autocorrelation time
        thin1 = []
        thin2 = []
        thin3 = []

        #get the autocorrelation time we use for all 3 parameters
        if p1[0] > p2[0]:
            time = p1[0]
        else:
            time = p2[0]
            if p3[0] > time:
                time = p3[0]
        time = int(time)
                
        for i in range(len(flat2)):
            if i % time == 0:
                thin1.append(flat1[i])
                thin2.append(flat2[i])
                thin3.append(flat3[i])
                            
        #array thinned samples
        thin1 = np.array(thin1)
        thin2 = np.array(thin2)
        thin3 = np.array(thin3)

        #stack traces back together
        thin = np.vstack((thin1, thin2, thin3))

        #median calculation
        median_1 = statistics.median(thin[0,:])
        median_2 = statistics.median(thin[1,:])
        median_3 = statistics.median(thin[2,:])

        #mean calculation
        mean_1 = np.mean(thin[0,:])
        mean_2 = np.mean(thin[1,:])
        mean_3 = np.mean(thin[2,:])

        #MAP calculation
        map_1 = np.argmax(thin[0, :])
        map_2 = np.argmax(thin[1, :])
        map_3 = np.argmax(thin[2, :])

        #arrays
        mean_results = np.array([mean_1, mean_2, mean_3])
        median_results = np.array([median_1, median_2, median_3])
        map_results = np.array([map_1, map_2, map_3])

        print('The median values are: {}'.format(median_results))
        print('The mean values are: {}'.format(mean_results))
        print('The MAP values are: {}'.format(map_1, map_2, map_3))

        return mean_results, median_results, map_results

    
    def stats_trace(self, trace, ndim):

        '''
        A function to calculate the mean and credible intervals corresponding to
        each parameter. The trace plots for each parameter are plotted. 

        :Example:
            Mixing.stats_trace(trace=np.array([]), ndim=len(trace))

        Parameters:
        -----------
        trace : numpy.ndarray
            The trace from the sampler object that was generated when estimating the
            parameters of the mixing function.

        ndim : int
            The number of parameters in the mixing function that were sampled. To
            find this, simply use len(trace) in the main code and pass it here.

        Returns:
        --------
        mean : numpy.ndarray
            The array of mean values for each parameter.
        
        ci : numpy.ndarray
            The array of sets of credible interval bounds for each parameter. 
        '''

        #calculate mean and credible intervals
        mean = []
        ci = []

        for i in range(ndim):
            mean.append(np.mean(trace[i].T))
            ci.append(Mixing.credible_intervals(self, trace[i], 0.95))

        mean = np.asarray(mean)
        ci = np.asarray(ci)

        #dpi setting
        dpi = int(input('Set a dpi for the figure.'))

        #plot traces with mean and credible intervals
        fig, ax = plt.subplots(ndim,1,figsize=(7,4*ndim), dpi=dpi)

        for irow in range(ndim):
            ax[irow].plot(trace[irow].T, 'k')
            ax[irow].set_ylabel('Parameter {0}'.format(irow+1), fontsize=22)
            ax[irow].set_title('Trace plot: Parameter {0}'.format(irow+1), fontsize=22)

            ax[irow].axhline(y=mean[irow], color='b', linestyle='solid', label='Mean')
            ax[irow].axhline(y=ci[irow, 0], color='b', linestyle='dashed')
            ax[irow].axhline(y=ci[irow, 1], color='b', linestyle='dashed')

        answer = input('Do you want to display the median with the mean? (yes/no)')

        if answer == 'yes':
            med = []

            for i in range(ndim):
                med.append(np.median(trace[i].T))

            med = np.asarray(med)

            for irow in range(ndim):
                ax[irow].axhline(y=med[irow], color='r', linestyle='solid', label='Median')

                ax[irow].legend(loc='upper right')

        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)

        #corner plots
        fig, axs = plt.subplots(ndim,ndim, figsize=(8,8), dpi=600)
        label = ["Parameter 1", "Parameter 2", "Parameter 3"]
        corner.corner(trace.T,labels=label, \
            quantiles=[0.16, 0.5, 0.84],fig=fig,show_titles=True, label_kwargs=dict(fontsize=16))
        
        return mean, ci 


    @staticmethod
    def logistic(params, g):
        
        '''
        A basic logistic function often used in machine learning, implemented here with two free
        parameters to be determined via sampling.
        
        :Example:
            logistic(params=np.array(), g=0.5)
            
        Parameters:
        -----------
        params : numpy.ndarray
            The array of parameters the sampler will determine (here labelled beta0 and beta1, where
            beta0 controls the location of the function and beta1 controls the slope). 

        Returns:
        --------
        mixing : float        
            The result of the logistic function given the value g.
        '''
        beta0, beta1 = params
    
        mixing = (1.0 + np.exp(-(beta0 + g*beta1)))**(-1.0)
    
        return mixing
    
    @staticmethod
    def cdf(params, g):
        
        '''
        The cumulative distribution function of a standard normal distribution, with two free parameters
        determined by sampling.
        
        :Example:
            cdf(params=np.array(), g=0.5)
        
        Parameters:
        -----------
        params : numpy.ndarray
            The array of parameters the sampler will determine (here labelled beta0 and beta1, where
            beta0 controls the location of the function and beta1 controls the slope). 
        
        g : float
            The value of g the cdf is calculated at.

        Returns:
        --------
        function : float         
            The result of the cdf function at the value of g. 
        '''
        beta0, beta1 = params
    
        function = (1.0 + math.erf(beta0 + g*beta1)/np.sqrt(2.0)) / 2.0
    
        return function

    @staticmethod
    def hypertan(params, g):

        #params unpack
        theta0, theta1 = params

        #set up the tanh function
        hyperfunction = 0.5 + 0.5*np.tanh(-(theta0 + g*theta1))

        return hyperfunction
    
    @staticmethod
    def switchcos(params, g):
        
        '''
        A piecewise function using two constants at either end, and two cosine functions in the centre,
        to be used as a mixing function. One free parameter, g3, is found by sampling. 
        
        :Example:
            switchcos(params=np.array(), g=0.5)
            
        Parameters:
        -----------
        params : numpy.ndarray
            The array of parameters to be determined by the sampler (here labelled g1, g2, and g3, where
            g1 is the separation point between the first constant function and the first cosine function, 
            g2 is the separation point between the second cosine function and the second constant function, 
            and g3 is the point between the two cosine functions). 

        g : float
            The value of g that this cosine function is calculated at.

        Returns:
        --------
            The value of the function at a specific point in g. 
        '''

        #unpack the parameters
        g1, g2, g3 = params

        if g1 > g2 or g2 < g3 or g1 > g3:
            return -np.inf

        if g <= g1:
            return 1.0
    
        elif g <= g3:
            return (1.0 + np.cos((np.pi/2.0) * ((g - g1)/(g3 - g1))))/2.0
    
        elif g < g2:
            return 0.5 + np.cos((np.pi/2.0) * (1.0 + ((g - g3)/(g2 - g3))))/2.0
    
        else:
            return 0.0


    def ppd(self, trace, g, loworder):
        
        '''
        A function to calculate the posterior predictive distribution (PPD) for any chosen mixing function 
        defined in this class. 
        
        :Example:
            Mixing.ppd(trace, g=np.linspace(0.0, 0.5, 100), loworder=5)
            
        Parameters:
        -----------
        trace : numpy.ndarray
            The trace of each of the parameters from the sampler.
            
        g : linspace
            The linspace desired to calculate the PPD across.
            
        loworder : numpy.ndarray, int, float
            The order of the small-g expansion to be calculated in the mixing model.
             
        Returns:
        --------
        result_array : numpy.ndarray
            The array of results of the PPD for each of the points in the linspace g. 
        '''

        result_array = np.empty([len(g), len(trace[0].T)])
        gmax = max(g)

        #determine which mixing function was used
        if self.choice == 'logistic' or self.choice == 'cdf':
    
            for i in range(len(g)):
                for j in range(len(trace[0].T)):
            
                    if (Models.low_g(self, g[i], loworder) - Models.high_g(self, g[i]))\
                    > 0.1 and g[i] > (0.25*gmax):
                        result_array[i,j] = Models.high_g(self, g[i])
                    
                    elif (Models.low_g(self, g[i], loworder) - Models.high_g(self, g[i])) > 0.1:
                        result_array[i,j] = Models.low_g(self, g[i], loworder)
                    
                    else:
                        params = np.array([trace[0, j], trace[1, j]])

                        result_array[i,j] = self.f(params, g[i])*Models.low_g(self, g[i], loworder) \
                                        + (1.0 - self.f(params, g[i])) \
                                        *Models.high_g(self, g[i])
        
            return result_array    

        elif self.choice == 'cosine':

            params = np.array([np.mean(trace[0,:]), np.mean(trace[1,:]), np.mean(trace[2,:])])

            for i in range(len(g)):
                for j in range(len(trace[0].T)):
                    
                    params = np.array([trace[0, j], trace[1, j], trace[2, j]])
                
                    result_array[i,j] = self.switchcos(params, g[i]) * Models.low_g(self, g[i], loworder) \
                                    + (1.0 - self.switchcos(params, g[i])) \
                                    * Models.high_g(self, g[i])

            return result_array
    
    
    def plot_ppd(self, results, g_data, g_true, g_ppd, data, ppd_results, ppd_intervals, percent, loworder):
        
        '''
        A plotting function that can be used to plot the posterior predictive distribution (PPD) results (mean and 
        credible interval) obtained from calling the functions above in the main code, as well as data generated, 
        the true model, and the small- and large-g expansions chosen for the mixed model calculation. 
        
        :Example:
            Mixing.plot_ppd(g_data=np.linspace(0.0, 0.5, 20), g_true=np.linspace(0.0, 0.5, 100), 
            g_ppd=np.linspace(0.0, 0.5, 200), data=np.array(), ppd_results=np.array(), ppd_intervals=np.array(),
            loworder=5)
            
        Parameters:
        -----------
        results : numpy.ndarray
            The mean or the median of the estimated parameters from the posterior draws. 

        g_data : linspace
            The linspace used to generate the data.
        
        g_true : linspace
            The linspace used to calculate the true model over the input space, as well as the expansions.
        
        g_ppd : linspace
            The linspace chosen to calculate the PPD over. 
        
        data : numpy.ndarray
            An array of data either generated or supplied by the user.
            
        ppd_results : numpy.ndarray
            An array of the mean of the PPD at each point in the g_ppd linspace.
        
        ppd_intervals : numpy.ndarray
            A 2D array of the credibility interval calculated for the PPD (containing both bounds).
            
        percent : float
            The percent credibility interval calculated for the variable ppd_intervals (used in the plot
            legend). 
        
        loworder : numpy.ndarray, int, float
            The order of the small-g expansion used in the mixed model.
          
        Returns:
        --------
        None.
        '''
        
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=6)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)

        ax.set_xlim(0.0, 1.0)
 
        ax.set_ylim(1.2, 3.2)
        ax.set_yticks([1.2, 1.6, 2.0, 2.4, 2.8, 3.2])

        ax.plot(g_data, data, 'k.', label='Data set')  
        ax.plot(g_true, Models.true_model(self, g_true), 'k', label='Exact')

        ax.plot(g_true, Models.low_g(self, g_true, loworder)[0,:], 'r--', label=r'$f_s$ ($N_s$ = {})'.format(loworder[0]))
        ax.plot(g_true, Models.high_g(self, g_true)[0,:], 'b--', label=r'$f_l$ ($N_l$ = {})'.format(self.highorder[0]))

        ax.plot(g_ppd, ppd_results, 'g', label='Mixed model')
        ax.plot(g_ppd, ppd_intervals[:,0], 'g', linestyle='dotted', label=r'{}\% CI (HPD)'.format(percent))
        ax.plot(g_ppd, ppd_intervals[:,1], 'g', linestyle='dotted')

        ax.fill_between(g_ppd, ppd_intervals[:,0], ppd_intervals[:,1], color='green', alpha=0.2)

        #parameter results
        ax.axvline(x=results[0], color='darkviolet', alpha=0.35, label=r"$\theta_{1}$, $\theta_{2}$, $\theta_{3}$")
        ax.axvline(x=results[1], color='darkviolet', alpha=0.35)
        ax.axvline(x=results[2], color='darkviolet', alpha=0.35)

        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        answer = input('Would you like to save the plot to a file (yes/no)?')

        if answer == 'yes':
            name = input('Enter a file name to use (include file type as .png, .pdf, etc.).')
            fig.savefig(name, bbox_inches='tight')
        else:
            pass
        
        return None


if __name__=="__main__":
    pass
