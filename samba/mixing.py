import numpy as np
import math
import time
import emcee
import corner
import warnings
import statistics
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .priors import Priors
from .models import Models, Uncertainties

__all__ = ['LMM']


class LMM(Models, Uncertainties): 
    
    
    def __init__(self, loworder, highorder, error_model='informative'):
        
        r'''
        This class is designed with all of the necessary functions for creating 
        a data set, plotting it along with the true model, and calculating expansions 
        of specific orders of the true model to mix. Dependent on the Models class to 
        run the expansion functions. 
    
        Example:            
            LMM(loworder=np.array([2]), highorder=np.array([2]), error_model='informative')
            
        Parameters:
            loworder (numpy.ndarray, int): The truncation order to which we calculate 
                the small-g expansion. 

            highorder (numpy.ndarray, int): The truncation order to which we calculate the 
                large-g expansion. 

            error_model (str): The error model chosen for this calculation. Can be either 
                'uninformative' or 'informative'. Default is 'informative'. 
            
        Returns:
            None.
        '''    

        #check type and create class variables
        if isinstance(loworder, float) == True or isinstance(loworder, int) == True:
            loworder = np.array([loworder])

        if isinstance(highorder, float) == True or isinstance(highorder, int) == True:
            highorder = np.array([highorder])

        self.loworder = loworder 
        self.highorder = highorder

        #instantiate the Models() and Priors() classes here
        self.m = Models(self.loworder, self.highorder)
        self.u = Uncertainties(error_model)
        self.p = Priors()

        return None
        
        
    def add_data(self, g_true, g_data, data=None, sigma=None, error=None, plot=True):
        
        r'''
        A data generation function that generates data based on the g_data linspace 
        provided (with the number of points chosen by the user) and the error desired on 
        each point (also input by the user), or accepts the user's input of an array of data 
        and standard deviations of the data points. 
        
        Example:
            LMM.add_data(g_true=np.linspace(0.0, 0.5, 100), g_data=np.linspace(0.0, 0.5, 20),
            error=0.01, plot=False)
        
        Parameters:
            g_true (linspace): The linspace desired for the true model to be calculated.
            
            g_data (linspace): The linspace input for the data to be generated within. 

            data (numpy.ndarray): The data array entered by the user; if user wishes to 
                generate data, this remains set to None.
            
            sigma (numpy.ndarray): The standard deviation array entered by the user; if 
                user wishes to generate data, this will remain set to None. 

            error (float): The error to put on the data set if the data set is not being 
                given by the user. Enter in decimal form (0.01 = 1%). Default is None. 

            plot (bool): The option to plot the data. Default is True. 
            
        Returns:
            data (numpy.ndarray): The array of data (generated or entered by the user).
                
            sigma (numpy.ndarray): The standard deviation at each data point (generated or 
                entered by the user).
        '''
        
        #if user has an array of data, skip data generation
        if data is None:

            if error is None:
                raise ValueError('Please enter a error in decimal form for the data set generation.')
            elif error < 0.0 or error > 1.0:
                raise ValueError('Error must be between 0.0 and 1.0.')

            #generate fake data  
            data = self.true_model(g_data)
            rand = np.random.RandomState()
            var = error*rand.randn(len(g_data))
            data = data*(1 + var)
        
            #calculate standard deviation
            sigma = error*data
        
        #plot the data and true model
        if plot is True:
            self.plot_data(g_true, g_data, data)
    
        return data, sigma
    
    
    def plot_data(self, g_true, g_data, data):
        
        r'''
        The plotting function to display the generated data and true model. 
        
        Example:
            LMM.plot_data(g_true=np.linspace(0.0, 0.5, 100), g_data=np.linspace(0.0, 0.5, 20), 
            data=np.array([]))
            
        Parameters:
            g_true (linspace): The linspace desired for the true model to be calculated.
                
            g_data (linspace): The linspace over which the data was generated. 
                
            data (numpy.ndarray): The array of data generated using the LMM.add_data function.
            
        Returns:
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
        ax.set_ylim(1.2,3.2)
        
        ax.plot(g_data, data, 'k.', label='Data set')
        ax.plot(g_true, self.m.true_model(g_true), 'k', label='True model')
        
        ax.legend(fontsize=18)
        plt.show()

        #save figure option
        # response = input('Would you like to save this figure? (yes/no)')

        # if response == 'yes':
        #     name = input('Enter a file name (include .jpg, .png, etc.)')
        #     fig.savefig(name, bbox_inches='tight')
     
        return None
        
    
    def likelihood_low(self, g_data, data, sigma, siglow):
        
        r'''
        The likelihood function for the data using the small-g expansion as the model in the 
        chi-squared.
        
        Example:
            LMM.likelihood_low(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), 
                sigma=np.array(), loworder=5)
            
        Parameters:
            g_data (linspace): A linspace used to generate data points. 
                
            data (numpy.ndarray): An array of data points generated or supplied by 
                the user.
                
            sigma (numpy.ndarray): An array of standard deviations at each point in 'data'. 
            
        Returns:
            An array of the likelihood calculated at each data point. 
        '''

        #set up the uncertainties using experimental & theory errors
        sigma_t = np.sqrt(sigma**2.0 + siglow**2.0)
    
        prelow = (np.sqrt(2.0 * np.pi) * sigma_t)**(-1.0)
        insidelow = -0.5 * ((data - self.m.low_g(g_data))/(sigma_t))**2.0
        
        return prelow*np.exp(insidelow)

    
    def likelihood_high(self, g_data, data, sigma, sighigh):
        
        r'''
        The likelihood function for the data using the large-g expansion as the model in the 
        chi-squared.
        
        Example:
            LMM.likelihood_high(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), 
                sigma=np.array(), highorder=23)
            
        Parameters:
            g_data (linspace): A linspace used to generate data points. 
                
            data (numpy.ndarray): An array of data points generated or supplied by 
                the user.
                
            sigma (numpy.ndarray): An array of standard deviations at each point 
                in 'data'. 
            
        Returns:
            An array of the likelihood calculated at each data point. 
        '''

        #set up the uncertainties using experimental & theory errors
        sigma_t = np.sqrt(sigma**2.0 + sighigh**2.0)
    
        prehigh = (np.sqrt(2.0 * np.pi) * sigma_t)**(-1.0)
        insidehigh = -0.5 * ((data - self.m.high_g(g_data))/(sigma_t))**2.0
    
        return prehigh*np.exp(insidehigh)


    def sampler_mix(self, params, g_data, data, sigma, siglow, sighigh):

        r'''
        The model mixing function sent to the sampler to find the values of the parameters in the 
        selected mixing function. 

        Example:
            emcee.EnsembleSampler(nwalkers, self.sampler_mix,
                args=[g_data, data, sigma])

        Parameters:
            params (numpy.ndarray): The parameters that are being determined by the 
                sampler, in an array (not input).

            g_data (linspace): The linspace used to generate the data.

            data (numpy.ndarray): An array of data either generated or supplied by 
                the user. 

            sigma (numpy.ndarray): An array of standard deviations for each data point.

        Returns:
            mixed_results (numpy.ndarray): The results of the mixing function for the 
                entire linspace in g, in an array format.
        '''

        #set up arrays
        mixed_likelihood = np.empty([len(g_data)])
        log_ml = np.empty([len(g_data)])

        #test prior first
        logprior = self.p.lpdf(params)

        if math.isnan(logprior) == True or np.isinf(-logprior) == True:
            return -np.inf

        else:

            #likelihood mixing
            for i in range(len(g_data)):
                mixed_likelihood[i] = self.f(params, g_data[i]) * \
                                    LMM.likelihood_low(self, g_data[i], data[i], sigma[i], siglow[i]) \
                                    + (1.0- self.f(params, g_data[i])) * \
                                    LMM.likelihood_high(self, g_data[i], data[i], sigma[i], sighigh[i])

                if mixed_likelihood[i] <= 0.0:
                    return -np.inf

                log_ml[i] = np.log(mixed_likelihood[i])

            total_lml = np.sum(log_ml)

            #add the priors
            mixed_results = total_lml + self.p.lpdf(params)

            return mixed_results

        
    def mixed_model(self, g_data, data, sigma, mixing_function='cosine', nsteps=1000):
        
        r'''
        A function that will run the emcee ensemble sampler for a given mixed model to determine at least one
        unknown parameter in the mixing function selected. The function asks the user to decide which mixing
        function to use, and runs the subsequent code to use the correct one. Functions sent to the sampler are
        static methods defined at the end of this class.
        
        Example:
            LMM.mixed_model(g_data=np.linspace(0.0, 0.5, 20), data=np.array(), sigma=np.array(),
                mixing_function='cosine', nsteps=3000)
            
        Parameters:
            g_data (linspace): The linspace over which the data was generated.
        
            data (numpy.ndarray): An array of data points, either generated or supplied by 
                the user.
                
            sigma (numpy.ndarray): An array of standard deviations at each data point.

            mixing_function (str): The name of the mixing function to use for the LMM method. 
                Default is the piecewise cosine. 

            nsteps (int): The number of steps per walker for the sampler to use. 
    
        Returns:
            sampler_mixed (object): The sampler results, contained in a sampler object, from 
                the determination of the unknown parameter. 
            
            emcee_trace_mixed (numpy.ndarray): The trace of each parameter, with burnin samples 
                extracted.
        '''

        #dictionary of LMM functions
        self.function_mappings = {
            'step': self.step,
            'logistic': self.logistic,
            'cdf': self.cdf,
            'cosine': self.switchcos,
        }
        
        #ask user which mixing function to use
        self.choice = mixing_function

        #determine number of hyperparameters 
        if self.choice == 'step':
            self.ndim = 1
        elif self.choice == 'logistic' or self.choice == 'cdf':
            self.ndim = 2 
        elif self.choice == 'cosine':
            self.ndim = 3
        else:
            raise ValueError('Mixing function requested is not found. Enter one of the valid options.')

        #theory errors via error models
        siglow = np.sqrt(self.u.variance_low(g_data, self.loworder[0]))
        sighigh = np.sqrt(self.u.variance_high(g_data, self.highorder[0]))

        #set up sampler
        nwalkers = 2*int(3*self.ndim + 1)

        #show total samples while running
        total_samples = nwalkers * nsteps
        print('Using {} walkers with {} steps each, for a total of {} samples.'.format(nwalkers, nsteps, total_samples))

        #set starting points per parameter
        starting_points = np.zeros((nwalkers, self.ndim))

        #generalise for ndim=1,2,...!=3 and specify for 3
        if self.ndim == 3:
            starting_points[:,0] = np.random.uniform(0.12, 0.18, nwalkers)
            starting_points[:,2] = np.random.uniform(0.19, 0.24, nwalkers)
            starting_points[:,1] = np.random.uniform(0.25, 0.30, nwalkers)
        else:
            for i in range(self.ndim):
                starting_points[:,i] = np.random.uniform(0.0, 1.0, nwalkers)

        #set the mixing function
        self.f = self._select_function(self.choice)
        
        #call emcee
        sampler_mixed = emcee.EnsembleSampler(nwalkers, self.ndim, self.sampler_mix, \
                                            args=[g_data, data, sigma, siglow, sighigh])
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
        emcee_trace_mixed = self.burnin_trace(sampler_mixed, nsteps)

        return sampler_mixed, emcee_trace_mixed

    
    def ppd(self, trace, param_values, g_data, g, data, ci, plot=True):
        
        r'''
        A function to calculate the posterior predictive distribution (PPD) 
        for any chosen mixing function defined in this class. 
        
        Example:
            LMM.ppd(trace, param_values=np.array([]),g_data=np.linspace(1e-6,1.0,10), 
                g_ppd=np.linspace(0.0, 0.5, 100), ci=68)
            
        Parameters:
            trace (numpy.ndarray): The trace of each of the parameters from 
                the sampler.

            param_values (numpy.ndarray): The mean, median, or MAP values of the 
                parameters. 

            g_data (numpy.linspace): The linspace in g from which the data set 
                was calculated.

            g (numpy.linspace): The linspace over which the PPD result will be 
                calculated.

            data (numpy.ndarray): The data set used to calculate the mixed model. 

            ci (int): The desired credibility interval. Can be either 68 or 95.

            plot (bool): The option to plot the PPD result with the series expansions
                and true model. Default is True. 
           
        Returns:
            switch_med_results (numpy.ndarray): The array of median values from the 
                PPD at each point in g.

            switch_g_intervals (numpy.ndarray): The array of credibility interval values 
                for the median results of the PPD.
        '''

        #convert list to array
        trace = np.asarray(trace)

        #check interval
        if ci == 68:
            ci = 0.68
        elif ci == 95:
            ci = 0.95

        if self.choice != 'step':
            result_array = np.empty([len(g), len(trace[0].T)])
        elif self.choice == 'step':
            result_array = np.empty([len(g), len(trace)])
        gmax = max(g)

        #determine which mixing function was used
        if self.choice == 'step':
            for i in range(len(g)):
                for j in range(len(trace)):
                    result_array[i,j] = self.step(trace[j], g[i]) * self.m.low_g(g[i]) \
                        + (1.0 - self.step(trace[j], g[i])) \
                            * self.m.high_g(g[i])

        elif self.choice == 'logistic' or self.choice == 'cdf':
    
            for i in range(len(g)):
                for j in range(len(trace[0].T)):
            
                    if (self.m.low_g(g[i]) - self.m.high_g(g[i]))\
                    > 0.1 and g[i] > (0.25*gmax):
                        result_array[i,j] = self.m.high_g(g[i])
                    
                    elif (self.m.low_g(g[i]) - self.m.high_g(g[i])) > 0.1:
                        result_array[i,j] = self.m.low_g(g[i])
                    
                    else:
                        params = np.array([trace[0, j], trace[1, j]])

                        result_array[i,j] = self.f(params, g[i])*self.m.low_g(g[i]) \
                                        + (1.0 - self.f(params, g[i])) \
                                        *self.m.high_g(g[i])

        elif self.choice == 'cosine':

            params = np.array([np.mean(trace[0,:]), np.mean(trace[1,:]), np.mean(trace[2,:])])

            for i in range(len(g)):
                for j in range(len(trace[0].T)):
                    
                    params = np.array([trace[0, j], trace[1, j], trace[2, j]])
                
                    result_array[i,j] = self.switchcos(params, g[i]) * self.m.low_g(g[i]) \
                                    + (1.0 - self.switchcos(params, g[i])) \
                                    * self.m.high_g(g[i])
        
        #define the credibility intervals
        switch_med_results = np.empty([len(g)])
        switch_g_intervals = np.empty([len(g), 2])

        for i in range(len(g)):
            switch_med_results[i] = statistics.median(result_array[i,:])
            switch_g_intervals[i, :] = self.hpd_interval(result_array[i,:], ci)  # this is what I need (put in M-R curves)

        #plot the PPD results
        if plot is True:
            self.plot_ppd(param_values, g_data, g, data, switch_med_results, switch_g_intervals, percent=68)

        return switch_med_results, switch_g_intervals
    
    
    def plot_ppd(self, results, g_data, g_ppd, data, ppd_results, ppd_intervals, percent):
        
        r'''
        A plotting function that can be used to plot the posterior predictive distribution (PPD) results (mean and 
        credible interval) obtained from calling the functions above in the main code, as well as data generated, 
        the true model, and the small- and large-g expansions chosen for the mixed model calculation. 
        
        Example:
            LMM.plot_ppd(g_data=np.linspace(0.0, 0.5, 20), g_true=np.linspace(0.0, 0.5, 100), 
                g_ppd=np.linspace(0.0, 0.5, 200), data=np.array(), ppd_results=np.array(), 
                ppd_intervals=np.array(), percent=68)
            
        Parameters:
            results (numpy.ndarray): The mean or the median of the estimated parameters 
                from the posterior draws. 

            g_data (linspace): The linspace used to generate the data.
            
            g_ppd (linspace): The linspace chosen to calculate the PPD over. 
            
            data (numpy.ndarray): An array of data either generated or supplied by 
                the user.
                
            ppd_results (numpy.ndarray): An array of the mean of the PPD at each point 
                in the g_ppd linspace.
            
            ppd_intervals (numpy.ndarray): A 2D array of the credibility interval calculated 
                for the PPD (containing both bounds).
                
            percent (int): The percent credibility interval calculated for the variable 
                ppd_intervals (used in the plot legend). 
          
        Returns:
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

        #plot the data and true model
        ax.plot(g_data, data, 'k.', label='Data set')  
        ax.plot(g_ppd, self.m.true_model(g_ppd), 'k', label='True model')

        #plot the expansions
        ax.plot(g_ppd, self.m.low_g(g_ppd)[0,:], 'r--', label=r'$f_s$ ($N_s$ = {})'.format(self.loworder[0]))
        ax.plot(g_ppd, self.m.high_g(g_ppd)[0,:], 'b--', label=r'$f_l$ ($N_l$ = {})'.format(self.highorder[0]))

        #plot the PPD results from mixing
        ax.plot(g_ppd, ppd_results, 'g', label='Mixed model')
        ax.plot(g_ppd, ppd_intervals[:,0], 'g', linestyle='dotted', label=r'{}\% CI (HPD)'.format(percent))
        ax.plot(g_ppd, ppd_intervals[:,1], 'g', linestyle='dotted')
        ax.fill_between(g_ppd, ppd_intervals[:,0], ppd_intervals[:,1], color='green', alpha=0.2)

        #parameter results (vertical lines)
        if self.choice == 'step':
            ax.axvline(x=results, color='darkviolet', alpha=0.35, label=r'$\theta_{1}$')
        else:
            ax.axvline(x=results[0], color='darkviolet', alpha=0.35, label=r"$\theta_{1}$, $\theta_{2}$, $\theta_{3}$")
            ax.axvline(x=results[1], color='darkviolet', alpha=0.35)

        if len(results) == 3:
            ax.axvline(x=results[2], color='darkviolet', alpha=0.35)

        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        # answer = input('Would you like to save the plot to a file (yes/no)?')

        # if answer == 'yes':
        #     name = input('Enter a file name to use (include file type as .png, .pdf, etc.).')
        #     fig.savefig(name, bbox_inches='tight')
       
        return None


    def _select_function(self, x):

        r'''
        A function that selects the proper mixing function given the input from the user
        in the function 'LMM.mixed_model'. 

        *Internal function only for use inside the LMM class.*

        Example:
            LMM._select_function(x=self.choice)
        
        Parameters:
            x (string): The string that is input by the user to select one of the 
                available mixing functions. 
        
        Returns:
            self.function_mappings[x]: The correct function label for the chosen mixing 
                function, converted from string to object.
        '''
        while True:
            try:
                return self.function_mappings[x]
            except KeyError:
                print('Invalid function, try again.')
    
    
    def _autocorrelation(self, chain, max_lag=100):

        r'''
        Borrowed from Christian Forssen's nuclear TALENT school work on 
        'Learning from Data: Bayesian Methods and Machine Learning' course
        in June 2019 (see https://github.com/NuclearTalent/Bayes2019). 

        *Internal function, only for use inside the LMM class.*

        Example:
            LMM._autocorrelation(chain, max_lag=200)

        Parameters:
            chain (numpy.ndarray): The array of samples from the emcee chain object that 
                are returned by the sampler. 

            max_lag (int): The maximum lagtime for the autocorrelation length. 
        
        Returns:
            acors (numpy.ndarray): The array of autocorrelation calculated.
        '''

        #determine the autocorrelation length
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
     

    def burnin_trace(self, sampler_object, nsteps):
        
        r'''
        A small function to take the burn-in samples off of the sampler chain from the 
        LMM.mixed_model function, and to send back the trace of the sampler chain to 
        LMM.mixed_model.
        
        Example:
            LMM.burnin_trace(sampler_object=sampler_mixed, nsteps=3000)
            
        Parameters:
            sampler_object (object): The chain sent back by the emcee sampler after 
                it finishes running through the samples and walkers.

            nsteps (int): The number of steps per walker.
                
        Returns:
            emcee_trace_mixed (numpy.ndarray): The trace of the sampler chain with 
                the user's desired number of burn-in samples removed.
        '''
        
        nburnin = int((1/15) * nsteps)

        #throw out the burn-in and reshape again
        emcee_trace_mixed = sampler_object.chain[:, nburnin:, :].reshape(-1, self.ndim).T
        
        return emcee_trace_mixed


    def stats_chain(self, chain, plot=True):

        r'''
        Calculates the autocorrelation time and thins the samples
        accordingly for a better estimate of the mean, median, and MAP values. 

        Example: 
            LMM.stats_chain(chain=emcee.object, plot=False)

        Parameters:
            chain (object): The object resulting from sampling the parameters
                using emcee. The chain of samples must be extracted
                from it. 

            plot (bool): The option to plot the traces of the sample
                chains and the corner plot of the parameter
                distributions. Default is True.  

        Returns:
            thin (numpy.ndarray): The array of thinned samples per parameter. 
                Used externally to calculate the MAP values.

            median_results (numpy.ndarray): Each of the median parameter 
                values found from the sampling.

            mean_results (numpy.ndarray): Each of the mean parameter values found 
                from the sampling. 
        '''

        #retrieve the chain
        chain_result = chain.chain[:,:,:]

        #"quick and dirty" method; will finish later more generally
        if self.ndim == 1:
            
            #set up arrays
            chain1 = chain_result[:,:,0]
            
            #flatten each individual array
            flat1 = chain1.flatten()

            #call autocorrelation to find the lengths
            post_acors1 = self._autocorrelation(flat1, max_lag=200)

            #determine the autocorrelation time
            post_rho1 = post_acors1[25:35]

            post_y = np.arange(10)
            post_x1 = -np.log(post_rho1)

            #linear fits
            p1, _ = np.polyfit(post_x1, post_y, 1, cov=True)

            #thin the samples given the determined autocorrelation time
            thin1 = []
            time = p1[0]

            time = int(time)
                    
            for i in range(len(flat1)):
                if i % time == 0:
                    thin1.append(flat1[i])
                                
            #array thinned samples
            thin = np.array(thin1)

            #call stats_trace for plots
            if plot is True:
                _, _ = LMM.stats_trace(self, thin)

            #median calculation
            median_1 = statistics.median(thin)

            #mean calculation
            mean_1 = np.mean(thin)

            #arrays
            mean_results = np.array([mean_1])
            median_results = np.array([median_1])
          
            return thin, mean_results, median_results

        else:

            #set up lists and arrays
            chains = []
            post_acors = []
            post_rho = []
            post_x = []
            p = []

            post_y = np.arange(10)

            #create list using parameter arrays and flatten
            for i in range(self.ndim):
                chains.append(chain_result[:,:,i].flatten())

            #max lag determination
            max_lag = 200
            if max_lag > int(0.2*len(chains[0])):
                max_lag = int(0.15*len(chains[0]))

            #determine autocorrelation
            for i in range(len(chains)):
                post_acors.append(self._autocorrelation(chains[i], max_lag=max_lag))
                post_rho.append(post_acors[i][25:35])
                post_x.append(-np.log(post_rho[i]))
                p_temp, _ = np.polyfit(post_x[i], post_y, 1, cov=True)
                p.append(p_temp[0])

        #thin the samples based on the results above
        thin = []

        #get the autocorrelation time for all parameters
        for i in range(len(chains)-1):
            if p[i] > p[i+1]:
                time = int(p[i])
            else:
                time = int(p[i+1])
                
        #thin the samples for each parameter  ---> double counting, need to separate params correctly
        for i in range(self.ndim):
            thin_temp = []
            for j in range(len(chains[0])):
                if j % time == 0:
                    thin_temp.append(chains[i][j])
            
            thin.append(thin_temp)

        #plot the traces
        if plot is True:
                _, _ = LMM.stats_trace(self, thin)

        #median, mean calculations
        median = []
        mean = []

        for i in range(len(thin)):
            median.append(statistics.median(thin[i]))
            mean.append(np.mean(thin[i]))
    
        return thin, mean, median


    def MAP_values(self, thin, g, g_data, data, sigma, plot=True):

        r'''
        A function to calculate the MAP values of sampled distributions 
        of parameters. Will calculate for as many parameters as are present 
        and return results in an array. 

        Example:
            LMM.MAP_values(thin=np.array([]), g_data=np.linspace(),
                g=np.linspace(), data=np.array([]), sigma=np.array([]))

        Parameters:
            thin (numpy.ndarray): The array of thinned samples from the 
                stats_chain() function.

            g (numpy.linspace): The input space over which the mixing 
                is calculated.

            g_data (numpy.linspace): The array of input points in g for 
                the data set. 

            data (numpy.ndarray): The data set being used for the mixing 
                calculation.

            sigma (numpy.ndarray): The data error set being used for the 
                mixing calculation.

            plot (bool): The option to plot the weights over the input space 
                in g. Default is True. 

        Returns:
            map_values (numpy.ndarray): The MAP values of each parameter. 
        '''
    
        #calculate theory error needed 
        siglow = np.sqrt(self.u.variance_low(g_data, self.loworder[0]))
        sighigh = np.sqrt(self.u.variance_high(g_data, self.highorder[0]))

        #find the posterior using the parameters
        thetas = np.asarray(thin)

        if self.choice == 'step':
            posterior = np.zeros(len(thetas))
            for i in range(len(thetas)):
                posterior[i] = self.sampler_mix(thetas[i], g_data, data, sigma, siglow, sighigh)
            
            #MAP value for parameter
            theta_index = np.argmax(posterior)
            map_values = np.array([thetas[theta_index]])

        else:
            posterior = np.zeros([len(thetas[0,:])])
            for i in range(len(thetas[0,:])):
                posterior[i] = self.sampler_mix(thetas[:,i], g_data, data, sigma, siglow, sighigh)
            
            #MAP value calculation
            theta_index = np.argmax(posterior)
            map_values = thetas[:, theta_index]

        #plot the weight function and MAP values
        if plot is True:
            self.plot_MAP(g, map_values)

        return map_values

    
    def plot_MAP(self, g, map_values):

        r'''
        A simple rough plotter to plot the weight/mixing function
        for the LMM method using the mixing function calculated at
        the points in g and the MAP values of its parameters. 

        Example:
            LMM.plot_MAP(g=np.linspace(), map_values=numpy.ndarray([]))

        Parameters:
            g (numpy.linspace): The input space over which the mixing 
                is calculated.

            map_values (numpy.ndarray): The results of the MAP_values() 
                function (MAP values of each parameter in the mixing function 
                selected).

        Returns:
            None.
        '''

        #set up figure
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0,1.0)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel(r'$\alpha$(g; $\theta$)', fontsize=22)

        #solve the mixing function first and plot
        switch = np.zeros([len(g)])
        for i in range(len(g)):
            switch[i] = self.f(map_values, g[i])
        ax.plot(g, switch, 'k', linewidth=3, label=r'$\alpha(g; \theta)$') 

        #plot the MAP value lines
        if len(map_values) == 1:
            ax.axvline(x=map_values, color='darkorange', linestyle='dashed', label=r'$\theta_{1}$')

        elif len(map_values) == 2:
            ax.axvline(x=map_values[0], color='darkorange', linestyle='dashed', label=r'$\theta_{1}$')
            ax.axvline(x=map_values[1], color='darkviolet', linestyle='dashdot', label=r'$\theta_{2}$')

        elif len(map_values) == 3:
            ax.axvline(x=map_values[0], color='darkorange', linestyle='dashed', label=r'$\theta_{1}$')
            ax.axvline(x=map_values[2], color='darkviolet', linestyle='dashdot', label=r'$\theta_{2}$')
            ax.axvline(x=map_values[1], color='darkgreen', linestyle='dashed', label=r'$\theta_{3}$')

        ax.legend(loc='upper right', fontsize=18)

        #fig.savefig('mixing_plot_MAP_5v5.pdf', bbox_inches='tight')

        plt.show()

        return None 

    
    def stats_trace(self, trace):

        r'''
        A function to calculate the mean and credible intervals corresponding to
        each parameter. The trace plots for each parameter are plotted. 

        Example:
            LMM.stats_trace(trace=np.array([]))

        Parameters:
            trace (numpy.ndarray): The trace from the sampler object that was generated 
                when estimating the parameters of the mixing function.

        Returns:
            mean (numpy.ndarray): The array of mean values for each parameter.
            
            ci (numpy.ndarray): The array of sets of credible interval bounds for 
                each parameter. 
        '''

        #calculate mean and credible intervals
        mean = []
        ci = []

        if self.ndim == 1:
            mean.append(np.mean(trace))
            ci.append(self.hpd_interval(trace, 0.95))
        else:
            for i in range(self.ndim):
                mean.append(np.mean(trace[i]))
                ci.append(self.hpd_interval(trace[i], 0.95))

        mean = np.asarray(mean)
        ci = np.asarray(ci)

        #plot traces with mean and credible intervals
        fig, ax = plt.subplots(self.ndim,1,figsize=(7,4*self.ndim), dpi=600)

        if self.ndim == 1:
            ax.plot(trace, 'k')
            ax.set_ylabel('Parameter 1', fontsize=22)
            ax.set_title('Trace plot: Parameter 1', fontsize=22)

            ax.axhline(y=mean, color='b', linestyle='solid', label='Mean')
            ax.axhline(y=ci[0, 0], color='b', linestyle='dashed')
            ax.axhline(y=ci[0, 1], color='b', linestyle='dashed')

            #plot the median over the mean
            med = []
            med.append(np.median(trace[0]))
            med = np.asarray(med)

            ax.axhline(y=med, color='r', linestyle='solid', label='Median')
            ax.legend(loc='upper right')

            plt.show()
            

        if self.ndim > 1:
            for irow in range(self.ndim):
                ax[irow].plot(trace[irow], 'k')
                ax[irow].set_ylabel('Parameter {0}'.format(irow+1), fontsize=22)
                ax[irow].set_title('Trace plot: Parameter {0}'.format(irow+1), fontsize=22)

                ax[irow].axhline(y=mean[irow], color='b', linestyle='solid', label='Mean')
                ax[irow].axhline(y=ci[irow, 0], color='b', linestyle='dashed')
                ax[irow].axhline(y=ci[irow, 1], color='b', linestyle='dashed')

            #plot the median over the mean
            med = []

            for i in range(self.ndim):
                med.append(np.median(trace[i]))

            med = np.asarray(med)

            for irow in range(self.ndim):
                ax[irow].axhline(y=med[irow], color='r', linestyle='solid', label='Median')

                ax[irow].legend(loc='upper right')

            plt.show()

            #corner plots for hyperparameter posteriors
            fig, axs = plt.subplots(self.ndim,self.ndim, figsize=(8,8), dpi=600)
            label = ["Parameter 1", "Parameter 2", "Parameter 3"]
            trace = np.asarray(trace)
            corner.corner(trace.T,labels=label, \
                quantiles=[0.16, 0.5, 0.84],fig=fig,show_titles=True, label_kwargs=dict(fontsize=16))
            plt.show()
            
        return mean, ci 

    
    @staticmethod
    def hpd_interval(trace, fraction):
        
        r'''
        A function to calculate the Bayesian credible intervals of a posterior distribution. This function
        uses the HPD (highest posterior density) method.
        
        Example:
            LMM.hpd_interval(trace=emcee_trace, fraction=0.95)
            
        Parameters:
            trace (numpy.ndarray): The trace generated by a sampler when sampling a variable to obtain 
                its posterior distribution.
            
            fraction (float): The percent (in decimal form) requested by the user to set the credibility 
                interval. 
            
        Returns:
            interval (numpy.ndarray): The credibility interval bounds in a numpy array (format: [min, max]).
        '''
    
        sort_list = np.sort(np.copy(trace))
        total_samples = len(trace)
    
        int_samples = np.floor(fraction * total_samples).astype(int)
        int_width = sort_list[int_samples:] - sort_list[:total_samples-int_samples]  # cutting the tails off
    
        min_int = np.argmin(int_width)  # tells me the location
        
        interval = np.array([sort_list[min_int], sort_list[min_int+int_samples]])  # gives me the interval
    
        return interval
    

    @staticmethod
    def logistic(params, g):
        
        r'''
        A basic logistic function often used in machine learning, implemented here with two free
        parameters to be determined via sampling.
        
        Example:
            logistic(params=np.array(), g=0.5)
            
        Parameters:
            params (numpy.ndarray): The array of parameters the sampler will determine (here 
                labelled beta0 and beta1, where beta0 controls the location of the function and 
                beta1 controls the slope). 

        Returns:
            mixing (float): The result of the logistic function given the value g.
        '''
        beta0, beta1 = params
    
        mixing = (1.0 + np.exp(-(beta0 + g*beta1)))**(-1.0)
    
        return mixing
    
    @staticmethod
    def cdf(params, g):
        
        r'''
        The cumulative distribution function of a standard normal distribution, with two free 
        parameters determined by sampling.
        
        Example:
            cdf(params=np.array(), g=0.5)
        
        Parameters:
            params (numpy.ndarray): The array of parameters the sampler will determine (here 
                labelled beta0 and beta1, where beta0 controls the location of the function and 
                beta1 controls the slope). 
            
            g : float
                The value of g the cdf is calculated at.

        Returns:
            function (float): The result of the cdf function at the value of g. 
        '''
        beta0, beta1 = params
    
        function = (1.0 + math.erf((beta0 + g*beta1)/np.sqrt(2.0))) / 2.0
    
        return function

    
    @staticmethod
    def switchcos(params, g):
        
        r'''
        A piecewise function using two constants at either end, and two cosine functions in the centre,
        to be used as a mixing function. One free parameter, g3, is found by sampling. 
        
        Example:
            switchcos(params=np.array(), g=0.5)
            
        Parameters:
            params (numpy.ndarray): The array of parameters to be determined by the sampler 
                (here labelled g1, g2, and g3, where g1 is the separation point between the first constant 
                function and the first cosine function, g2 is the separation point between the second 
                cosine function and the second constant function, 
                and g3 is the point between the two cosine functions). 

            g (float): The value of g that this cosine function is calculated at.

        Returns:
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

    @staticmethod
    def step(params, g):

        r'''
        A step mixing function to switch between two models. 
        ***Only useful for two models.***

        Example:
            step(params, g=0.2)
        
        Parameters:
            params (np.ndarray): One single parameter to determine 
                where the step function will break from one model to the other.

            g (float): One value of the input space. 

        Returns:
            The value of the step function at a specific 
            point in g. 
        '''

        #enable step function
        if g < params:
            return 1.0 
        else:
            return 0.0 
