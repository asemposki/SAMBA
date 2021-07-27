import numpy as np
from scipy import special, integrate, stats
from cycler import cycler
import math as math
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
        
        Results:
        --------
        None.
        '''
        
        pass
    
    
    def logistic(self, beta0, beta1, g):
        
        '''
        A basic logistic function often used in machine learning, implemented here with two free
        parameters to be determined via sampling.
        
        :Example:
            Switching.logistic(beta0=param[0], beta1=param[1], g=0.5)
            
        Parameters:
        -----------
        beta0          
            One free parameter that is found by sampling---controls the location of the function's 
            halfway point in g space.
            
        beta1         
            The other free parameter found by sampling---controls the steepness of the function. 
        
        Returns:
        --------
        mixing          
            The result of the logistic function given the value g.
        '''
    
        mixing = (1.0 + np.exp(-(beta0 + g*beta1)))**(-1.0)
    
        return mixing
    

    def cdf(self, beta0, beta1, g):
        
        '''
        The cumulative distribution function of a standard normal distribution, with two free parameters
        determined by sampling.
        
        :Example:
            Switching.cdf(beta0=params[0], beta1=params[1], g=0.5)
        
        Parameters:
        -----------
        beta0        
             One free parameter that is found by sampling---controls the location of the function's 
            halfway point in g space.
            
        beta1    
            The other free parameter found by sampling---controls the steepness of the function. 
                   
        Returns:
        --------
        function          
            The result of the cdf function at the value of g. 
        '''
    
        function = (1.0 + math.erf(beta0 + g*beta1)/np.sqrt(2.0)) / 2.0
    
        return function
    
    
    #later on ---> make g1 and g2 parameters to be found via sampling as well!
    def switchcos(self, g1, g2, g3, g):
        
        '''
        A piecewise function using two constants at either end, and two cosine functions in the centre,
        to be used as a switching function. One free parameter, g3, is found by sampling. 
        
        :Example:
            Switching.switchcos(g1=0.12, g2=0.20, g3=param[0], g=0.5)
            
        Parameters:
        -----------
        g1        
            The point at which the function switches from a constant to the first cosine function. 
        
        g2    
            The point at which the function switches from the second cosine function to a constant.
        
        g3         
            The free parameter that will be determined by sampling that resides between the two cosine 
            functions.
            
        Returns:
        --------
            The value of the function at a specific point in g. 
        '''
    
        if g <= g1:
            return 1.0
    
        elif g <= g3:
            return (1.0 + np.cos((np.pi/2.0) * ((g - g1)/(g3 - g1))))/2.0
    
        elif g < g2:
            return 0.5 + np.cos((np.pi/2.0) * (1.0 + ((g - g3)/(g2 - g3))))/2.0
    
        else:
            return 0.0
        
    
    def log_mix(self, params, g_data, data, sigma, loworder, highorder, mu0, mu1, sig0, sig1):
        
        '''
        The mixing function that will be sent to the sampler that implements Switching.logistic as 
        the switching function.
        
        :Example:
            Switching.log_mix(params, g_data=np.linspace(0.0, 0.5, 20), data=np.array(), sigma=np.array(),
            loworder=5, highorder=23, mu0=0.16, mu1=0.5, sig0=0.02, sig1=0.1)
            
        Parameters:
        -----------
        params
            The two parameters, beta0 and beta1, that are being determined by the sampler, in an array.
            
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
            
        mu0
            The mean given to the prior for beta0.
            
        mu1
            The mean given to the prior for beta1.
            
        sig0 
            The standard deviation given to the prior on beta0.
        
        sig1
            The standard deviation given to the prior on beta1.
        
        Returns:
        --------
        mixed_results
            The results of the mixing function for the entire linspace in g, in an array format.
        '''
        
        #unpack parameters
        beta0 = params[0]
        beta1 = params[1]       
    
        #set up arrays
        mixed_likelihood = np.empty([len(g_data)])
        log_ml = np.empty([len(g_data)])
    
        #likelihood mixing
        for i in range(len(g_data)):
        
            mixed_likelihood[i] = self.logistic(beta0, beta1, g_data[i]) * \
                                  Mixing.likelihood_low(self, g_data[i], data[i], sigma[i], loworder) \
                                  + (1.0 - self.logistic(beta0, beta1, g_data[i])) * \
                                  Mixing.likelihood_high(self, g_data[i], data[i], sigma[i], highorder)
                
            log_ml[i] = np.log(mixed_likelihood[i])
        
        total_lml = np.sum(log_ml)
    
        #add the priors
        mixed_results = total_lml + Mixing.prior_gauss(self, beta0, mu0, sig0) + Mixing.prior_gauss(self, beta1, mu1, sig1)
    
        return mixed_results
    
    
    def cdf_mix(self, params, g_data, data, sigma, loworder, highorder, mu0, mu1, sig0, sig1):
        
        '''
        The mixing function that will be sent to the sampler that implements Switching.cdf as 
        the switching function.
        
        :Example:
            Switching.cdf_mix(params, g_data=np.linspace(0.0, 0.5, 20), data=np.array(), sigma=np.array(),
            loworder=5, highorder=23, mu0=0.16, mu1=0.5, sig0=0.02, sig1=0.1)
            
        Parameters:
        -----------
        params
            The two parameters, beta0 and beta1, that are being determined by the sampler, in an array.
            
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
            
        mu0
            The mean given to the prior for beta0.
            
        mu1
            The mean given to the prior for beta1.
            
        sig0 
            The standard deviation given to the prior on beta0.
        
        sig1
            The standard deviation given to the prior on beta1.
        
        Returns:
        --------
        mixed_results
            The results of the mixing function for the entire linspace in g, in an array format.
        '''
        
        #unpack parameters
        beta0 = params[0]
        beta1 = params[1]
    
        #set up arrays
        mixed_likelihood = np.empty([len(g_data)])
        log_ml = np.empty([len(g_data)])
    
        #likelihood mixing
        for i in range(len(g_data)):
        
            mixed_likelihood[i] = self.cdf(beta0, beta1, g_data[i]) * \
                                  Mixing.likelihood_low(self, g_data[i], data[i], sigma[i], loworder) \
                                  + (1.0 - self.cdf(beta0, beta1, g_data[i])) * \
                                  Mixing.likelihood_high(self, g_data[i], data[i], sigma[i], highorder)
                
            log_ml[i] = np.log(mixed_likelihood[i])
        
        total_lml = np.sum(log_ml)
    
        #add the priors
        mixed_results = total_lml + Mixing.prior_gauss(self, beta0, mu0, sig0) + Mixing.prior_gauss(self, beta1, mu1, sig1)
    
        return mixed_results
    
    #later on ---> add the g1 and g2 information
    def cosine_mix(self, params, g_data, data, sigma, loworder, highorder, g3mu, g3sig):
        
        '''
        The mixing function that will be sent to the sampler that implements Switching.cosine_mix 
        as the switching function.
        
        :Example:
            Switching.cosine_mix(params=param[0], g_data=np.linspace(0.0, 0.5, 20), data=np.array(),
            sigma=np.array(), loworder=5, highorder=23, g3mu=0.15, g3sig=0.01)
        
        Parameters:
        -----------
        params
            The parameter g3 that is being determined by the sampler, in an array.
            
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
            
        g3mu
            The mean given to the prior for g3.
            
        g3sig
            The standard deviation given to the prior for g3.
        
        Returns:
        --------
        mixed_results
            The results of the mixing function for the entire linspace in g, in an array format.
            
        '''
    
        #unpack parameters
        g1 = 0.12
        g2 = 0.20
        g3 = params
    
        #set up arrays
        mixed_likelihood = np.empty([len(g_data)])
        log_ml = np.empty([len(g_data)])
  
        #likelihood mixing
        for i in range(len(g_data)):
        
            mixed_likelihood[i] = self.switchcos(g1, g2, g3, g_data[i]) * \
                                  Mixing.likelihood_low(self, g_data[i], data[i], sigma[i], loworder) \
                                  + (1.0 - self.switchcos(g1, g2, g3, g_data[i])) * \
                                  Mixing.likelihood_high(self, g_data[i], data[i], sigma[i], highorder)
                        
            log_ml[i] = np.log(mixed_likelihood[i])
        
        total_lml = np.sum(log_ml)
    
        #add the priors
        mixed_results = total_lml + Mixing.prior_gauss(self, g3, g3mu, g3sig)
    
        return mixed_results
    
    
    def ppd_log(self, beta0, beta1, g, loworder, highorder):
        
        '''
        A function to calculate the posterior predictive distribution (PPD) when the switching function
        is the logistic function defined above in this class. 
        
        :Example:
            Switching.ppd_log(beta0=emcee_trace[0,:], beta1=emcee_trace[1,:], g=np.linspace(0.0, 0.5, 100), 
            loworder=5, highorder=23)
            
        Parameters:
        -----------
        beta0 
            The trace of one of the parameters determined by sampling 
            (controls the location of the logistic function in g_data space).
        
        beta1
            The trace of the other parameter determined by sampling 
            (controls the slope of the logistic function).
        
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
    
        result_array = np.empty([len(g), len(beta0.T)])
        gmax = max(g)
    
        for i in range(len(g)):
            for j in range(len(beta0.T)):
            
                if (Mixing.low_g(self, g[i], loworder) - Mixing.high_g(self, g[i], highorder))\
                > 0.1 and g[i] > (0.5*gmax):
                    result_array[i,j] = Mixing.high_g(self, g[i], highorder)
                
                elif (Mixing.low_g(self, g[i], loworder) - Mixing.high_g(self, g[i], highorder)) > 0.1:
                    result_array[i,j] = Mixing.low_g(self, g[i], loworder)
                
                else:
                    result_array[i,j] = self.logistic(beta0[j], beta1[j], g[i])*Mixing.low_g(self, g[i], loworder) \
                                      + (1.0 - self.logistic(beta0[j], beta1[j], g[i]))\
                                      *Mixing.high_g(self, g[i], highorder)
        
        return result_array

    
    def ppd_cdf(self, beta0, beta1, g, loworder, highorder):
        
        '''
        A function to calculate the posterior predictive distribution (PPD) when the switching function
        is the cdf function defined above in this class. 
        
        :Example:
            Switching.ppd_cdf(beta0=emcee_trace[0,:], beta1=emcee_trace[1,:], g=np.linspace(0.0, 0.5, 100), 
            loworder=5, highorder=23)
            
        Parameters:
        -----------
        beta0 
            The trace of the first parameter determined by sampling 
            (controls the location of the cdf 
            function in g_data space).
        
        beta1
            The trace of the other parameter determined by sampling 
            (controls the slope of the cdf function).
        
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
    
        result_array = np.empty([len(g), len(beta0.T)])
        gmax = max(g)
    
        for i in range(len(g)):
            for j in range(len(beta0.T)):
            
                if (Mixing.low_g(self, g[i], loworder) - Mixing.high_g(self, g[i], highorder))\
                > 0.1 and g[i] > (0.5*gmax):
                    result_array[i,j] = Mixing.high_g(self, g[i], highorder)
                
                elif (Mixing.low_g(self, g[i], loworder) - Mixing.high_g(self, g[i], highorder)) > 0.1:
                    result_array[i,j] = Mixing.low_g(self, g[i], loworder)
                
                else:
                    result_array[i,j] = self.cdf(beta0[j], beta1[j], g[i])*Mixing.low_g(self, g[i], loworder) \
                                      + (1.0 - self.cdf(beta0[j], beta1[j], g[i])) \
                                      *Mixing.high_g(self, g[i], highorder)
        
        return result_array
    
    
    def ppd_cos(self, g1, g2, g3, g, loworder, highorder):
        
        '''
        A function to calculate the posterior predictive distribution (PPD) when the switching function
        is the cosine function defined in this class. 
        
        :Example:
            Switching.ppd_cos(g1=0.12, g2=0.2, g3=emcee_trace, g=np.linspace(0.0, 0.5, 100), loworder=5,
            highorder=23)
            
        Parameters:
        -----------
        g1        
            The point at which the function switches from a constant to the first cosine function.  
        
        g2    
            The point at which the function switches from the second cosine function to a constant.
            
        g3
            The trace of the parameter determined by sampling. 
            
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
    
        result_array = np.empty([len(g), len(g3.T)])

        for i in range(len(g)):
            for j in range(len(g3.T)):
            
                result_array[i,j] = self.switchcos(g1, g2, g3[j], g[i]) * Mixing.low_g(self, g[i], loworder) \
                                  + (1.0 - self.switchcos(g1, g2, g3[j], g[i])) * Mixing.high_g(self, g[i], highorder)
            
        return result_array


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
        
        
    def generate_data(self, g_true, g_data):
        
        '''
        A data generation function that generates data based on the g_data linspace provided (with the number of 
        points chosen by the user) and the error desired on each point (also input by the user).
        
        :Example:
            Mixing.generate_data(g_true=np.linspace(0.0, 0.5, 100), g_data=np.linspace(0.0, 0.5, 20))
        
        Parameters:
        -----------
        g_true         
            The linspace desired for the true model to be calculated.
        
        g_data    
            The linspace input for the data to be generated within. 
            
        Returns:
        --------
        data         
            The array of data generated.
            
        sigma    
            The standard deviation at each data point.
        '''
        
        #ask for error desired
        error = float(input('What percent error would you like on your data (please enter a decimal)?'))
        
        #generate fake data  
        self.data = Mixing.true_model(self, g_data)
        rand = np.random.RandomState()
        var = error*rand.randn(len(g_data))
        self.data = self.data*(1 + var)
        
        #calculate standard deviation
        sigma = error*self.data
        
        #plot the data and true model
        Mixing.plot_data(self, g_true, g_data, self.data)
    
        return self.data, sigma
    
    
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
            The array of data generated using the Mixing.generate_data function.
            
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
        
        #ask user which switching function to use
        response = input('What switching function would you like to use: logistic, cdf, or cosine?')
        
        #switch statement implementation
        if response == "logistic":
            
            #enter mu and sigma for both parameters
            mu0 = float(input("Enter a guess for the mean of the parameter beta0: "))
            sig0 = float(input("Enter a guess for the standard deviation of the parameter beta0: "))
            mu1 = float(input("Enter a guess for the mean of the parameter beta1: "))
            sig1 = float(input("Enter a guess for the standard deviation of the parameter beta1: "))
            
            #set up sampler
            ndim = 2
            nwalkers = 5
            nsteps = 1000

            total_samples = nwalkers * nsteps

            print('Using {} walkers with {} steps each, for a total of {} samples.'.format(nwalkers, nsteps, total_samples))

            starting_points = np.random.randn(nwalkers, ndim)

            #call emcee
            sampler_mixed = emcee.EnsembleSampler(nwalkers, ndim, self.log_mix, \
                                                  args=[g_data, data, sigma, loworder, highorder, mu0, mu1, sig0, sig1])
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
            
            return emcee_trace_mixed
            
        elif response == "cdf":
            
            #enter mu and sigma for both parameters
            mu0 = float(input("Enter a guess for the mean of the parameter beta0: "))
            sig0 = float(input("Enter a guess for the standard deviation of the parameter beta0: "))
            mu1 = float(input("Enter a guess for the mean of the parameter beta1: "))
            sig1 = float(input("Enter a guess for the standard deviation of the parameter beta1: "))
            
            #set up sampler
            ndim = 2
            nwalkers = 5
            nsteps = 1000

            total_samples = nwalkers * nsteps

            print('Using {} walkers with {} steps each, for a total of {} samples.'.format(nwalkers, nsteps, total_samples))

            starting_points = np.random.randn(nwalkers, ndim)

            #call emcee
            sampler_mixed = emcee.EnsembleSampler(nwalkers, ndim, self.cdf_mix, \
                                                  args=[g_data, data, sigma, loworder, highorder, mu0, mu1, sig0, sig1])
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
            
            return emcee_trace_mixed     
            
        elif response == "cosine":
            
            #enter mu and sigma for the prior on g3
            mu = float(input("Enter a guess for the mean of the parameter g3: "))
            sig = float(input("Enter a guess for the standard deviation of the parameter g3: "))
            
            #set up the sampler here
            ndim = 1
            nwalkers = 10
            nsteps = 1000

            total_samples = nwalkers * nsteps

            print('Using {} walkers with {} steps each, for a total of {} samples.'.format(nwalkers, nsteps, total_samples))

            starting_points = np.random.randn(nwalkers, ndim)

            #call emcee
            sampler_mixed = emcee.EnsembleSampler(nwalkers, ndim, self.cosine_mix, \
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
            
            return emcee_trace_mixed
        
        else:
            print('Please select one of the options listed above.')
            
            
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
        emcee_trace_mixed = emcee_trace_mixed[0]
        
        return emcee_trace_mixed
            
        
if __name__=="__main__":
    pass