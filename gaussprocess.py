import numpy as np 
import GPy as gp
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mixing import Mixing, Models
from discrepancy import Discrepancy 

class GP(Mixing):

    #need to set this function so that the user can determine the kernel
    def __init__(self, dim=1, lengthscale=1., variance=1.):

        '''
        A class that will pull from the Models, Mixing, and Discrepancy classes
        to perform GP emulation on two models from the small-g expansion region
        to the large-g expansion region. The parameter settings of the kernel
        will be set by the user in this initial function.

        :Example:
            GP()

        Parameters:
        -----------
        dim : int
            The number of dimensions of the input parameters. This sets up the
            dimension of the kernel to be used. Default is 1. 

        lengthscale : float, numpy.ndarray
            The lengthscale of the kernel that will initially be set. If dim > 1, 
            an array can be sent in to this variable with values corresponding to 
            the desired lengthscales of each dimension. Default is 1. 
        
        variance : float, numpy.ndarray
            The variance of the kernel that will be initially set. If dim > 1, an
            array can be sent in with values corresponding to the desired variances
            of each dimension. Default is 1.      

        Returns:
        -------
        None.
        '''

        #kernel set-up for the rest of the class
        krbf = gp.kern.RBF(input_dim=dim, variance=variance, lengthscale=lengthscale)
        kwhite = gp.kern.White(input_dim=1)

        self.kern = krbf + kwhite

        #for input_dim=1, plot the kernel
        fig = plt.figure(figsize=(8,6), dpi=100)
        ax = plt.axes()
        ax.set_title('RBF + white noise kernels')
        self.kern.plot(ax=ax, color='blue', label=f'l = {lengthscale}, v = {variance}')
        ax.legend()
        plt.show()

        return None

    def model_data(self, g, loworder, highorder):

        '''
        Description:

        :Example:
            GP.model_data(g=np.linspace(1e-6,0.5,100), loworder=np.array([5]), highorder=np.array([5]))

        Parameters:
        -----------
        g : numpy.linspace
            The range over which we are investigating the models. 
        
        loworder : int, float, numpy.ndarray
            The highest order to which the small-g expansion is calculated. 

        highorder : int, float, numpy.ndarray
            The highest order to which the large-g expansion is calculated. 

        Returns:
        --------
        data : numpy.ndarray
            The data that we generate from the two expansions.

        sigma : numpy.ndarray
            The variances of each of the data points in the two expansions.
        '''

        #stop the data generation where the gap begins
        for i in range(len(g)):
            if Mixing.low_g(self, g[i], loworder) > 2.6:
                lowmax = g[i-1]
                print(lowmax)
                break

        for i in range(len(g)):
            if Mixing.high_g(self, g[i], highorder) < 1.8:
                highmin = g[i+1]
            else:
                break

        #create new ranges in g
        self.lowg = np.linspace(min(g), lowmax, 20)
        self.highg = np.linspace(highmin, max(g), 20)

        #calculate the variances from the next order
        response = input('Which error model do you want to use? (uninformative/informative)')
        if response == 'uninformative':
            error_model = 1
        elif response == 'informative':
            error_model = 2
        else:
            raise ValueError('Please choose a valid error model.')

        lowvariance = Discrepancy.variance_low(self, self.lowg, loworder[0], error_model)
        lowsigma = np.sqrt(lowvariance)
        highvariance = Discrepancy.variance_high(self, self.highg, highorder[0], error_model)
        highsigma = np.sqrt(highvariance)

        #calculate the data points from the models
        self.lowdata = Mixing.low_g(self, self.lowg, loworder)[0,:]
        self.highdata = Mixing.high_g(self, self.highg, highorder)[0,:]

        #merge data sets for the GP
        gdata = np.concatenate((self.lowg, self.highg))
        data = np.concatenate((self.lowdata, self.highdata))
        sigma = np.concatenate((lowsigma, highsigma))

        #plot the results
        dpi = int(input('Set a dpi for the figure.'))
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(1.8,2.6)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): mixed model', fontsize=22)
        ax.plot(g, Models.true_model(self, g), 'k', label='True model')

        #plot the data
        ax.plot(self.lowg, self.lowdata, 'r.', label=r'$f_s$ ({}) data'.format(loworder[0]))
        ax.plot(self.highg, self.highdata, 'b.', label=r'$f_l$ ({}) data'.format(highorder[0]))
        ax.errorbar(self.lowg, self.lowdata, yerr=lowsigma, color='red', fmt='o', markersize=4, capsize=4)
        ax.errorbar(self.highg, self.highdata, yerr=highsigma, color='blue', fmt='o', markersize=4, capsize=4)

        ax.legend(fontsize=14, loc='lower left')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)

        return gdata, data, sigma

    
    def training(self, gdata, data, loworder, highorder):

        '''
        A function that links the model data and the training function in GPy.
        This serves as a temporary wrapper to GPy---plans to replace this with 
        surmise have been formed.

        :Example:
            GP.training(gdata=np.linspace(0.0, 0.5, 20), data=np.array())

        Parameters:
        -----------
        gdata : numpy.linspace
            An array of the coupling constant values corresponding to the data
            array points.

        data : numpy.ndarray
            An array of data points taken from the two expansion models to be
            emulated. 

        Returns:
        --------
        m : GPy object
            The object storing all training information from the GPy regression
            performed on the data.
        '''

        #set up the proper format
        gdatatr = gdata[::2]
        datatr = data[::2]
        index = int(0.33*len(gdatatr))
        gs = np.concatenate((gdatatr[0:index+1], gdatatr[2*(index)+1:]))
        datas = np.concatenate((datatr[0:index+1], datatr[2*(index)+1:]))
        gc = gs.reshape(-1,1)
        datac = datas.reshape(-1,1)

        #use GPR and kernel to train
        m = gp.models.GPRegression(gc, datac, self.kern)
        #m['sum.rbf.lengthscale'] = 100.0
        #m['sum.rbf.variance'] = 0.01
        m.optimize(messages=True)
        print(m)

        #change plotting library for matplotlib use
        gp.plotting.change_plotting_library('matplotlib')

        #plot the training results
        fig = m.plot(figsize=(8,6), dpi=100)
        ax = plt.gca()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(1.8,2.6)
        ax.set_title('GP Results', fontsize=22)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel(r'F(g)', fontsize=22)
        ax.plot(gdata, Models.true_model(self, gdata), 'k', label='True model')
        ax.plot(self.lowg, self.lowdata, 'r.', label=r'$f_s$ ({}) data'.format(loworder[0]))
        ax.plot(self.highg, self.highdata, 'b.', label=r'$f_l$ ({}) data'.format(highorder[0]))
        ax.legend(fontsize=14)
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            ax.figure.savefig(name)

        return m


    def validate(self, m, g, gdata, data, loworder, highorder):

        '''
        '''

        #set up proper validation format
        gdataval = gdata[1::2]
        dataval = data[1::2]
        gc = gdataval.reshape(-1,1)
        datac = dataval.reshape(-1,1)

        #predict the results for the validation data
        meanp, vp = m.predict(gc)
        meanp = meanp[:,0]
        vp = vp[:,0]

        #calculate the interval for the predictions
        intervals = np.zeros([len(meanp), 2])
        intervals[:,0] = meanp - 1.0*np.sqrt(vp)
        intervals[:,1] = meanp + 1.0*np.sqrt(vp)

        #plot the results
        dpi = int(input('Set a dpi for the figure.'))
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(1.8,2.6)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): mixed model', fontsize=22)
        ax.plot(g, Models.true_model(self, g), 'k', label='True model')

        #plot the data
        ax.plot(self.lowg, self.lowdata, 'r.', label=r'$f_s$ ({}) data'.format(loworder[0]))
        ax.plot(self.highg, self.highdata, 'b.', label=r'$f_l$ ({}) data'.format(highorder[0]))
        ax.plot(gdataval, dataval, color='black', marker='+', linestyle=' ', label='Prediction data')
        ax.plot(gdataval, meanp, 'g', label='Predictions')
        ax.plot(gdataval, intervals[:,0], color='green', linestyle='dotted', label=r'68$\%$ interval')
        ax.plot(gdataval, intervals[:,1], color='green', linestyle='dotted')
        ax.fill_between(gdataval, intervals[:,0], intervals[:,1], color='green', alpha=0.2)

        ax.legend(fontsize=14, loc='lower left')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)

        return meanp, vp