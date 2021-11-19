import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mixing import Mixing, Models
from discrepancy import Discrepancy 

class GP(Mixing):


    def __init__(self, g):

        '''
        A class that will pull from the Models, Mixing, and Discrepancy classes
        to perform GP emulation on two models from the small-g expansion region
        to the large-g expansion region. The parameter settings of the kernel
        will be set by the user in this initial function. This class 'wraps' the
        scikit learn package. 

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
        
        #set up the prediction array as a class variable for use later
        self.gpredict = g

        #kernel set-up for the rest of the class (one-dimensional)
        kconstant = kernels.ConstantKernel(1.0)
        krbf = kernels.RBF(length_scale=0.10, length_scale_bounds=(1e-5,1e5))
        self.kern = kconstant * krbf
        print('Initializing standard Constant * RBF kernel.')

        return None

    
    def training(self, loworder, highorder, error=False):

        '''
        A function that links the model data and the training function in 
        scikit learn.

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

        sigma : numpy.ndarray
            An array of the data point uncertainty from the two expansion models.

        error : bool
            A boolean variable to toggle use of the data uncertainty in the 
            kernel during training. 

        Returns:
        --------
        sk : scikit learn object
            The object storing all training information from the sklearn regression
            performed on the data.
        '''

        #set up the training set from the prediction set (offset by midpoint)
        midpoint = (self.gpredict[1] - self.gpredict[0]) / 2.0
        gtrainingset = np.linspace(min(self.gpredict)+midpoint, max(self.gpredict)+midpoint, len(self.gpredict))

        #stop the training set before large values are obtained
        for i in range(len(gtrainingset)):
            if Mixing.low_g(self, gtrainingset[i], loworder) > 2.6:
               # lowmax = gtrainingset[i-1]
                lowindex = i-1
                break

        for i in range(len(gtrainingset)):
            if Mixing.high_g(self, gtrainingset[i], highorder) < 1.8:
               # highmin = gtrainingset[i+1]
                highindex = i+1
            else:
                break

        #slice the training set for the two models
        self.gtrlow = gtrainingset[:lowindex]
        self.gtrhigh = gtrainingset[highindex:]

        #calculate the data at each point
        self.datatrlow = Mixing.low_g(self, self.gtrlow, loworder)[0,:]
        self.datatrhigh = Mixing.high_g(self, self.gtrhigh, highorder)[0,:]

        #calculate the variance at each point from the next term
        response = input('Which error model do you want to use? (uninformative/informative)')
        if response == 'uninformative':
            error_model = 1
        elif response == 'informative':
            error_model = 2
        else:
            raise ValueError('Please choose a valid error model.')

        lowvariance = Discrepancy.variance_low(self, self.gtrlow, loworder[0], error_model)
        self.lowsigma = np.sqrt(lowvariance)
        highvariance = Discrepancy.variance_high(self, self.gtrhigh, highorder[0], error_model)
        self.highsigma = np.sqrt(highvariance)

        #concatenate the arrays for use in the GP
        gtr = np.concatenate((self.gtrlow, self.gtrhigh))
        datatr = np.concatenate((self.datatrlow, self.datatrhigh))
        sigmatr = np.concatenate((self.lowsigma, self.highsigma))

        #choose a place to stop training around the gap
        index = int(0.33*len(gtr))
        gs = np.concatenate((gtr[0:index+1], gtr[int(2.5*(index)+1):]))
        datas = np.concatenate((datatr[0:index+1], datatr[int(2.5*(index)+1):]))
        sigmas = np.concatenate((sigmatr[0:index+1], sigmatr[int(2.5*(index)+1):]))

        #set up the proper format
        self.n_skip = 18
        gs = gs[::self.n_skip]
        datas = datas[::self.n_skip]
        sigmas = sigmas[::self.n_skip]

        #take out specific points
        # gs = np.concatenate((gs[0:2], gs[-2:]))
        # datas = np.concatenate((datas[0:2], datas[-2:]))
        # sigmas = np.concatenate((sigmas[0:2], sigmas[-2:]))

        #make column vectors for the regressor
        gc = gs.reshape(-1,1)
        datac = datas.reshape(-1,1)

        #take the data point uncertainty into the kernel 
        if error == True:
            alpha = np.square(sigmas)
        else:
            alpha = 1e-12

        #use GPR and kernel to train
        m = GaussianProcessRegressor(kernel=self.kern, alpha=alpha, n_restarts_optimizer=20, normalize_y=True)

        #fit the GP to the training data
        sk = m.fit(gc, datac)

        #print the optimized parameters for the user
        print('Optimized parameters: {}, {}'.format(m.kernel_.k1, m.kernel_.k2))

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
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): mixed model', fontsize=22)
        ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, yerr=self.lowsigma, color='red', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_s$ ({}) data'.format(loworder[0]))
        ax.errorbar(self.gtrhigh, self.datatrhigh, yerr=self.highsigma, color='blue', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_l$ ({}) data'.format(highorder[0]))

        #plot the chosen training points over the whole training set
        ax.errorbar(gs, datas, yerr=sigmas, color='black', fmt='o', markersize=4, capsize=4, label='Training data')

        ax.legend(fontsize=14, loc='lower left')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)

        return sk


    def validate(self, sk, loworder, highorder, interval=68):

        '''
        A wrapper function for scikit learn's GP prediction function. This will 
        predict the GP results with an interval and plot against the expansions.

        :Example:
            GP.validate(sk, g=np.linspace(1e-6,0.5,100), gdata=np.linspace(), 
            data=np.array([]), loworder=5, highorder=5)

        Parameters:
        -----------
        sk : scikit learn object
            The GP object created from training the GP on the data. 

        g : numpy.linspace
            The full range of g used in the investigation.

        gdata : numpy.linspace
            The range over which the data set exists.
        
        data : numpy.ndarray
            The data set formed from the model expansions.

        loworder : int
            The order at which the small-g expansion data was generated.

        highorder : int
            The order at which the large-g expansion data was generated.

        interval : float
            The credible interval desired. 68 or 95 available.

        Return:
        -------
        meanp : numpy.ndarray
            The mean array of the GP prediction results.

        sigp : numpy.ndarray
            The standard deviation array of the GP prediction results. 
        '''

        #make the prediction values into a column vector
        gpred = self.gpredict.reshape(-1,1)

        #predict the results for the validation data (vp = std)
        meanp, sigp = sk.predict(gpred, return_std=True)
        meanp = meanp[:,0]

        #calculate the interval for the predictions
        if interval == 68:
            factor = 1.0
        elif interval == 95:
            factor = 1.96
        intervals = np.zeros([len(meanp), 2])
        intervals[:,0] = meanp - factor*sigp
        intervals[:,1] = meanp + factor*sigp

        #plot the results
        dpi = int(input('Set a dpi for the figure.'))
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, 0.5)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): mixed model', fontsize=22)
        ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, self.lowsigma, fmt="r.", markersize=4, capsize=4, alpha = 0.4, label=r"$f_s$ ({})".format(loworder[0]), zorder=1)
        ax.errorbar(self.gtrhigh, self.datatrhigh, self.highsigma, fmt="b.", markersize=4, capsize=4, alpha=0.4, label=r"$f_l$ ({})".format(highorder[0]), zorder=1)
        ax.plot(self.gpredict, meanp, 'g', label='Predictions', zorder=2)
        ax.plot(self.gpredict, intervals[:,0], color='green', linestyle='dotted', label=r'{}$\%$ interval'.format(interval), zorder=2)
        ax.plot(self.gpredict, intervals[:,1], color='green', linestyle='dotted', zorder=2)
        ax.fill_between(self.gpredict, intervals[:,0], intervals[:,1], color='green', alpha=0.3, zorder=10)

        ax.legend(fontsize=14)
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)

        return meanp, sigp 