import numpy as np 
import math
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mixing import Models
from discrepancy import Discrepancy 

__all__ = ['GP']

class GP(Models):


    def __init__(self, g, kernel="RBF", fix_length=False):

        '''
        A class that will pull from the Models and Discrepancy classes
        to perform GP emulation on two models from the small-g expansion region
        to the large-g expansion region. The parameter settings of the kernel
        will be set by the user in this initial function. This class 'wraps' the
        scikit learn package. 

        :Example:
            GP()

        Parameters:
        -----------
        g : numpy linspace
            The linspace across the coupling constant space used for the GP.
        
        kernel : str
            The type of kernel the user wishes to use. Default is the RBF kernel;
            possible choices are RBF, Matern, and Rational Quadratic. 

        fix_length : bool
            If True, will fix the lengthscale to a specific value entered. 
            If False, will optimize the lengthscale (default). 

        Returns:
        -------
        None.
        ''' 
        
        #set up the prediction array as a class variable for use later
        self.gpredict = g

        #kernel set-up for the rest of the class (one-dimensional)
        kconstant = kernels.ConstantKernel(1.0)

        #fix lengthscale option
        if fix_length == True:
            lsc = float(input('Enter the lengthscale value.'))

            if kernel == "RBF":
                k = kernels.RBF(length_scale=lsc, length_scale_bounds=(lsc,lsc))
            elif kernel == "Matern":
                nu = float(input('Enter a value for nu (standard: 0.5, 1.5, 2.5).'))
                k = kernels.Matern(length_scale=lsc, length_scale_bounds=(lsc,lsc), nu=nu)
            elif kernel == "Rational Quadratic":
                k = kernels.RationalQuadratic(length_scale=lsc, length_scale_bounds=(lsc,lsc), alpha=1)
            else:
                raise ValueError('Please choose an available kernel.')

        else:
            if kernel == "RBF":
                k = kernels.RBF(length_scale=0.10, length_scale_bounds=(1e-5,1e5))
            elif kernel == "Matern":
                nu = float(input('Enter a value for nu (standard: 0.5, 1.5, 2.5).'))
                k = kernels.Matern(length_scale=0.4, length_scale_bounds=(1e-5,1e5), nu=nu)
            elif kernel == "Rational Quadratic":
                k = kernels.RationalQuadratic(length_scale=1.0, alpha=1)
            else:
                raise ValueError('Please choose an available kernel.')
            
        self.kern = kconstant * k

        print('Initializing standard Constant * {} kernel.'.format(kernel))

        return None

    
    def training(self, loworder, highorder, error=False):

        '''
        A function that links the model data and the training function in 
        scikit learn.

        :Example:
            GP.training(gdata=np.linspace(0.0, 0.5, 20), data=np.array())

        Parameters:
        -----------
        loworder : numpy.ndarray, float, int
            The truncation order for the low-g expansion.

        highorder : numpy.ndarray, float, int
            The truncation order for the high-g expansion.

        error : bool
            A boolean variable to toggle use of the data uncertainty in the 
            kernel during training. Default is False. 

        Returns:
        --------
        sk : scikit learn object
            The object storing all training information from the sklearn regression
            performed on the data.
        '''

        #call the training set generator function
        gs, datas, sigmas = self.training_set(loworder, highorder)

        #choose specific training points
        #2 vs 2 perfect points
        gs = np.array([gs[6], gs[15], gs[58], gs[65]])
        datas = np.array([datas[6], datas[15], datas[58], datas[65]])
        sigmas = np.array([sigmas[6], sigmas[15], sigmas[58], sigmas[65]])
        #5 vs 10 perfect points
        # gs = np.array([gs[6], gs[9], gs[17], gs[20]])
        # datas = np.array([datas[6], datas[9], datas[17], datas[20]])
        # sigmas = np.array([sigmas[6], sigmas[9], sigmas[17], sigmas[20]])
        #5 vs 5 perfect points
        # gs = np.array([gs[6], gs[9], gs[25], gs[30]])
        # datas = np.array([datas[6], datas[9], datas[25], datas[30]])
        # sigmas = np.array([sigmas[6], sigmas[9], sigmas[25], sigmas[30]])

        print(gs)

        #make column vectors for the regressor
        gc = gs.reshape(-1,1)
        datac = datas.reshape(-1,1)

        #take the data point uncertainty into the kernel 
        if error == True:
            self.alpha = np.square(sigmas)
        else:
            self.alpha = 1e-12

        #use GPR and kernel to train
        m = GaussianProcessRegressor(kernel=self.kern, alpha=self.alpha, n_restarts_optimizer=20, normalize_y=True)

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
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, max(self.gpredict))
        ax.set_ylim(-2.0,5.0)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): training set', fontsize=22)
        ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, yerr=self.lowsigma, color='red', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_s$ ($N_s$ = {}) data'.format(loworder[0]))
        ax.errorbar(self.gtrhigh, self.datatrhigh, yerr=self.highsigma, color='blue', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_l$ ($N_l$ = {}) data'.format(highorder[0]))

        #plot the chosen training points over the whole training set
        ax.errorbar(gs, datas, yerr=sigmas, color='black', fmt='o', markersize=4, capsize=4, label='Training data')

        ax.legend(fontsize=14, loc='upper right')
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
            GP.validate(sk, loworder=5, highorder=5)

        Parameters:
        -----------
        sk : scikit learn object
            The GP object created from training the GP on the data. 

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
        
        cov : numpy.ndarray
            The covariance matrix of the GP prediction results. 
        '''

        #make the prediction values into a column vector
        self.gpred = self.gpredict.reshape(-1,1)

        #predict the results for the validation data
        self.meanp, self.sigp = sk.predict(self.gpred, return_std=True)
        meanc, self.cov = sk.predict(self.gpred, return_cov=True)
        self.meanp = self.meanp[:,0]

        #calculate the interval for the predictions
        if interval == 68:
            factor = 1.0
        elif interval == 95:
            factor = 1.96
        intervals = np.zeros([len(self.meanp), 2])
        intervals[:,0] = self.meanp - factor*self.sigp
        intervals[:,1] = self.meanp + factor*self.sigp

        #compare standard deviations for testing set and model values
        test_lowg = self.gpred[:4]
        test_highg = self.gpred[4:]
        var_low = Discrepancy.variance_low(self, test_lowg, loworder[0], error_model=2)[:,0]
        self.stdev_low = np.sqrt(var_low)
        var_high = Discrepancy.variance_high(self, test_highg, highorder[0], error_model=2)[:,0]
        self.stdev_high = np.sqrt(var_high)

        #plot the results
        dpi = int(input('Set a dpi for the figure.'))
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, max(self.gpredict))
        ax.set_ylim(1.0,3.0)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): GP predictions', fontsize=22)
        ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, self.lowsigma, color="red", fmt='o', markersize=4, \
            capsize=4, alpha = 0.4, label=r"$f_s$ ($N_s$ = {})".format(loworder[0]), zorder=1)
        ax.errorbar(self.gtrhigh, self.datatrhigh, self.highsigma, color="blue", fmt='o', markersize=4, \
             capsize=4, alpha=0.4, label=r"$f_l$ ($N_l$ = {})".format(highorder[0]), zorder=1)
        ax.plot(self.gpred, self.meanp, 'g', label='Predictions', zorder=2)
        ax.plot(self.gpred, intervals[:,0], color='green', linestyle='dotted', label=r'{}$\%$ interval'.format(interval), zorder=2)
        ax.plot(self.gpred, intervals[:,1], color='green', linestyle='dotted', zorder=2)
        ax.fill_between(self.gpred[:,0], intervals[:,0], intervals[:,1], color='green', alpha=0.3, zorder=10)

        ax.legend(fontsize=14, loc='upper right')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name)

        return self.meanp, self.sigp, self.cov

    
    def training_set(self, loworder, highorder):

        '''
        An internal function to calculate the necessary training data set from
        the input prediction set. 

        :Example:
            GP.training_set(loworder=np.array([2]), highorder=np.array([2])) 

        Parameters:
        -----------

        Returns:
        -------
        gs : numpy.ndarray
            The modified array of input values for the training. 

        datas : numpy.ndarray
            The modified array of data values for the training. 

        sigmas : numpy.ndarray 
            The modified array of the truncation errors for the training. 
        '''

        #set up the training set from the prediction set (offset by midpoint)
        midpoint = (self.gpredict[1] - self.gpredict[0]) / 2.0
        gtrainingset = np.linspace(min(self.gpredict)+midpoint, max(self.gpredict)+midpoint, len(self.gpredict))

        #stop the training set, negative curvature
        if loworder[0] % 4 == 2 or loworder[0] % 4 == 3:
            for i in range(len(gtrainingset)):
                if Models.low_g(self, gtrainingset[i], loworder) < 1.0:
                    lowindex = i-1
                    break
        #stop the training set, positive curvature
        elif loworder[0] % 4 == 0 or loworder[0] % 4 == 1:
            for i in range(len(gtrainingset)):
                if Models.low_g(self, gtrainingset[i], loworder) > 3.0:
                    lowindex = i-1
                    break
        #stop the training set, even orders (positive curvature)
        if highorder[0] % 2 == 0:
            for i in range(len(gtrainingset)):
                if Models.high_g(self, gtrainingset[i], highorder) > 3.0:
                    highindex = i+1
                else:
                    break
        #stop the training set, odd orders (negative curvature)
        else:
            for i in range(len(gtrainingset)):
                if Models.high_g(self, gtrainingset[i], highorder) < 1.0:
                    highindex = i+1
                else:
                    break

        #slice the training set for the two models
        self.gtrlow = gtrainingset[:lowindex]
        self.gtrhigh = gtrainingset[highindex:]

        #calculate the data at each point
        self.datatrlow = Models.low_g(self, self.gtrlow, loworder)[0,:]
        self.datatrhigh = Models.high_g(self, self.gtrhigh, highorder)[0,:]

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

        #stop training around the gap
        for i in range(len(gtr)):
            if gtr[i+1] < gtr[i]:
                index_low = i+1
                index_high = i
                break
        for i in range(len(gtr), -1, -1):
            if gtr[i-1] <= gtr[index_high]:
                index_end = i
                break

        #find left end of the gap
        all_indices = np.where(gtr == gtr[index_low])[0]

        #set up the new arrays
        gs = np.concatenate((gtr[0:all_indices[0]], gtr[index_end:]))
        datas = np.concatenate((datatr[0:all_indices[0]], datatr[index_end:]))
        sigmas = np.concatenate((sigmatr[0:all_indices[0]], sigmatr[index_end:]))

        return gs, datas, sigmas 


    def ref_dist(self, mean, cov):

        '''
        Constructs a multivariate normal distribution to act
        as a reference distribution for the Mahalanobis distance
        calculation. 

        :Example:
            Diagnostics.ref_dist(mean=np.array([]), cov=np.array([]))

        Parameters:
        -----------
        mean : numpy.ndarray
            The mean of the GP (given by the prediction set). 
        
        cov : numpy.ndarray
            The covariance matrix of the GP (given by the prediction
            set). 

        Returns:
        --------
        dist : stats object
            A multivariate normal distribution that can be used to 
            generate samples for the reference distribution. 
        '''

        dist = stats.multivariate_normal(mean=mean, cov=cov)

        return dist

    
    def sample_ref(self, dist, n_curves):

        '''
        Generate some sample curves from the reference distribution.

        :Example:
            Diagnostics.sample_ref(dist, n_curves=10)

        Parameters:
        -----------
        dist : stats object
            The reference distribution object. 

        n_curves : int
            The number of draws from the reference distribution.

        Returns:
        --------
        samples : numpy.ndarray
            The array of curves from the distribution. 
        '''

        samples = dist.rvs(n_curves).T

        return samples


    def MD_set(self, sigmas):

        '''
        ***FINISH DOCUMENTATION & SPLIT CLASS***
        A function that takes in the errors from the training set and 
        uses them to cut the prediction set from GP.validate() to the 
        approximate region of the gap where the GP is most important. 

        Example:
            GP.MD_set(sigmas=np.array([]))
        '''

        #select the lhs of the gap
        logsigl = np.zeros([len(sigmas)])
        for i in range(len(sigmas)):
            logsigl[i] = math.floor(math.log10(sigmas[i]))
            if logsigl[i] >= -2.:
                index = i
                if sigmas[index+1]/sigmas[index] <= abs(1.5):
                    first_index = index+1
                    print(first_index)
                    break 

        #select the rhs of the gap
        logsigr = np.zeros([len(sigmas)])
        for i in range(len(sigmas)-1, -1, -1):
            logsigr[i] = math.floor(math.log10(sigmas[i]))
            if logsigr[i] >= -2.:
                index = i
                if sigmas[index-1]/sigmas[index] >= abs(0.01):
                    second_index = index-1
                    print(second_index)
                    break

        #reduce the prediction set (g) to the gap 
        md_g = self.gpredict[first_index:second_index]
#        print(self.gpredict[first_index], self.gpredict[second_index])

        #reduce the GP mean, sig, cov to the gap
        md_mean = self.meanp[first_index:second_index]
        md_sig = self.sigp[first_index:second_index]
        md_cov = self.cov[first_index:second_index, first_index:second_index]  

        #plot the result to check
        plt.xlim(0.,1.)
        plt.plot(md_g, np.ones(len(md_g)), 'k.')

#        print(np.shape(md_cov))

        return md_g, md_mean, md_sig, md_cov


    def Mahalanobis(self, y, mean, cov):

        '''
        A diagnostic testing function that can calculate the Mahalanobis 
        distance for a given set of mean, covariance data and a vector. 

        Uses: 1). Calculate the MD of the predictions of the GP;
              2). Calculate the MD of the predictions to construct a 
                  reference distribution. 

        :Example:
            GP.MD(y=np.array([]), mean=np.array([]), cov=np.array([2,2]))

        Parameters:
        -----------
        y : numpy.ndarray
            An array of predicted values from the emulator.

        mean : numpy.ndarray
            An array of true values from the true model (simulator).

        cov : numpy.ndarray 
            A 2D covariance matrix from the emulator. 

        Returns:
        --------
        md : float
            The Mahalanobis distance. 
        '''

        y = np.atleast_2d(y)

        md = np.squeeze(np.sqrt(np.diag((y - mean) @ np.linalg.inv(cov) @ (y - mean).T)))

        return md


    def md_plot(self, g, mean, cov):

        '''
        Histogram plotter for the Mahalanobis distance and the GP's
        reference distribution. 

        :Example:
            Diagnostics.md_plot(g=np.linspace(), mean=np.array([]), cov=np.array([]))

        Parameters:
        -----------
        g : numpy.ndarray
            The input prediction set to the GP. 

        mean : numpy.ndarray
            The array of the mean values from the prediction
            set of the GP. 

        cov : numpy.ndarray
            The covariance matrix from the prediction set of 
            the GP. 

        Returns:
        --------
        None. 
        '''

        #call the reference distribution functions
        n_curves = 500
        dist = self.ref_dist(mean, cov)
        samples = self.sample_ref(dist, n_curves)

        #calculate MD^2 for the reference distribution
        md_ref = np.ones([n_curves])
        for i in range(n_curves):
            md_ref[i] = self.Mahalanobis(samples[:,i].T, mean, cov)

        #MD^2 for the true value and the GP  
        ftrue = Models.true_model(g)
        md_true = self.Mahalanobis(ftrue, mean, cov)**2.0

        #histogram plot
        fig = plt.figure(figsize=(8,6), dpi=200)
        ax = plt.axes()
        ax.set_xlabel('MD', fontsize=14)
        ax.set_title('Mahalanobis distance: reference distribution', fontsize=14)
        ax.set_xlim(0.0, max(md_ref))
        ax.hist(md_ref, bins=50, histtype='bar', facecolor='black', ec='white', label='Reference distribution')

        #check the value of md_true
        if md_true - md_ref <= 100:
            ax.plot(md_true, 0.0, 'r', marker='o', markersize=10)

        #plot the chi-squared distribution over this histogram
        ax.plot(np.linspace(0.0,max(md_ref),200), 250*stats.chi2.pdf(np.linspace(0.0,max(md_ref),200), df=int(len(g))), \
            'r', linewidth=2, label=r'$\chi^2$ (df={})'.format(len(g)))
        ax.legend(loc='upper right', fontsize=12)
        plt.show()

        return None

'''
    def plotter(self, x_label, y_label, title, y_lim, *args, show_true=True, **kwargs):

        #set up the simple bits
        dpi = int(input('Set a dpi for the figure.'))
        fig = plt.figure(figsize=(8,6), dpi=dpi)
        ax = plt.axes()
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, max(self.gpredict))
        #ax.set_ylim(-2.0,5.0)
        ax.set_ylim(y_lim)
        ax.set_xlabel(x_label, fontsize=22)
        ax.set_ylabel(y_label, fontsize=22)
        ax.set_title(title, fontsize=22)

        #plot true model if indicated
        if (show_true):
            ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the rest

        return None
'''