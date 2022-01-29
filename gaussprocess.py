import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
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
        gs = np.array([gs[6], gs[15], gs[58], gs[65]])
        datas = np.array([datas[6], datas[15], datas[58], datas[65]])
        sigmas = np.array([sigmas[6], sigmas[15], sigmas[58], sigmas[65]])

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
        meanp, sigp = sk.predict(self.gpred, return_std=True)
        meanc, cov = sk.predict(self.gpred, return_cov=True)
        meanp = meanp[:,0]

        #calculate the interval for the predictions
        if interval == 68:
            factor = 1.0
        elif interval == 95:
            factor = 1.96
        intervals = np.zeros([len(meanp), 2])
        intervals[:,0] = meanp - factor*sigp
        intervals[:,1] = meanp + factor*sigp

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
        ax.plot(self.gpred, meanp, 'g', label='Predictions', zorder=2)
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

        return meanp, sigp, cov

    
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

    
    def MD_set(self, sigmas):

        #select the gap by checking variances
        for i in range(len(sigmas)):
            if sigmas[i+3]/sigmas[i] >= 0.1:
                first_index = i
                break 
        
        # for i in range(len(sigmas), -1, -1):
        #     if sigmas[i+3]/sigmas[i] >= 0.1:
        #         second_index = i
        #         break

        #reduce the prediction set to the gap 
        md_data = self.gpredict[first_index:]

        #plot the result to check
        plt.plot(md_data, np.ones(len(md_data)), 'k.')

        return md_data


    def MD(self, fval, mean, cov):

        '''
        A diagnostic testing function that calculates the Mahalanobis
        distance of the predictions from the GP. 

        :Example:
            GP.MD(fval=np.array([]), mean=np.array([]), cov=np.array([2,2]))

        Parameters:
        -----------
        fval : numpy.ndarray
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

        #reduce the number of points used in the MD
        thin = 20
        fval = fval[::thin]
        mean = mean[::thin]
        print(np.shape(cov), np.shape(mean), cov)
        cov = cov[::thin,::thin]
        print(np.shape(cov), cov)

        #calculate the reference distribution
        m = GaussianProcessRegressor(kernel=self.kern, alpha=self.alpha, n_restarts_optimizer=20, normalize_y=True)
        toy_data_full = m.sample_y(self.gpred, n_samples=4, random_state=2).T
        mean_full, cov_full = m.predict(self.gpred, return_cov=True)

        # toy_data_full *= (ratio**orders)[:, None]
        mask = np.array([i % 5 == 0 for i in range(len(self.gpred))])
        toy_data = toy_data_full[:, mask]
        meanplt = mean_full[mask]
        covplt = cov_full[mask][:, mask]
        X = self.gpred[mask]

        plt.plot(self.gpred.ravel(), toy_data_full.T)
        plt.plot(X.ravel(), toy_data.T, ls='', marker='o', fillstyle='none', markersize=10, c='gray')

        #calculate the Mahalanobis distance
        md = (fval - mean).T @ np.linalg.inv(cov) @ (fval - mean)
        print(fval-mean)
        md = np.sqrt(md)

        return md


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