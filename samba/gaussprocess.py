import numpy as np
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy import stats
from numpy.linalg import cholesky
import scipy.linalg as scl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from .models import Models, Uncertainties
from .discrepancy import Bivariate

# set savefig color for all plots
plt.rcParams['savefig.facecolor']='white'

__all__ = ['GP']


class GP(Bivariate):


    def __init__(self, g, loworder, highorder, kernel="RBF", nu=None, ci=68, error_model='informative'):

        '''
        A class that will pull from the Models class to perform GP emulation on 
        two models from the small-g expansion region to the large-g expansion region. 
        The parameter settings of the kernel will be set by the user in this 
        initial function. This class 'wraps' the scikit learn package. 

        :Example:
            GP(g=np.linspace(1e-6,1.0,100), loworder=5, highorder=2, kernel="Matern",
            ci=68, error_model='informative')

        Parameters:
        -----------
        g : numpy linspace
            The linspace across the coupling constant space used for the GP.
        
        highorder : numpy.ndarray, float, int
            The truncation order of the large-g expansion. 
        
        kernel : str
            The type of kernel the user wishes to use. Default is the RBF kernel;
            possible choices are RBF, Matern, and Rational Quadratic. 

        nu : float
            The value of the Matern kernel used, if kernel="Matern". Otherwise, 
            default is None.

        ci : int
            The uncertainty interval to use. Must be 68 or 95. 

        error_model : str
            The error model to be used in the calculation. Options are
            'uninformative' and 'informative'. Default is 'informative'. 

        Returns:
        -------
        None.
        ''' 
        
        #set up the prediction array as a class variable for use later
        self.gpredict = np.copy(g)

        #extract uncertainty interval for later use
        self.ci = ci 

        #check type and assign class variables
        if isinstance(loworder, float) == True or isinstance(loworder, int) == True:
            loworder = np.array([loworder])
        
        if isinstance(highorder, float) == True or isinstance(highorder, int) == True:
            highorder = np.array([highorder])
        
        self.loworder = loworder 
        self.highorder = highorder 

        #Models(), Uncertainties()
        self.m = Models(self.loworder, self.highorder)
        self.u = Uncertainties(error_model)

        #instantiate the class variable error_model for ease class crossing
        self.error_model = self.u.error_model

        #integral length
        self.gint = np.empty([])

        #kernel set-up for the rest of the class (one-dimensional)
        kconstant = kernels.ConstantKernel(1.0)

        if kernel == "RBF":
            k = kernels.RBF(length_scale=0.5, length_scale_bounds=(1e-5,1e5))
        elif kernel == "Matern":
            if nu is None:
                raise ValueError('Matern kernel must be supplied a value for nu.')
            else:
                k = kernels.Matern(length_scale=0.4, length_scale_bounds=(1e-5,1e5), nu=nu)
        elif kernel == "Rational Quadratic":
            k = kernels.RationalQuadratic(length_scale=1.0, alpha=1)
        else:
            raise ValueError('Please choose an available kernel.')
        
        self.kern = kconstant * k

        return None

    
    def training(self, error=True, method=2, plot=True):

        '''
        A function that links the model data and the training function in 
        scikit learn, and plots the training data using GP.plot_training().

        :Example:
            GP.training(error=False, method=3)

        Parameters:
        -----------
        error : bool
            A boolean variable to toggle use of a truncation error model in the 
            kernel during training. Default is True.

        method : int  
            The method used for determining the training points. Options: 1,2,3.
            For an extensive explanation of the methods, see the paper.

        plot : bool
            Option to plot the training set with series expansions and true model.
            Default is True. 

        Returns:
        --------
        sk : scikit learn object
            The object storing all training information from the sklearn regression
            performed on the data.
        '''

        #first set the method
        self.method = method 

        #call the training set generator function
        gs, datas, sigmas = self.training_set()

        #make a gs class variable for weights use
        self.gs = gs 

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
        self.sk = m.fit(gc, datac)

        #print the optimized parameters for the user
        print('Gaussian process parameters: {}'.format(m.kernel_))

        #plot the results
        if plot is True:
            self.plot_training(gs, datas, sigmas)

        return self.sk


    def validate(self, plot=True, run_taweret=False):

        '''
        A wrapper function for scikit learn's GP prediction function. This will 
        predict the GP results with an interval and plot against the expansions
        using GP.plot_validate().

        :Example:
            GP.validate()

        Parameters:
        -----------
        plot : bool
            The option to plot the GP mean and variance over the testing
            set and true model. Default is True. 

        Returns:
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
        self.meanp, self.sigp = self.sk.predict(self.gpred, return_std=True)
        _, self.cov = self.sk.predict(self.gpred, return_cov=True)
        
        # issues right here when running wrapped in Taweret
        if run_taweret is False:
            self.meanp = self.meanp[:,0]

        #calculate the interval for the predictions
        if self.ci == 68:
            factor = 1.0
        elif self.ci == 95:
            factor = 1.96
        intervals = np.zeros([len(self.meanp), 2])
        intervals[:,0] = self.meanp - factor*self.sigp
        intervals[:,1] = self.meanp + factor*self.sigp

        #plot the results
        if plot is True:
            self.plot_validate(intervals)

        return self.meanp, self.sigp, self.cov


    def plot_training(self, gs, datas, sigmas):

        '''
        A simple plotter to plot the trained GP results and models, 
        as well as the points at which the GP was trained. 

        :Example:
            GP.plot_training(gs=np.array([]), datas=np.array([]),
            sigmas=np.array([]))

        Parameters:
        ------------
        gs : numpy.ndarray
            Points chosen by GP.training_set() in input space g.
        
        datas : numpy.ndarray
            Corresponding values of the series expansions at gs.
        
        sigmas : numpy.ndarray
            Corresponding error model results at each training
            point.

        Returns:
        --------
        None.
        '''

        #set up the plot
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        fig.patch.set_facecolor('white')
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.locator_params(nbins=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0.0, max(self.gpredict))
        ax.set_ylim(1.0,3.0)
        ax.set_xlabel('g', fontsize=22)
        ax.set_ylabel('F(g)', fontsize=22)
        ax.set_title('F(g): training set', fontsize=22)
        ax.plot(self.gpredict, self.m.true_model(self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, yerr=self.lowsigma, color='red', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_s$ ($N_s$ = {}) data'.format(self.loworder[0]))
        ax.errorbar(self.gtrhigh, self.datatrhigh, yerr=self.highsigma, color='blue', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_l$ ($N_l$ = {}) data'.format(self.highorder[0]))

        #plot the chosen training points over the whole training set
        ax.errorbar(gs, datas, yerr=sigmas, color='black', fmt='o', markersize=4, capsize=4, label='Training data')

        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        # response = input('Would you like to save this figure? (yes/no)')

        # if response == 'yes':
        #     name = input('Enter a file name (include .jpg, .png, etc.)')
        #     fig.savefig(name, bbox_inches='tight')

        return None

    
    def plot_validate(self, intervals):

        '''
        A simple plotter to show the results of the GP 
        predictions at new points in g. 

        :Example:
            GP.plot_validate(intervals=np.array([,]))

        Parameters:
        -----------
        intervals : numpy.ndarray
            The uncertainty band around the prediction set.

        Returns:
        --------
        None.
        '''

        #plot the results
        fig = plt.figure(figsize=(8,6), dpi=600)
        ax = plt.axes()
        fig.patch.set_facecolor('white')
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
        ax.plot(self.gpredict, self.m.true_model(self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, self.lowsigma, color="red", fmt='o', markersize=4, \
            capsize=4, alpha = 0.4, label=r"$f_s$ ($N_s$ = {})".format(self.loworder[0]), zorder=1)
        ax.errorbar(self.gtrhigh, self.datatrhigh, self.highsigma, color="blue", fmt='o', markersize=4, \
             capsize=4, alpha=0.4, label=r"$f_l$ ($N_l$ = {})".format(self.highorder[0]), zorder=1)
        ax.plot(self.gpred, self.meanp, 'g', label='Predictions', zorder=2)
        ax.plot(self.gpred, intervals[:,0], color='green', linestyle='dotted', label=r'{}$\%$ CI'.format(self.ci), zorder=2)
        ax.plot(self.gpred, intervals[:,1], color='green', linestyle='dotted', zorder=2)
        ax.fill_between(self.gpred[:,0], intervals[:,0], intervals[:,1], color='green', alpha=0.3, zorder=10)

        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        # response = input('Would you like to save this figure? (yes/no)')

        # if response == 'yes':
        #     name = input('Enter a file name (include .jpg, .png, etc.)')
        #     fig.savefig(name, bbox_inches='tight')

        return None

    
    def training_set(self):

        '''
        An internal function to calculate the necessary training data set from
        the input prediction set. 

        :Example:
            GP.training_set() 

        Parameters:
        -----------
        None. 

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
        self.midpoint = (self.gpredict[1] - self.gpredict[0]) / 2.0
        gtrainingset = np.linspace(min(self.gpredict)+self.midpoint, max(self.gpredict)+self.midpoint, len(self.gpredict))
    
        #stop the training set, negative curvature
        if self.loworder[0] % 4 == 2 or self.loworder[0] % 4 == 3:
            for i in range(len(gtrainingset)):
                if self.m.low_g(gtrainingset[i]) < -1.0:
                    lowindex = i-1
                    break

        #stop the training set, positive curvature
        elif self.loworder[0] % 4 == 0 or self.loworder[0] % 4 == 1:
            for i in range(len(gtrainingset)):
                if self.m.low_g(gtrainingset[i]) > 3.0:
                    lowindex = i-1
                    break

        #stop the training set, even orders (positive curvature)
        if self.highorder[0] % 2 == 0:
            for i in range(len(gtrainingset)):
                if self.m.high_g(gtrainingset[i]) > 3.0:
                    highindex = i+1
                else:
                    break

        #stop the training set, odd orders (negative curvature)
        else:
            for i in range(len(gtrainingset)):
                if self.m.high_g(gtrainingset[i]) < -1.0:
                    highindex = i+1
                else:
                    break
           
        #slice the training set for the two models
        self.gtrlow = gtrainingset[:lowindex]
        self.gtrhigh = gtrainingset[highindex:]

        #calculate the data at each point
        self.datatrlow = self.m.low_g(self.gtrlow)[0,:]
        self.datatrhigh = self.m.high_g(self.gtrhigh)[0,:]

        #calculate the variance at each point from the next term
        lowvariance = self.u.variance_low(self.gtrlow, self.loworder[0])
        self.lowsigma = np.sqrt(lowvariance)
        highvariance = self.u.variance_high(self.gtrhigh, self.highorder[0])
        self.highsigma = np.sqrt(highvariance)

        #find the values of g in the other set to determine location of points
        index_ghigh = (np.where(self.gtrhigh == self.gtrlow[-1])[0])[0]
    
        #value of g at the optimal red points
        pt1 = 0.0656575
        pt2 = 0.1161625

        #method 1: using g=0.6 as a training point
        pttest = 0.6  
        indexptest = self.nearest_value(self.gtrhigh, pttest) 

        #method 3: finding based on error (5%)
        for i in range(len(self.gtrhigh)-1, -1, -1):
            if self.highsigma[i] >= 0.05*self.datatrhigh[i]:
                indexerror = i
                break 

        #find the values in the training array closest to the points
        indexpt1 = self.nearest_value(self.gtrlow, pt1)
        indexpt2 = self.nearest_value(self.gtrlow, pt2)

        #create two points on either side (highpoint = 20)
        glowtr = np.array([self.gtrlow[indexpt1], self.gtrlow[indexpt2]])
        datalowtr = np.array([self.datatrlow[indexpt1], self.datatrlow[indexpt2]])
        sigmalowtr = np.array([self.lowsigma[indexpt1], self.lowsigma[indexpt2]])

        #choose training points depending on method entered
        if self.method == 1:
            ghightr = np.array([self.gtrhigh[indexptest], self.gtrhigh[-1]])
            datahightr = np.array([self.datatrhigh[indexptest], self.datatrhigh[-1]])
            sigmahightr = np.array([self.highsigma[indexptest], self.highsigma[-1]])

        elif self.method == 2:
            ghightr = np.array([self.gtrhigh[index_ghigh], self.gtrhigh[-1]])
            datahightr = np.array([self.datatrhigh[index_ghigh], self.datatrhigh[-1]])
            sigmahightr = np.array([self.highsigma[index_ghigh], self.highsigma[-1]])

        elif self.method == 3:
            ghightr = np.array([self.gtrhigh[indexerror], self.gtrhigh[-1]])
            datahightr = np.array([self.datatrhigh[indexerror], self.datatrhigh[-1]])
            sigmahightr = np.array([self.highsigma[indexerror], self.highsigma[-1]])

        #concatenate these arrays and send back
        gtr = np.concatenate((glowtr, ghightr))
        datatr = np.concatenate((datalowtr, datahightr))
        sigmatr = np.concatenate((sigmalowtr, sigmahightr))

        return gtr, datatr, sigmatr 


    def MD_set(self, pts=3, plot=False):

        '''
        Takes the training set of points and uses them to cut the
        testing set to their limits. This reduces the MD calculation
        to the region of interest.  

        Example:
            GP.MD_set()

        Parameters:
        -----------
        pts : int
            The number of points to use to calculate the Mahalanobis
            distance. Can be any number up to the size of self.gpredict. 

        plot : bool
            The option to plot the MD points across the input space. 
            Default is False. 

        Returns:
        --------
        md_g : numpy.ndarray
            The input values used in the MD calculation.

        md_mean : numpy.ndarray
            The mean values from the GP corresponding to the 
            md_g points.

        md_sig : numpy.ndarray
            The error bars corresponding to the md_g points.

        md_cov : numpy.ndarray
            The covariance matrix corresponding to the md_g points.
        '''

        #import the GP mean, cov, and errors for the prediction set
        GP_mean = self.meanp
        GP_err = self.sigp
        GP_cov = self.cov

        #calculate the variance at each expansion point from the next term
        lowvar = self.u.variance_low(self.gpredict, self.loworder[0])
        lowerr = np.sqrt(lowvar)
        highvar = self.u.variance_high(self.gpredict, self.highorder[0])
        hierr = np.sqrt(highvar)

        #compare the values and choose where the gap is
        for i in range(len(lowerr)):
            if GP_err[i] < lowerr[i]:
                index_lowerr = i
                break

        for i in range(len(hierr)-1, -1, -1):
            if GP_err[i] < hierr[i]: 
                index_hierr = i 
                break

        #cut the GP array into the gap
        md_g = self.gpredict[index_lowerr:index_hierr]
        self.gint = md_g.copy()
        md_mean = GP_mean[index_lowerr:index_hierr]
        md_sig = GP_err[index_lowerr:index_hierr]
        md_cov = GP_cov[index_lowerr:index_hierr, index_lowerr:index_hierr]

        #select points in g
        self.lenpts = pts
        points = self.create_points(int(self.lenpts), md_g[0], md_g[-1])
        #print('Location of MD points in g: ', points)

        #find the indices
        indices = np.zeros([self.lenpts])
        for i in range(self.lenpts):
            indices[i] = self.nearest_value(md_g, points[i])

        #convert to integer array
        indices = indices.astype(int)
        
        #pick the points out of the arrays
        md_g = md_g[indices]
        md_mean = md_mean[indices]
        md_sig = md_sig[indices]
        md_cov = md_cov[np.ix_(indices, indices)]

        #plot the check the location of the points
        if plot is True:
            plt.xlim(0.,1.)
            plt.plot(md_g, np.ones(len(md_g)), 'k.')

        return md_g, md_mean, md_sig, md_cov


    def md_squared(self, md_g, md_mean, md_cov, n_curves=1000):

        '''
        A wrapper for the Mahalanobis distance calculation for the
        reference distribution and the GP curve. To calculate the 
        Cholesky decomposition or to perform an SVD analysis, consult
        GP.mahalanobis() below. 

        :Example:
            GP.md_squared(md_g=np.linspace, md_mean=np.array([]), 
                          md_cov=np.array([,]), n_curves=1000)

        Parameters:
        -----------
        md_g : numpy.linspace
            The points in input space g from the GP.MD_set() function. 

        md_mean : numpy.ndarray
            The values of the GP mean at the md_g points. 

        md_cov : numpy.ndarray
            The values of the GP covariance matrix at the md_g points. 

        n_curves : int
            The number of curves from the reference distribution that
            are drawn for the MD^2 calculation (md_ref). 

        Returns:
        --------
        md_gp : float
            The individual MD^2 value for the GP curve. 

        md_ref : numpy.ndarray  
            The array of MD^2 values from the reference distribution.
        '''

        #calculate the ref distribution MDs
        dist = self.ref_dist(md_mean, md_cov)
        y = self.sample_ref(dist, n_curves)
        md = np.ones([n_curves])
        for i in range(n_curves):
            md[i] = self.mahalanobis(y[:,i].T, md_mean, inv=md_cov, chol=False, svd=False)
        
        #MD^2 (ref)
        md_ref = md**2.0 

        #calculate the GP MD 
        fval = self.m.true_model(md_g)
        mdgp = self.mahalanobis(fval.T, md_mean, inv=md_cov, chol=False, svd=False)

        #MD^2 (GP)
        md_gp = mdgp**2.0

        return md_gp, md_ref

    
    def md_plotter(self, md_gp, md_ref, md_mean=None, md_cov=None, hist=True, box=False):

        '''
        A plotting function that allows the Mahalanobis distance
        to be plotted using either a histogram or a box and whisker
        plot, or both. 

        Box and whisker plot code heavily drawn from J. Melendez' gsum
        code (https://github.com/buqeye/gsum).

        :Example:
            GP.md_plotter(md_gp=np.array([]), md_ref=np.array([]),
            hist=False, box=True)
        
        Parameters:
        -----------
        md_gp : float
            The MD^2 value for the GP curve. 

        md_ref : numpy.ndarray
            The array of MD^2 values for the reference
            distribution.
        
        md_mean : numpy.ndarray
            The values of the GP mean at the md_g points. Only used
            for box and whisker option; default is None. 

        md_cov : numpy.ndarray
            The values of the GP covariance matrix at the md_g points. 
            Only used for box and whisker option; default is None.

        hist : bool
            Toggle for plotting a histogram. Default is True. 

        box : bool
            Toggle for plotting a box plot. Default is False. 

        Returns:
        --------
        None.
        '''
        
        title = 'Mahalanobis Distance'
        xlabel = r'$\mathrm{D}_{\mathrm{MD}}^{2}$'
        
        #histogram option
        if hist is True:
            fig = plt.figure(figsize=(8,6), dpi=600)
            fig.patch.set_facecolor('white')
            ax = plt.axes()
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_title(title, fontsize=22)
            ax.set_xlim(0.0, max(md_ref))
            ax.hist(md_ref, bins=50, density=True, histtype='bar', facecolor='black', \
                    ec='white', label='Reference distribution')
            ax.plot(md_gp, 0.0, marker='o', color='r', markersize=10)

            #add chi-squared to histogram
            n = 200
            x = np.linspace(0.0, max(md_ref), n)
            ax.plot(x, stats.chi2.pdf(x, df=self.lenpts), 'r', linewidth=2, label=r'$\chi^2$ (df={})'.format(self.lenpts))

            #include legend
            legend = True

        #box-and-whisker option
        if box is True:

            dist = self.ref_dist(md_mean, md_cov)
            
            legend = False
        
            #set up the figure
            fig = plt.figure(figsize=(8,6), dpi=100)
            ax = plt.axes()
            ax.set_xlabel(xlabel, fontsize=18)
            
            #reference distribution (using chi2, NOT md_ref)
            boxartist = self.ref_boxplot(dist, ax=ax, patch_artist=True, widths=0.8)
            gray = 'gray'
            for box in boxartist['boxes']:
                box.update(dict(facecolor='lightgrey', edgecolor=gray))
            for whisk in boxartist["whiskers"]:
                whisk.update(dict(color=gray))
            for cap in boxartist["caps"]:
                cap.update(dict(color=gray))
            for med in boxartist["medians"]:
                med.update(dict(color=gray))
            
            #ax.boxplot(md_ref, showfliers=False)
            ax.get_xaxis().set_ticks([])
            ax.tick_params(direction='in')
            ax.set_ylim(0,20)
            ax.set_aspect(0.25)
            sns.despine(offset=0, bottom=True, ax=ax)
            
            #plot the individual GP MD value
            ax.plot(1.0, md_gp, color='red', marker='o', markersize=10)
            
        #finish up plot
        if legend is True:
            ax.legend(loc='upper right', fontsize=18)
            
        plt.show()
        
        return None

    
    @staticmethod
    def mahalanobis(y, mean, inv=None, chol=False, svd=False):

        '''
        A diagnostic testing function that can calculate the Mahalanobis 
        distance for a given set of mean, covariance data and a vector. 

        Uses: 1). Calculate the MD of the predictions of the GP using
                  the inverse covariance matrix (usual method);
              2). Calculate the MD of the predictions to construct a 
                  reference distribution using the inverse covariance
                  matrix (usual method);
              3). Calculate the Cholesky decomposition of the MD
                  information;
              4). Perform an SVD analysis and send back the MD 
                  calculated via SVD. 

        :Example:
            GP.MD(y=np.array([]), mean=np.array([]), inv=numpy.ndarray([]),
            chol=False, svd=False)

        Parameters:
        -----------
        y : numpy.ndarray
            An array of predicted values from the emulator.

        mean : numpy.ndarray
            An array of true values from the true model (simulator).

        inv : numpy.ndarray
            The covariance matrix to be inverted in the MD calculation.
        
        chol : bool
            The option to calculate the Cholesky decomposition
            of the data. 

        svd : bool
            An option to perform the SVD analysis of the MD data.
            To use, must also have a covariance matrix sent to inv. 

        Returns:
        --------
        md : float (if calculating MD)
            The Mahalanobis distance. 

        chol_decomp : numpy.ndarray (if calculating Cholesky
                                     decomposition)
            The Cholesky decomposition results. 

        svderrs : numpy.ndarray (if calculating SVD)
            The SVD errors at each point in the MD testing set. 

        svd_md : float (if calculating SVD)
            The Mahalanobis distance. 
        '''

        y = np.atleast_2d(y)

        #cholesky option (solves for Cholesky decomposition)
        if (inv is not None) and (chol is True):

            chol = cholesky(inv)
            errs = scl.solve_triangular(chol, (y-mean).T, lower=True).T
            chol_decomp = np.linalg.norm(errs, axis=-1)

            return chol_decomp 

        #SVD option
        if (svd is True) and (inv is not None):
        
            #perform SVD
            _, s, vh = np.linalg.svd(inv)
            print('Eigenvalues: ',s)
            sinv = np.linalg.inv(np.diag(s))   #inverse of eigenvalue matrix
            one = vh @ (y-mean).T
            svd_md = np.squeeze(one.T @ sinv @ one)
            print('MD^2 (SVD): ', svd_md)
            
            #SVD errors
            svderrs = np.zeros([len(s)])
            for i in range(len(s)):
                svderrs[i] = np.square(1.0/np.sqrt(s[i]) * np.dot(vh[i,:],(y-mean).T))

            return svderrs, svd_md
    
        #inverse option (normal MD calculation)
        if (chol is False) and (svd is False) and (inv is not None):
            
            md = np.squeeze(np.sqrt(np.diag((y - mean) @ np.linalg.inv(inv) @ (y - mean).T)))

            return md

        #if nothing is selected
        if (inv is None):
            raise ValueError('Please input a covariance matrix.')


    @staticmethod
    def nearest_value(array, value):

        '''
        A static method to find the index of the nearest value
        of an array to a desired value. 

        :Example:
            GP.nearest_value(array=numpy.ndarray, value=5)

        Parameters:
        -----------
        array : numpy.ndarray
            The array of values to search. 

        value : int
            The desired value to search the array for. 

        Returns:
        --------
        index : int
            The index of the nearest value of the array
            to the desired value. 
        '''

        #calculate the difference between each point
        abs_val = np.abs(array - value)

        #find the smallest difference in the array
        index = abs_val.argmin()

        return index


    @staticmethod
    def ref_dist(mean, cov):

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


    @staticmethod
    def sample_ref(dist, n_curves):

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


    @staticmethod
    def create_points(N, a, b):

        '''
        A code to create a given number of points from a 
        linspace evenly from points a to b. 

        :Example:
            GP.create_points(N=3, a=0.0, b=1.0)

        Parameters:
        -----------
        N : int
            The number of points desired.

        a : float, int
            The left endpoint of the region of interest. 

        b : float, int
            The right endpoint of the region of interest. 

        Returns:
        --------
        pts : numpy.ndarray
            The resulting array of points. 
        '''
        
        #create the linspace with endpoints
        pts_array = np.linspace(a, b, N+2)

        #remove the first and last point
        pts = pts_array[1:-1]

        return pts


    @staticmethod 
    def ref_boxplot(dist, q1=0.25, q3=0.75, whislo=0.025, whishi=0.975, ax=None, **kwargs):

        '''
        Taken from the gsum code written by J. Melendez (https://github.com/buqeye/gsum).
        '''

        stat_dict = [{'med': dist.median(), 'q1': dist.ppf(q1), 'q3': dist.ppf(q3),
                      'whislo': dist.ppf(whislo), 'whishi': dist.ppf(whishi)}]
    
        return ax.bxp(stat_dict, showfliers=False, **kwargs)
