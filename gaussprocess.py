import numpy as np 
import seaborn as sns
import docrep
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from scipy import stats, misc
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mixing import Models
from uncertainties import Uncertainties

#set savefig color for all plots
plt.rcParams['savefig.facecolor']='white'


__all__ = ['GP', 'Diagnostics']

docstrings = docrep.DocstringProcessor()


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

        #integral length
        self.gint = np.empty([])

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
                k = kernels.RBF(length_scale=0.5, length_scale_bounds=(1e-5,1e5))
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

    
    def training(self, loworder, highorder, error=False, method=1):

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

        method : int    #TEMPORARY 
            The method used for determining the training points. Options: 1,2,3.

        Returns:
        --------
        sk : scikit learn object
            The object storing all training information from the sklearn regression
            performed on the data.
        '''

        #first set the method
        self.method = method 

        #call the training set generator function
        gs, datas, sigmas = self.training_set(loworder, highorder)

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
        print('Gaussian process parameters: {}'.format(m.kernel_))

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
        ax.set_title('F(g): training set', fontsize=22)
        ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, yerr=self.lowsigma, color='red', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_s$ ($N_s$ = {}) data'.format(loworder[0]))
        ax.errorbar(self.gtrhigh, self.datatrhigh, yerr=self.highsigma, color='blue', fmt='o', markersize=4, \
                    capsize=4, label=r'$f_l$ ($N_l$ = {}) data'.format(highorder[0]))

        #plot the chosen training points over the whole training set
        ax.errorbar(gs, datas, yerr=sigmas, color='black', fmt='o', markersize=4, capsize=4, label='Training data')

        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name, bbox_inches='tight')

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
        _, self.cov = sk.predict(self.gpred, return_cov=True)
        self.meanp = self.meanp[:,0]

        #calculate the interval for the predictions
        if interval == 68:
            factor = 1.0
        elif interval == 95:
            factor = 1.96
        intervals = np.zeros([len(self.meanp), 2])
        intervals[:,0] = self.meanp - factor*self.sigp
        intervals[:,1] = self.meanp + factor*self.sigp

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
        ax.plot(self.gpredict, Models.true_model(self, self.gpredict), 'k', label='True model')

        #plot the data
        ax.errorbar(self.gtrlow, self.datatrlow, self.lowsigma, color="red", fmt='o', markersize=4, \
            capsize=4, alpha = 0.4, label=r"$f_s$ ($N_s$ = {})".format(loworder[0]), zorder=1)
        ax.errorbar(self.gtrhigh, self.datatrhigh, self.highsigma, color="blue", fmt='o', markersize=4, \
             capsize=4, alpha=0.4, label=r"$f_l$ ($N_l$ = {})".format(highorder[0]), zorder=1)
        ax.plot(self.gpred, self.meanp, 'g', label='Predictions', zorder=2)
        ax.plot(self.gpred, intervals[:,0], color='green', linestyle='dotted', label=r'{}$\%$ CI'.format(interval), zorder=2)
        ax.plot(self.gpred, intervals[:,1], color='green', linestyle='dotted', zorder=2)
        ax.fill_between(self.gpred[:,0], intervals[:,0], intervals[:,1], color='green', alpha=0.3, zorder=10)

        ax.legend(fontsize=18, loc='upper right')
        plt.show()

        #save figure option
        response = input('Would you like to save this figure? (yes/no)')

        if response == 'yes':
            name = input('Enter a file name (include .jpg, .png, etc.)')
            fig.savefig(name, bbox_inches='tight')

        return self.meanp, self.sigp, self.cov

    
    def training_set(self, loworder, highorder):

        '''
        An internal function to calculate the necessary training data set from
        the input prediction set. 

        :Example:
            GP.training_set(loworder=np.array([2]), highorder=np.array([2])) 

        Parameters:
        -----------
        loworder : numpy.ndarray
            The truncation order of the small-g expansion.

        highorder : numpy.ndarray
            The truncation order of the large-g expansion.

        Returns:
        -------
        gs : numpy.ndarray
            The modified array of input values for the training. 

        datas : numpy.ndarray
            The modified array of data values for the training. 

        sigmas : numpy.ndarray 
            The modified array of the truncation errors for the training. 
        '''

        #print the method used for training
        print('Current training method: {}'.format(self.method))

        #set up the training set from the prediction set (offset by midpoint)
        self.midpoint = (self.gpredict[1] - self.gpredict[0]) / 2.0
        gtrainingset = np.linspace(min(self.gpredict)+self.midpoint, max(self.gpredict)+self.midpoint, len(self.gpredict))

        #stop the training sets using derivatives
        dg = gtrainingset[1] - gtrainingset[0]
        lowarray = Models.low_g(self, gtrainingset, loworder)[0]
        higharray = Models.high_g(self, gtrainingset, highorder)[0]
        dfsdg = np.gradient(lowarray, dg)
        dfldg = np.gradient(higharray, dg)    
       
        for i in range(len(gtrainingset)):
            if np.abs(dfsdg[i]) >= 10.0:
                lowindex = i-1
                #print(lowindex)
                break 

        for i in range(len(gtrainingset)-1, -1, -1):
            if np.abs(dfldg[i]) >= 10.0:
                highindex = i+1
                #print(highindex)
                break 

        #slice the training set for the two models
        self.gtrlow = gtrainingset[:lowindex]
        self.gtrhigh = gtrainingset[highindex:]

      #  print('**Length of training sets: {} and {}'.format(len(self.gtrlow), len(self.gtrhigh)))

        #calculate the data at each point
        self.datatrlow = Models.low_g(self, self.gtrlow, loworder)[0,:]
        self.datatrhigh = Models.high_g(self, self.gtrhigh, highorder)[0,:]

        #calculate the variance at each point from the next term
        obj = Uncertainties()
        lowvariance = obj.variance_low(self.gtrlow, loworder[0])
        self.lowsigma = np.sqrt(lowvariance)
        highvariance = obj.variance_high(self.gtrhigh, highorder[0])
        self.highsigma = np.sqrt(highvariance)

        #find the values of g in the other set to determine location of points
        index_glow = (np.where(self.gtrlow == self.gtrhigh[0])[0])[0]
        index_ghigh = (np.where(self.gtrhigh == self.gtrlow[-1])[0])[0]
     #   print('***Index: {} ***'.format(index_ghigh))
     #   print('Values to compare to MD: {} and {}.'.format(self.gtrlow[index_glow], self.gtrhigh[index_ghigh]))

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

     #   print('Index of error at 5%: {}; g-value = {}; error = {}'.format(indexerror, self.gtrhigh[indexerror], self.highsigma[indexerror]))

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

    @staticmethod
    def nearest_value(array, value):

        #calculate the difference between each point
        abs_val = np.abs(array - value)

        #find the smallest difference in the array
        index = abs_val.argmin()

        return index

#class Diagnostics(GP):


#     def __init__(self, g):

#         #redefine self.gpredict here
#         self.gpredict = g
        
#         #***GET GPREDICT PASSED HERE***
#         print('Available diagnostic tests: Mahalanobis distance.')

#         return None

 
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


    def MD_set(self, loworder, highorder, test=False):

        '''
        Takes the training set of points and uses them to cut the
        testing set to their limits. This reduces the MD calculation
        to the region of interest.  

        Example:
            GP.MD_set(loworder=np.array([2]), highorder=np.array([2]))

        Parameters:
        -----------
        loworder : numpy.ndarray
            The truncation order for the small-g expansion.

        highorder : numpy.ndarray
            The truncation order for the large-g expansion. 
        
        test : bool
            The option to use this function to create a test MD set. 

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
        mdobj = Uncertainties()
        lowvar = mdobj.variance_low(self.gpredict, loworder[0])
        lowerr = np.sqrt(lowvar)
        highvar = mdobj.variance_high(self.gpredict, highorder[0])
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
      #  print(f'md_g : {md_g} Type of object: {type(md_g)}')
        self.gint = md_g.copy()
        md_mean = GP_mean[index_lowerr:index_hierr]
        md_sig = GP_err[index_lowerr:index_hierr]
        md_cov = GP_cov[index_lowerr:index_hierr, index_lowerr:index_hierr]

        #select 4 points in g
        self.lenpts = int(input('Enter the number of training points.'))
        points = self.create_points(int(self.lenpts), md_g[0], md_g[-1])
        print(points)

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
        plt.xlim(0.,1.)
        plt.plot(md_g, np.ones(len(md_g)), 'k.')

        return md_g, md_mean, md_sig, md_cov


    @staticmethod
    def create_points(N, a, b):
        
        #create the linspace with endpoints
        pts_array = np.linspace(a, b, N+2)

        #remove the first and last point
        pts = pts_array[1:-1]

        return pts


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


    #develop MD plotter and copy into the class
    def md_plotter(self, dist, md_gp, md_ref, hist=True, box=False):

        '''
        ***FINISH DOCUMENTATION***
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

        #box-and-whisker option
        if box is True:
            
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


    def ref_boxplot(self, dist, q1=0.25, q3=0.75, whislo=0.025, whishi=0.975, ax=None, **kwargs):

        '''
        ***FINISH DOCUMENTATION***
        '''

        stat_dict = [{'med': dist.median(), 'q1': dist.ppf(q1), 'q3': dist.ppf(q3),
                      'whislo': dist.ppf(whislo), 'whishi': dist.ppf(whishi)}]
    
        return ax.bxp(stat_dict, showfliers=False, **kwargs)


    #function for paper plot (vertical)
    def vertical_gp(self, g, loworder, highorder):

        #set up the vertical subplots
        

        return None

'''
    def plotter(self, x_label, y_label, title, y_lim, *args, show_true=True, **kwargs):

        #set up the simple bits
        fig = plt.figure(figsize=(8,6), dpi=600)
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