import numpy as np
from scipy import stats 

__all__ = ['Priors']

class Priors:

    def __init__(self):

        r'''
        Prior class for the three mixing functions currently
        in the package -> logistic, cdf, and cosine. These are 
        found in the LMM() class. In future, users can input 
        these functions themselves for their choice of mixing 
        function and data set. 

        Example:
            Priors()

        Parameters:
            None.

        Returns:
            None.
        '''
        return None 

    
    def luniform(self, theta, a, b):

        r'''
        General uniform prior to be used to truncate the normal
        distributions used in the parameter priors. 

        Parameters:
            theta (float): The parameter in question.

            a (float): The lower cutoff of the uniform prior.

            b (float): The upper cutoff of the uniform prior.

        Returns:
            The value of the log uniform prior given the 
                value of the hyperparameter theta.
        '''

        if theta > a and theta < b:
            return 0.0
        else:
            return -np.inf

    
    def lpdf(self, params):

        r'''
        Log pdf of the priors for the parameters. Must be truncated
        for the sampler to walk in valid regions. 

        Parameters:
            params (numpy.ndarray): The hyperparameters of the 
                mixing function that are being estimated.

        Returns:
            The value of the total log pdf of the priors on the 
                hyperparameters.
        '''

        if isinstance(params, float) == True:
            params = np.array([params])

        if len(params) == 1:
            param_1 = self.luniform(params[0], 0.0, 1.0)

            return param_1

        if len(params) == 2:
            # param_1 = self.luniform(params[0], -20, 20) #0, 10
            # param_2 = self.luniform(params[1], -20, 20) #-50, 0
            param_1 = stats.norm.logpdf(params[0], 10.0, 2.0)
            param_2 = stats.norm.logpdf(params[1], -20.0, 10.0)

            return (param_1 + param_2)

        elif len(params) == 3:
            #g1 truncated between (0, 0.35) 
            g1 = self.luniform(params[0], 0.01, 0.3) + stats.norm.logpdf(params[0], 0.1, 0.05)   #0.1 for 2 v 2, #0.1 for 5 v 5

            #g3 truncated between (g1, 0.35)
            g3 = self.luniform(params[2], params[0], 0.55) + stats.norm.logpdf(params[2], 0.4, 0.05)   #0.4 for 2 v 2, #0.25 for 5 v 5

            #g2 truncated between (g3, 0.35)
            g2 = self.luniform(params[1], params[2], 0.8) + stats.norm.logpdf(params[1], 0.6, 0.05)  #0.6 for 2 v 2, #0.4 for 5 v 5

            return (g1 + g2 + g3)

        else:
            raise ValueError('The number of parameters does not match any available switching function.')


if __name__ == "_main__":
    pass 
