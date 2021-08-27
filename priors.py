import numpy as np
from scipy import stats

'''TEST FILE: User will write their own prior class
   and "send" it to the mixed model/sampler_mix functions.''' 


class Priors:

    def __init__(self):

        '''
        Prior class for the cosine function -> 3 parameters.
        '''
        pass

    
    def luniform(self, theta, a, b):

        '''
        General uniform prior to be used to truncate the normal
        distributions used in the parameter priors. 
        '''

        if theta > a and theta < b:
            return 0.0
        else:
            return -np.inf

    
    def lpdf(self, params):

        '''
        Log pdf of the priors for the parameters. Must be truncated
        for the sampler to walk in valid regions. 
        '''

        if len(params) == 2:
            param_1 = stats.norm.logpdf(params[0], 10.0, 2.0)
            param_2 = stats.norm.logpdf(params[1], -20.0, 10.0)

            return (param_1 + param_2)

        elif len(params) == 3:

            #g1 truncated between (0, inf)
            g1 = self.luniform(params[0], 0.1, np.inf) + stats.norm.logpdf(params[0], 0.12, 0.5)

            #g3 truncated between (g1, inf)
            g3 = self.luniform(params[2], params[0], np.inf) + stats.norm.logpdf(params[2], 0.16, 0.5)

            #g2 truncated between (g3, inf)
            g2 = self.luniform(params[1], params[2], np.inf) + stats.norm.logpdf(params[1], 0.2, 0.5)

            return (g1 + g2 + g3)

        else:
            raise ValueError('The number of parameters does not match any available switching function.')


if __name__ == "_main__":
    pass 