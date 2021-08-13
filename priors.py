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

    
    def lpdf(self, params):

        '''
        Log pdf of the priors for the three parameters.
        '''

        param_1 = stats.norm.logpdf(params[0], 0.11, 0.01)
        param_2 = stats.norm.logpdf(params[1], 0.2, 0.01)
        param_3 = stats.norm.logpdf(params[2], 0.16, 0.01)

        return (param_1 + param_2 + param_3)


if __name__ == "_main__":
    pass 