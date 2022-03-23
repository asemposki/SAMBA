import numpy as np
from sympy import Q
from mixing import Models
from gaussprocess import GP 

class Pade(Models):

    
    def __init__(self):

        return None 

    
    def padef(self, g, loworder, highorder):

        '''
        Pade approximant function from Honda (2014) with 
        a,b fixed to 0, -1/2 respectively, to match the
        toy model series expansions. 

        :Example:
            Pade.padef(g=np.linspace(), loworder=2, highorder=2)

        Parameters:
        -----------
        g : numpy.linspace
            The input space over which the Pade is calculated.

        loworder : int
            The truncation order of the small-g expansion.

        highorder : int
            The truncation order of the large-g expansion.

        Returns:
        --------
        pmn : numpy.ndarray
            The result of the Pade approximant. 
        '''

        #set up the constants
        a = 0
        b = -0.5
        
        #must be less than or equal to loworder, highorder
        m = loworder[0]
        n = highorder[0]

        #p, q
        p = 0.5 * (m + n + 1.0 + (b-a))
        q = 0.5 * (m + n + 1.0 - (b-a))

        #s0---the first coefficient in the small-g expansion
        s0 = np.sqrt(2.0) * 0.5

        #coefficients c and d
        c = np.zeros(len(p))
        d = np.zeros(len(q))

        #pade equation
        pmn = np.zeros(len(g))

        for i in range(len(g)):

            pmn[i] = s0 * g[i]**(a) * \
                 (1.0 + np.sum(c[k] * g[i]**(k) for k in p))/ \
                     (1.0 + np.sum(d[k] * g[i]**(k) for k in q))

        return pmn