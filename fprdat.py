#Explicit formula for interpolating functions
#in the paper "On Pertubation theory improved by Strong coupling expansion"
#by Masazumi Honda (Harish-Chandra Research Institute)

#All the interpolating functions are taking the form of Fractional Power of Rational function Subscript[F**(\[Alpha]), m,n](g).
#We denote this by FPR[\[Alpha],m,n,g]
#Zero-dimensional \[Phi]**4-theory

#packages needed
import numpy as np
import scipy.special as sp
from gaussprocess import GP

__all__ = ['FPR']


class FPR(GP):

    def __init__(self, g):

        self.g = g

        return None

    
    def fprset(self, key):

        '''
        Call the proper FPR function desired and obtain 
        an array of the results in the input space, g. 

        :Example: 
            FPR.fprset(key='(2,4)^(1/8)')

        Parameters:
        -----------
        key : str
            The preferred FPR function. Enter a key in the
            convention: '(m,n)^(\alpha)', where m,n are orders
            less than or equal to N_s and N_l (loworder, highorder
            in the other classes). \alpha is the value the FPR is 
            raised to in Eq. (2.7) (Honda 2014). 

        Returns:
        --------
        fpr : numpy.ndarray
            Results of the FPR function in an array. 
        '''

        #switch-case statement for calling the proper FPR function
        fpr = np.zeros(len(self.g))

        match key:
            case '(0,0)^(1/2)':
                #FPR[1/(2*1),0,0,g_] 
                fpr = np.sqrt(2 * np.pi * np.sqrt(1/(1 + (8 * self.g * np.pi)/sp.gamma(1/4)**2)))
                print(fpr)          

        #functions for each order and alpha       
        #FPR[1/(2*1),0,0,g_] = 
        # np.sqrt(2 * np.pi * np.sqrt(1/(1 + (8 * self.g * np.pi)/sp.gamma(1/4)**2)))
        # #FPR[1/(2*1),1,1,g_] = 
        # np.sqrt(2 * np.pi * sp.gamma(1/4)) * np.sqrt((2 * np.pi * sp.gamma(-(1/4)) + 8 * self.g * np.pi * sp.gamma(1/4) + sp.gamma(1/4)**3) / (64 * self.g**2 * np.pi**2 + 2 * np.pi * sp.gamma(-(1/4)) * sp.gamma(1/4) + 8 * self.g * np.pi * sp.gamma(1/4)**2 + sp.gamma(1/4)**4))
        # #FPR[1/(2*1),2,2,g_] = 
        # 2.5066282746310002 * np.sqrt((1. + 10.1531607808241 * self.g + 37.91166947810798 * self.g**2) / (1. + 10.1531607808241 * self.g + 43.91166947810798 * self.g**2 + 72.48541321812907 * self.g**3))
        # #FPR[1/(2*1),3,3,g_] = 
        # 2.5066282746310002 * np.sqrt((1. + 16.030388229931486 * self.g + 110.26077245790317 * self.g**2 + 324.0187225963292 * self.g**3) / (1. + 16.030388229931486 * self.g + 116.26077245790317 * self.g**2 + 420.2010519759181 * self.g**3 + 619.509278307239 * self.g**4))
        # #FPR[1/(2*1),4,4,g_] = 
        # 2.5066282746310002 * np.sqrt(( 1. + 22.874503929271544 * self.g + 238.23905635876318 * self.g**2 + 1303.4929582331083 * self.g**3 + 3224.5631655188554 * self.g**4)/( 1. + 22.874503929271544 * self.g + 244.23905635876318 * self.g**2 + 1440.7399818087376 * self.g**3 + 4575.997503671434 * self.g**4 + 6165.220279617642 * self.g**5))
        # #FPR[1/(2*3),1,1,g_] = 
        # 2.5066282746310002 (1/( 1. + 7.086913042848253 * self.g**2 + 6.989291097242859 * self.g**3))**(1/6)
        # #FPR[1/(2*3),2,2,g_] = 
        # 2.5066282746310002 (( 1. + 3.7875802399388747 * self.g)/( 1. + 3.7875802399388747 * self.g + 18.000000000000007 * self.g**2 + 33.83154290049999 * self.g**3 + 26.47250085109775 * self.g**4))**(1/6)
        # #FPR[1/(2*3),3,3,g_] = 
        # 2.5066282746310002 (( 1. + 8.099704591178746 * self.g + 28.252523713142793 * self.g**2)/( 1. + 8.099704591178746 * self.g + 46.25252371314279 * self.g**2 + 145.7946826412175 * self.g**3 + 256.83437198547387 * self.g**4 + 197.46511246291166 * self.g**5))**(1/6)
        # #FPR[1/(2*3),4,4,g_] =
        # 2.5066282746310002 (( 1. + 13.470628417091639 * self.g + 87.53888949233587 * self.g**2 + 255.1535591483978 * self.g**3)/( 1. + 13.470628417091639 * self.g + 105.53888949233587 * self.g**2 + 497.6248706560473 * self.g**3 + 1449.7000108620534 * self.g**4 + 2420.0858672492423 * self.g**5 + 1783.3424993857257 * self.g**6))**(1/6)
        # #FPR[1/(2*5),2,2,g_] = 
        # 2.5066282746310002 (1/( 1. + 30. * self.g**2 + 32.14821200212872 * self.g**3 + 43.17787580118569 * self.g**4 + 25.54986136647706 * self.g**5))**(1/10)
        # #FPR[1/(2*5),4,4,g_] = 
        # 2.5066282746310002 (( 1. + 5.072505666220409 * self.g + 14.43685326985559 * self.g**2)/( 1. + 5.072505666220409 * self.g + 44.43685326985559 * self.g**2 + 152.17516998661225 * self.g**3 + 403.10559809566917 * self.g**4 + 708.6889005862954 * self.g**5 + 752.9544739983669 * self.g**6 + 368.85959961298136 * self.g**7))**(1/10)
        # #FPR[1/(2*7),3,3,g_] = 
        # 2.5066282746310002 (1/( 1. + 42. * self.g**2+155.75843284764994 * self.g**4 + 239.21559267499774 * self.g**5 + 220.9758065100799 * self.g**6 + 93.39937438057375 * self.g**7))**(1/14)
        # #FPR[1/(2*9),4,4,g_] = 
        # 2.5066282746310002 (1/( 1. + 54. * self.g**2+594. * self.g**4+780.7879756756589 * self.g**5 + 1294.340979801729 * self.g**6 + 1475.3510504866329 * self.g**7 + 1038.5911468627608 * self.g**8 + 341.42819835916043 * self.g**9))**(1/18)

        return fpr