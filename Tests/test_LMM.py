###########################################################
# Tutorial example for LMM BMM
###########################################################

import numpy as np
import pytest 

#matplotlib settings for Latex plots
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

import sys
sys.path.append('../')

#import classes needed
from samba.mixing import LMM

#set up the linspace for input variable g and N_s and N_l
g = np.linspace(1e-6,1.0,100)
ns = np.array([2])
nl = np.array([2])

#call the plot_models() function from Models()
m = LMM(ns, nl, error_model='informative')

##
#test LMM class object generation, ns, nl passing
def test_LMM():
    assert LMM(ns, nl, error_model='informative') is not None
    assert m.loworder is ns 
    assert m.highorder is nl 
##

#make linspace for data
g_data = np.linspace(0.1, 0.6, 10)

#call the add_data() function to create the data
data, sigma = m.add_data(g, g_data, error=0.01, plot=False)

##
#check data, sigma generation
def test_add_data():
    assert data is not None
    assert sigma is not None 
##

#call mixed_model()
chain, trace = m.mixed_model(g_data, data, sigma, mixing_function='cosine', nsteps=3000)

##
#test chain, trace
def test_mixed_model():
    assert chain is not None 
    assert trace is not None 
##

thin_array, mean, median = m.stats_chain(chain, parameters=3, plot=False)

##
#simple check of results
def test_stats_chain():
    assert thin_array is not None 
    assert mean is not None
    assert median is not None 
##

map_values = m.MAP_values(thin_array, g, g_data, data, sigma, plot=False)

##
#test the function
def test_MAP_values():
    assert m.MAP_values(thin_array, g, g_data, data, sigma, plot=False) is not None
##

#PPD linspace
g_ppd = np.linspace(1e-6, 1.0, 200)

#PPD calculation using ppd() and MAP parameter values
ppd_result = m.ppd(thin_array, map_values, g_data, g_ppd, data, 0.68, plot=False)

##
#check PPD
def test_ppd():
    assert ppd_result is not None 
##