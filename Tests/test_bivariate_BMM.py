###########################################################
# Tutorial example for Bivariate BMM (simplified)
###########################################################

#import packages
import numpy as np
import math
import pytest
from contextlib import contextmanager
import statistics
from scipy import stats, special, integrate

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#matplotlib settings for Latex plots
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

#load SAMBA classes
from samba.models import Models, Uncertainties
from samba.discrepancy import Bivariate

#set up linspace and expansions
g = np.linspace(1e-6,1.0,200)
ns = 2
nl = 2

#instantiate first object for N_s = N_l = 2
error_model='informative'
model1 = Bivariate(ns, nl, error_model)

##
#test class object
@contextmanager
def does_not_raise():
    yield

@pytest.mark.parametrize(
    "ns, nl, error_model, expectation",
    [(ns, nl, error_model, does_not_raise()),],
)

def test_Bivariate(ns, nl, error_model, expectation):
    with expectation:
        assert Bivariate(ns, nl, error_model) is not None
##

#call plot_models() from Models() class
model1.plot_models(g)

##
#test ns, nl passing
def test_class_variables(expectation):
    with expectation:
        assert Models.loworder == ns
        assert Models.highorder == nl

#check figure generation
def test_plot_models(g, expectation):
    with expectation:
        assert model1.plot_models(g) is not None
##

#call plot_mix() function to mix
mean1, intervals1 = model1.plot_mix(g, plot_fdagger=True)

##
#check mean and interval
def test_plot_mix(ns, nl, error_model, expectation):
    with expectation:
        assert mean1 is not None
        assert intervals1 is not None
##