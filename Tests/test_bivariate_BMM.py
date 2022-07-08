###########################################################
# Tutorial example for Bivariate BMM
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


#call plot_models() from Models() class
model1.plot_models(g)

#TODO: check ns, nl passing

#check figure generation
def test_plot_models(g, expectation):
    with expectation:
        assert model1.plot_models(g) is not None


#call plot_mix() function to mix
mean1, intervals1 = model1.plot_mix(g, plot_fdagger=True)

#TODO: check mean and interval and see what is going on inside here (check ns, nl passing)


#instantiate a new object for N_s = N_l = 5
ns = 5
nl = 5
error_model = 'informative'
model2 = Bivariate(ns, nl, error_model)


#check class object again
@pytest.mark.parametrize(
    "ns, nl, error_model, expectation",
    [(ns, nl, error_model, does_not_raise()),],
)

def test_Bivariate(ns, nl, error_model, expectation):
    with expectation:
        assert Bivariate(ns, nl, error_model) is not None


#call plot_mix()
mean2, intervals2 = model2.plot_mix(g, plot_fdagger=True)

#TODO: again check mean2, interval2
#TODO: again check ns, nl passing 