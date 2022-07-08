##simple example to run pytest with Bivariate BMM method

#import packages
import numpy as np
import math
import pytest
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
model1 = Bivariate(ns, nl, error_model='informative')

#assert that model1 is not None (unless these are NoneType objects?)

#call plot_models() from Models() class
model1.plot_models(g)

#assert that this is not None (creates axes), check ns, nl passing

#call plot_mix() function to mix
mean1, intervals1 = model1.plot_mix(g, plot_fdagger=True)

#check mean and interval and see what is going on inside here (check ns, nl passing)

## start a new test

#instantiate a new object for N_s = N_l = 5
ns = 5
nl = 5
model2 = Bivariate(ns, nl, error_model='informative')

#check model2 type

#call plot_mix()
mean2, intervals2 = model2.plot_mix(g, plot_fdagger=True)

#again check mean2, interval2, and ns, nl passing 