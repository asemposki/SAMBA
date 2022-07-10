###########################################################
# Tutorial example for trivariate BMM with a GP
###########################################################

#import packages
import numpy as np
import pytest

#matplotlib settings for Latex plots
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

#import the SAMBA classes needed
from samba.gaussprocess import GP

#define g and series expansions
g = np.linspace(1e-6,1.0,200)
ns = np.array([3])
nl = np.array([3])

#instantiate GP object for N_s = N_l = 3
obj1 = GP(g, ns, nl, kernel="Matern", nu=1.5, ci=68, error_model='informative')

##
#check ns, nl, g passing
# @contextmanager
# def does_not_raise():
#     yield

# @pytest.mark.parametrize(
#     "g, ns, nl, expectation",
#     [(g, ns, nl, does_not_raise()),],
# )

def test_GP():
    assert np.array_equal(obj1.gpredict, g) 
    assert np.array_equal(obj1.loworder, ns)
    assert np.array_equal(obj1.highorder, nl)
##

#call the training() function from the GP() class
obj_tr1 = obj1.training(error=True, method=2, plot=False)

##
#check return object
def test_training():
    assert obj1.training() is not None
##

#call the validate() function
mean1, sig1, cov1 = obj1.validate(plot=False)

##
#check mean1, sig1, cov1
def test_validate():
    assert mean1 is not None 
    assert sig1 is not None 
    assert cov1 is not None 
##

#call plot_mix() to mix in the GP
mixed_mean, mixed_intervals = obj1.plot_mix(g, plot_fdagger=True, \
    plot_true=True, GP_mean=mean1, GP_var=np.square(sig1))

##
#check returned variables
def test_plot_mix():
    assert mixed_mean is not None
    assert mixed_intervals is not None
##

#call subplot_mix() and plot no GP results next to GP results
obj1.subplot_mix(g, GP_mean=mean1, GP_var=np.square(sig1))

##
#check NoneType object
def test_subplot_mix():
    assert obj1.subplot_mix(g, GP_mean=mean1, GP_var=np.square(sig1)) \
        is None
##

#calculate the Mahalanobis points
md_g, md_mean, md_sig, md_cov = obj1.MD_set(pts=3, plot=False)

##
#check each output
def test_MD_set():
    assert md_g is not None 
    assert md_mean is not None 
    assert md_sig is not None 
    assert md_cov is not None 
##

#use the points to calculate the Mahalanobis distance for our GP
md_gp, md_ref = obj1.md_squared(md_g, md_mean, md_cov, n_curves=1000)

##
#check return output
def test_md_squared():
    assert md_gp is not None 
    assert md_ref is not None 
##

#call 
obj1.md_plotter(md_gp, md_ref, hist=True)

##
#check NoneType object
def test_md_plotter():
    assert obj1.md_plotter(md_gp, md_ref) is None
##