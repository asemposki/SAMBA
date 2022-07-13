###########################################################
# Tutorial example for Bivariate BMM 
###########################################################

#import packages
import numpy as np

#matplotlib settings for Latex plots
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

#load SAMBA classes
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
# @contextmanager
# def does_not_raise():
#     yield

# @pytest.mark.parametrize(
#     "ns, nl, error_model, expectation",
#     [(ns, nl, error_model, does_not_raise()),],
# )

def test_Bivariate():
    assert Bivariate(ns, nl, error_model) is not None
##

#call plot_models() from Models() class
model1.plot_models(g)

##
#test ns, nl passing
def test_class_variables():
    if isinstance(ns, int) or isinstance(ns, float):
        assert model1.loworder[0] == ns
    else:
        assert np.array_equal(model1.loworder, ns)
    if isinstance(nl, int) or isinstance(nl, float):
        assert model1.highorder[0] == nl
    else:
        assert np.array_equal(model1.highorder, nl)

#check NoneType object
def test_plot_models():
    assert model1.plot_models(g) is None
##

#call plot_mix() function to mix
mean1, intervals1 = model1.plot_mix(g, plot_fdagger=True)

##
#check mean and interval
def test_plot_mix():
    assert mean1 is not None
    assert intervals1 is not None
##