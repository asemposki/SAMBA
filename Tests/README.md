# Testing SAMBA

Welcome to the SAMBA pytest suite! This is a developing folder where, currently, three pytest files exist that will run basic assertion tests over the three methods of BMM in this package. They are based on the three tutorial notebooks in the `Tutorials` folder, and cover most of the functions in the package. To run any test file alone, type `pytest <file name>` into a terminal in the `Tests` folder. If you wish to run them all at once, type `pytest` in the same terminal and it will automatically grab all test files and run them in sequence.  

## Common warnings

All tests are currently passing, but some warnings may show up from external packages in the pytest run. The ones that have been seen so far are below.


*test_GP_BMM.py warnings:*

- DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
    if LooseVersion(mpl.__version__) >= "3.0":

- DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
    other = LooseVersion(other)


*test_LMM.py warnings*:

- DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To 
silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
  Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    dtype = np.float

- DeprecationWarning: 'chain' is deprecated. Use 'get_chain()' instead.
    deprecation_warning(msg)

If you see any of these while running pytest, you can presently ignore them---as long as there are no failed tests or errors, SAMBA is working correctly.