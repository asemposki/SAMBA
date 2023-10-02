from distutils.core import setup, Extension

setup(
    name='samba',
    packages=['samba'],
    # cmdclass={'build_ext': CustomBuildExtCommand},
    # ext_modules=ext_modules,
    version='0.1.1',
    description='Sandbox for testing methods of Bayesian Model Mixing',
    author='Alexandra Semposki',
    author_email='as727414@ohio.edu',
    license='MIT',
    url='https://www.github.com/asemposki/SAMBA',
    download_url='',
    keywords='BAND nuclear physics model mixing gaussian process uncertainty quantification',
    classifiers=[
        'Development Status :: 1 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific :: Physics',
        'Topic :: Scientific :: Statistics'
        ],
    install_requires=[
        'numpy>=1.20.3',
        'matplotlib',
        'scipy>=1.7.0',
        'seaborn',
        'emcee',
        'corner',
        'scikit-learn',
        'cycler',
        'statistics',
    ]
)
