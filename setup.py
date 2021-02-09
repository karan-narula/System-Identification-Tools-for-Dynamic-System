#!/usr/bin/env python

from distutils.core import setup

setup(name='SysIdTools',
      version='1.0',
      description='Some potentially useful tools for performing system identification of dynamic systems',
      author='Karan Narula',
      url='https://github.com/karan-narula/System-Identification-Tools-for-Dynamic-System',
      packages=['SysIdTools'],
      package_dir={'SysIdTools': 'src'},
      package_data={'SysIdTools': ['matlab_sim_data/*']},
      license='MIT',
      scripts=['scripts/test_estimators', 'scripts/matlab_sim_estimator', 'scripts/lugre_force_versus_slip_plot']
      )
