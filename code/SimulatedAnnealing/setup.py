from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("SimAnneal", ["SimAnneal.c", "_SimAnneal.c"])

setup(
    name= 'SimAnneal',
    version = '1.0',
    description = 'C extension for Simulated Annealing',
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs()
)