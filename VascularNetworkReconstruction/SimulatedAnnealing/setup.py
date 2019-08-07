from distutils.core import setup, Extension

c_ext = Extension("_SimAnneal", ["_SimAnneal.c", "SimAnneal.c"])

setup(
    ext_modules=[c_ext],
)