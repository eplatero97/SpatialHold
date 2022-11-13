import os 
from os.path import basename, splitext
from glob import glob
from setuptools import setup
from setuptools import find_packages


setup(
    name="spatial_bias_assignment",
    version = "0.0", 
    description = "spatial bias assignment implementation in MOT setting",
    keywords = "spatial-bias",
    packages = find_packages("src"),
    package_dir = {'': "src"},
    py_modules = [splitext(basename(path))[0] for path in glob('src/*.py')]

)



