# -*- coding: utf-8 -*-
"""Example setup.py file

This example setup file is adapted from https://github.com/pypa/sampleproject
and is not supposed to be and exhaustive list of options accepted by the
setuptools' function `setup`. You should read the respective documentation,
which can be found at https://setuptools.readthedocs.io/en/latest/setuptools.html

The original License and Copyright for this setup file follows:

Copyright (c) 2016 The Python Packaging Authority (PyPA)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='pynumpm',
    version='0.1',
    description='''Numerical phasematching routines''',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='None',
    author='Matteo Santandrea',
    author_email='matteo.santandrea@upb.de',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'numpy', 'matplotlib', 'scipy', 'warnings'
    ],
    packages=find_packages(),
)
