# coding: utf-8
from __future__ import print_function

from os.path import dirname, join
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import subprocess

def _get_version():
    """Return the project version from VERSION file."""

    with open(join(dirname(__file__), 'imret/VERSION'), 'rb') as f:
        version = f.read().decode('ascii').strip()
    return version

setup(
    name='imret',
    version=_get_version(),
    url='',
    description='Image retrieval by region connection calculus',
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    author='Danilo Nunes',
    maintainer='Danilo Nunes',
    maintainer_email='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)