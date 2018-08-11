# Agamemnon Krasoulis 2018
# sklearn-ext
# Author: Agammenon Krasoulis
#
# License: MIT

from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import sklearn_ext

VERSION = sklearn_ext.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')


setup(name = "sklearn-ext",
      version=VERSION,
      description = ("Machine learning extensions for scikit-learn"),
      author = "Agamemnon Krasoulis",
      author_email='agamemnon.krasoulis@gmail.com',
      url='https://github.com/rasbt/sklearn-ext',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt']
                    },
      include_package_data=True,
      install_requires=install_reqs,
      extras_require={'testing': ['nose'],
                      'docs': ['mkdocs']},
      license='MIT',
      platforms='any',
      long_description="""
A library of extensions for sklearn.
Currently only intended for personal use.
""")
