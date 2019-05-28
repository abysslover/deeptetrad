"""
The build/compilations setup

>> pip install -r requirements.txt
>> python setup.py install
"""
import pip
import logging
import pkg_resources
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]

with open("README.md", "r") as fh:
    long_description = fh.read()
    
# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

print('[main] req: {}'.format(install_reqs))
setup(
    name='deeptetrad',
    version='1.0',
    url='https://github.com/abysslover/deeptetrad',
    author='Eun-Cheon Lim',
    author_email='abysslover@gmail.com',
    license='GPL3.0',
    description='DeepTetrad: deeplearning model for pollen-tetrad analysis',
    packages=["deeptetrad"],
    install_requires=install_reqs,
    include_package_data=True,
    python_requires='>=3.4',
    entry_points = {
        'console_scripts': ['deeptetrad=deeptetrad.deeptetrad:main'],
    },
    long_description=long_description,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Image Segmentation",
        "Programming Language :: Python :: 3",
    ],
    keywords="tetrad crossover interference tensorflow keras",
)