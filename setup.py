from setuptools import setup, find_packages

from modelagnosic_superior_training import __version__

readme = open("README.md").read()

install_requires = [
    "numpy>=1.23.5",
    "scikit_learn>=1.0.2",
    "scipy>=1.10.0",
    "torch>=1.13.1",
    'gpytorch @ git+https://github.com/DPanknin/gpytorch.git'
]

setup(
    name='modelagnosic_superior_training',
    version=__version__,
    
    description="An implementation of the model-agnostic superior training framework based on a mixture of Gaussian processes model in gPyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",

    url='https://github.com/DPanknin/modelagnosic_superior_training',
    author='Danny Panknin',
    author_email='danny.panknin@gmail.com',

    license="MIT",
    
    packages=find_packages(),
    
    python_requires=">=3.9",
    install_requires=install_requires,
)