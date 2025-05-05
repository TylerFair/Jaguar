# setup.py
from setuptools import setup, find_packages

setup(
    name="jaguar",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax[cuda12]<=0.4.31",
        "jaxopt",
        "jaxoplanet",
        "numpyro",
        "numpyro_ext",
        "arviz",            
        "exotic_ld",
        "astropy",          
        "corner",           
        "jmespath",         
        "celerite2",        
        "matplotlib",       
        "pandas",           
        "pyyaml",           
    ],
    entry_points={
        "console_scripts": [
            "run-stage4 = jaguar.Stage4.runstage4:main",
        ],
    },
)
