"""
The package setup script.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from setuptools import find_packages, setup

setup(
    name="forecaster",
    version="1.0.0",
    author="Alvis Tang",
    author_email="alvis@hilbert.space",
    description="A collection of libraries for forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/alvis/forecaster",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # data processing
        "pandas==1.*",
        # plotting
        "plotly==4.*",
        # machine learning
        "torch==1.*",
        "pytorch_lightning==1.*",
    ],
)
