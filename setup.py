from setuptools import setup, find_packages
from os import path
from io import open

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qedward",
    version="0.0.3",
    author="Robert Tucci",
	keywords = ('Quantum Neural Networks'),
    author_email="Robert.Tucci@artiste-qb.net",
    description="Python tools for supervised learning by Quantum Neural Networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artiste-qb-net/Quantum_Edward",
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
	install_requires=[
        'numpy',
		'matplotlib',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)