"""
Setup script for CLV-ML package.

This allows SageMaker to install the package and its dependencies.
"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='clv-ml',
    version='1.0.0',
    description='Customer Lifetime Value Prediction Models',
    author='Cendyn AI Team',
    author_email='ggiosa@cendyn.com',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'train-clv=src.training.train_all:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
