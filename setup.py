from setuptools import setup
import os

PACKAGE_PATH = os.path.dirname(__file__)

setup(
    name='password_detector',
    version='0.1.0',
    packages=['passgan', 'passgan.models', 'passgan.datasets'],
    url='',
    license='',
    author='James McHugh',
    author_email='mchugh.jamec1@gmail.com',
    description='A GAN for generating passwords and detecting strings that '
                'look like passwords.',
    install_requires=[
        'torch', 'numpy'
    ]
)
