from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GunDetect",
    version="0.1",
    author="vaibhav",
    packages=find_packages(),
    install_requires = requirements,
)