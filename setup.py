from setuptools import setup, find_packages

setup(
    name="clique-cover-lp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.8",
        "numpy>=1.22",
        "matplotlib>=3.5",
        "pandas>=1.4",
        "pulp>=2.7",
    ],
)
