from setuptools import setup, find_packages

setup(
    name="pso",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    zip_safe=False,
    install_requires=['numpy'])
