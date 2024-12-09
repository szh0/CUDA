import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
__version__  =  '0.0.1'   
sources = ['src/pybind.cu']
setup(
    name = 'mytensor',
    version = __version__,
    author = 'Eureka',
    author_email = '2300012969@pku.edu.cn',
    packages = find_packages(),
    zip_safe = False,
    install_requires = ['torch'],
    python_requires = '>=3.8',
    license = 'MIT',
    ext_modules=[
        CUDAExtension(
            name='mytensor',
            sources=sources)
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)