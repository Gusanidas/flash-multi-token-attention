import os
from setuptools import setup, find_packages

setup(
    name="flash-multi-token-attention",
    version="0.1.0",
    author="Alejandro Alvarez",
    description="Flash Attention algorithm variations with efficient CUDA kernels",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "ninja",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)