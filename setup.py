#!/usr/bin/env python3
"""
Setup script for Clippy AdaGrad implementation.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clippy-adagrad",
    version="1.0.0",
    author="Wei-Han Tu",
    author_email="b03608027@gmail.com",
    description="Clippy AdaGrad optimizer for multitask learning in recommender systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/weihantu/clippy-adagrad",
    project_urls={
        "Bug Reports": "https://github.com/weihantu/clippy-adagrad/issues",
        "Source": "https://github.com/weihantu/clippy-adagrad",
        "Documentation": "https://github.com/weihantu/clippy-adagrad/tree/main/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "deep-learning",
        "pytorch",
        "optimizer",
        "gradient-clipping",
        "multitask-learning",
        "recommender-systems",
        "clippy",
        "adagrad",
    ],
    entry_points={
        "console_scripts": [
            "clippy-train=test:main",
        ],
    },
) 