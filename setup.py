#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from setuptools import setup

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("offlinerl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

setup(
    name='offlinerl',
    description="A Library for Offline RL (Batch RL)",
    version=get_version(),
    python_requires=">=3.7",
    install_requires=[
        "fire==0.4.0",
        "loguru==0.5.3",
        "gym==0.18.3",
        "sklearn==0.23.2",
        "gtimer",
        "numpy==1.23.1",
        "tianshou==0.4.2",
        "torch==1.8.0"
    ],
    
)
