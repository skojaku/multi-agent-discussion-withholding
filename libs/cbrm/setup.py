#!/usr/bin/env python
# encoding: utf-8

from setuptools import find_packages, setup

setup(
    name="cbrm",
    version="0.1",
    description=(
        "Measure concealment, internalisation and repair in multi-agent "
        "transcripts, and place the team on the CBRM phase diagram"
    ),
    author="Sadamori Kojaku",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
    ],
)
