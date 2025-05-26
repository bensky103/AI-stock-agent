# File: setup.py
from setuptools import setup, find_packages

setup(
    name="ai_stock_agent",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'yfinance',
        'pyyaml',
        'pytz',
        'pytest'
    ],
)

