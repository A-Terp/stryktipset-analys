from setuptools import find_packages, setup

setup(
    name="stryktips-analys",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A project for analyzing Stryktips data and developing betting strategies",
    keywords="stryktips, betting, analysis, machine learning",
    python_requires=">=3.8",
) 