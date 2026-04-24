from setuptools import setup, find_packages
import os

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="NeuralNetwork",
    version="0.4.0",
    packages=find_packages(),
    install_requires=["numpy>=1.21.0"],
    extras_require={"dev": ["matplotlib>=3.5.0", "python-dotenv"]},
    author="elJulioDev",
    description="Deep Learning vectorizado con API Keras, caches externos y persistencia portable",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
)