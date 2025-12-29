from setuptools import setup, find_packages
import os

# Forma segura de leer el readme
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="NeuralNetwork",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "python-dotenv"
    ],
    author="elJulioDev",
    description="Una librería de Deep Learning ligera construida desde cero",
    long_description=long_description, # Usamos la variable segura
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)