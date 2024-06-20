from setuptools import setup, find_packages

setup(
    name="cifar10_classifier",
    version="1.0.0",
    description="CIFAR10 classifier",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    author="Raph et Maui",
    author_email="maui.tadeje@epita.fr",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HeyMaui",
    python_requires=">=3.6",
    install_requires=[
        "joblib",
        "numpy",
        "opencv-python",
        "scikit-learn",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [],
    },
)
