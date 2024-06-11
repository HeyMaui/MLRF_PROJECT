from setuptools import setup, find_packages

setup(
    name="cifar-10-classifier",
    version="0.1",
    description="CIFAR-10 classifier",
    packages=find_packages(),
    classifiers=["Programming Language :: Python :: 3", "License :: OSI Approved :: MIT License"],
    author="Raph et Maui",
    author_email="your.email@example.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HeyMaui/HeyMaui",  # Replace with your GitHub repo URL
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            # 'your-command=your_module:main_function',
        ],
    },
)
