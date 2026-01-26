from setuptools import setup, find_packages

setup(
    name="npTensor",
    version="0.1.5",
    description="A minimal tensor and autograd library (experimental)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pawan Adhikari",
    author_email="pawan.adk7@gmail.com",
    url="https://github.com/Pawan-Adhikari/MLMaths-TorchfromScratch",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)