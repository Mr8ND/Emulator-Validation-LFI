import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Regr2STest-Mr8ND",
    version="0.0.1",
    author="Niccolo Dalmasso",
    author_email="ndalmass@andrew.cmu.edu",
    description="Computes the regression two sample test using probabilistic classifiers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mr8ND/Emulator-Validation-LFI/tree/master/python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
