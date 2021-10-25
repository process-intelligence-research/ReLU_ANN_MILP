from setuptools import find_packages, setup
import pathlib
# package version
__version__ = "0.2.0"

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="reluMIP",
    version="0.2.0",
    description="Embed tensorflow ReLU neural networks in MIP optimization problems.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ChemEngAI/ReLU_ANN_MILP",
    author="Laurens Lueg, Artur Schweidtmann",
    author_email="A.Schweidtmann@tudelft.nl.de",
    license="Eclipse",
    classifiers=[
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "tensorflow", "tqdm", "matplotlib", "pyomo", "gurobipy"]
)
