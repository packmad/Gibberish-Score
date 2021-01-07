import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gibberish_score",
    version="0.1.7",
    author="Simone Aonzo",
    author_email="simone.aonzo@gmail.com",
    description="Gibberish score and non-gibberish generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/packmad/Gibberish-Score",
    packages=setuptools.find_packages(),
    install_requires=['networkx'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
