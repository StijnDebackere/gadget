import setuptools

setuptools.setup(
    name="gadget",
    version="0.0.1",
    author="Stijn Debackere",
    author_email="debackere@strw.leidenuniv.nl",
    description="A package to read in different gadget simulations.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/StijnDebackere/gadget",
    packages=['gadget'],
    install_requires=[
        "astropy",
        "h5py",
        "numpy",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
