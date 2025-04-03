import setuptools

extensions = ['*.txt','*.dat','*.md','*.py']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EmulateIt",
    version="1.0",
    description="emulate the thing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NoahSailer/EmulateIt",
    packages=['EmulateIt'],
    package_data={'EmulateIt': extensions},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy'],
    entry_points={
        "console_scripts": ["train-nn=EmulateIt.evaluate_nn:train_neural_network"]
    },
)