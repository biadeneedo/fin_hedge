from setuptools import setup, find_packages

setup(
    name="fin_hedge",
    version="0.0.2",
    author="edob",
    author_email="biadene.edoardo@gmail.com",
    description="A package for quant finance",
    long_description="x",
    long_description_content_type="text/markdown",
    url="https://github.com/biadeneedo/fin_hedge",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
