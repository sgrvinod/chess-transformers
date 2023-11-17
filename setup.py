from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="chess-transformers",
    version="0.1.1",
    author="Sagar Vinodababu",
    author_email="sgrvinod@gmail.com",
    description="Chess Transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/sgrvinod/chess-transformers",
    download_url="https://github.com/sgrvinod/chess-transformers",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        "setuptools>=65.5.1",
        "beautifulsoup4==4.11.1",
        "chess==1.9.4",
        "colorama==0.4.5",
        "IPython>=8.10",
        "Markdown==3.3.4",
        "regex==2022.7.9",
        "tables==3.6.1",
        "tabulate==0.8.10",
        "torch==2.1.0",
        "tqdm==4.64.1",
        "scipy>=1.10.0",
        "gdown==4.7.1"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformer networks chess pytorch deep learning",
)
