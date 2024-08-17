from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="chess-transformers",
    version="0.3.0",
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
        "beautifulsoup4==4.12.3",
        "chess==1.10.0",
        "colorama==0.4.5",
        "ipython==8.17.2",
        "Markdown==3.3.4",
        "py_cpuinfo==9.0.0",
        "regex==2024.7.24",
        "scipy==1.13.1",
        "setuptools==69.0.3",
        "tables==3.9.2",
        "tabulate==0.9.0",
        "torch==2.4.0",
        "tqdm==4.64.1",
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
