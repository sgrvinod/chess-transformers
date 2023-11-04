from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="chess-transformers",
    version="0.0.1",
    author="Sagar Vinodababu",
    author_email="sv2414@columbia.edu",
    description="Chess Transformers",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/sgrvinod/chess-transformers",
    download_url="https://github.com/sgrvinod/chess-transformers",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=["tables==3.6.1", "torch==2.1.0.dev20230809+cu118", "tqdm==4.64.1", "python-chess==1.999"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformer networks chess pytorch deep learning",
)
