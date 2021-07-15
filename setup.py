import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neinsteinpy",
    version="0.0.1",
    author="Wenyin Wei",
    author_email="wenyin.wei@ipp.ac.cn",
    description="The neinsteinpy package is a numeric version of einsteinpy, which is well-known due to its good ability on tensor symbol operation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenyinWei/neinsteinpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)