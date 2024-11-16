from setuptools import setup, find_packages

setup(
    name="InfantVisionSimulator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
        "matplotlib"
    ],
    include_package_data=True,
    python_requires=">=3.9",
)
