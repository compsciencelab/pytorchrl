import setuptools
import subprocess

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except Exception as e:
    print("Could not get version tag. Defaulting to version 0")
    version = "0"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

if __name__ == "__main__":
    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="pytorchrl",
        version=version,
        author="albertbou92",
        author_email="",
        description="Disributed RL implementations with ray and pytorch.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/PyTorchRL/pytorchrl/",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        packages=setuptools.find_packages(include=["pytorchrl*"], exclude=[]),
        install_requires=requirements,
    )
