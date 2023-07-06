from setuptools import setup, find_packages

setup(
    name="meso_dtfa",
    version="0.1",
    description="",
    author="Cyrus Vahidi",
    author_email="c.vahidi@qmul.ac.uk",
    include_package_data=True,
    packages=find_packages(exclude=['scripts', 'tests.*']),
    url="https://github.com/cyrusvahidi/meso-dtfa",
    install_requires=["librosa", 
                      "torch",
                      "tqdm",
                      "fire"]
)
