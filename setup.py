from setuptools import setup

setup(
    name="meso_dtfa",
    version="0.1",
    description="",
    author="Cyrus Vahidi",
    author_email="c.vahidi@qmul.ac.uk",
    include_package_data=True,
    packages=['meso_dtfa'],
    url="https://github.com/cyrusvahidi/meso_dtfa-gpu",
    install_requires=["gin-config",
                      "librosa", 
                      "torch",
                      "tqdm",
                      "fire"]
)
