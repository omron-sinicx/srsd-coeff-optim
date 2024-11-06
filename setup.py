from setuptools import setup

import srsd_coeff_optim

DESCRIPTION = "srsd-coeff-optim is a Python package for a coefficient optimization in symbolic regression."
NAME = "srsd-coeff-optim"
AUTHOR = "Masahiro Negishi"
AUTHOR_EMAIL = ""
URL = "https://github.com/omron-sinicx/srsd-coeff-optim"
LICENSE = "MIT"
DOWNLOAD_URL = "https://github.com/omron-sinicx/srsd-coeff-optim"
VERSION = srsd_coeff_optim.__version__
PYTHON_REQUIRES = ">=3.11"
INSTALL_REQUIRES = ["sympy==1.12.0", "scipy==1.11.4"]
PACKAGES = ["srsd_coeff_optim"]

setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=PACKAGES,
)
