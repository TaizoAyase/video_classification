import os
from distutils.core import setup

from setuptools import find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()
with open("requirements-dev.txt") as f:
    extra_deps = f.read().splitlines()
    extra_deps = [e for e in extra_deps if not e.startswith("./")]

here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, "video_classification", "_version.py")).read())

extras_require = {"dev": extra_deps}

setup(
    name="video-classification",
    version=__version__,  # NOQA
    description="test package",
    author="TaizoAyase",
    author_email="zxcvbnmkind@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    extras_require=extras_require,
    python_requires=">=3.9,<4.0",
)
