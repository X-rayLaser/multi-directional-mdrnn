import setuptools
import os


current_path = os.path.abspath(os.path.dirname(__file__))
version_path = os.path.join(current_path, 'mdrnn', '__version__.py')

about = {}

with open(version_path, 'r') as f:
    exec(f.read(), about)

with open("README.md", "r") as f:
    long_description = f.read()

packages = ['mdrnn', 'mdrnn._layers', 'mdrnn._util']

setuptools.setup(
    name=about['__name__'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about['__url__'],
    packages=packages,
    install_requires=[
        'numpy',
        'tensorflow>=2'
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
