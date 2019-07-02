#!/usr/bin/env python
import os

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    about = {}
    with open(os.path.join(src_dir, "vivarium_unimelb_tobacco_intervention_comparison", "__about__.py")) as f:
        exec(f.read(), about)

    with open(os.path.join(base_dir, "README.rst")) as f:
        long_description = f.read()

    install_requirements = [
        'vivarium==0.8.21',
        'vivarium_public_health==0.9.13',
        'vivarium_cluster_tools==1.0.14',
        'vivarium_inputs[data]==3.0.0',

        # These are pinned for internal dependencies on IHME libraries
        'numpy<=1.15.4',
        'tables<=3.4.0',

        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'jupyter',
        'jupyterlab',
        'pytest',
        'pytest-mock',
    ]

    setup(
        name=about['__title__'],
        version=about['__version__'],

        description=about['__summary__'],
        long_description=long_description,
        license=about['__license__'],
        url=about["__uri__"],

        author=about["__author__"],
        author_email=about["__email__"],

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,

        install_requires=install_requirements,

        zip_safe=False,
    )
