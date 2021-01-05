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
        'vivarium==0.9.1',
        'vivarium_public_health==0.10.1',

        # These are pinned for internal dependencies on IHME libraries
        'numpy<=1.15.4',
        'tables<=3.3',
        'pandas<0.25',
        
        'scipy',
        'jinja2',
        'click',
    ]

    test_requirements = [
        'pytest',
        'pytest-mock',
    ]

    extra_requirements = [
        'sphinx',
        'sphinx-autodoc-typehints',
        'sphinx-rtd-theme',
        'seaborn',
        'matplotlib',
        'jupyter',
        'jupyterlab',

        'vivarium_cluster_tools==1.0.14',
        'vivarium_inputs[data]==3.0.0',
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
        tests_require=test_requirements,
        extras_require={
            'test': test_requirements,
            'extra':  extra_requirements + test_requirements,
        },

        entry_points="""
            [console_scripts]
            make_artifacts=vivarium_unimelb_tobacco_intervention_comparison.cli:make_artifacts
            make_model_specifications=vivarium_unimelb_tobacco_intervention_comparison.cli:make_model_specifications
            run_uncertainty_analysis=vivarium_unimelb_tobacco_intervention_comparison.cli:run_uncertainty_analysis
        """,


        zip_safe=False,
    )
