===============================
vivarium_unimelb_tobacco_intervention_comparison
===============================

Research repository for the vivarium_unimelb_tobacco_intervention_comparison project.

.. contents::
   :depth: 1

Installation
------------

To set up a new research environment, open up a terminal on the cluster and
run::

    $> conda create --name=vivarium_unimelb_tobacco_intervention_comparison python=3.6 redis
    ...standard conda install stuff...
    $> conda activate vivarium_unimelb_tobacco_intervention_comparison
    (vivarium_unimelb_tobacco_intervention_comparison) $> git clone git@github.com:ihmeuw/vivarium_unimelb_tobacco_intervention_comparison.git
    ...you may need to do username/password stuff here...
    (vivarium_unimelb_tobacco_intervention_comparison) $> cd vivarium_unimelb_tobacco_intervention_comparison
    (vivarium_unimelb_tobacco_intervention_comparison) $> pip install -e .


Usage
-----

You'll find four directories inside the main
``src/vivarium_unimelb_tobacco_intervention_comparison`` package directory:

- ``components``

  This directory is for Python modules containing custom components for
  the vivarium_unimelb_tobacco_intervention_comparison project. You should work with the
  engineering staff to help scope out what you need and get them built.

- ``external_data``

  If you have **small scale** external data for use in your sim or in your
  results processing, it can live here. This is almost certainly not the right
  place for data, so make sure there's not a better place to put it first.

- ``model_specifications``

  This directory should hold all model specifications and branch files
  associated with the project.

- ``verification_and_validation``

  Any post-processing and analysis code or notebooks you write should be
  stored in this directory.

