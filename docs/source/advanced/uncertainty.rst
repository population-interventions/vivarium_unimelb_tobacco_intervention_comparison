.. _uncertainty_analyses:

Uncertainty analyses
====================

Uncertainty analyses are a **bespoke process**, because you need to decide
which input data should be correlated (e.g., the incidence rate for a single
disease, across all age groups, sex, and ethnicity).
To build data artifacts that contain 2000 draws for each input rate/value, run
the following command:

.. code:: console

   make_artifacts uncertainty

.. note:: This can take a long time to complete, and generates data artifacts
   that are around **3 GB** in size.

We have also provided a command that runs multiple simulations for a single
model specification file, where each simulation uses a different draw from the
data artifact.
This script can be used as follows:

.. code:: console

   run_uncertainty_analysis --draws 2000 --spawn 16 modelA.yaml modelB.yaml [...]

This will run 2000 simulations for each of the model specifications
(modelA.yaml, modelB.yaml, etc) and will simultaneously run 16 simulations at
a time.
Each simulation will produce distinct output files (``modelA_mm_1.csv``,
``modelA_mm_2.csv``, etc).
