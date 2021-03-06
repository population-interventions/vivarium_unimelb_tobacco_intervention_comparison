Tobacco smoking: effect of interventions
========================================

Each chronic and acute disease that is affected by tobacco smoking is modelled
as a separate component, so that interventions on tobacco smoking can affect
the morbidity and mortality of these diseases.
We also need to inform the tobacco component which diseases it should affect;
this is done in the configuration section.
The resulting simulation definition is quite long, simply because there are
many diseases to include.

.. literalinclude:: /_static/mslt_tobacco_bau.yaml
   :language: yaml
   :caption: The simulation definition for the BAU scenario.
   :name: tobacco_bau_yaml

Tobacco eradication
-------------------

.. py:currentmodule:: vivarium_public_health.mslt.intervention

We add the :class:`TobaccoEradication` component, and specify at what year it
comes into effect.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   components:
       vivarium_public_health:
           mslt:
               # Other components ...
               intervention:
                   TobaccoEradication()

   configuration:
       # Other configuration settings ...
       tobacco_eradication:
           year: 2011

These simulations are already defined in the following files:

+ ``mslt_tobacco_maori_20-years_decreasing_erad.yaml``
+ ``mslt_tobacco_non-maori_20-years_decreasing_erad.yaml``

Tobacco-free generation
-----------------------

We add the :class:`TobaccoFreeGeneration` component, and specify at what year
it comes into effect.
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   components:
       vivarium_public_health:
           mslt:
               # Other components ...
               intervention:
                   TobaccoFreeGeneration()

   configuration:
       # Other configuration settings ...
       tobacco_free_generation:
           year: 2011

These simulations are already defined in the following files:

+ ``mslt_tobacco_maori_20-years_decreasing_tfg.yaml``
+ ``mslt_tobacco_non-maori_20-years_decreasing_tfg.yaml``

Tobacco tax
-----------

.. py:currentmodule:: vivarium_public_health.mslt.delay

We enable the ``tobacco_tax`` option of the tobacco risk factor
(:class:`DelayedRisk`).
Shown below are the new lines that are added to the simulation definition
:ref:`for the BAU scenario <tobacco_bau_yaml>`.

.. code:: yaml

   configuration:
       # Other configuration settings ...
       tobacco:
           tobacco_tax: True

These simulations are already defined in the following files:

+ ``mslt_tobacco_maori_20-years_decreasing_tax.yaml``
+ ``mslt_tobacco_non-maori_20-years_decreasing_tax.yaml``

Intervention comparison
-----------------------

If you run all of these simulations, you can then compare them by the gains
that they provide in LYs and HALYs, and the reductions that they provide in
ACMR and YLDR, using the data analysis software of your choice.

As an example, here are some of the results obtained for non-Maori males aged
50-54 in 2011, for the tobacco eradication intervention:

.. _tobacco_eradication_bau_table1:

.. csv-table:: Results for the tobacco eradication intervention, which yields
   gains in LYs, HALYs, ACMR, and YLDR.
   :file: ../_static/table_mslt_tobacco_non-maori_20-years_decreasing_erad_mm.csv
   :header-rows: 1
