Chronic heart disease
=====================

.. _mslt_reduce_chd:

Intervention: a reduction in CHD incidence
------------------------------------------

.. note:: In this example, we will also use components from the
   :mod:`vivarium_public_health.mslt.disease` module.

Compared to the :ref:`previous simulation <mslt_reduce_acmr>`, we will now add
a chronic disease component, and replace the all-cause mortality rate
intervention with an intervention that affects CHD incidence.

To add CHD as a separate cause of morbidity and mortality, we use the
:class:`~vivarium_public_health.mslt.disease.Disease` component:

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :lines: 7-9,14-15
   :caption: Add a chronic disease.

.. py:currentmodule:: vivarium_public_health.mslt.intervention

We then replace the :class:`ModifyAllCauseMortality` intervention with the
:class:`ModifyDiseaseIncidence` intervention.
We give this intervention a name (``reduce_chd``) and identify the disease
that it affects (``CHD``).
In the configuration settings, we identify this intervention by name
(``reduce_chd``) and specify the scaling factor for CHD incidence
(``CHD_incidence_scale``).

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :lines: 7-9,16-17,22,39-42
   :caption: Add an intervention that reduces CHD incidence.

.. py:currentmodule:: vivarium_public_health.mslt.observer

Finally, we add an observer to record CHD incidence, prevalence, and deaths,
in both the BAU scenario and the intervention scenario.
We use the :class:`Disease` observer, identify the disease of interest by name
(``CHD``), and specify the prefix for output files (``mslt_reduce_chd``).

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :lines: 7-9,18,20,21-22,43-44
   :caption: Record CHD incidence, prevalence, and deaths.

Putting all of these pieces together, we obtain the following simulation
definition:

.. literalinclude:: /_static/mslt_reduce_chd.yaml
   :language: yaml
   :caption: The simulation definition for the BAU scenario and the
      intervention.

Running the model simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above simulation is already defined in ``mslt_reduce_chd.yaml``. Run this
simulation with the following command:

.. code-block:: console

   simulate run model_specifications/mslt_reduce_chd.yaml

When this has completed, the output recorded by the
:class:`MorbidityMortality` observer will be saved in the file
``mslt_reduce_chd_mm.csv``.

We can now plot the survival of this cohort in both the BAU and intervention
scenarios, relative to the starting population, and see how the survival rate
has increased as a result of this intervention.

.. _mslt_reduce_chd_fig:

.. figure:: /_static/mslt_reduce_chd_survival.png
   :alt: The survival rates in the BAU and intervention scenarios, and the
      difference between these two rates.

   The impact of reducing the CHD incidence rate by 5% on survival rate.
   Results are shown for the cohort of males aged 50-54 in 2010.
   Compare this to the impact of
   :ref:`reducing all-cause mortality rate by 5% <mslt_reduce_acmr_fig>`.

The output recorded by the :class:`Disease` observer will be saved in the file
``reduce_chd_disease.csv``.
The contents of this file will contain the following results:

.. csv-table:: An extract of the CHD statistics, showing a subset of rows for
   the cohort of males aged 50-54 in 2010.
   :file: ../_static/table_mslt_reduce_chd_chd.csv
   :header-rows: 1
