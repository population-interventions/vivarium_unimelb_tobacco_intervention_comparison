.. _getting-started:

Getting started
===============

1. You need to have Python 3.6 installed. If you don't already have this
   version of Python installed, the easiest option is to use
   `Anaconda <https://www.anaconda.com/distribution/>`__.
   Once Anaconda is installed:

   1. Create a new virtual environment:

      .. code-block:: sh

         conda create --name=mslt_tobacco python-3.6

   2. Activate this Conda environment:

      .. code-block:: sh

         conda activate mslt_tobacco

2. Download the Vivarium MSLT Tobacco Intervention Comparison project.

   - You can clone this project using ``git``; this will create a new
     directory called **vivarium_unimelb_tobacco_intervention_comparison**.

     .. code-block:: sh

        git clone https://github.com/population-interventions/vivarium_unimelb_tobacco_intervention_comparison.git

   - Alternatively, you download the project as a
     `zip archive <https://github.com/population-interventions/vivarium_unimelb_tobacco_intervention_comparison/archive/master.zip>`__
     and unzip its contents; this will create a new directory called
     **vivarium_unimelb_tobacco_intervention_comparison-master**.

3. Open a terminal and install the project using ``pip``.

   - If you used ``git`` to clone the repository:

      .. code-block:: sh

         cd vivarium_unimelb_tobacco_intervention_comparison
         pip install -e

   - If you downloaded the zip archive:

      .. code-block:: sh

         cd vivarium_unimelb_tobacco_intervention_comparison-master
         pip install -e

4. Create the data artifacts, which will be stored in the ``artifacts``
   directory:

   .. code-block:: sh

      make_artifacts minimal

5. Create the model specification files, which will be stored in the
   ``model_specifications`` directory:

   .. code-block:: sh

      make_model_specifications

Once you have completed these steps, you will be able to run all of the
simulations described in these tutorials. For each simulation there will be a
model specification file, whose file name ends in ``.yaml``. These are
plain text files, that you can edit in any text editor. To run the simulation
described in one of these files, run the following command in a command prompt
or terminal, from within the project directory:

.. code-block:: sh

   simulate run model_specifications/model_file.yaml

.. note:: Each simulation will produce one or more output CSV files. You can
   then extract relevant subsets from these data files and plot them using
   your normal plotting tools. This allows you to easily examine outcomes of
   interest for specific cohorts and/or over specific time intervals.

   The figures shown in these tutorials were created using external tools, not
   included in the Vivarium Public Health package and not documented here. Any
   plotting software could be used to produce similar figures.
