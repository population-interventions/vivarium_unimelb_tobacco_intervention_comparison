#!/usr/bin/env python

"""
This script generates the simulation definition files for each experiment.
"""

import jinja2

from .utilities import get_model_specification_template_file


def create_model_specifications(output_dir):
    """
    Construct the data artifacts requires for the tobacco simulations.
    """

    template_file = get_model_specification_template_file()
    with template_file.open('r') as f:
        template_contents = f.read()

    template = jinja2.Template(template_contents,
                               trim_blocks=True,
                               lstrip_blocks=True)

    out_format = 'mslt_tobacco_{}_{}-years_{}_{}.yaml'

    # The simulation populations.
    populations = ['non-maori', 'maori']

    # The simulation BAUs:
    #   1. normal (20 years, decreasing tobacco prevalence)
    #   2. immediate (0 years, decreasing tobacco prevalence)
    #   3. constant (20 years, no cessation)
    baus = [(20, 'decreasing'), (0, 'decreasing'), (20, 'constant')]

    # The tobacco interventions:
    interventions = {'erad': 'TobaccoEradication',
                     'tfg': 'TobaccoFreeGeneration',
                     'tax': None}

    for population in populations:
        for (delay, tobacco_prev) in baus:
            for interv_label, interv_class in interventions.items():
                out_file = output_dir / out_format.format(population, delay,
                                                          tobacco_prev,
                                                          interv_label)

                template_args = {
                    'output_root': str(out_file.parent.parent),
                    'basename': out_file.stem,
                    'population': population,
                    'constant_prevalence': tobacco_prev == 'constant',
                    'delay': delay,
                    'intervention_class': interv_class,
                    'tobacco_tax': interv_label == 'tax',
                }
                if interv_class is None:
                    del template_args['intervention_class']
                out_content = template.render(template_args)
                with out_file.open('w') as f:
                    f.write(out_content)
