import logging
from pathlib import Path

import click

from vivarium_unimelb_tobacco_intervention_comparison.external_data import assemble_tobacco_artifacts
from vivarium_unimelb_tobacco_intervention_comparison.external_data import (create_model_specifications,
                                                                            create_reduce_acmr_specification,
                                                                            create_reduce_chd_specification)
from vivarium_unimelb_tobacco_intervention_comparison.external_data import run_many


@click.command()
@click.argument('scenario', type=click.Choice(['minimal', 'uncertainty']))
def make_artifacts(scenario):
    """Generate artifacts for the MSLT tobacco intervention simulations."""
    logging.basicConfig(level=logging.INFO)

    output_path = Path('.').resolve() / 'artifacts'
    output_path.mkdir(exist_ok=True)
    draws = 0 if scenario == 'minimal' else 2000

    logging.info(f'Generating artifact for scenario {scenario} with {draws} '
                 f'draws at {str(output_path)}')

    assemble_tobacco_artifacts(draws, output_path)


@click.command()
def make_model_specifications():
    """Generate model specifications for the MSLT tobacco intervention simulations."""
    logging.basicConfig(level=logging.INFO)

    output_path = Path('.').resolve() / 'model_specifications'
    output_path.mkdir(exist_ok=True)

    logging.info(f'Generating model_specifications at {str(output_path)}')

    create_model_specifications(output_path)
    create_reduce_acmr_specification(output_path)
    create_reduce_chd_specification(output_path)


@click.command()
@click.option('-d', '--draws', default=5)
@click.option('-s', '--spawn', default=1)
@click.argument('spec_file', type=click.Path(exists=True), nargs=-1)
def run_uncertainty_analysis(draws, spawn, spec_files):
    """Run MSLT tobacco intervention simulations simulations for multiple value draws."""
    logging.basicConfig(level=logging.INFO)

    run_many(spec_files, num_draws, num_procs)
