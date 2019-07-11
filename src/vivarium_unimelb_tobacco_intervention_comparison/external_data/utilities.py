from pathlib import Path


def get_data_dir(population):
    here = Path(__file__).resolve()
    return here.parent / population


def get_model_specification_template_file():
    here = Path(__file__).resolve()
    return here.parent / 'yaml_template.in'
