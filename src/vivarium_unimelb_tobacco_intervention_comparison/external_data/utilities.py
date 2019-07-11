from pathlib import Path


def get_data_dir(population):
    here = Path(__file__).resolve()
    return here.parent / population


def get_model_specification_template_file():
    here = Path(__file__).resolve()
    return here.parent / 'yaml_template.in'


def get_reduce_acmr_specification_template_file():
    here = Path(__file__).resolve()
    return here.parent / 'mslt_reduce_acmr.in'


def get_reduce_chd_specification_template_file():
    here = Path(__file__).resolve()
    return here.parent / 'mslt_reduce_chd.in'
