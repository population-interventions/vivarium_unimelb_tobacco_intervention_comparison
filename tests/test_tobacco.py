#Integration tests for tobacco simulations

import pytest
from os import path
from time import time

from vivarium.framework.engine import run_simulation

def get_model_specification(model_name):
    model_specification_file = (path.dirname(path.abspath(__file__)) + 
    '/../model_specifications/' + model_name +'.yaml')
    return model_specification_file


def run_timed_simulation(model_specification_file):
    start_time = time()
    simulation = run_simulation(model_specification_file, None, None, None)
    total_time = time() - start_time
    return total_time


def compare_results(results_filename):
    expected_results_file = (path.dirname(path.abspath(__file__)) + 
    '/expected_results/' + results_filename)
    obtained_results_file = (path.dirname(path.abspath(__file__)) + 
    '/../results/' + results_filename)

    with open(expected_results_file, 'r') as expected_results: 
        with open(obtained_results_file, 'r') as obtained_results:
            expected_lines = expected_results.readlines()
            obtained_lines = obtained_results.readlines()
    
    assert len(expected_lines) == len(obtained_lines)

    for i in range(len(expected_lines)):
        assert expected_lines[i] == obtained_lines[i]


def test_reduce_acmr_simulation():
    model_specification_file = get_model_specification('mslt_reduce_acmr')
    sim_time = run_timed_simulation(model_specification_file)
    assert sim_time/60 <= 0.3
    compare_results('mslt_reduce_acmr_mm.csv')


def test_reduce_chd_simulation():
    model_specification_file = get_model_specification('mslt_reduce_chd')
    sim_time = run_timed_simulation(model_specification_file)
    assert sim_time/60 <= 0.6
    compare_results('mslt_reduce_chd_mm.csv')
    compare_results('mslt_reduce_chd_chd.csv')


@pytest.mark.parametrize('model, expected_time',
                         [('mslt_test',0.3)
                          ('mslt_tobacco_maori_0-years_decreasing_erad',6),
                          ('mslt_tobacco_maori_0-years_decreasing_tax',6),
                          ('mslt_tobacco_maori_0-years_decreasing_tfg',6),
                          ('mslt_tobacco_maori_20-years_decreasing_erad',6),
                          ('mslt_tobacco_maori_20-years_decreasing_tax',6),
                          ('mslt_tobacco_maori_20-years_decreasing_tfg',6),
                          ('mslt_tobacco_maori_20-years_constant_erad',6),
                          ('mslt_tobacco_maori_20-years_constant_tax',6),
                          ('mslt_tobacco_maori_20-years_constant_tfg',6)
                         ])
def test_tobacco_simulation(model, expected_time):
    model_specification_file = get_model_specification(model)
    sim_time = run_timed_simulation(model_specification_file)
    assert sim_time/60 <= expected_time
    compare_results(model+'_mm.csv')
