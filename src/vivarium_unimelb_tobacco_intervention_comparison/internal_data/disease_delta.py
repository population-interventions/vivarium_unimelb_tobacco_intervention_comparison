"""
This script generates disease delta values from mortality and prevalence tables.
"""
import os
import numpy as np
import pandas as pd

import pdb, traceback, sys

from vivarium.framework.artifact.artifact import Artifact


class chronicDiseaseDeltaCalculator:

    def __init__(self, model, artifact_path, disease_list):
        self.model = model
        self.disease_list = disease_list
        
        self.key_columns = ['age_start', 'age_end', 'sex',
                            'year_start', 'year_end']

        self.disease_results_prefix = 'disease_results/'
        self.bau_scenario = 'BAU'

        self.disease_disability_data = {}
        self.bau_mortality_tables = {}
        self.bau_prevalence_tables = {}
        self.mortality_deltas = pd.DataFrame()
        self.yld_deltas = pd.DataFrame()

        artifact = Artifact(artifact_path)

        for disease in disease_list:
            self.disease_disability_data[disease] = self.load_disability_data(artifact, disease)
            self.bau_mortality_tables[disease] = (self.load_simulated_disease_data(self.model, 
                                                                                   self.bau_scenario, 
                                                                                   disease,
                                                                                   'mortality')
                                                 .set_index(['age','sex','year'])
            )
            self.bau_prevalence_tables[disease] = (self.load_simulated_disease_data(self.model, 
                                                                                    self.bau_scenario, 
                                                                                    disease,
                                                                                    'prevalence')
                                                 .set_index(['age','sex','year'])
            )          


    def compute_mortality_deltas(self, start_year, end_year, scenario):
        total_delta = 0

        for disease in self.disease_list:
            mortality_risk_int = (self.load_simulated_disease_data(self.model, 
                                                                          scenario, 
                                                                          disease,
                                                                          'mortality')
                                    .set_index(['age','sex','year'])
            )
            mortality_risk = self.bau_mortality_tables[disease]
            delta = np.log((1 - mortality_risk) / (1 - mortality_risk_int))
            total_delta += delta
        
        self.mortality_deltas = total_delta


    def compute_yld_deltas(self, start_year, end_year, scenario):
        total_delta = 0

        for disease in self.disease_list:
            disability_table = self.disease_disability_data[disease]
            disease_delta = pd.DataFrame()
            prevalence_rate_int = (self.load_simulated_disease_data(self.model, 
                                                                    scenario, 
                                                                    disease,
                                                                    'prevalence')
                                  .set_index(['age','sex','year'])
            )
            for year in range(start_year, end_year+1):
                prevalence_rate_subtable = self.bau_prevalence_tables[disease].xs(year, 
                                                                                level = 'year', 
                                                                                drop_level=False)

                prevalence_rate_int_subtable = prevalence_rate_int.xs(year, 
                                                                      level = 'year', 
                                                                      drop_level=False)
                disability_subtable = disability_table.loc[prevalence_rate_subtable.index]

                delta = disability_subtable*(prevalence_rate_int_subtable 
                                            - prevalence_rate_subtable)

                disease_delta = disease_delta.append(delta)
            
            total_delta += disease_delta

        self.yld_deltas = total_delta

                                
    def load_simulated_disease_data(self, model, scenario, disease, rate_name):
        '''rate_name is either 'mortality' or 'prevalence'
        '''
        disease_rate_filename = '{}_{}_{}_{}.csv'.format(model, 
                                                         scenario, 
                                                         disease, 
                                                         rate_name)
        return pd.read_csv(self.disease_results_prefix + disease_rate_filename)
    

    def load_disability_data(self, artifact, disease):
        key = f'chronic_disease.{disease}.morbidity'
        disability_data = pivot_load(artifact, key)

        # Check that the morbidity table includes required columns.
        if ( 
            set(self.key_columns) 
            & set(disability_data.columns) 
            != set(self.key_columns)
        ):
            msg = 'Missing index columns for morbidity rates'
            raise ValueError(msg) 

        disability_data.rename(columns={'age_start': 'age'}, inplace=True)
        disability_data.drop(['year_start','year_end','age_end', 'draw'], axis=1, inplace=True)
        disability_data.set_index(['age','sex'], inplace=True)

        return disability_data


    def write_delta_table(self, output_prefix, scenario):
        directory = './{}{}'.format(output_prefix, self.model)
        make_directory(directory)
        pif_table_filename = os.path.join(directory, '{}_deltas_{}.csv'.format(self.model, scenario))

        delta_index = pd.MultiIndex.from_tuples(self.mortality_deltas.index).set_names(['age','sex','year'])
        output_table = self.mortality_deltas.rename(columns={'value':'mortality_delta'})
        output_table['yld_delta'] = self.yld_deltas['value']
        output_table.set_index(delta_index, drop=True, inplace=True)

        output_table.reset_index().to_csv(pif_table_filename, index=False)


    def write_delta_series(self, output_prefix, scenario, rate_name):
        parent_directory = './{}{}'.format(output_prefix, self.model)
        make_directory(parent_directory)
        sub_directory = os.path.join(parent_directory, rate_name)
        make_directory(sub_directory)

        if rate_name == 'mortality':
            delta_table = self.mortality_deltas
        else:
            delta_table = self.yld_deltas

        delta_table_filename = os.path.join(sub_directory, '{}_{}_deltas_{}.csv'.format(self.model, 
                                                                                rate_name, 
                                                                                scenario))

        delta_index = pd.MultiIndex.from_tuples(delta_table.index).set_names(['age','sex','year'])
        table = delta_table.set_index(delta_index, drop=True)       
        
        output_series = table['value']
        output_series.reset_index().to_csv(delta_table_filename, index=False)


def make_directory(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def pivot_load(artifact, entity_key):
    """Helper method for loading dataframe from artifact.

    Performs a long to wide conversion if dataframe has an index column
    named 'measure'.

    """
    data = artifact.load(entity_key).reset_index()
    index = [i for i in data.columns if i not in ['measure','value']]

    if 'measure' in data.columns :
        data  = (
            data.pivot_table(index=index, columns='measure', values='value')
            .rename_axis(None,axis = 1)
            .reset_index()
        )
    
    return data

model = 'test_exposure'
artifact_path = 'C:\\Users\\andersenp\\Documents\\Python_CEB_cloudless\\vivarium_tobacco\\vivarium_unimelb_tobacco_intervention_comparison/artifacts/mslt_tobacco_maori_20-years.hdf'
disease_list = ['CHD', 'Stroke']
#disease_list = ['CHD']
scenario = 'tax'
output_prefix = 'disease_delta_results/'

chronic_delta_calc = chronicDiseaseDeltaCalculator(model, artifact_path, disease_list) 
chronic_delta_calc.compute_mortality_deltas(2011, 2020, scenario)
chronic_delta_calc.compute_yld_deltas(2011, 2020, scenario)
chronic_delta_calc.write_delta_series(output_prefix, scenario, 'mortality')
chronic_delta_calc.write_delta_series(output_prefix, scenario, 'yld')
#chronic_delta_calc.write_delta_table(output_prefix, scenario)



