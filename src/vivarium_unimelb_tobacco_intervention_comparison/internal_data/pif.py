"""
This script generates PIF values from exposure tables.
"""
import os
import pandas as pd

from vivarium.framework.artifact.artifact import Artifact


class PIFCalculator:

    def __init__(self, model, artifact_path, exposure_list, disease_dict):
        self.model = model
        self.exposure_list = exposure_list
        self.disease_dict = disease_dict
        
        self.rr_key_columns = ['age_start', 'age_end', 'sex',
                            'year_start', 'year_end']

        self.exposure_table_prefix = 'exposure_results/'
        self.bau_scenario = 'BAU'
        self.pif_table = pd.DataFrame()

        self.disease_rr_data = {}
        self.bau_exp_tables = {}
        artifact = Artifact(artifact_path)

        for exposure in exposure_list:
            key = f'risk_factor.{exposure}.disease_relative_risk'
            self.disease_rr_data[exposure] = pivot_load(artifact, key)
            # Check that the relative risk table includes required columns..
            if ( 
                set(self.rr_key_columns) 
                & set(self.disease_rr_data[exposure].columns) 
                != set(self.rr_key_columns)
            ):
                msg = 'Missing index columns for disease-specific relative risks'
                raise ValueError(msg)

            self.bau_exp_tables[exposure] = (
                                            self.load_exposure_table(self.model, self.bau_scenario, exposure)
                                            .set_index(['age','sex','year'])
            )          


    def compute_PIFs(self, start_year, end_year, scenario):

        #Pre-load exposure tables for intervention scenario
        int_exp_tables = {}
        for exposure in self.exposure_list:
            int_exp_tables[exposure] = (
                                            self.load_exposure_table(self.model, scenario, exposure)
                                            .set_index(['age','sex','year'])
            )

        for disease in self.disease_dict:
            pif_prod = None

            for exposure in self.disease_dict[disease]:
                rr_table = self.load_disease_rr(disease, exposure)
                exposure_pif = pd.Series()

                for year in range(start_year, end_year+1):
                    bau_exp_subtable = self.bau_exp_tables[exposure].xs(year, level = 'year', drop_level=False)
                    int_exp_subtable = int_exp_tables[exposure].xs(year, level = 'year', drop_level=False)
                    rr_subtable = rr_table.loc[bau_exp_subtable.index]
                    
                    bau_rr_values = bau_exp_subtable*rr_subtable
                    int_rr_values = int_exp_subtable*rr_subtable
                
                    # Calculate the mean relative-risk for the BAU scenario.
                    # Sum over all of the bins in each row.
                    mean_bau_rr = bau_rr_values.sum(axis=1) / bau_exp_subtable.sum(axis=1)
                    # Handle cases where the population size is zero.
                    mean_bau_rr = mean_bau_rr.fillna(1.0)

                    # Calculate the mean relative-risk for the intervention scenario.
                    # Sum over all of the bins in each row.
                    mean_int_rr = int_rr_values.sum(axis=1) / int_exp_subtable.sum(axis=1)
                    # Handle cases where the population size is zero.
                    mean_int_rr = mean_int_rr.fillna(1.0)

                    # Calculate the disease incidence PIF for the intervention scenario.
                    pif = (mean_bau_rr - mean_int_rr) / mean_bau_rr
                    pif = pif.fillna(0.0)

                    exposure_pif = exposure_pif.append(pif)
                
                if pif_prod is not None:
                    pif_prod *= 1-exposure_pif 
                else:
                    pif_prod = 1-exposure_pif
            
            self.pif_table[disease] = 1 - pif_prod

                                
    def load_exposure_table(self, model, scenario, exposure):
        exposure_table_filename = '{}_{}_{}.csv'.format(model,scenario,exposure)
        return pd.read_csv(self.exposure_table_prefix + exposure_table_filename)
    

    def load_disease_rr(self, disease, exposure):
        dis_rr_data = self.disease_rr_data[exposure]

        dis_columns = [c for c in dis_rr_data.columns if c.startswith(disease)]
        dis_keys = [c for c in dis_rr_data.columns if c in self.rr_key_columns]

        if not dis_columns or not dis_keys:
            msg = 'No relative risks for disease {}'
            raise ValueError(msg.format(disease))
        
        rr_data = dis_rr_data.loc[:, dis_keys + dis_columns]
        dis_prefix = '{}_'.format(disease)
        exp_prefix =  '{}.'.format(exposure)

        exp_col = {c: c.replace(dis_prefix, exp_prefix).replace('post_', '')
                    for c in dis_columns}
        exp_col['age_start'] = 'age'

        rr_data.rename(columns=exp_col, inplace=True)
        rr_data.drop(['year_start','year_end','age_end'], axis=1, inplace=True)
        rr_data.set_index(['age','sex'], inplace=True)

        return rr_data

    def write_pif_table(self, output_prefix, scenario):
        directory = './{}{}'.format(output_prefix, self.model)
        make_directory(directory)
        pif_table_filename = os.path.join(directory, '{}_pifs_{}.csv'.format(self.model, scenario))

        pif_index = pd.MultiIndex.from_tuples(self.pif_table.index).set_names(['age','sex','year'])
        output_table = self.pif_table.set_index(pif_index, drop=True)

        output_table.reset_index().to_csv(pif_table_filename, index=False)


    def write_pif_series(self, output_prefix, scenario, diseases):
        for disease in diseases:
            parent_directory = './{}{}'.format(output_prefix, self.model)
            make_directory(parent_directory)
            sub_directory = os.path.join(parent_directory, disease)
            make_directory(sub_directory)

            pif_table_filename = os.path.join(sub_directory, '{}_pifs_{}_{}.csv'.format(self.model, 
                                                                                    disease, 
                                                                                    scenario))

            pif_index = pd.MultiIndex.from_tuples(self.pif_table.index).set_names(['age','sex','year'])
            table = (
                           self.pif_table
                           .set_index(pif_index, drop=True)
                           .rename(columns={disease:'value'})  
            )             
            
            output_series = table['value']
            output_series.reset_index().to_csv(pif_table_filename, index=False)


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
exposure_list = ['tobacco']
disease_dict = {'CHD':['tobacco'], 'Stroke':['tobacco'],}
scenario = 'tax'
output_prefix = 'pif_results/'

pif_calc = PIFCalculator(model, artifact_path, exposure_list, disease_dict) 
pif_calc.compute_PIFs(2011, 2020, scenario)
#pif_calc.write_pif_table(output_prefix, scenario)
pif_calc.write_pif_series(output_prefix, scenario, ['CHD','Stroke'])

