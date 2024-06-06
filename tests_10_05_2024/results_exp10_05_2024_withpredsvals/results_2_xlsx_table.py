import subprocess, ipdb, os, re
import yaml

import subprocess
import pandas as pd
import numpy as np

def parse_results(path, organism):
    #regex_pattern = r"(^[a-zA-Z]+)|(\-?[0-9]+.[0-9]+)"
    regex_pattern = r"(^[a-zA-Z]+)|((\-?[0-9]+.[0-9]+)|(\d+\.))"
    with open(path) as reader:
        lines = reader.readlines()
    
    res_info = {}
    #import ipdb; ipdb.set_trace

    for j in lines:
        match = re.findall(regex_pattern, j)
        #res_info[ max( match[0] ) ] = float(  ''.join([*match[1]])  )
        res_info[ max( match[0] ) ] = float(  max(match[1])  )

    
    return res_info#{organism:res_info}


def get_organism_res_dict(folder, contribution, method, list_organism):
    #import ipdb; ipdb.set_trace()
    method_dict = dict()
    for org in list_organism:
        method_dict[org] = parse_results(f"{folder}{contribution}/{method}/{org}.txt.txt", \
                                     org)
    
    return {method:method_dict}#method_dict


def get_global_res_dict(folder, contribution, methods):
    #import ipdb; ipdb.set_trace()
    method_dict = dict()
    for method in methods:
        method_dict[method] = parse_results(f"{folder}{contribution}/res_AllMeltome_{method}.txt.txt", \
                                     method)
            
    return {'global':method_dict}
    

# Function to create the combined DataFrame
def create_combined_dataframe(*dicts):
    # Extract method names and metrics from the dictionaries
    method_dicts = [{method: data} for d in dicts for method, data in d.items()]
    
    # Create a list of all organisms
    organisms = sorted(set(organism for method_dict in method_dicts for data in method_dict.values() for organism in data.keys()))
    
    # Initialize an empty dictionary to store the combined data
    combined_data = {organism: {} for organism in organisms}
    
    # Populate the combined data dictionary
    for method_dict in method_dicts:
        for method, data in method_dict.items():
            for organism in organisms:
                for metric in ['Spearman', 'MSE', 'RMSE', 'MAE']:
                    combined_data[organism][f'{method}_{metric}'] = data.get(organism, {}).get(metric, None)
    
    # Create a DataFrame from the combined data
    df = pd.DataFrame.from_dict(combined_data, orient='index')
    
    # Reorder columns
    methods = [method for method_dict in method_dicts for method in method_dict.keys()]
    cols = [f'{method}_{metric}' for method in methods for metric in ['Spearman', 'MSE', 'RMSE', 'MAE']]
    df = df[cols]
    
    means=df.mean()
    df.loc['MEANS']= means
    
    return df

"""
# Function to save multiple dataframes to an Excel file
def save_dataframes_to_excel(dataframes, filename):
    with pd.ExcelWriter(filename) as writer:
        for i, df in enumerate(dataframes):
            sheet_name = f'Sheet{i+1}'
            df.to_excel(writer, sheet_name=sheet_name)
"""
# Function to save multiple dataframes to an Excel file with specified sheet names
def save_dataframes_to_excel(dataframes, sheet_names, filename):
    with pd.ExcelWriter(filename) as writer:
        for df, sheet_name in zip(dataframes, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name)
            

def species_type_df_group_creation(folder_global_species_res, contributions, methods, list_organism):

    ESM_info = [ get_organism_res_dict(folder_global_species_res, contributions[0], ii, list_organism) for ii in methods]
    PiFold_info = [ get_organism_res_dict(folder_global_species_res, contributions[1], ii, list_organism) for ii in methods]
    Combination_info = [ get_organism_res_dict(folder_global_species_res, contributions[2], ii, list_organism) for ii in methods]
    
    df_ESM = create_combined_dataframe(*ESM_info)
    df_PiFold = create_combined_dataframe(*PiFold_info)
    df_Both = create_combined_dataframe(*Combination_info)
    
    return [df_ESM, df_PiFold, df_Both]

def global_group_creation(folder_global, contributions, gmethods):

    ESM_global_info = [ get_global_res_dict(folder_global, contributions[0], gmethods) ]
    PiFold_global_info = [ get_global_res_dict(folder_global, contributions[1], gmethods) ]
    Combination_global_info = [ get_global_res_dict(folder_global, contributions[2], gmethods) ]
    
    df_ESM = create_combined_dataframe(*ESM_global_info);  df_ESM.drop('MEANS', inplace=True)
    df_PiFold = create_combined_dataframe(*PiFold_global_info);  df_PiFold.drop('MEANS', inplace=True)
    df_Both = create_combined_dataframe(*Combination_global_info);  df_Both.drop('MEANS', inplace=True)

    return [df_ESM, df_PiFold, df_Both]
    
    
if __name__ == '__main__':
    
    folder_spec_species_res = 'species/'
    folder_global_species_res = 'global_species/'
    folder_global = 'global/'
    
    contributions = ['ESM_only', 'PiFold_only', 'Both']
    methods = ['balanceMSE', 'biasg', 'MSE', 'rankN']
    gmethods = ['balanceMSE', 'biasg', 'MSE', 'rank_l2_l1_MSE_001_t2']

    
    
    list_organism = [ 'Caenorhabditis_elegans_lysate', 
                'Mus_musculus_BMDC_lysate',  
                'Danio_rerio_Zenodo_lysate', 
                'Geobacillus_stearothermophilus_NCA26_lysate', 
                'Mus_musculus_liver_lysate', 
                'Drosophila_melanogaster_SII_lysate', 
                'Arabidopsis_thaliana_seedling_lysate', 
                'Bacillus_subtilis_168_lysate_R1',  
                'Escherichia_coli_cells', 
                'Ecoli_Lysate', 
                'Oleispira_antarctica_RB-8_lysate_R1', 
                'Saccharomyces_cerevisiae_lysate', 
                'Thermus_thermophilus_HB27_cells', 
                'Thermus_thermophilus_HB27_lysate', 
                'Picrophilus_torridus_DSM9790_lysate', 
                'HepG2', 'HAOEC', 
                'HEK293T', 'HL60', 
                'HaCaT', 'Jurkat', 
                'pTcells', 
                'colon_cancer_spheroids',
                'U937', 'K562']

    import ipdb; ipdb.set_trace()
    
    '''------------------------------------------------------------------------------------------------------------------------'''
    
    '''  GLOBAL RESULTS OVER ALL FLIP PARTITION RELATED TO MELTOME AND TM'''
    df_info_global = global_group_creation(folder_global, contributions, gmethods)
    save_dataframes_to_excel(df_info_global, contributions, 'global.xlsx')
    '''-------------------------------------------------------------------------------------------------------------------------'''
    
    '''-------------------------------------------------------------------------------------------------------------------------'''

    '''  RESULTS OVER ALL FLIP PARTITION RELATED TO MELTOME AND TM USING GLOBAL PRETRAINED MODELS APPLIED TO EACH ORGANISM'''


    df_info_global_species = species_type_df_group_creation(folder_global_species_res, contributions, methods, list_organism)
    save_dataframes_to_excel(df_info_global_species, contributions, 'global_expecies.xlsx')
    
    '''-------------------------------------------------------------------------------------------------------------------------'''
    
    
    '''-------------------------------------------------------------------------------------------------------------------------'''

    '''  RESULTS OVER ALL FLIP PARTITION RELATED TO MELTOME AND TM USING GLOBAL PRETRAINED MODELS APPLIED TO EACH ORGANISM'''


    df_info_spec_species = species_type_df_group_creation(folder_spec_species_res, contributions, methods, list_organism)
    save_dataframes_to_excel(df_info_spec_species, contributions, 'spec_expecies.xlsx')
    
    '''-------------------------------------------------------------------------------------------------------------------------'''

    #save_dataframes_to_excel([df_ESM, df_PiFold, df_Both], contributions, 'global_expecies.xlsx')

    #df = create_combined_dataframe(aa, bb, aa)
    #cols = [f'd{i}_{metric}' for i in range(1, len([aa, bb]) + 1) for metric in ['Spearman', 'MSE', 'RMSE', 'MAE']]
    #df = df[cols]
    
    #print(df)
    #print(df_info_global)
    #print(df_PiFold)
    #print(df_Both)