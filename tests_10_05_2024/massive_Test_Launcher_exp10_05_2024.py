import subprocess, ipdb, os
import yaml

import subprocess

# Bash commands to set environment variables
bash_commands = [
    "SLURM_CPUS_PER_TASK=15",
    "N_THREADS=$SLURM_CPUS_PER_TASK",
    "export MKL_NUM_THREADS=${N_THREADS}",
    "export NUMEXPR_NUM_THREADS=${N_THREADS}",
    "export OMP_NUM_THREADS=${N_THREADS}",
    "export OPENBLAS_NUM_THREADS=${N_THREADS}"
]
import ipdb; ipdb.set_trace()
# Execute each Bash command
for cmd in bash_commands:
    subprocess.run(cmd, shell=True)

def launch_process(job_name, py_file, out_file, config_path, nodelist='pika', 
                   gpures='gpu:L40:1', cpu_res='15'):
    r = subprocess.run(
        [
            "sbatch", 
            "--partition=high", 
            "--nodelist={}".format(nodelist), 
            "--gres={}".format(gpures), 
            "--job-name={}".format(job_name),
            "--ntasks=1", 
            "--cpus-per-task={}".format(str(cpu_res)), 
            "papermill", "{}".format(py_file), "{}".format(out_file),
            "-p", "config_path", "{}".format(config_path),
            "--log-output", "--log-level", "DEBUG", "--progress-bar" 
        ] )

    
def execute_bulky_species_test(setup_tuples, list_bacteria, nodelist='ai'):
    
    import ipdb; ipdb.set_trace()
    py_file, job_lexem, config_path, \
        folder_path_out, out_file_lexem = setup_tuples

    
    if not os.path.exists(folder_path_out):
        os.makedirs(folder_path_out)
        print("Directory '%s' created" %folder_path_out)
        
    '''
        - loss1: just the first chunck of loss of bias example, sumation of 2
                outputs with target in MSE
    '''
    for i in list_bacteria:
        launch_process(job_name=job_lexem+i, 
                       py_file=py_file, 
                       out_file=folder_path_out+out_file_lexem+i+'.ipynb', 
                       config_path=config_path+i+'.yaml', 
                       #nodelist='koala', 
                       #gpures='gpu:A5000:1',
                       nodelist=nodelist, 
                       #gpures='gpu:L40:1',
                       gpures='shard:40',  
                       cpu_res='15')
        print(f'Running {i} on slurm from:    {config_path+i+".yaml"}')

if __name__ == '__main__':

    
    list_bacteria = [ 'Caenorhabditis_elegans_lysate', 
                'Mus_musculus_BMDC_lysate',  
                'Danio_rerio_Zenodo_lysate', 
                'Geobacillus_stearothermophilus_NCA26_lysate', 
                'Mus_musculus_liver_lysate', 
                'Drosophila_melanogaster_SII_lysate', 
                'Arabidopsis_thaliana_seedling_lysate', 
                'Bacillus_subtilis_168_lysate_R1',  
                'Escherichia_coli_cells', 
                'Escherichia_coli_lysate', 
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
    
    
    #list_bacteria = [ 'HAOEC']
    #list_bacteria = [ 'Saccharomyces_cerevisiae_lysate']
    #list_bacteria = [ 'Mus_musculus_BMDC_lysate']
    
    '''----------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------------------------------------------------'''    

    import ipdb; ipdb.set_trace()
    
    experimental_setup = 'configs/experiments_10_05_2024/experimental_setup.yaml'
    with open(experimental_setup, 'r') as file:
        config = yaml.safe_load(file)
        
    def load_configuration_yaml(config, tag_type):
        pyfile = config['py_file']

        keys_methods = config[tag_type].keys()
        setup_list = [ [pyfile, \
               config[tag_type][algo_type]['job_lexem'],\
               config[tag_type][algo_type]['config_path'],\
               config[tag_type][algo_type]['folder_path_out'],\
               config[tag_type][algo_type]['out_file_lexem'] ] for algo_type in keys_methods]
        return setup_list, keys_methods
    
    
    
    #setup_list, methods = load_configuration_yaml(config, 'spec_species')
    setup_list, methods = load_configuration_yaml(config, 'global_species')

    
    for setup, method in zip(setup_list, methods):
        import ipdb; ipdb.set_trace()
        print(f'\n launching ....{method}..... for each species...\n' )
        execute_bulky_species_test(setup, list_bacteria, nodelist='ai')
        #py_file, job_lexem, config_path, folder_path_out, out_file_lexem = setup
        


    


