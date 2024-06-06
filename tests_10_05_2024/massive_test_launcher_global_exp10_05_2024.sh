#!/bin/bash

SLURM_CPUS_PER_TASK=15

N_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

sbatch \
--partition=high \
--nodelist=ai \
--gres=shard:40 \
--job-name=sgal_global_meltome_MSE \
--ntasks=1 \
--cpus-per-task=15 \
papermill 'LAMLP_Benchmark_all_taxonomy_08_April_2024.ipynb' \
'results_exp10_05_2024/res_global_MSE.ipynb' \
-p config_path 'configs/experiments_10_05_2024/global/global_allmeltome_MSE.yaml' \
--log-output --log-level DEBUG --progress-bar 

sbatch \
--partition=high \
--nodelist=ai \
--gres=gpu:1 \
--job-name=sgal_global_meltome_rankN \
--ntasks=1 \
--cpus-per-task=15 \
papermill 'LAMLP_Benchmark_all_taxonomy_08_April_2024.ipynb' \
'results_exp10_05_2024/res_global_rankN.ipynb' \
-p config_path 'configs/experiments_10_05_2024/global/global_allmeltome_rankN.yaml' \
--log-output --log-level DEBUG --progress-bar 

sbatch \
--partition=high \
--nodelist=ai \
--gres=shard:40 \
--job-name=sgal_global_meltome_biasg \
--ntasks=1 \
--cpus-per-task=15 \
papermill 'LAMLP_Benchmark_all_taxonomy_08_April_2024.ipynb' \
'results_exp10_05_2024/res_global_biasg.ipynb' \
-p config_path 'configs/experiments_10_05_2024/global/global_allmeltome_biasg.yaml' \
--log-output --log-level DEBUG --progress-bar 

sbatch \
--partition=high \
--nodelist=ai \
--gres=shard:40 \
--job-name=sgal_global_meltome_balance \
--ntasks=1 \
--cpus-per-task=15 \
papermill 'LAMLP_Benchmark_all_taxonomy_08_April_2024.ipynb' \
'results_exp10_05_2024/res_global_balance.ipynb' \
-p config_path 'configs/experiments_10_05_2024/global/global_allmeltome_balance_batch.yaml' \
--log-output --log-level DEBUG --progress-bar 