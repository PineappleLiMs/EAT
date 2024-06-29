'''fun script for factorial design'''

import numpy as np
import pfsp
import algorithm_run
import pandas as pd
import os


# run on 2024-04-02: run all levels among factors on all instances
# '''
## set basic information
run_day = '20240402'
save_path = '../data/results/factorial_design/' + run_day + '/'
## set instances: all 80 expensive instances from Taillard benchmark
# a = np.arange(41, 61) # first trail on the first 20 instances
a = np.arange(61, 91) # first trail on instances with 100 jobs
instance_index = a.reshape([2*int(len(a)/10),-1])
instance_index1 = instance_index[::2].reshape([1,-1])[0]
instance_index2 = instance_index[1::2].reshape([1,-1])[0]
run_index = instance_index1
bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[0]

## set parameters
ini_type = ''  # random initial solution
num_rep = 20
all_algorithm_config = {'neigh': 'insert', 'max_time_parameter': 0.03, 'transfer_interval': 5, 'elite_number': 5}

## set factors and levels
sampling_strategies = ['LSP', 'LST', 'KK1', 'KK2', 'RND', 'SR0', 'SR1', 'SR2']
sample_scales = [10, 20, 30, 40, 50, 60, 70, 80, 90]
patching_strategies = ['RI', 'EI', 'OI', 'AI']

EMT_names = ['MFEA', 'PMFEA', 'MFEA-II', 'GMFEA']
for EMT_name in EMT_names:
    if EMT_name in ['MFEA-II', 'GMFEA', 'MFEA']:
        mfea_config = {'pop_size': 100, 'max_ls': 40, 'type': 'rk', 'improve': 1, 'rmp': 0.3}
    elif EMT_name in ['PMFEA']:
        mfea_config = {'pop_size': 30, 'max_ls': 40, 'type': 'perm', 'improve': 0.1, 'rmp': 0.7}
    for sampling_strategy in sampling_strategies:
        for sample_scale in sample_scales:
            for patching_strategy in patching_strategies:
                for i in run_index:
                    ## load instance and set parameters
                    instance_path = '../data/benchmark/ta/ta{:03d}.json'.format(i)
                    instance = pfsp.import_instance(instance_path)
                    task = pfsp.Pfsp(instance)
                    running_time = 0.03 * task.num_job * task.num_mac
                    if EMT_name in ['mfea', 'MFEA', 'pmfea', 'PMFEA', 'MFEA-II', 'mfea-II', 'MFEA2', 'mfea2']:
                        algorithm_config = (100, all_algorithm_config['neigh'], running_time, all_algorithm_config['transfer_interval'], patching_strategy, all_algorithm_config['elite_number'])
                    elif EMT_name in ['gmfea', 'GMFEA']:
                        algorithm_config = (100, all_algorithm_config['neigh'], running_time, all_algorithm_config['transfer_interval'], patching_strategy, all_algorithm_config['elite_number'])
                    # algorithm_run
                    raw_data, dealed_data = algorithm_run.repeat_on_instance2(EMT_name, instance_path, bsf_list[i-1], mfea_config, algorithm_config, num_rep, eco_ratio=sample_scale/100, eco_method=sampling_strategy, ini=ini_type, eco_num=None, use_seed=True)
                    ## save data
                    done_mark = EMT_name+'_ta{:03d}'.format(i)+'_'+sampling_strategy+'-'+str(sample_scale)+'-'+patching_strategy
                    np.save(save_path+'raw_data/'+done_mark+'.npy', raw_data)
                    np.save(save_path+'dealed_data/'+done_mark+'.npy', dealed_data)
                    print(done_mark)
# '''


# deal data
'''
## set stored results
all_raw_data = {'EMT': [], 'sampling_strategy': [], 'sample_scale': [], 'patching_strategy': [], 'instance': [], 'repeat': [], 'makespan': [], 'RE': []}
all_dealed_data = {'EMT': [], 'sampling_strategy': [], 'sample_scale': [], 'patching_strategy': [], 'instance': [], 'makespan': [], 'STD': [], 'ARE': [], 'BRE': [], 'WRE': []}
convergence_trend = {}
## settings
run_day = '20240402'
save_path = '../data/results/factorial_design/' + run_day + '/'
len_num = 20  # length of convergence
num_rep = 20
sampling_strategies = ['LSP', 'LST', 'KK1', 'KK2', 'RND', 'SR0', 'SR1', 'SR2']
sample_scales = [10, 20, 30, 40, 50, 60, 70, 80, 90]
patching_strategies = ['RI', 'EI', 'OI', 'AI']
EMT_names = ['MFEA', 'PMFEA', 'MFEA-II', 'GMFEA']
run_index = np.arange(41, 61)
bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[0]

for EMT_name in EMT_names:
    for sampling_strategy in sampling_strategies:
        for sample_scale in sample_scales:
            for patching_strategy in patching_strategies:
                convergence_trend_tem = np.zeros([len(run_index), len_num])
                for i in run_index:
                    ## load data
                    done_mark = EMT_name+'_ta{:03d}'.format(i)+'_'+sampling_strategy+'-'+str(sample_scale)+'-'+patching_strategy
                    raw_data = np.load(save_path+'raw_data/'+done_mark+'.npy', allow_pickle=True)
                    dealed_data = np.load(save_path+'dealed_data/'+done_mark+'.npy', allow_pickle=True).item()
                    bsf = bsf_list[i-1]
                    ## write dealed_data
                    all_dealed_data['EMT'].append(EMT_name)
                    all_dealed_data['sampling_strategy'].append(sampling_strategy)
                    all_dealed_data['sample_scale'].append(sample_scale)
                    all_dealed_data['patching_strategy'].append(patching_strategy)
                    all_dealed_data['instance'].append('ta{:03d}'.format(i))
                    all_dealed_data['makespan'].append(dealed_data['mean_value'])
                    all_dealed_data['STD'].append(dealed_data['std_value'])
                    all_dealed_data['ARE'].append(dealed_data['are'])
                    all_dealed_data['BRE'].append(dealed_data['bre'])
                    all_dealed_data['WRE'].append(dealed_data['wre'])
                    ## write raw_data
                    trend_instance = np.zeros([len(raw_data), len_num])
                    for j in range(num_rep):
                        single_run_data = raw_data[j]
                        value = single_run_data['best_value'][0]
                        all_raw_data['EMT'].append(EMT_name)
                        all_raw_data['sampling_strategy'].append(sampling_strategy)
                        all_raw_data['sample_scale'].append(sample_scale)
                        all_raw_data['patching_strategy'].append(patching_strategy)
                        all_raw_data['instance'].append('ta{:03d}'.format(i))
                        all_raw_data['repeat'].append(j)
                        all_raw_data['makespan'].append(value)
                        all_raw_data['RE'].append(100 * (value - bsf) / bsf)
                        ## write trend
                        trend_instance_tem = single_run_data['list'][0]
                        sample_index = np.linspace(0,len(trend_instance_tem)-1,len_num,dtype=int)
                        trend_instance[j] = 100 *(np.array(trend_instance_tem)[sample_index] - bsf) / bsf
                    convergence_trend_tem[i-run_index[0]] = np.mean(trend_instance, axis=0)
                convergence_trend[EMT_name+'_'+sampling_strategy+'-'+str(sample_scale)+'-'+patching_strategy] = np.mean(convergence_trend_tem, axis=0)
all_raw_data = pd.DataFrame(all_raw_data)
all_raw_data.to_csv(save_path+'raw_data.txt', sep='\t')
all_dealed_data = pd.DataFrame(all_dealed_data)
all_dealed_data.to_csv(save_path+'dealed_data.txt', sep='\t')
np.save(save_path+'convergence_trend.npy', convergence_trend)
'''