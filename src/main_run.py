'''Main file for performing comparisive experiment'''
import numpy as np
import pfsp
import algorithm_run
import pandas as pd
import os

def main_experiment(EMT_names, instance_indices, eco_strategies, kt_methods, save_folder, patching_strategies=['RI'], num_rep=20, max_time_parameter=0.03, instance_type ='ta'):
    '''
    Main function for performing comparisive experiment

    Parameters:
        EMT_names: list of EMT_name (list)
        instance_indices: list of instance indices (list)
        eco_strategies: list of strategies to generate economical auxiliary tasks. Elements are tuples like (sampling strategies (str), sampling ratio (float)) (list) 
        kt_methods: methods for performing knowledge transfer (list)
        save_folder: path to save results
        patching_strategies: when ct_method is 'patching', define patching strategies (list, default: 'RI')
        num_rep: number of repeated runs (int, default: 20)
        max_time_parameter: parameter to determine maximum running time (float, default: 0.03)
        instance_type: type of instances (str, default: 'ta')
    Returns:
        this function has no return. It will directly save results to given path and print process information.
    '''
    if save_folder[-1] != '/':
        save_folder += '/'
    if instance_type == 'ta':
        bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[0]
        instance_head = '../data/benchmark/ta/ta'
    elif instance_type == 'vrf_small':
        bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[1][:240]
        instance_head = '../data/benchmark/vrf_small/vrf_small'
    elif instance_type == 'vrf_large':
        bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[1][240:]
        instance_head = '../data/benchmark/vrf_large/vrf_large'
    for EMT_name in EMT_names:
        os.makedirs(save_folder + EMT_name + '/raw_data/', exist_ok=True)
        os.makedirs(save_folder + EMT_name + '/dealed_data/', exist_ok=True)
        for eco_strategy in eco_strategies:
            eco_method, eco_ratio = eco_strategy[0], eco_strategy[1]
            if (type(eco_ratio) != str) and (eco_ratio > 1):
                real_eco_ratio = eco_ratio / 100
            else:
                real_eco_ratio = eco_ratio
            for kt_method in kt_methods:
                for patching_strategy in patching_strategies:
                    for instance_index in instance_indices:
                        bsf = bsf_list[instance_index-1]
                        instance_path = instance_head + '{:03d}.json'.format(instance_index)
                        if instance_type == 'ta' and instance_index in list(range(61,71))+list(range(91,101))+list(range(111,121)) and eco_method == 'RndTsk3':
                            continue
                        raw_data, dealed_data = algorithm_run.repeat_run(EMT_name, instance_path, real_eco_ratio, eco_method, bsf, num_rep, kt_method, True, '', max_time_parameter, patching_strategy)
                        if kt_method != 'patching':
                            done_mark = EMT_name+'_ta{:03d}'.format(instance_index)+'_'+eco_method+'-'+str(eco_ratio)+'-'+kt_method
                        else:
                            done_mark = EMT_name+'_ta{:03d}'.format(instance_index)+'_'+eco_method+'-'+str(eco_ratio)+'-'+'RI'
                        np.save(save_folder + EMT_name + '/raw_data/' + done_mark + '.npy', raw_data)
                        np.save(save_folder + EMT_name + '/dealed_data/' + done_mark + '.npy', dealed_data)
                        print(done_mark)


def process_data(EMT_names, instance_indices, eco_strategies, kt_methods, save_folder, patching_strategies=['RI'], num_rep=20, instance_type ='ta', len_num=20):
    '''
    Main function for performing comparisive experiment

    Parameters:
        EMT_names: list of EMT_name (list)
        instance_indices: list of instance indices (list)
        eco_strategies: list of strategies to generate economical auxiliary tasks. Elements are tuples like (sampling strategies (str), sampling ratio (float)) (list) 
        kt_methods: methods for performing knowledge transfer (list)
        save_folder: path to save results
        patching_strategies: when ct_method is 'patching', define patching strategies (list, default: 'RI')
        num_rep: number of repeated times (int, default: 20)
        instance_type: type of instances (str, default: 'ta')
        len_num: length of convergence (int, default: 20)
    Returns:
        all_raw_data: aggregated raw data (pd.DataFrame)
        all_dealed_data: aggregated dealed data (pd.DataFrame)
        convergence_trend: aggregated convergence_trend (dict)
    '''
    all_raw_data = {'EMT': [], 'sampling_strategy': [], 'sample_scale': [], 'patching_strategy': [], 'instance': [], 'repeat': [], 'makespan': [], 'RE': [], 'kt_method': []}
    all_dealed_data = {'EMT': [], 'sampling_strategy': [], 'sample_scale': [], 'kt_method': [], 'patching_strategy': [], 'instance': [], 'makespan': [], 'STD': [], 'ARE': [], 'BRE': [], 'WRE': []}
    convergence_trend = {}
    if save_folder[-1] != '/':
        save_folder += '/'
    if instance_type == 'ta':
        bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[0]
        instance_name_head = 'ta'
    elif instance_type == 'vrf_small':
        bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[1][:240]
        instance_name_head = 'vrf_small'
    elif instance_type == 'vrf_large':
        bsf_list = np.load('../data/benchmark/bsf.npy', allow_pickle=True)[1][240:]
        instance_name_head = 'vrf_large'
    for EMT_name in EMT_names:
        for eco_strategy in eco_strategies:
            eco_method, eco_ratio = eco_strategy[0], eco_strategy[1]
            for kt_method in kt_methods:
                for patching_strategy in patching_strategies:
                    convergence_trend_tem = np.zeros([len(instance_indices), len_num])
                    for instance_index in instance_indices:
                        if instance_type == 'ta' and instance_index in list(range(61,71))+list(range(91,101))+list(range(111,121)) and eco_method == 'RndTsk3':
                            continue
                        bsf = bsf_list[instance_index-1]
                        instance_name = instance_name_head + '{:03d}'.format(instance_index)
                        done_mark = EMT_name+'_ta{:03d}'.format(instance_index)+'_'+eco_method+'-'+str(eco_ratio)+'-'+kt_method+'_'+patching_strategy
                        raw_data = np.load(save_folder + EMT_name + '/raw_data/' + done_mark + '.npy', allow_pickle=True)
                        dealed_data = np.load(save_folder + EMT_name + '/dealed_data/' + done_mark + '.npy', allow_pickle=True).item()
                        # write all_dealed data
                        all_dealed_data['EMT'].append(EMT_name)
                        all_dealed_data['instance'].append(instance_name)
                        all_dealed_data['sampling_strategy'].append(eco_method)
                        all_dealed_data['sample_scale'].append(eco_ratio)
                        all_dealed_data['kt_method'].append(kt_method)
                        all_dealed_data['patching_strategy'].append(patching_strategy)
                        all_dealed_data['makespan'].append(dealed_data['mean_value'])
                        all_dealed_data['STD'].append(dealed_data['std_value'])
                        all_dealed_data['ARE'].append(dealed_data['are'])
                        all_dealed_data['BRE'].append(dealed_data['bre'])
                        all_dealed_data['WRE'].append(dealed_data['wre'])
                        # write raw data and convergence trends
                        trend_instance = np.zeros([len(raw_data), len_num])
                        for j in range(num_rep):
                            single_run_data = raw_data[j]
                            value = single_run_data['best_value'][0]
                            all_raw_data['EMT'].append(EMT_name)
                            all_raw_data['sampling_strategy'].append(eco_method)
                            all_raw_data['sample_scale'].append(eco_ratio)
                            all_raw_data['kt_method'].append(kt_method)
                            all_raw_data['patching_strategy'].append(patching_strategy)
                            all_raw_data['instance'].append(instance_name)
                            all_raw_data['repeat'].append(j)
                            all_raw_data['makespan'].append(value)
                            all_raw_data['RE'].append(100 * (value - bsf) / bsf)
                            trend_instance_tem = single_run_data['list'][0]
                            sample_index = np.linspace(0,len(trend_instance_tem)-1,len_num,dtype=int)
                            trend_instance[j] = 100 *(np.array(trend_instance_tem)[sample_index] - bsf) / bsf
                        convergence_trend_tem[instance_index-instance_indices[0]] = np.mean(trend_instance, axis=0)
                    algorithm_name = EMT_name+'_'+eco_method+'-'+str(eco_ratio)+'-'+kt_method+'_'+patching_strategy
                    convergence_trend[algorithm_name] = np.mean(convergence_trend_tem, axis=0)
    all_raw_data = pd.DataFrame(all_raw_data)
    all_raw_data.to_csv(save_folder+'raw_data.txt', sep='\t')
    all_dealed_data = pd.DataFrame(all_dealed_data)
    all_dealed_data.to_csv(save_folder+'dealed_data.txt', sep='\t')
    np.save(save_folder+'convergence_trend.npy', convergence_trend)
    return all_raw_data, all_dealed_data, convergence_trend


if __name__ == '__main__':
    # run on 2024-04-25: run (4 EMTs) * (10 task pairs) * (cl) on all 80 expensive Taillard instances
    
    run_day = '20240425'
    save_folder = '../data/results/comparison/' + run_day + '/'
    a = np.arange(41, 121)
    instance_index = a.reshape([2*int(len(a)/10),-1])
    instance_index1 = instance_index[::2].reshape([1,-1])[0]
    instance_index2 = instance_index[1::2].reshape([1,-1])[0]
    run_index = instance_index1

    EMT_names = ['MFEA-I', 'PMFEA', 'MFEA-II', 'GMFEA']
    eco_strategies = [('LSP', 20), ('LST', 20), ('KK2', 20), ('LSP', 30), ('KK1', 20), ('LST', 30), ('KK2', 30), ('RndTsk1', ''), ('RndTsk2', ''), ('RndTsk3', '')]
    kt_methods = ['CL']
    # 如果想运行全部实验，需要将上面一行改成 kt_methods = ['IK','CL','patching']
    patching_strategies=['RI']
    '''
    main_experiment(EMT_names, run_index, eco_strategies, kt_methods, save_folder, patching_strategies, num_rep=20, max_time_parameter=0.03, instance_type ='ta')
    '''
    process_data(EMT_names, a, eco_strategies, kt_methods, save_folder, patching_strategies=['RI'], num_rep=20, instance_type ='ta', len_num=20)
    