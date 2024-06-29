'''
Integrtate performance of each algorithm
'''
import numpy as np
import pfsp
import EMT_algorithms
import multiprocessing as mp
import random


def single_run_on_instance(algorithm_name, instance_path, eco_ratio, eco_method, kt_method=None, seed=None, ini='', max_time_parameter=0.03, patching_strategy='RI'):
    '''
    Integrate parameter assignment, object formulation and algorithnm performing in this function.

    Parameters:
        algorithm_name: name of selected algorithm (str)
        instance_path: path of file containing the expensive instance (str)
        eco_ratio: sampling ratio (scale) to generate economical auxiliary task (float)
        eco_method: sampling strategy to generate economical auxiliary task (str)
        kt_method: method for knowledge transfer (str, default: None)
        seed: whether to use pre-determined seed. (int, default: None)
        ini: method to get initial solution (str, default: '')
        max_time_parameter: parameter to determine the maximum CPU time (float)
        patching_strategy: patching strategy used when kt_method=='patching' (str, default: 'RI')
    Returns:
        single_run_result: a dictionary containing best_value, running_time and fitness value list (dict)
    '''
    # set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # import instance
    instance = pfsp.import_instance(instance_path)
    task = pfsp.Pfsp(instance)
    # formulate auxiliary task
    if eco_method == 'RndTsk1':
        # randomly select from other instances with the same size
        select_index = random.randint(1, 10)
        instance2_index = 10 * int((int(instance_path[-8:-5])-1)/10) + select_index
        instance2_path = instance_path[:-8] + '{:03d}.json'.format(instance2_index)
        eco_instance = pfsp.import_instance(instance2_path)
        eco = pfsp.Pfsp(eco_instance)
    elif eco_method == 'RndTsk2':
        # randomly select from instances with fewer jobs but identical number of machines
        size_range = {'20-5': [1, 10], '20-10': [11, 20], '20-20': [21, 30], '50-5': [31, 40], '50-10': [41, 50], '50-20': [51, 60], '100-5': [61, 70], '100-10': [71, 80], '100-20': [81, 90], '200-10': [91, 100], '200-20': [101, 110], '500-20': [111, 120]}
        n = task.num_job
        m = task.num_mac
        job_list = []
        for job_num in [20, 50, 100, 200, 500]:
            if job_num < n:
                job_list.append(job_num)
        if len(job_list) == 0:
            raise Exception("no instances has fewer jobs than given instance")
        n2 = random.choice(job_list)
        n2_range = size_range[str(n2)+'-'+str(m)]
        select_index = random.randint(n2_range[0], n2_range[1])
        eco_instance = pfsp.import_instance(instance_path[:-8] + '{:03d}.json'.format(select_index))
        eco = pfsp.Pfsp(eco_instance)
        eco.eco_index = np.arange(n) + 1
        eco.eco_type = 'eco'
    elif eco_method == 'RndTsk3':
        # randomly select from instances with more jobs but identical number of machines
        size_range = {'20-5': [1, 10], '20-10': [11, 20], '20-20': [21, 30], '50-5': [31, 40], '50-10': [41, 50], '50-20': [51, 60], '100-5': [61, 70], '100-10': [71, 80], '100-20': [81, 90], '200-10': [91, 100], '200-20': [101, 110], '500-20': [111, 120]}
        n = task.num_job
        m = task.num_mac
        job_list = []
        for job_num in [20, 50, 100]:
            if job_num > n:
                job_list.append(job_num)
        for job_num in [200, 500]:
            if job_num == 200 and m == 5:
                continue
            elif job_num == 500 and m in [5, 10]:
                continue
            else:
                job_list.append(job_num)
        if len(job_list) == 0:
            raise Exception("no instances has more jobs than given instance")
        n2 = random.choice(job_list)
        n2_range = size_range[str(n2)+'-'+str(m)]
        select_index = random.randint(n2_range[0], n2_range[1])
        eco_instance = pfsp.import_instance(instance_path[:-8] + '{:03d}.json'.format(select_index))
        eco = pfsp.Pfsp(eco_instance)
    elif eco_method in ['LSP', 'lsp', 'LST', 'lst', 'RND', 'rnd', 'KK1', 'kk1', 'KK2', 'kk2', 'SR0', 'sr0', 'SR1', 'sr1', 'SR2', 'sr2']:
        eco = task.get_eco(int(task.num_job*eco_ratio), eco_method)
    else:
        raise ValueError('unsupported eco_method: ' + eco_method)
    # formulate multi-task list
    tasks = [task, eco]
    # set initial solution
    EMT_config = {}
    if ini == 'neh':
        ini_solution = [[task.NEH()[0]], [np.append(eco.NEH()[0], eco.eco_index[eco.num_job:])]]
        EMT_config.update({'ini': ini_solution})
    # determine knowledge transfer config
    if kt_method == 'patching':
        kt_config = {'kt_method': 'patching', 'kt_interval': 5, 'transfer_num': 5, 'patching_strategy': patching_strategy}
    elif kt_method == 'cl':
        kt_config = {'kt_method': 'cl', 'kt_interval': 5}
    else:
        kt_config = {}
    # formulate object and run on each algorithm
    if algorithm_name in ['mfea', 'MFEA', 'mfea-i', 'MFEA-I', 'MFEA1', 'MFEA1']:
        EMT_config_tem = {'pop_size': 100, 'max_ls': 40}
        EMT_config.update(EMT_config_tem)
        algorithm_config = {'max_time': max_time_parameter*task.num_job*task.num_mac, 'rmp': 0.3, 'improve_type': 'probability', 'improve_parameter': 1}
        algorithm_object = EMT_algorithms.MFEA(algorithm_config, kt_config, tasks, EMT_config)
        single_run_result = algorithm_object.run()
    elif algorithm_name in ['MFEA-II', 'mfea-II', 'MFEA2', 'mfea2']:
        EMT_config_tem = {'pop_size': 100, 'max_ls': 40}
        EMT_config.update(EMT_config_tem)
        algorithm_config = {'max_time': max_time_parameter*task.num_job*task.num_mac, 'rmp': 0.3, 'improve_type': 'probability', 'improve_parameter': 1}
        algorithm_object = EMT_algorithms.MFEA2(algorithm_config, kt_config, tasks, EMT_config)
        single_run_result = algorithm_object.run()
    elif algorithm_name in ['gmfea', 'GMFEA', 'G-MFEA', 'g-mfea']:
        EMT_config_tem = {'pop_size': 100, 'max_ls': 40}
        EMT_config.update(EMT_config_tem)
        algorithm_config = {'max_time': max_time_parameter*task.num_job*task.num_mac, 'rmp': 0.3, 'improve_type': 'probability', 'improve_parameter': 1}
        algorithm_object = EMT_algorithms.GMFEA(algorithm_config, kt_config, tasks, EMT_config)
        single_run_result = algorithm_object.run()
    elif algorithm_name in ['pmfea', 'PMFEA', 'P-MFEA', 'p-mfea']:
        EMT_config_tem = {'pop_size': 30, 'max_ls': 40}
        EMT_config.update(EMT_config_tem)
        algorithm_config = {'max_time': max_time_parameter*task.num_job*task.num_mac, 'rmp': 0.7, 'improve_type': 'probability', 'improve_parameter': 0.1}
        algorithm_object = EMT_algorithms.PMFEA(algorithm_config, kt_config, tasks, EMT_config)
        single_run_result = algorithm_object.run()
    else:
        raise ValueError('Unsupported algorithm_name: ' + algorithm_name)
    return single_run_result


def repeat_run(algorithm_name, instance_path, eco_ratio, eco_method, bsf, num_rep=20, kt_method=None, random_parallel_seed=True, ini='', max_time_parameter=0.03, patching_strategy='RI'):
    '''
    Repeat running given algorithm on given instance for num_rep times. Parallel is used to perform the repeated run.

    Parameters:
        algorithm_name: name of selected algorithm (str)
        instance_path: path of file containing the expensive instance (str)
        eco_ratio: sampling ratio (scale) to generate economical auxiliary task (float)
        eco_method: sampling strategy to generate economical auxiliary task (str)
        bsf: best-so-far makespan of given instance (int)
        num_rep: number of repeat runs (int, default: 20)
        kt_method: method for knowledge transfer (str, default: None)
        seed: whether to use pre-determined seed. (int, default: None)
        ini: method to get initial solution (str, default: '')
        max_time_parameter: parameter to determine the maximum CPU time (float)
        patching_strategy: patching strategy used when kt_method=='patching' (str, default: 'RI')
    Returns:
        raw_data: a list containing results of each repeated run, each result is a dictionary containing best_value, running_time and fitness value list (list)
        dealed_data: a dictionary containing mean_value, std_value, are, bre, are and mean_time of all repeated runs (dict).
    '''
    if random_parallel_seed:
        # to ensure randomness under parallel environment, we randomly generate random seeds for each parallel core
        seeds = np.random.randint(1, 10000, num_rep)
    else:
        seeds = [None] * num_rep
    single_run_parameters = [(algorithm_name, instance_path, eco_ratio, eco_method, kt_method, seeds[i], ini, max_time_parameter, patching_strategy) for i in range(num_rep)]
    p = mp.Pool(processes=num_rep)
    raw_data = p.starmap(single_run_on_instance, single_run_parameters)
    raw_value, re, raw_time, raw_trend = [], [], [], []
    ismt = True
    if algorithm_name == 'stea':
        ismt = False
    for i in raw_data:
        if ismt:
            value = i['best_value'][0]
            trend = i['list'][0]
            time = i['running_time']
        else:
            value = i[0]['best_value']
            trend = i[0]['list']
            time = i[0]['running_time']
        raw_value.append(value)
        re.append(100*(value-bsf)/bsf)
        raw_time.append(time)
        raw_trend.append(trend)
    dealed_data = {'mean_value': np.mean(raw_value), 'std_value': np.std(raw_value), 'are': np.mean(re), 'bre': min(re), 'wre': max(re), 'mean_time': np.mean(raw_time)}
    return raw_data, dealed_data