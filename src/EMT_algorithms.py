"""
EMT algorithms
"""
from abc import abstractmethod
import numpy as np
import time
from scipy.optimize import fminbound
from scipy.stats import norm
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import random


class EMT(object):
    '''
    Class for EMT, containing main procedures and common operations

    Attributes:
        tasks: list of tasks to be optimized (list).
        num_task: number of tasks (int).
        dimensions: list of dimensions of each task (list).
        D: dimension of chromosomes (int).
        pop_size: size of population. It shoule be divided by num_task (int).
        sbx_index: spread factor distribution index of SBX (float).
        pm_index: index of polynomial mutation (float).
        improve_num: number of offspring individuals to be improved for each task (int).
        num_eva: number of evaluations (int).
        lists: list of best value found by each iteration of population (np array).
        best_solutions: best solutions found for each task (list).
        best_values: values of best solutions (list).
        max_iter: maximum local search iterations (int).

        population: population of EA (2-D np array, float).
        factorial_cost: factorial cost of each individual (2-D np array, int).
        scalar_fitness: scalar fitness of each individual (np array, float).
        skill_factor: skill factor of each individual (np array, int).
    '''


    def __init__(self, tasks, config):
        # initialize tasks and their properties
        self.tasks = tasks
        self.num_task = len(tasks)
        self.dimensions = [task.num_job for task in tasks]
        self.D = np.max(self.dimensions)
        # initialize MFEA parameters
        self.pop_size = config['pop_size']
        self.num_eva = 0
        self.lists = np.zeros([self.num_task, 0], dtype='int32')
        self.best_solutions = []
        self.max_ls = config['max_ls']
        if 'ini' in config:
            self.ini = config['ini']
        else:
            self.ini = [[] for _ in range(self.num_task)]
        # intialize MFEA population
        self.population = np.empty([self.pop_size, self.D])
        self.factorial_cost = np.full([self.pop_size, self.num_task], np.inf)
        self.scalar_fitness = np.empty(self.pop_size)
        self.skill_factor = np.empty(self.pop_size)


    ### basic operators in EMTs: decoding, encoding, generic operators and local improvement
    def decoding(self, task_index, code):
        """
        Function for decoding
        
        Parameters:
            task_index: index of task to be decoded (int)
            code: code to be decoded (np array)
        Returns:
            solution: solution after decoding
            value: fitness value of solution
            decode_index: index used in decoding
        """
        task = self.tasks[task_index]
        if self.type == 'perm':
            # sequence form
            decode_index = code - 1
            solution = []
            for i in code:
                if i in task.eco_index[:task.num_job]:
                    solution.append(i)
            solution = np.array(solution, dtype='int32')
        elif self.type == 'rk':
            # random key
            decode_index = np.argsort(code)
            solution_tem = decode_index + 1
            solution = []
            for i in solution_tem:
                if i in task.eco_index[:task.num_job]:
                    solution.append(i)
            solution = np.array(solution, dtype='int32')
        value = task.calculate_makespan(solution)
        return solution, value, decode_index

    
    def encoding(self, task_index, sequence, original_sequence=None, original_code=None, decode_index=None):
        """
        Function for encoding.

        Parameters:
            task_index: index of task to be encoded (int)
            sequence: job sequence (np array).
            original_sequence: sequence obtained by decoding original code (np array)
            original_code: original code (np array)  
            decode_index: index used in decoding original code
        Returns:
            new_code: code after encoding (np array)
        """
        task = self.tasks[task_index]
        if original_code is None:
            if self.type == 'perm':
                solution_tem = sequence.copy()
                return np.concatenate([solution_tem, task.eco_index[task.num_job:]])
            elif self.type == 'rk':
                solution_tem = sequence.copy()
                return np.concatenate([solution_tem, task.eco_index[task.num_job:]]) / self.D
            else:
                raise ValueError("unsupported type! Only 'perm' and 'rk' are suported")
        else:
            new_code = original_code.copy()
            for i in range(len(original_sequence)):
                index_in_original_code = np.where(decode_index==original_sequence[i]-1)
                index_in_new_code = np.where(decode_index==sequence[i]-1)
                new_code[index_in_original_code] = original_code[index_in_new_code]
            return new_code
        
    
    def sbx_crossover(self, p1, p2):
        """SBX crossover operator
        Parameter:
            p1 & p2: two parents (np array)
        Returns:
            c1 & c2: kids generated by SBX (np array)
        """
        u = np.random.rand(1, self.D)
        cf = np.empty([1, self.D])
        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (self.sbx_index + 1)))
        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (self.sbx_index + 1)))
        c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
        c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)
        c1 = np.clip(c1, 0, 1)
        c2 = np.clip(c2, 0, 1)
        return c1[0], c2[0]
    

    def pm_mutate(self, p):
        """polynomial mutation operator
        Parameter:
            p: parent chromosome (np array)
        Returns:
            c: kid chromosome (np array)
        """
        mp = float(1. / self.D)
        u = np.random.uniform(size=[self.D])
        r = np.random.uniform(size=[self.D])
        c = np.copy(p)
        for i in range(p.shape[0]):
            if r[i] < mp:
                if u[i] < 0.5:
                    delta = (2*u[i]) ** (1/(1+self.pm_index)) - 1
                    c[i] = p[i] + delta * p[i]
            else:
                delta = 1 - (2 * (1 - u[i])) ** (1/(1+self.pm_index))
                c[i] = p[i] + delta * (1 - p[i])
        c = np.clip(c, 0, 1)
        return c

    
    def ox1(self, p1, p2):
        '''ordered crossover 1: a crossover operator on permutated-form solutions
        
        Parameters:
            p1 & p2: 2 parents coded in permutation (np array)
        Returns:
            c1 & c2: kids (np array)
        '''
        length = len(p1)
        points = random.sample(list(range(length)), 2)
        point1, point2 = min(points), max(points)
        c1_tem, c2_tem = p1[point1: point2+1], p2[point1: point2+1]
        p1_tem = np.hstack((p1[point2+1:], p1[:point2+1]))
        p2_tem = np.hstack((p2[point2+1:], p2[:point2+1]))
        for i in p1_tem:
            if i not in p2[point1: point2+1]:
                c2_tem = np.append(c2_tem, i)
        for i in p2_tem:
            if i not in p1[point1: point2+1]:
                c1_tem = np.append(c1_tem, i)
        c1 = np.hstack((c1_tem[length-point1:], c1_tem[:length-point1]))
        c2 = np.hstack((c2_tem[length-point1:], c2_tem[:length-point1]))
        return c1, c2


    def swap_mutate(self, p):
        '''swap mutation for permutation code
        
        Parameters: 
            p: parent (np array)
        Returns:
            c: kid after mutation (np array)
        ''' 
        points = random.sample(list(range(len(p))), 2)
        c = p.copy()
        c[points[0]], c[points[1]] = p[points[1]], p[points[0]]
        return c


    def local_improvement(self, start, task_index, neigh):
        """local improvement
        
        Parameter:
            start: starting point (np array)
            task_index: index of task (int)
            neigh: neighbourhood (str)
            rule: search rule (str, default: 'best_improve')
        Returns:
            new_solution: solution after local search
            value: fitness value of new solution
        """
        initial_sequence, initial_value, decode_index = self.decoding(task_index, start)
        task = self.tasks[task_index]
        new_sequence, value = task.local_search(neigh, self.max_ls, initial_sequence)
        self.num_eva += self.max_ls
        return self.encoding(task_index, new_sequence, initial_sequence, start, decode_index), value
    

    def sort_best_from_pop(self, current_solution, current_value):
        '''sort best individual from population

        Parameters:
            current_solution: current best solutions for each task (2D np array)
            current_value: current best values for each task (1D np array)
        Returns:
            best_solution: best solution for each task (2D np array)
            best_value: best fitness value for each task (1D np array)
        '''
        if len(current_solution) == 0:
            best_solution = np.empty([self.num_task, self.D])
            best_value = np.empty(self.num_task)
        else:
            best_solution = current_solution.copy()
            best_value = current_value.copy()
        for sf in range(self.num_task):
            index_for_sf = np.where(self.skill_factor == sf)[0]
            pop_for_sf = self.population[index_for_sf]
            fc_for_sf = self.factorial_cost[index_for_sf]
            scalar_fitness_for_sf = self.scalar_fitness[index_for_sf]
            best_index = np.argmax(scalar_fitness_for_sf)
            if len(current_solution) == 0:
                best_solution[sf] = pop_for_sf[best_index]
                best_value[sf] = fc_for_sf[best_index, sf]
            else:
                if fc_for_sf[best_index, sf] <= best_value[sf]:
                    best_solution[sf] = pop_for_sf[best_index]
                    best_value[sf] = fc_for_sf[best_index, sf]
        self.lists = np.column_stack([self.lists, best_value])
        return best_solution, best_value
        
    
    ### every procedure of EMT algorithms
    def get_initial_population(self, size_subpopulation=None):
        """get initial population from uniform sampling.
        
        Parameters:
            size_population: size of each subpolulation. If not given, the whole population will be divided equally for each subpopulation. (list, default: None)
        """
        if self.type == 'rk':
            self.population = np.random.rand(self.pop_size, self.D)
        elif self.type == 'perm':
            self.population = np.zeros([self.pop_size, self.D], dtype='int32')
            for i in range(self.pop_size):
                self.population[i] = np.random.permutation(np.arange(1, self.D+1, dtype='int32'))
        if size_subpopulation == None:
            self.skill_factor = np.array([i % self.num_task for i in range(self.pop_size)])
        else:
            index_tem = random.sample(range(self.pop_size), self.pop_size)
            skill_factor_tem = [0] * self.pop_size
            for i in range(self.pop_size):
                tmp = 0
                for j in range(len(size_subpopulation)):
                    tmp += size_subpopulation[j]
                    if i < tmp:
                        skill_factor_tem[index_tem[i]] = j
                        break
                    else:
                        continue
            self.skill_factor = np.array(skill_factor_tem, dtype='int32')
        # assign given initial solutions if provided
        for i in range(self.num_task):
            ini = self.ini[i]
            if len(ini) != 0:
                index = np.where(self.skill_factor == i)[0]
                for j in range(min(len(ini), len(index))):
                    self.population[index[j]] = ini[j]
        for i in range(self.pop_size):
            sf = self.skill_factor[i]
            _, self.factorial_cost[i, sf], _ = self.decoding(sf, self.population[i])
        self.scalar_fitness = self.calculate_scalar_fitness(self.factorial_cost)
        self.num_eva += self.pop_size
        # sort
        sort_index = np.argsort(self.scalar_fitness)[::-1]
        self.population = self.population[sort_index]
        self.skill_factor = self.skill_factor[sort_index]
        self.factorial_cost = self.factorial_cost[sort_index]
        self.scalar_fitness = self.scalar_fitness[sort_index]
    

    @abstractmethod
    def generate_offspring(self, **kwargs):
        '''
        Generate offspring. Different EMT algorithms may have different ways to generate offsprings. Notice it is an abstract method, which is detailed in each subclass.

        **Kwargs: may contain the following parameters:
            pop: current popoluation (2D np array, defualt: self.       populaiton)
            permutate: permutate order of individuals before mating. (Boolean, default: True)
            shuffle: apply variable shuffle, a strategy used in MCEEA (Boolean, default: False)
            direction: direction used in variable translation
        Returns:
            offspring: offspring population (2-D np array)
            offspring_skill_factor: skill factors of each offspring (np array)
            offspring_factorial_cost: factorial cost of children (2-D np array)
        '''
        pass


    def offspring_ls(self, offspring, offspring_skill_factor, offspring_factorial_cost, neigh='insert', ls_type='probability', improve_parameter=1):
        '''
        Perform local search on offpsring to improve their performance

        Paremeters:
            offspring: offspring population (2-D np array)
            offspring_skill_factor: skill factors of each offspring (np array)
            offspring_factorial_cost: factorial cost of offspring (2-D np array)
            neigh: neighborhood to perform local search (str, default: 'insert')
            ls_type: type of performing local search. 'probability': each offspring can be improved at a given probability; 'elite': only the best individuals in offspring can be improved (default: probability).
            improve_parameter: parameter for improve. For 'probability', it is the probability of each individual's improving; for 'elite', it is the number of elite individuals. (float or int, default: 1)
        Returns:
            new_offspring: offspring after local search (2-D np array)
            new_offspring_factorial_cost: factorial cost of offspring after local search (2-D np array)
        '''
        new_offspring, new_offspring_factorial_cost = offspring.copy(), np.full(np.shape(offspring_factorial_cost), np.inf)
        if ls_type == 'probability':
            for offspring_index in range(len(offspring)):
                individual_skill_factor = offspring_skill_factor[offspring_index]
                individual = offspring[offspring_index]
                if random.random() < improve_parameter:
                    new_offspring[offspring_index], new_offspring_factorial_cost[offspring_index, individual_skill_factor] = self.local_improvement(individual, individual_skill_factor, neigh)
                else:
                    new_offspring_factorial_cost[offspring_index, individual_skill_factor] = offspring_factorial_cost[offspring_index, individual_skill_factor]
        elif ls_type == 'elite':
            ls_elite_num = improve_parameter
            for task_index in range(self.num_task):
                # sort offspring individuals
                index_for_task = np.where(offspring_skill_factor == task_index)[0]  # index of individuals corresponding to the task among population
                offspring_for_task = offspring[index_for_task]
                cost_for_task = offspring_factorial_cost[index_for_task, task_index]
                sort_index = np.argsort(cost_for_task)
                for i in range(ls_elite_num):
                    elite_individual = offspring_for_task[sort_index[i]]
                    improve_p, improve_cost = self.local_improvement(elite_individual, task_index, neigh)
                    new_offspring[index_for_task[sort_index[i]]] = improve_p
                    new_offspring_factorial_cost[index_for_task[sort_index[i]], task_index] = improve_cost
        else:
            raise ValueError('Unsupported ls_type: currently only support "probability" and "elite"')
        return new_offspring, new_offspring_factorial_cost


    def calculate_scalar_fitness(self, factorial_cost):
        """calculate scalar fitness value of given factorial cost
        
        Parameters:
            factorial_cost: factorial cost (2-D np array)
        Returns:
            scalar fitness of given populations
        """
        return 1 / np.min(np.argsort(np.argsort(factorial_cost, axis=0), axis=0) + 1, axis=1)


    def population_updation(self, offspring_pop, offspring_skill_factor, offspring_factorial_costs, pop=None):
        '''
        Combine current population and offspring population, then update population by steady replacement and elite selection

        Parameters:
            offspring_pop: offspring population.
            offspring_skill_factor: skill factors of offspring.
            offspring_factorial_costs: factorial costs of offspring.
            pop: current popoluation (2D np array, defualt: self.populaiton).
        Returns:
            new_pop: new population
        Atrributes updation:
            self.population: updated by new_pop
            self.skill_factor: updated by skill factors of new population
            self.factorial_cost: updated by factorial cost of new population
            self.scalar_fitness: updated by scalar fitness of new population
        '''
        # concatenate current population and offsprings
        if pop is None:
            pop = self.population
        # concatenate current population and offsprings
        intermediate_pop = np.append(pop, offspring_pop, axis=0)
        intermediate_skill_factor = np.append(self.skill_factor, offspring_skill_factor, axis=0)
        intermediate_factorial_costs = np.append(self.factorial_cost, offspring_factorial_costs, axis=0)
        # calculate scalar fitness and sort individuals to get new population
        intermediate_scalar_fitness = self.calculate_scalar_fitness(intermediate_factorial_costs)
        sort_index = np.argsort(intermediate_scalar_fitness)[::-1]
        new_pop = intermediate_pop[sort_index[:self.pop_size]]
        self.skill_factor = intermediate_skill_factor[sort_index[:self.pop_size]]
        self.factorial_cost = intermediate_factorial_costs[sort_index[:self.pop_size]]
        self.scalar_fitness = intermediate_scalar_fitness[sort_index[:self.pop_size]]
        return new_pop
    

    ### knowledge transfer
    def knowledge_transfer(self, num_gen, transfer_interval, method, **kwargs):
        '''
        Perform knowledge transfer

        Parameters:
            num_gen: number of current generation (int)
            transfer_interval: interval of knowledge transfer (int)
            method: method to perform knowledge transfer, can be 'patching': partial solution patching strategy, 'cl': centralized learning strategy, None (default): no knowledge transfer.
            kwargs: may contain the following parameters:
                transfer_num: number of solutions to be selected (int)
                patching_strategy: when method=='patching', input strategy for patching partial solutions (str, default: 'RI')
                source_index: index of source task (int).
                target_index: index of target task (int).
                model: model used for centralized learning (str, default: None)
                best_ratio: ratio of best individuals in current population (float, default: 0.5)
        Returns:
            transfer_population: transferred population (np array)
            transfer_skill_factor: skill factor of transferred population (np array)
            transfer_factorial_cost: factorial costs of transferred population (np array)
        '''
        if num_gen % transfer_interval == 0:
            if method == 'patching':
                if kwargs.get('source_index') is None:
                    source_index = 1
                else:
                    source_index = kwargs.get('source_index')
                if kwargs.get('target_index') is None:
                    target_index = 0
                else:
                    target_index = kwargs.get('target_index')
                if self.tasks[source_index].eco_type != 'eco':
                    raise ValueError('when using "patching" method, source task must be eco')
                index_for_source = np.where(self.skill_factor == source_index)[0]
                pop_for_source = self.population[index_for_source]
                transfer_num = kwargs.get('transfer_num')
                patching_strategy = kwargs.get('patching_strategy')
                pop_to_transfer = pop_for_source[:transfer_num]
                transfer_population = pop_to_transfer.copy()
                transfer_factorial_cost = np.full([transfer_num, self.num_task], np.inf)
                for i in range(len(pop_to_transfer)):
                    source_code = pop_to_transfer[i]
                    source_sequence, _, decode_index = self.decoding(source_index, source_code)
                    transfer_sequence, transfer_factorial_cost[i, target_index] = self.tasks[source_index].p2p_transfer(self.tasks[target_index], source_sequence, patching_strategy)
                    transfer_population[i] = self.encoding(target_index, transfer_sequence, self.decoding(target_index, source_code)[0], source_code, decode_index)
                transfer_skill_factor = np.full(transfer_num, target_index)
            elif method in ['cl', 'CL']:
                raise ValueError('Currently not support')
            elif method in {'ik', 'IK'}:
                raise ValueError('implicit knowledge transfer is integreted in function "generate_offspring"')
            else:
                transfer_population, transfer_skill_factor, transfer_factorial_cost = None, None, None
            return transfer_population, transfer_skill_factor, transfer_factorial_cost
        else:
            return None, None, None
    

    @abstractmethod
    def run(self, **kwarg):
        '''
        Run EMT algorithms. Notice it is an abstract method, which is detailed in each subclass.

        Basically, run in each subclass has the following procedure:
        1. initialize population (in self.get_initial_population)
        2. generate offspring (in generating_offspring of each subclass)
        3. perform local search on offspring (in self.offspring_ls)
        4. generate transfer population (in self.knowledge_transfer)
        5. combine population, offspring and transfer population (in self.population_updation)
        6. population updation (in self.population_updation)

        Parameters: kwarg may contain the following inputs
            max_time: maximum CPU time in seconds (float)
            max_gen: maximum generation (int)
            neigh: neighbourhood used for local improvement (str, default: 'insert')
        Returns: 
            result: dictionary containing best_value, running_time and fitness value list (dic)
        '''


class MFEA(EMT):
    '''
    Class for MFEA-I algorithm

    Attributes:
        Attributes in mfea_config:
        max_gen: maximum generation (int).
        max_time: maximum CPU time in seconds (float).
        neigh: neighbourhood used for local improvement (str, default: 'insert').
        rmp: random mating probability (float).
        type: type of coding (fixed as 'rk')
        sbx_index: a parameter used in SBX (float, default: 15)
        pm_index: a parameter used in polynomial mutation (float, default: 15)
        improve_type: type of imporve ('str', default: 'probability')
        improve_parameter: parameter for improve. For 'probability', it is the probability of each individual's improving; for 'elite', it is the number of elite individuals. (float or int, default: 1)
        use_ik: use implicit knowledge transfer or not (Boolean, default: True)

        Attributes in kt_config:
        kt_method: method for knowledge transfer (str, default: None).
        kt_interval: interval of knowledge transfer (int).
        transfer_num: number of solutions to be transferred in 'patching' method (int)
        patching_strategy: strategy used for patching partial solution in 'patching" method (str).
        model: model used for centralized learning in 'ct' method (str, default: None).
        best_ratio: ratio of best individuals in current population in 'ct' method (float, default: 0.5).
    '''

    def __init__(self, mfea_config, kt_config, tasks, super_config):
        '''
        Parameters:
            mfea_config: containing configurations of MFEA-I (dict).
            kt_config: containing configurations for knowledge transfer (dict).
            tasks: list of tasks (list).
            super_config: configurations for super class (dict).
        '''
        if mfea_config.get('max_gen') is None:
            self.max_gen = np.inf
        else:
            self.max_gen = mfea_config['max_gen']
        if mfea_config.get('max_time') is None:
            self.max_time = np.inf
        else:
            self.max_time = mfea_config['max_time']
        if self.max_gen == np.inf and self.max_time == np.inf:
            raise ValueError('max_gen or max_time must be a finite value')
        if mfea_config.get('rmp') is None:
            self.rmp = 0.3
        else:
            self.rmp = mfea_config.get('rmp')
        if mfea_config.get('sbx_index') is None:
            self.sbx_index = 15
        else:
            self.sbx_index = mfea_config.get('sbx_index')
        if mfea_config.get('pm_index') is None:
            self.pm_index = 15
        else:
            self.pm_index = mfea_config.get('sbx_index')
        self.type = 'rk'
        if mfea_config.get('improve_type') is None:
            self.improve_type = 'probability'
        else:
            self.improve_type = mfea_config.get('improve_type')
        if mfea_config.get('improve_parameter') is None:
            self.improve_parameter = 1
        else:
            self.improve_parameter = mfea_config.get('improve_parameter')
        if mfea_config.get('use_ik') is None:
            self.use_ik = True
        else:
            self.use_ik = mfea_config.get('use_ik')

        self.kt_method = kt_config.get('kt_method')
        if kt_config.get('kt_interval') is None:
            self.kt_interval = 5
        else:
            self.kt_interval = kt_config.get('kt_interval')
        if kt_config.get('transfer_num') is None:
            self.transfer_num = 1
        else:
            self.transfer_num = kt_config.get('transfer_num')
        self.patching_strategy = kt_config.get('patching_strategy')
        self.model = kt_config.get('model')
        self.best_ratio = kt_config.get('best_ratio')

        super().__init__(tasks, super_config)


    def generate_offspring(self, **kwargs):
        '''
        Use assorsative mating to generate offspring population in MFEA-I.
        
        Parameters (contained in kwargs):
            permutate: permutate order of individuals before mating. (Boolean, default: False)
        Returns:
            offspring: offspring population (2-D np array)
            offspring_skill_factor: skill factors of each offspring (np array)
            offspring_factorial_cost: factorial cost of children (2-D np array)
        '''
        if kwargs.get('permutate') is None:
            permutate = False
        if permutate:
            permutation_index = np.random.permutation(self.pop_size)
            self.population = self.population[permutation_index]
            self.skill_factor = self.skill_factor[permutation_index]
            self.factorial_cost = self.factorial_cost[permutation_index]
            self.scalar_fitness = self.scalar_fitness[permutation_index]

        # generate offsprings
        offspring = self.population.copy()
        offspring_skill_factor = self.skill_factor.copy()
        offspring_factorial_cost = np.full(np.shape(self.factorial_cost), np.inf)
        if self.use_ik:
            # use_ik: perform assortative mating
            for i in range(0, self.pop_size, 2):
                p1 = self.population[i]
                p2 = self.population[i+1]
                skill_factor1, skill_factor2 = self.skill_factor[i], self.skill_factor[i+1]
                if skill_factor1 == skill_factor2 or np.random.rand() < self.rmp:
                    # generate 2 offsprings by crossover operator
                    offspring[i], offspring[i+1] = self.sbx_crossover(p1, p2)
                    if np.random.rand() < 0.5:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                    else:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                else:
                    # generate 2 offsprings by mumating each parent
                    offspring[i], offspring[i+1] = self.pm_mutate(p1), self.pm_mutate(p2)
                    offspring_skill_factor[i], offspring_skill_factor[i+1] = skill_factor1, skill_factor2
                offspring_factorial_cost[i,offspring_skill_factor[i]] = self.decoding(offspring_skill_factor[i], offspring[i])[1]
                offspring_factorial_cost[i+1,offspring_skill_factor[i+1]] = self.decoding(offspring_skill_factor[i+1], offspring[i+1])[1]
        else:
            # not use ik: genetic operators occur only among individuals of the same task
            raise ValueError('not support use_ik=False yet')
        return offspring, offspring_skill_factor, offspring_factorial_cost


    def run(self, neigh='insert', **kwarg):
        '''
        Run MFEA-I algorithm on given tasks

        Parameters:
            neigh: neighborhood used for local improvement (str, default: 'insert')
        '''
        start_time = time.process_time()
        # initialize population
        self.get_initial_population()
        num_gen = 1
        # main loop
        best_solution, best_value = [], []
        best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
        used_time = time.process_time() - start_time
        while num_gen <= self.max_gen and used_time < self.max_time:
            # generate offspring
            offspring, offspring_skill_factor, offspring_factorial_cost = self.generate_offspring(**kwarg)
            # perform local search on offspring
            offspring, offspring_factorial_cost = self.offspring_ls(offspring, offspring_skill_factor, offspring_factorial_cost, neigh, self.improve_type, self.improve_parameter)
            # perform knowledge transfer
            if self.kt_method == 'patching':
                transfer_parameters = {'transfer_num': self.transfer_num, 'patching_strategy': self.patching_strategy}
            elif self.kt_method == 'cl':
                transfer_parameters = {'model': self.model, 'best_ratio': self.best_ratio}
            else:
                transfer_parameters = {}
            transfer_population, transfer_skill_factor, transfer_factorial_cost = self.knowledge_transfer(num_gen, self.kt_interval, self.kt_method, **transfer_parameters)
            if transfer_population is not None:
                offspring = np.concatenate([offspring, transfer_population], axis=0)
                offspring_skill_factor = np.concatenate([offspring_skill_factor, transfer_skill_factor], axis=0)
                offspring_factorial_cost = np.concatenate([offspring_factorial_cost, transfer_factorial_cost], axis=0)
            # combine and update population
            self.population_updation(offspring, offspring_skill_factor, offspring_factorial_cost)
            best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
            num_gen += 1
            used_time = time.process_time() - start_time
        return {'best_value': best_value, 'running_time': time.process_time() - start_time, 'list': self.lists}
    

class MFEA2(EMT):
    '''
    Class for MFEA-II algorithm

    Attributes:
        Attributes in mfea2_config:
        max_gen: maximum generation (int).
        max_time: maximum CPU time in seconds (float).
        neigh: neighbourhood used for local improvement (str, default: 'insert').
        type: type of coding (fixed as 'rk')
        rmp: random mating probability (float).
        sbx_index: a parameter used in SBX (float, default: 15)
        pm_index: a parameter used in polynomial mutation (float, default: 15)
        improve_type: type of imporve ('str', default: 'probability')
        improve_parameter: parameter for improve. For 'probability', it is the probability of each individual's improving; for 'elite', it is the number of elite individuals. (float or int, default: 1)
        use_ik: use implicit knowledge transfer or not (Boolean, default: True)

        Attributes in kt_config:
        kt_method: method for knowledge transfer (str, default: None).
        kt_interval: interval of knowledge transfer (int).
        transfer_num: number of solutions to be transferred in 'patching' method (int)
        patching_strategy: strategy used for patching partial solution in 'patching" method (str).
        model: model used for centralized learning in 'ct' method (str, default: None).
        best_ratio: ratio of best individuals in current population in 'ct' method (float, default: 0.5).
    '''

    def __init__(self, mfea2_config, kt_config, tasks, super_config):
        '''
        Parameters:
            mfea2_config: containing configurations of MFEA-II (dict).
            kt_config: containing configurations for knowledge transfer (dict).
            tasks: list of tasks (list).
            super_config: configurations for super class (dict).
        '''
        if mfea2_config.get('max_gen') is None:
            self.max_gen = np.inf
        else:
            self.max_gen = mfea2_config['max_gen']
        if mfea2_config.get('max_time') is None:
            self.max_time = np.inf
        else:
            self.max_time = mfea2_config['max_time']
        if self.max_gen == np.inf and self.max_time == np.inf:
            raise ValueError('max_gen or max_time must be a finite value')
        if mfea2_config.get('rmp') is None:
            self.rmp = 0.3
        else:
            self.rmp = mfea2_config.get('rmp')
        if mfea2_config.get('sbx_index') is None:
            self.sbx_index = 15
        else:
            self.sbx_index = mfea2_config.get('sbx_index')
        if mfea2_config.get('pm_index') is None:
            self.pm_index = 15
        else:
            self.pm_index = mfea2_config.get('sbx_index')
        self.type = 'rk'
        if mfea2_config.get('improve_type') is None:
            self.improve_type = 'probability'
        else:
            self.improve_type = mfea2_config.get('improve_type')
        if mfea2_config.get('improve_parameter') is None:
            self.improve_parameter = 1
        else:
            self.improve_parameter = mfea2_config.get('improve_parameter')
        if mfea2_config.get('use_ik') is None:
            self.use_ik = True
        else:
            self.use_ik = mfea2_config.get('use_ik')

        self.kt_method = kt_config.get('kt_method')
        if kt_config.get('kt_interval') is None:
            self.kt_interval = 5
        else:
            self.kt_interval = kt_config.get('kt_interval')
        if kt_config.get('transfer_num') is None:
            self.transfer_num = 1
        else:
            self.transfer_num = kt_config.get('transfer_num')
        self.patching_strategy = kt_config.get('patching_strategy')
        self.model = kt_config.get('model')
        self.best_ratio = kt_config.get('best_ratio')

        super().__init__(tasks, super_config)

    # functions used to learn random mating probability matrix
    def log_likehood_v0(self, rmp, prob_matrix):
        '''calculate log likehood value in MFEA-II. This is version 0, only support 2 tasks
        
        Parameters:
            prob_matrix: probability matrix (np array)
        Returns:
            value: log likehood value
        '''
        posterior_matrix = prob_matrix.copy()
        value = 0
        for k in range(2):
            for j in range(2):
                if k == j:
                    posterior_matrix[k][:,j] = posterior_matrix[k][:, j] * (1 - 0.5 * (self.num_task - 1) * rmp / float(self.num_task))
                else:
                    posterior_matrix[k][:, j] = posterior_matrix[k][:, j] * 0.5 * (self.num_task - 1) * rmp / float(self.num_task)
            value += np.sum(-np.log(np.sum(posterior_matrix[k], axis=1)))
        return value


    def get_density(self, subpop, mean, std):
        '''get density distribution of subpopulation
        
        Parameters:
            subpop: sub population (2D np array)
            mean: mean value of norm distribution (array)
            std: std value of norm distribution (array)
        Returns:
            prob: probability distribution
        '''
        prob = np.ones([len(subpop)])
        for d in range(self.D):
            prob *= norm.pdf(subpop[:, d], loc=mean[d], scale=std[d])
        return prob
    

    def learn_rmp_v0(self):
        '''Online RMP learning for MFEA-II. This is version 0, only support 2 tasks.
        '''
        prob_models = []
        for skill_factor in range(self.num_task):
            # get sub population for each task
            subpop_index = np.where(self.skill_factor == skill_factor)[0]
            subpop = self.population[subpop_index]
            # calulate distribution models
            num_random_sample = int(np.floor(0.1 * len(subpop_index)))
            rand_pop = np.random.rand(num_random_sample, self.D)
            mean = np.mean(np.concatenate([subpop, rand_pop]), axis=0)
            std = np.std(np.concatenate([subpop, rand_pop]), axis=0)
            prob_models.append([subpop, mean, std])
        
        for k in range(self.num_task):
            for j in range(k+1, self.num_task):
                # get probability matrix
                probmatrix = [np.ones([len(prob_models[k][0]), 2]), 
                    np.ones([len(prob_models[j][0]), 2])]
                probmatrix[0][:,0] = self.get_density(prob_models[k][0], prob_models[k][1], prob_models[k][2])
                probmatrix[0][:,1] = self.get_density(prob_models[k][0], prob_models[j][1], prob_models[j][2])
                probmatrix[1][:,0] = self.get_density(prob_models[j][0], prob_models[k][1], prob_models[k][2])
                probmatrix[1][:,1] = self.get_density(prob_models[j][0], prob_models[j][1], prob_models[j][2])
                # optimize log likehood
                rmp = fminbound(lambda rmp: self.log_likehood_v0(rmp, probmatrix), 0, 1)
                rmp += np.random.randn() * 0.01
                rmp = np.clip(rmp, 0, 1)
        self.rmp = rmp


    def generate_offspring(self, **kwargs):
        '''
        Update random mating probability and use assorsative mating to generate offspring population in MFEA-II.
        
        Parameters (contained in kwargs):
            permutate: permutate order of individuals before mating. (Boolean, default: False)
        Returns:
            offspring: offspring population (2-D np array)
            offspring_skill_factor: skill factors of each offspring (np array)
            offspring_factorial_cost: factorial cost of children (2-D np array)
        '''
        if kwargs.get('permutate') is None:
            permutate = False
        if permutate:
            permutation_index = np.random.permutation(self.pop_size)
            self.population = self.population[permutation_index]
            self.skill_factor = self.skill_factor[permutation_index]
            self.factorial_cost = self.factorial_cost[permutation_index]
            self.scalar_fitness = self.scalar_fitness[permutation_index]

        # update random mating probability
        self.learn_rmp_v0()
        # generate offsprings
        offspring = self.population.copy()
        offspring_skill_factor = self.skill_factor.copy()
        offspring_factorial_cost = np.full(np.shape(self.factorial_cost), np.inf)
        if self.use_ik:
            # use_ik: perform assortative mating
            for i in range(0, self.pop_size, 2):
                p1 = self.population[i]
                p2 = self.population[i+1]
                skill_factor1, skill_factor2 = self.skill_factor[i], self.skill_factor[i+1]
                if skill_factor1 == skill_factor2 or np.random.rand() < self.rmp:
                    # generate 2 offsprings by crossover operator
                    offspring[i], offspring[i+1] = self.sbx_crossover(p1, p2)
                    if np.random.rand() < 0.5:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                    else:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                else:
                    # generate 2 offsprings by mumating each parent
                    offspring[i], offspring[i+1] = self.pm_mutate(p1), self.pm_mutate(p2)
                    offspring_skill_factor[i], offspring_skill_factor[i+1] = skill_factor1, skill_factor2
                offspring_factorial_cost[i,offspring_skill_factor[i]] = self.decoding(offspring_skill_factor[i], offspring[i])[1]
                offspring_factorial_cost[i+1,offspring_skill_factor[i+1]] = self.decoding(offspring_skill_factor[i+1], offspring[i+1])[1]
        else:
            # not use ik: genetic operators occur only among individuals of the same task
            raise ValueError('not support use_ik=False yet')
        return offspring, offspring_skill_factor, offspring_factorial_cost


    def run(self, neigh='insert', **kwarg):
        '''
        Run MFEA-II algorithm on given tasks

        Parameters:
            neigh: neighborhood used for local improvement (str, default: 'insert')
        '''
        start_time = time.process_time()
        # initialize population
        self.get_initial_population()
        num_gen = 1
        # main loop
        best_solution, best_value = [], []
        best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
        used_time = time.process_time() - start_time
        while num_gen <= self.max_gen and used_time < self.max_time:
            # generate offspring
            offspring, offspring_skill_factor, offspring_factorial_cost = self.generate_offspring(**kwarg)
            # perform local search on offspring
            offspring, offspring_factorial_cost = self.offspring_ls(offspring, offspring_skill_factor, offspring_factorial_cost, neigh, self.improve_type, self.improve_parameter)
            # perform knowledge transfer
            if self.kt_method == 'patching':
                transfer_parameters = {'transfer_num': self.transfer_num, 'patching_strategy': self.patching_strategy}
            elif self.kt_method == 'cl':
                transfer_parameters = {'model': self.model, 'best_ratio': self.best_ratio}
            else:
                transfer_parameters = {}
            transfer_population, transfer_skill_factor, transfer_factorial_cost = self.knowledge_transfer(num_gen, self.kt_interval, self.kt_method, **transfer_parameters)
            if transfer_population is not None:
                offspring = np.concatenate([offspring, transfer_population], axis=0)
                offspring_skill_factor = np.concatenate([offspring_skill_factor, transfer_skill_factor], axis=0)
                offspring_factorial_cost = np.concatenate([offspring_factorial_cost, transfer_factorial_cost], axis=0)
            # combine and update population
            self.population_updation(offspring, offspring_skill_factor, offspring_factorial_cost)
            best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
            num_gen += 1
            used_time = time.process_time() - start_time
        return {'best_value': best_value, 'running_time': time.process_time() - start_time, 'list': self.lists}
    

class GMFEA(EMT):
    '''
    Class for G-MFEA algorithm

    Attributes:
        Attributes in gmfea_config:
        max_gen: maximum generation (int).
        max_time: maximum CPU time in seconds (float).
        neigh: neighbourhood used for local improvement (str, default: 'insert').
        type: type of coding (fixed as 'rk')
        rmp: random mating probability (float).
        sbx_index: a parameter used in SBX (float, default: 15)
        pm_index: a parameter used in polynomial mutation (float, default: 15)
        improve_type: type of imporve ('str', default: 'probability')
        improve_parameter: parameter for improve. For 'probability', it is the probability of each individual's improving; for 'elite', it is the number of elite individuals. (float or int, default: 1)
        use_ik: use implicit knowledge transfer or not (Boolean, default: True)
        variable_transfer_frequence: frequency of performing variable transfer strategy (int, default: None)
        start_threshold: threshold of performing variable transfer strategy (int, default: None)

        Attributes in kt_config:
        kt_method: method for knowledge transfer (str, default: None).
        kt_interval: interval of knowledge transfer (int).
        transfer_num: number of solutions to be transferred in 'patching' method (int)
        patching_strategy: strategy used for patching partial solution in 'patching" method (str).
        model: model used for centralized learning in 'ct' method (str, default: None).
        best_ratio: ratio of best individuals in current population in 'ct' method (float, default: 0.5).
    '''

    def __init__(self, gmfea_config, kt_config, tasks, super_config):
        '''
        Parameters:
            gmfea_config: containing configurations of G-MFEA (dict).
            kt_config: containing configurations for knowledge transfer (dict).
            tasks: list of tasks (list).
            super_config: configurations for super class (dict).
        '''
        if gmfea_config.get('max_gen') is None:
            self.max_gen = np.inf
        else:
            self.max_gen = gmfea_config['max_gen']
        if gmfea_config.get('max_time') is None:
            self.max_time = np.inf
        else:
            self.max_time = gmfea_config['max_time']
        if self.max_gen == np.inf and self.max_time == np.inf:
            raise ValueError('max_gen or max_time must be a finite value')
        if gmfea_config.get('rmp') is None:
            self.rmp = 0.3
        else:
            self.rmp = gmfea_config.get('rmp')
        if gmfea_config.get('sbx_index') is None:
            self.sbx_index = 15
        else:
            self.sbx_index = gmfea_config.get('sbx_index')
        if gmfea_config.get('pm_index') is None:
            self.pm_index = 15
        else:
            self.pm_index = gmfea_config.get('sbx_index')
        self.type = 'rk'
        if gmfea_config.get('improve_type') is None:
            self.improve_type = 'probability'
        else:
            self.improve_type = gmfea_config.get('improve_type')
        if gmfea_config.get('improve_parameter') is None:
            self.improve_parameter = 1
        else:
            self.improve_parameter = gmfea_config.get('improve_parameter')
        if gmfea_config.get('use_ik') is None:
            self.use_ik = True
        else:
            self.use_ik = gmfea_config.get('use_ik')
        if gmfea_config.get('variable_transfer_frequence') is None:
            self.variable_transfer_frequence = 10
        else:
            self.variable_transfer_frequence = gmfea_config.get('variable_transfer_frequence')
        if gmfea_config.get('start_threshold') is None:
            self.start_threshold = 50
        else:
            self.start_threshold = gmfea_config.get('start_threshold')

        self.kt_method = kt_config.get('kt_method')
        if kt_config.get('kt_interval') is None:
            self.kt_interval = 5
        else:
            self.kt_interval = kt_config.get('kt_interval')
        if kt_config.get('transfer_num') is None:
            self.transfer_num = 1
        else:
            self.transfer_num = kt_config.get('transfer_num')
        self.patching_strategy = kt_config.get('patching_strategy')
        self.model = kt_config.get('model')
        self.best_ratio = kt_config.get('best_ratio')

        super().__init__(tasks, super_config)


    def variable_translation(self, current_cputime, scale_factor=1.25, center_point=[]):
        '''Decision variable translation strategy in GMFEA an MCEEA. In original experiment, translate_frequency=0.02*max_gen, strat_threshold=0.1*max_gen.
        
        Parameters:
            current_cputime: current CPU time or generation (float)
            scale_factor: parameter for translated direction (float, default: 1.25)
        Reutrns:
            translate_pop: translated population (2D np array)
            direction: direction for translation (2D np array)
        '''
        direction = np.zeros([self.num_task, self.D])
        if len(center_point) == 0:
            center_point = np.full(self.D, 0.5)
        if len(self.lists[0]) < self.start_threshold:
            return self.population, direction
        else:
            if len(self.lists[0]) % self.variable_transfer_frequence == 0:
                # calculate direction
                if self.max_gen is not None:
                    alpha = (len(self.lists[0]) / self.max_gen) ** 2
                else:
                    alpha = (current_cputime / self.max_time) ** 2
                for sf in range(self.num_task):
                    # select the 40% best solutions of each task and get thier averaged solution
                    index_for_sf = np.where(self.skill_factor == sf)[0]
                    num_best = int(np.ceil(0.2 * len(index_for_sf)))
                    best_pop_sf = self.population[index_for_sf[:num_best]]  # In main algorithm, population will be sorted before this step, so we don't need to do additional sort
                    estimated_optimum = np.mean(best_pop_sf, axis=0)
                    direction[sf] = scale_factor * alpha * (center_point - estimated_optimum)
            
            # translate indivuals
            translate_pop = np.empty([self.pop_size, self.D])
            for i in range(self.pop_size):
                p = self.population[i]
                sf = self.skill_factor[i]
                translate_pop[i,:] = p + direction[sf]
            return translate_pop, direction
        
    
    def variable_shuffling(self, p1_index, p2_index, translate_pop):
        '''Decision variable shuffling strategy in GMFEA and MECCA

        Parameters:
            p1_index & p2_index: indices of 2 parents
            translate_pop: translated population
        Returns:
            p1_new & p2_new: new parents
            order_list: order list for shuffling
        '''
        p1 = translate_pop[p1_index,:]
        p2 = translate_pop[p2_index,:]
        sf1 = self.skill_factor[p1_index]
        sf2 = self.skill_factor[p2_index]
        order_list = np.array([np.arange(self.D, dtype='int32')] * 2)

        p1_tem = p1
        sf2_tem = sf2
        p1_le_p2 = False  # whether p1 has more dimension than p2
        dimension_low = self.dimensions[sf2]
        if self.dimensions[sf2] < self.dimensions[sf1]:
            dimension_low = self.dimensions[sf2]
            p1_tem = p2
            sf2_tem = sf1
            p1_le_p2 = True

        [p1_new, p2_new] = [p1, p2]
        if self.dimensions[sf2] == self.dimensions[sf1]:
            return p1_new, p2_new, order_list
        else:
            # randomly select one individual with the same skill factor as p2_tem
            index_for_sf2 = np.where(self.skill_factor == sf2_tem)[0]
            pop_sf2 = self.population[index_for_sf2]
            p1_new = pop_sf2[np.random.randint(len(pop_sf2))]
            # assign p1_tem to p_tem
            np.random.shuffle(order_list[0])
            p1_new[order_list[0][:dimension_low]] = p1_tem[:dimension_low]
            if p1_le_p2:
                order_list = np.array([order_list[1], order_list[0]])
            return p1_new, p2_new, order_list
        
    
    def GMFEA_translate_back(self, offspring, skill_factor, direction, order_list):
        '''Translate individual back into original search space
        Parameters:
            offspring: individual to be translated (1-D np array)
            skill_factor: skill factor of given indiividual (int)
            direction: direction for decision variable translation (2D np array)
            order_list: order list for decision variable shuffling (1D np array)
        Returns:
            new_offspring: offspring in original search space
        '''
        offspring_tem = offspring - direction[skill_factor]
        offspring_new = offspring_tem[order_list]
        return offspring_new
    

    def generate_offspring(self, current_cputime, **kwargs):
        '''
        Apply variable transfer strategy, variable shuffling strategy and use assorsative mating to generate offspring population in G-MFEA.
        
        Parameters:
            current_cputime: current CPU time (float)
            Contained in kwargs:
            permutate: permutate order of individuals before mating. (Boolean, default: False)
        Returns:
            offspring: offspring population (2-D np array)
            offspring_skill_factor: skill factors of each offspring (np array)
            offspring_factorial_cost: factorial cost of children (2-D np array)
        '''
        if kwargs.get('permutate') is None:
            permutate = False
        if permutate:
            permutation_index = np.random.permutation(self.pop_size)
            self.population = self.population[permutation_index]
            self.skill_factor = self.skill_factor[permutation_index]
            self.factorial_cost = self.factorial_cost[permutation_index]
            self.scalar_fitness = self.scalar_fitness[permutation_index]

        # perform variable translation
        variable_transfer_pop, direction = self.variable_translation(current_cputime)
        # generate offsprings
        offspring = variable_transfer_pop.copy()
        offspring_skill_factor = self.skill_factor.copy()
        offspring_factorial_cost = np.full(np.shape(self.factorial_cost), np.inf)
        if self.use_ik:
            # use_ik: perform assortative mating
            for i in range(0, self.pop_size, 2):
                p1_shuffle, p2_shuffle, order_list = self.variable_shuffling(i, i+1, variable_transfer_pop)
                skill_factor1, skill_factor2 = self.skill_factor[i], self.skill_factor[i+1]
                if skill_factor1 == skill_factor2 or np.random.rand() < self.rmp:
                    # generate 2 offsprings by crossover operator
                    offspring[i], offspring[i+1] = self.sbx_crossover(p1_shuffle, p2_shuffle)
                    if np.random.rand() < 0.5:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                    else:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                else:
                    # generate 2 offsprings by mumating each parent
                    offspring[i], offspring[i+1] = self.pm_mutate(p1_shuffle), self.pm_mutate(p2_shuffle)
                    offspring_skill_factor[i], offspring_skill_factor[i+1] = skill_factor1, skill_factor2
                offspring[i] = self.GMFEA_translate_back(offspring[i], offspring_skill_factor[i], direction, order_list[0])
                offspring[i+1] = self.GMFEA_translate_back(offspring[i+1], offspring_skill_factor[i+1], direction, order_list[1])
                offspring_factorial_cost[i,offspring_skill_factor[i]] = self.decoding(offspring_skill_factor[i], offspring[i])[1]
                offspring_factorial_cost[i+1,offspring_skill_factor[i+1]] = self.decoding(offspring_skill_factor[i+1], offspring[i+1])[1]
        else:
            # not use ik: genetic operators occur only among individuals of the same task
            raise ValueError('not support use_ik=False yet')
        return offspring, offspring_skill_factor, offspring_factorial_cost
    

    def run(self, neigh='insert', **kwarg):
        '''
        Run G-MFEA algorithm on given tasks

        Parameters:
            neigh: neighborhood used for local improvement (str, default: 'insert')
        '''
        start_time = time.process_time()
        # initialize population
        self.get_initial_population()
        num_gen = 1
        # main loop
        best_solution, best_value = [], []
        best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
        used_time = time.process_time() - start_time
        while num_gen <= self.max_gen and used_time < self.max_time:
            # generate offspring
            offspring, offspring_skill_factor, offspring_factorial_cost = self.generate_offspring(used_time, **kwarg)
            # perform local search on offspring
            offspring, offspring_factorial_cost = self.offspring_ls(offspring, offspring_skill_factor, offspring_factorial_cost, neigh, self.improve_type, self.improve_parameter)
            # perform knowledge transfer
            if self.kt_method == 'patching':
                transfer_parameters = {'transfer_num': self.transfer_num, 'patching_strategy': self.patching_strategy}
            elif self.kt_method == 'cl':
                transfer_parameters = {'model': self.model, 'best_ratio': self.best_ratio}
            else:
                transfer_parameters = {}
            transfer_population, transfer_skill_factor, transfer_factorial_cost = self.knowledge_transfer(num_gen, self.kt_interval, self.kt_method, **transfer_parameters)
            if transfer_population is not None:
                offspring = np.concatenate([offspring, transfer_population], axis=0)
                offspring_skill_factor = np.concatenate([offspring_skill_factor, transfer_skill_factor], axis=0)
                offspring_factorial_cost = np.concatenate([offspring_factorial_cost, transfer_factorial_cost], axis=0)
            # combine and update population
            self.population_updation(offspring, offspring_skill_factor, offspring_factorial_cost)
            best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
            num_gen += 1
            used_time = time.process_time() - start_time
        return {'best_value': best_value, 'running_time': time.process_time() - start_time, 'list': self.lists}
    

class PMFEA(EMT):
    '''
    Class for P-MFEA algorithm

    Attributes:
        Attributes in pmfea_config:
        max_gen: maximum generation (int).
        max_time: maximum CPU time in seconds (float).
        neigh: neighbourhood used for local improvement (str, default: 'insert').
        rmp: random mating probability (float).
        type: type of coding (fixed as 'perm')
        improve_type: type of imporve ('str', default: 'probability')
        improve_parameter: parameter for improve. For 'probability', it is the probability of each individual's improving; for 'elite', it is the number of elite individuals. (float or int, default: 1)
        use_ik: use implicit knowledge transfer or not (Boolean, default: True)

        Attributes in kt_config:
        kt_method: method for knowledge transfer (str, default: None).
        kt_interval: interval of knowledge transfer (int).
        transfer_num: number of solutions to be transferred in 'patching' method (int)
        patching_strategy: strategy used for patching partial solution in 'patching" method (str).
        model: model used for centralized learning in 'ct' method (str, default: None).
        best_ratio: ratio of best individuals in current population in 'ct' method (float, default: 0.5).
    '''

    def __init__(self, pmfea_config, kt_config, tasks, super_config):
        '''
        Parameters:
            pmfea_config: containing configurations of P-MFEA (dict).
            kt_config: containing configurations for knowledge transfer (dict).
            tasks: list of tasks (list).
            super_config: configurations for super class (dict).
        '''
        if pmfea_config.get('max_gen') is None:
            self.max_gen = np.inf
        else:
            self.max_gen = pmfea_config['max_gen']
        if pmfea_config.get('max_time') is None:
            self.max_time = np.inf
        else:
            self.max_time = pmfea_config['max_time']
        if self.max_gen == np.inf and self.max_time == np.inf:
            raise ValueError('max_gen or max_time must be a finite value')
        if pmfea_config.get('rmp') is None:
            self.rmp = 0.7
        else:
            self.rmp = pmfea_config.get('rmp')
        self.type = 'perm'
        if pmfea_config.get('improve_type') is None:
            self.improve_type = 'probability'
        else:
            self.improve_type = pmfea_config.get('improve_type')
        if pmfea_config.get('improve_parameter') is None:
            self.improve_parameter = 0.1
        else:
            self.improve_parameter = pmfea_config.get('improve_parameter')
        if pmfea_config.get('use_ik') is None:
            self.use_ik = True
        else:
            self.use_ik = pmfea_config.get('use_ik')

        self.kt_method = kt_config.get('kt_method')
        if kt_config.get('kt_interval') is None:
            self.kt_interval = 5
        else:
            self.kt_interval = kt_config.get('kt_interval')
        if kt_config.get('transfer_num') is None:
            self.transfer_num = 1
        else:
            self.transfer_num = kt_config.get('transfer_num')
        self.patching_strategy = kt_config.get('patching_strategy')
        self.model = kt_config.get('model')
        self.best_ratio = kt_config.get('best_ratio')

        super().__init__(tasks, super_config)


    def generate_offspring(self, **kwargs):
        '''
        Use assorsative mating to generate offspring population in P-MFEA.
        
        Parameters (contained in kwargs):
            permutate: permutate order of individuals before mating. (Boolean, default: False)
        Returns:
            offspring: offspring population (2-D np array)
            offspring_skill_factor: skill factors of each offspring (np array)
            offspring_factorial_cost: factorial cost of children (2-D np array)
        '''
        if kwargs.get('permutate') is None:
            permutate = False
        if permutate:
            permutation_index = np.random.permutation(self.pop_size)
            self.population = self.population[permutation_index]
            self.skill_factor = self.skill_factor[permutation_index]
            self.factorial_cost = self.factorial_cost[permutation_index]
            self.scalar_fitness = self.scalar_fitness[permutation_index]

        # generate offsprings
        offspring = self.population.copy()
        offspring_skill_factor = self.skill_factor.copy()
        offspring_factorial_cost = np.full(np.shape(self.factorial_cost), np.inf)
        if self.use_ik:
            # use_ik: perform assortative mating
            for i in range(0, self.pop_size, 2):
                p1 = self.population[i]
                p2 = self.population[i+1]
                skill_factor1, skill_factor2 = self.skill_factor[i], self.skill_factor[i+1]
                if skill_factor1 == skill_factor2 or np.random.rand() < self.rmp:
                    # generate 2 offsprings by crossover operator
                    offspring[i], offspring[i+1] = self.ox1(p1, p2)
                    if np.random.rand() < 0.5:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                    else:
                        offspring_skill_factor[i] = skill_factor1
                        offspring_skill_factor[i+1] = skill_factor2
                else:
                    # generate 2 offsprings by mumating each parent
                    offspring[i], offspring[i+1] = self.swap_mutate(p1), self.swap_mutate(p2)
                    offspring_skill_factor[i], offspring_skill_factor[i+1] = skill_factor1, skill_factor2
                offspring_factorial_cost[i,offspring_skill_factor[i]] = self.decoding(offspring_skill_factor[i], offspring[i])[1]
                offspring_factorial_cost[i+1,offspring_skill_factor[i+1]] = self.decoding(offspring_skill_factor[i+1], offspring[i+1])[1]
        else:
            # not use ik: genetic operators occur only among individuals of the same task
            raise ValueError('not support use_ik=False yet')
        return offspring, offspring_skill_factor, offspring_factorial_cost


    def run(self, neigh='insert', **kwarg):
        '''
        Run P-MFEA algorithm on given tasks

        Parameters:
            neigh: neighborhood used for local improvement (str, default: 'insert')
        '''
        start_time = time.process_time()
        # initialize population
        self.get_initial_population()
        num_gen = 1
        # main loop
        best_solution, best_value = [], []
        best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
        used_time = time.process_time() - start_time
        while num_gen <= self.max_gen and used_time < self.max_time:
            # generate offspring
            offspring, offspring_skill_factor, offspring_factorial_cost = self.generate_offspring(**kwarg)
            # perform local search on offspring
            offspring, offspring_factorial_cost = self.offspring_ls(offspring, offspring_skill_factor, offspring_factorial_cost, neigh, self.improve_type, self.improve_parameter)
            # perform knowledge transfer
            if self.kt_method == 'patching':
                transfer_parameters = {'transfer_num': self.transfer_num, 'patching_strategy': self.patching_strategy}
            elif self.kt_method == 'cl':
                transfer_parameters = {'model': self.model, 'best_ratio': self.best_ratio}
            else:
                transfer_parameters = {}
            transfer_population, transfer_skill_factor, transfer_factorial_cost = self.knowledge_transfer(num_gen, self.kt_interval, self.kt_method, **transfer_parameters)
            if transfer_population is not None:
                offspring = np.concatenate([offspring, transfer_population], axis=0)
                offspring_skill_factor = np.concatenate([offspring_skill_factor, transfer_skill_factor], axis=0)
                offspring_factorial_cost = np.concatenate([offspring_factorial_cost, transfer_factorial_cost], axis=0)
            # combine and update population
            self.population_updation(offspring, offspring_skill_factor, offspring_factorial_cost)
            best_solution, best_value = self.sort_best_from_pop(best_solution, best_value)
            num_gen += 1
            used_time = time.process_time() - start_time
        return {'best_value': best_value, 'running_time': time.process_time() - start_time, 'list': self.lists}