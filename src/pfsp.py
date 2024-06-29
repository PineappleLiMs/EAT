"""
Class for permutation flowshop scheduling problem.
"""

import numpy as np
import random
import json
from cysource import calculations


def import_instance(path):
    """import instance from json file

    Parameters:
        path: path of json file
    Returns:
        instance: a dictionary containing all information of an instance. For PFSP benchamrk, its keys include 'matrix', 'num_job','num_mac', 'opt_index', 'LB', 'UB', 'bsf'
    """
    with open(path, 'r') as file:
        instance = json.load(file)
    return instance

def get_intertask_dm(matrix1, matrix2):
    """Calculate distance metrix between two PFSPs which own some size.

    Parameters:
        matrix1: processing time matrix of a PFSP (2-d np array).
        matrix2: processing time matrix of another PFSP (2-d np array).
    Returns:
        distance metrix between given two problems
    """
    matrix1_tmp = matrix1 - np.mean(matrix1)
    matrix2_tmp = matrix2 - np.mean(matrix2)
    tmp = np.sum(matrix1_tmp * matrix2_tmp) / (np.linalg.norm(matrix1_tmp) * np.linalg.norm(matrix2_tmp))
    return min(1, (1 - tmp) / np.sqrt(1 - tmp * tmp))
    # n = np.size(matrix1, axis=0)
    # m = np.size(matrix2, axis=1)
    # matrix1_tmp = matrix1 - np.mean(matrix1)
    # matrix2_tmp = matrix2 - np.mean(matrix2)
    # t0_1 = n*m*np.sum(matrix1*matrix2) - np.sum(matrix2)*np.sum(matrix2)
    # t0_2 = n*m*np.sum(matrix1*matrix1) - (np.sum(matrix1))**2
    # t0 = max(t0_1 / t0_2, 0)

    # tmp1 = np.linalg.norm(matrix1_tmp) - t0 * np.linalg.norm(matrix2_tmp)
    # tmp2 = np.linalg.norm(matrix1_tmp - t0*matrix2_tmp)
    # if tmp2 == 0:
    #     return 0
    # else:
    #     return tmp1 / tmp2


class Pfsp(object):
    """read instance with dictionary format, build functions and data for problem and solution

    Attributes:
        num_job: number of jobs (int).
        num_mac: number of machines (int).
        matrix: processing time matrix (2d np array).
        opt_index: whether optimal solution has been found or not (bool).
        LB: lower bound of optimal solution (int).
        UB: upper bound of optimal solution (int).
        makespan: makespan of given sequence (int, default: 0).
        idle_time: current idle time between jobs (int, default:0).
        sequence: sequence of jobs (np array, default:[]).
        type: orignal or economic type (str, default: 'orignal')
        eco_index: index of jobs for economic task (np array or list)
        bsf: best-so-far sequence
    """

    def __init__(self, instance):
        """import instance file and initialize variables

        Parameters:
            instance: dictionary containing all information of the instance (dict).
        """
        self.matrix = np.array(instance['matrix'], dtype='int32')
        if 'num_job' in instance:
            self.num_job = instance['num_job']
        else:
            self.num_job = len(self.matrix)
        if 'num_mac' in instance:
            self.num_mac = instance['num_mac']
        else:
            self.num_mac = len(self.matrix[0])
        if 'opt_index' in instance:
            self.opt_index = instance['opt_index']
        else:
            self.opt_index = None
        if 'LB' in instance:
            self.LB = instance['LB']
        else:
            self.LB = None
        if 'UB' in instance:
            self.UB = instance['UB']
        else:
            self.UB = None
        self.makespan = 0
        self.idle_time = 0
        self.sequence = np.array([], dtype='int32')
        self.eco_type = 'orignal'
        self.eco_index = np.arange(self.num_job) + 1
        if 'bsf' in instance and len(instance['bsf']) != 0:
            self.bsf = np.array(instance['bsf'][0], dtype='int32')
        else:
            self.bsf = np.array([], dtype='int32')
        
        
    def calculate_completion_times(self, sequence = []):
        """Calculate completion time."""
        if len(sequence) == 0:
            sequence = self.sequence
        memory_view_object = calculations.calculate_completion_times(sequence, self.matrix, self.num_mac, 1)
        return np.array(memory_view_object)[1:,1:]

    def calculate_makespan(self, sequence = None):
        """Calculate makespan for the sequence.

        Parameters:
            sequence: job sequence (np arrayt. default: None, which means self.sequence).
        Returns:
            self.makespan / makespan: makespan of self.sequence / given sequence
        """
        if sequence is None:
            self.makespan = calculations.calculate_completion_times(self.sequence, self.matrix, self.num_mac, 0)
            return self.makespan
        else:
            return calculations.calculate_completion_times(sequence, self.matrix, self.num_mac, 0)
        # self_flag = False
        # if len(sequence) == 0:
        #     sequence = self.sequence
        #     self_flag = True
        #     length = self.num_job
        # else:
        #     length = len(sequence)
        # ct = np.zeros((length, self.num_mac))
        # ct[0][0] = self.matrix[sequence[0] - 1][0]
        # for j in range(1, self.num_mac):
        #     ct[0][j] = ct[0][j - 1] + self.matrix[sequence[0] - 1][j]
        # for i in range(1, length):
        #     ct[i][0] = ct[i - 1][0] + self.matrix[sequence[i] - 1][0]
        # for i in range(1, length):
        #     for j in range(1, self.num_mac):
        #         ct[i][j] = max(ct[i - 1][j], ct[i][j - 1]) + self.matrix[sequence[i] - 1][j]
        # if self_flag:
        #     self.makespan = ct[-1][-1]
        # return ct[-1][-1]


    def insert_best_position(self, job, sequence = [], tie_breaking=0):
        """ Insert the given job in the position that minimize makespan.
        Parameters:
            job: job to be inserted (int).
            sequence: sequence to be inserted (np array, default: [], meaning self.sequence)
            tie_breaking: use tie breaking mechanism (0 or 1, default: 0).
        Returns:
            new_sequence: sequence after insertion (np array)
            makespan: makespan after inserting the job (int)
        """
        self_mark = False
        if len(sequence) == 0:
            self_mark = True
            sequence = self.sequence    
        best_position, makespan = calculations.taillard_acceleration(sequence, self.matrix, job, self.num_mac, tie_breaking)
        new_sequence = np.insert(sequence.copy(), best_position - 1, job)
        if self_mark:
            # if self.sequence if used, then update self.sequence and self.makespan
            self.sequence = new_sequence
            self.makespan = makespan
        return new_sequence, makespan

    # def insert_best_position(self, job, sequence = [], tie_breaking=0):
    #     """ Insert the given job in the position that minimize makespan.
    #     Parameters:
    #         job: job to be inserted (int).
    #         sequence: sequence to be inserted (np array, default: [], meaning self.sequence)
    #         tie_breaking: use tie breaking mechanism (0 or 1, default: 0).
    #     Returns:
    #         new_sequence: sequence after insertion (np array)
    #         makespan: makespan after inserting the job (int)
    #     """
    #     self_flag = False
    #     if len(sequence) == 0:
    #         sequence = self.sequence
    #         self_flag = True
    #     length = len(sequence)
    #     e = np.zeros((length, self.num_mac))  # earliest completion time
    #     q = np.zeros((length + 1, self.num_mac))  # tail
    #     f = np.zeros((length + 1, self.num_mac))  # earliest relative completion time
    #     e[0][0] = self.matrix[sequence[0] - 1][0]
    #     for i in range(1, length):
    #         e[i][0] = e[i - 1][0] + self.matrix[sequence[i] - 1][0]
    #     for j in range(1, self.num_mac):
    #         e[0][j] = e[0][j - 1] + self.matrix[sequence[0] - 1][j]
    #     for i in range(1, length):
    #         for j in range(1, self.num_mac):
    #             e[i][j] = max(e[i - 1][j], e[i][j - 1]) + self.matrix[sequence[i] - 1][j]
    #     for j in range(self.num_mac):
    #         q[length][j] = 0
    #     for i in range(length - 1, -1, -1):
    #         q[i][self.num_mac - 1] = q[i + 1][self.num_mac - 1] + self.matrix[sequence[i] - 1][self.num_mac - 1]
    #     for i in range(length - 1, -1, -1):
    #         for j in range(self.num_mac - 2, -1, -1):
    #             q[i][j] = max(q[i + 1][j], q[i][j + 1]) + self.matrix[sequence[i] - 1][j]
    #     f[0][0] = self.matrix[job - 1][0]
    #     for i in range(1, length + 1):
    #         f[i][0] = e[i - 1][0] + self.matrix[job - 1][0]
    #     for j in range(1, self.num_mac):
    #         f[0][j] = f[0][j - 1] + self.matrix[job - 1][j]
    #     for i in range(1, length + 1):
    #         for j in range(1, self.num_mac):
    #             f[i][j] = max(f[i][j - 1], e[i - 1][j]) + self.matrix[job - 1][j]
    #     m = np.amax(f + q, axis=1)
    #     best_makespan = np.amin(m)
    #     best_position_tem = np.where(m == best_makespan)
    #     best_position = best_position_tem[0][0]
    #     new_sequence = np.insert(sequence.copy(), best_position - 1, job)
    #     if self_flag:
    #         self.sequence = new_sequence
    #         self.makespan = best_makespan
    #     return new_sequence, best_makespan


    def NEH(self, tie_breaking=0, order_jobs='SD', given_order=[]):
        """get a solution with NEH heuristic
        Parameters:
            tie_breaking: use tie breaking mechanism (0 or 1, default: 0).
            order_jobs: priority order of jobs, possible values are:
                    SD: non-decreasing sum of processing times (default);
                    RD: random order.
            given_order: if other values are assigned to order_jobs, given_order will be selected as priority order. (array-like)
        Returns:
            sequence: NEH sequence (np array)
            makespan: makespan of NEH sequence (int)
        """
        # set job order
        if order_jobs == 'SD':
            total_processing_times = dict()
            for i in range(1, len(self.matrix)+1):
                if self.eco_type == 'eco' and i not in self.eco_index[:self.num_job]:
                    continue
                total_processing_times[i] = np.sum(self.matrix[i-1])
            sorted_jobs = sorted(total_processing_times, key=total_processing_times.get, reverse=True)
        elif order_jobs == 'RD':
            if self.eco_type == 'eco':
                sorted_jobs = self.eco_index[:self.num_job]
            else:
                sorted_jobs = list(range(1, self.num_job+1))
            random.shuffle(sorted_jobs)
        else:
            sorted_jobs = given_order
        # take jobs in order_jobs and insert them in turn in the place which minimize partial makespan
        sequence = np.array([sorted_jobs[0], sorted_jobs[1]], dtype = 'int32')
        makespan_tmp = self.calculate_makespan(sequence)
        sequence = np.array([sorted_jobs[1], sorted_jobs[0]], dtype = 'int32')
        if makespan_tmp < self.calculate_makespan(sequence):
            sequence = np.array([sorted_jobs[0], sorted_jobs[1]], dtype = 'int32')
            makespan = makespan_tmp
        for job in sorted_jobs[2:]:
            sequence, makespan = self.insert_best_position(job, sequence, tie_breaking)
        self.sequence = sequence
        self.makespan = makespan
        return sequence, makespan


    def neh_variants(self, method):
        '''variants of NEH heuristics.
        
        Parameters:
            method: method to generate priority index (str)
        Returns:
            sequence: sequence obtained (array-like)
            value: fitness value of sequence (int)
            priority_index: priority order used in NEH variants (array-like)
        '''
        if method == 'kk1':
            # NEHKK1
            c = dict()
            for i in range(1, len(self.matrix)+1):
                if self.eco_type == 'eco' and i not in self.eco_index[:self.num_job]:
                    continue
                a, b = 0, 0
                for j in range(self.num_mac):
                    a = a + ((self.num_mac-1) * (self.num_mac-2) / 2 + self.num_mac - j) * self.matrix[i-1,j]
                    b = b + ((self.num_mac-1) * (self.num_mac-2) / 2 + j - 1) * self.matrix[i-1,j]
                c[i] = a if a <= b else b
            priority_index = sorted(c, key=c.get, reverse=True)
        elif method == 'kk2':
            # NEHKK2
            s, t = int(np.floor(self.num_mac / 2)), int(np.ceil(self.num_mac / 2))
            c = dict()
            for i in range(1, len(self.matrix)+1):
                if self.eco_type == 'eco' and i not in self.eco_index[:self.num_job]:
                    continue
                u = 0
                for h in range(s):
                    u += (h-3/4) / (s-3/4) * (self.matrix[i-1, s+1-h] - self.matrix[i-1,t+h])
                a = np.sum(self.matrix[i-1]) + u
                b = np.sum(self.matrix[i-1]) - u
                c[i] = a if a <= b else b
            priority_index = sorted(c, key=c.get, reverse=True)
        sequence, makespan = self.NEH(order_jobs='', given_order=priority_index)
        return sequence, makespan, priority_index


    def local_search(self, neigh, max_iteration, sequence=None, makespan=None, temperature=None):
        """perform a complete local search
        
        Parameters:
            neigh: neighbourhood for local search (str).
            search_rule: rule for local search (str, default: 'first_improve')
            max_iteration: maximum iteration of one-step local search (int)
            sequence: job sequence (np arrayt. default: None, which means self.sequence).
            makespan: makespan of given sequence (int, default: None)
            temperature: temperature to control whether or not accept better solution (floet, default: None, which means accepting every better solution).
        Returns:
            new_sequence: new sequence after search
            new_makespan: makespan of new sequence
        """
        self_mark = False
        if sequence is None:
            sequence = self.sequence
            makespan = self.calculate_makespan()
            self_mark = True
        if makespan is None:
            makespan = self.calculate_makespan(sequence)
        # local search in given neighbourhood
        for _ in range(max_iteration):
            new_sequence = sequence.copy()
            if neigh == 'insert':
                index1 = random.randint(0, len(sequence) - 1)
                index2 = index1
                while index2 == index1:
                    index2 = random.randint(0, len(sequence) - 1)
                if index1 < index2:
                    new_sequence[index1:index2+1] = np.append(sequence[index1+1:index2+1], sequence[index1])
                if index1 > index2:
                    new_sequence[index2:index1+1] = np.append(sequence[index1], sequence[index2:index1])
            elif neigh == 'swap':
                index1 = random.randint(0, len(sequence) - 1)
                index2 = index1
                while index2 == index1:
                    index2 = random.randint(0, len(sequence) - 1)
                new_sequence[index1], new_sequence[index2] = sequence[index2], sequence[index1]
            elif neigh == 'inverse':
                sequence_tem = sequence[index1:index2+1]
                new_sequence[index1:index2+1] = sequence_tem[::-1]
            '''
            elif neigh == 'ig':
                # something wrong in this setting, do NOT use this option!
                index = random.randint(0, len(sequence) - 1)
                sequence_tem = new_sequence[:]
                np.delete(sequence_tem, np.where(sequence_tem==index))
                sequence_tem, makespan_tem = self.insert_best_position(index, sequence_tem)
            '''
            # accept new solution or not
            new_makespan = self.calculate_makespan(new_sequence)
            if new_makespan <= makespan:
                sequence = new_sequence.copy()
                makespan = new_makespan
            else:
                if temperature is not None:
                    diff = new_makespan - makespan
                    if random.random() < np.exp(-diff / temperature):
                        sequence = new_sequence.copy()
                        makespan = new_makespan
        new_sequence = sequence.copy()
        new_makespan = makespan
        # update attributes if self.sequence is used
        if self_mark:
            self.makespan = new_makespan
            self.sequence = new_sequence.copy()
        return new_sequence, new_makespan


    def get_eco(self, num_select, method='LSP', rows=[]):
        """Get economic task based on given methods or given rows (if self.type == 'eco', better not use this function)

        Parameters:
            num_select: number of jobs to be selected
            method: method to selection rows (str, default: 'dss')
            rows: rows that make up economic task (list or np array, default: [])
        Returns:
            eco_task: a pfsp objective of economic task (Pfsp object)
        """
        if method in ['LSP', 'lsp']:
            # descending order of square sum
            square_sum = np.sum(self.matrix * self.matrix, axis=1)
            index = np.argsort(square_sum)[::-1]
            rows = index[:num_select] + 1
        elif method in ['LST', 'lst']:
            # decending order of sum
            sum_matrix = np.sum(self.matrix, axis=1)
            index = np.argsort(sum_matrix)[::-1]
            rows = index[:num_select] + 1
        elif method in ['RND', 'rnd']:
            # random sample rows
            index = np.arange(0,self.num_job)
            random.shuffle(index)
            rows = index.copy()  + 1
            rows = rows[:num_select].copy()
        elif method == 'sl':
            # slope index
            s = {}
            for i in range(self.num_job):
                s[i] = (self.num_mac+1-2 * np.arange(1,self.num_mac+1)) @ self.matrix[i].T
            index = np.array(sorted(s, key=s.get, reverse=True), dtype='int32')
            rows = index[:num_select] + 1
        elif method in ['KK1', 'kk1']:
            # priority order used in NEHKK1
            _,_,index = self.neh_variants('kk1')
            index = np.array(index, dtype='int32') - 1
            rows = index[:num_select] + 1
        elif method in ['KK2', 'kk2']:
            # priority order used in NEHKK2
            _,_,index = self.neh_variants('kk2')
            index = np.array(index, dtype='int32') - 1
            rows = index[:num_select] + 1
        elif method in ['SR0', 'sr0']:
            # sequence obtained by NEH heuristics
            row_index, _ = self.NEH()
            index = row_index - 1
            rows = row_index[:num_select]
        elif method in ['SR1', 'sr1']:
            # sequence obtained by NEHKK1 heuristics
            row_index,_,_ = self.neh_variants('kk1')
            index = row_index - 1
            rows = row_index[:num_select]
        elif method in ['SR2', 'sr2']:
            # sequence obtianed by NEHKK2 heuristics
            row_index,_,_ = self.neh_variants('kk2')
            index = row_index - 1
            rows = row_index[:num_select]
        else:
            # if a set of given rows are wanted, input '' as method and a list of indices as rows. Make sure the length of rows is num_select
            rows = np.array(rows)
            index = rows - 1
            unselect = np.setdiff1d(np.arange(self.num_job), index, assume_unique=True)
            index = np.concatenate((index, unselect))
        
        eco_matrix = self.matrix.copy()
        for i in range(self.num_job):
            if (i+1) not in rows:
                eco_matrix[i] = np.zeros(self.num_mac)
        
        eco = Pfsp({'matrix': eco_matrix, 'num_job': num_select, 'num_mac': self.num_mac, 'opt_index': None, 'LB': None, 'UB': None, 'bsf': []})
        eco.eco_type = 'eco'
        eco.eco_index = (index + 1).astype(np.int32)
        return eco


    def p2p_transfer(self, target_task, sequence=[], method='RI'):
        '''
        transfer solution from eco/orignal task to task of another type

        Parameters:
            target_task: target task (pfsp object)
            sequence: sequence to be transferred (np.array, default: [], meaning self.sequence)
            method: transfer method ('str', default: 'best_position')
        Returns:
            target_sequence: sequence on target task after transfer
            target_makespan: makespan of target_sequence on target task
        '''
        if len(sequence) == 0:
            sequence = self.sequence.copy()
        if self.eco_type == 'orignal':
            # orignal to eco: delect unselected jobs
            target_sequence = []
            for i in sequence:
                if i in target_task.eco_index[:target_task.num_job]:
                    target_sequence.append(i)
            target_sequence = np.array(target_sequence, dtype='int32')
            target_makespan = target_task.calculate_makespan(target_sequence)
        else:
            # eco to orignal: insert unselect jobs into sequence
            target_sequence = sequence.copy()
            target_sequence = target_sequence.astype(np.int32)
            if method in ['RI', 'ri']:
                # recursely insertion: 
                target_makespan = 0
                for unselect_job in self.eco_index[self.num_job:]:
                    unselect_job = int(unselect_job)
                    target_sequence, target_makespan = target_task.insert_best_position(unselect_job, target_sequence, tie_breaking=0)
            elif method in ['EI', 'ei']:
                # insert at the end
                target_sequence = np.append(target_sequence, self.eco_index[self.num_job:])
                target_sequence = target_sequence.astype(np.int32)
                target_makespan = target_task.calculate_makespan(target_sequence)
            elif method in ['OI', 'oi']:
                # insert the selected job at the end if length of current sequence is odd, else insert it at position 0. See: "J. N. D. Gupta, “Heuristic Algorithms for Multistage Flowshop Scheduling Problem,” A I I E Transactions, vol. 4, no. 1, pp. 11-18, 1972/03/01, 1972."
                for unselect_job in self.eco_index[self.num_job:]:
                    if (len(target_sequence) % 2) == 1:
                        # length of current sequence is even
                         target_sequence = np.append(unselect_job,target_sequence)
                    else:
                        target_sequence = np.append(target_sequence, unselect_job)
                target_sequence = target_sequence.astype(np.int32)
                target_makespan = target_task.calculate_makespan(target_sequence)
            elif method in ['AI', 'ai']:
                # insert the selected job at a arbitrary position
                for unselect_job in self.eco_index[self.num_job:]:
                    position = random.randint(0,len(target_sequence))
                    target_sequence = np.insert(target_sequence, position, unselect_job)
                target_sequence = target_sequence.astype(np.int32)
                target_makespan = target_task.calculate_makespan(target_sequence)
            else:
                raise ValueError('not support current method:' + method)

        return target_sequence, target_makespan