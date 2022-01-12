#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Protein-ligand docking rescoring with OnionNet-SFCT.

Author:
    Liangzhen Zheng - June 4, 2021

Contact:
   astrozheng@gmail.com
"""

from multiprocessing import Pool, cpu_count

class ParallelSim(object):
    '''
    Run parallel featurization with multiprocess map function with progress reported.
    https://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call/41549217
    '''
    def __init__(self, processes=cpu_count(), verbose=False):
        self.pool = Pool(processes=processes)
        self.total_processes = 0
        self.completed_processes = 0
        self.results = []
        self.v = verbose

    def add(self, func, args):
        self.pool.apply_async(func=func, args=args, callback=self.complete)
        self.total_processes += 1.

    def complete(self, result):
        self.results.append(result)
        self.completed_processes += 1.
        if self.v:
            print('Progress: {:.2f}%'.format((self.completed_processes/self.total_processes)*100))

    def run(self):
        self.pool.close()
        self.pool.join()

    def get_results(self):
        return self.results
