#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:43:48 2020

@author: hashemghanem
"""
import numpy as np 

class graphlet_avg_features():
    '''
    Main class for graphlet (random) feature averaging.
    Instanciated with a graph_sampler S_k and a feature_map.
    with 'apply' method, this class:
        1. takes a set of graphs Gnx 
        2. sample samples_num subgraphs from each graph. 
        3. compute random features vector for each subgraph
        4. representing each graph by the average of its subgraphs' features 
        vector. 
        
    For each graph, graphlet sampling can be done by batch until samples_num 
    is reached (by default, only one batch). This is controled by batch_size 
    argument, which should 
    
    The subgraphs size is implicitly contained in sampler and feat_map 
    (of course, they should match)
    Formally, to instanciate the class you pass the following arguments:
        1. samples num: number of subgraphs to be sampled from each graph
        2. sampler: an instance of the graph sampling calss
        3. feat_map: an instance of the feature_map class
        4. batch size: how many subgraphs per each patch  
            0 < batch_size <= samples_num
    '''
    def __init__(self, samples_num, sampler, feat_map, batch_size= None, \
                 verbose=False):
        if batch_size is None:
            batch_size=samples_num
        self.num_batches=int(samples_num/batch_size)
        self.samples_num=self.num_batches*batch_size
        self.batch_size=batch_size
        self.sampler=sampler
        self.feat_map=feat_map
        self.verbose=verbose

    def calc_one_graph(self, G):
        for _ in range(self.num_batches):
            graphlets=self.sampler.sample(G, self.batch_size) # d*batch_size
            random_feature=self.feat_map.transform(graphlets) # m*batch_size
            result=random_feature.sum(axis=1) if _==0                   \
                else result + random_feature.sum(axis=1)
        return result/self.samples_num

    def apply(self, Gnx):
        for (i,G) in enumerate(Gnx):
            if self.verbose and np.mod(i,10)==0: 
                print('Graph {}/{}'.format(i,len(Gnx)))
            res=self.calc_one_graph(G)[:,None] if i==0 \
                else np.concatenate((res,self.calc_one_graph(G)[:,None]),\
                                    axis=1)
        return res