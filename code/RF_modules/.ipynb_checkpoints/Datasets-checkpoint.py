#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:02:01 2020

@author: hashemghanem
"""

from grakel.datasets import fetch_dataset
from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np

class dataset_loading:
    
    '''
    This class provide uploading the following datasets in Networkx graph form:
        1. Mutag dataset
        2. Stochastic Block Model (SBM) graph generator
        3. D&D dataset
        4. Reddit-B dataset. 
    For each dataset you pass the test_size as one of the arguments. 
    Based on that, the output is as follow:
        (training graphs, training labels), (testing graphs, testing labels)
    '''
    def __init__(self):
        pass 
    #mutag dataset
    def Mutag(self,test_size=0.1):
        Gnx_train=[];
        Gnx_test=[];
        MUTAG = fetch_dataset("MUTAG", verbose=True, as_graphs=False)
        G, y = MUTAG.data, MUTAG.target
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size= test_size)
        for i in range(len(G_train)):
            g_current=nx.Graph(list(G_train[i][2]));
            g_current.add_nodes_from(G_train[i][1])
            Gnx_train.append(g_current)
        for i in range(len(G_test)):
            g_current=nx.Graph(list(G_test[i][2]));
            g_current.add_nodes_from(G_test[i][1])
            Gnx_test.append(g_current)
        return (Gnx_train,y_train), (Gnx_test,y_test)

    #SBM generator
    def generate_SBM(self,Graphs_num=300, nodes_per_graph=60,block_size=10,\
                     fraction=0.3, r=1.2, avg_deg=10, test_size=0.2):
        blocks_num=int(nodes_per_graph/block_size)
        sizes=[block_size]*blocks_num
        G,y=[],[]
        for i in range (Graphs_num):                  
            p_in=fraction  if i <Graphs_num/2 else fraction*r
            p_out=(avg_deg-(block_size-1)*p_in)/(nodes_per_graph-block_size)
            p=p_out*np.ones([blocks_num]*2)+(p_in-p_out)*np.eye(blocks_num)
            #print(p_in,p_out)
            G.append(nx.stochastic_block_model(sizes, p))
            y.append(-1 if i<Graphs_num/2 else 1)            
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=test_size)
        return (G_train,y_train),(G_test,y_test)
        

    # DD dataset
    def DD(self, test_size=0.1):
        DD = fetch_dataset("DD", verbose=True)
        G, y = DD.data, DD.target
        Gnx_train=[];
        Gnx_test=[];          
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=test_size)
        for i in range(len(G_train)):
            g_current=nx.Graph(list(G_train[i][0]));
            g_current.add_nodes_from(G_train[i][1])
            Gnx_train.append(g_current)
        for i in range(len(G_test)):
            g_current=nx.Graph(list(G_test[i][0]));
            g_current.add_nodes_from(G_test[i][1])
            Gnx_test.append(g_current)
        return (Gnx_train,y_train), (Gnx_test,y_test)

    def Reddit_B(self,test_size=0.1): 
        Reddit_B= fetch_dataset("REDDIT-BINARY", verbose=True)
        G, y = Reddit_B.data, Reddit_B.target
        Gnx_train=[];
        Gnx_test=[];           
        G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=test_size)
        for i in range(len(G_train)):
            g_current=nx.Graph(list(G_train[i][0]));
            g_current.add_nodes_from(G_train[i][1])
            Gnx_train.append(g_current)
        for i in range(len(G_test)):
            g_current=nx.Graph(list(G_test[i][0]));
            g_current.add_nodes_from(G_test[i][1])
            Gnx_test.append(g_current)
        return (Gnx_train,y_train), (Gnx_test,y_test)