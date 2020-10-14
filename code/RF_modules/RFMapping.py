#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:10:11 2020

@author: hashemghanem
"""
from abc import ABC, abstractmethod
import numpy as np 


# LightOn related packages 
# note that you should have access to LightOn company's servers
# This can be done by signing up and paying for the necessary credits 

import warnings                                    
warnings.filterwarnings('ignore')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))          
from lightonml.projections.sklearn import OPUMap




class feature_map(ABC):
    '''
    Abstract class for (random) feature mappings.
    Ensure that the transform method is implemented.
    '''
    def __init__(self, input_dim, features_num):
        self.input_dim=input_dim
        self.output_dim=features_num
    
    @abstractmethod
    def transform(self, A):
        '''
        In: A (input_dim * batch_size)
        Out: B (output_dim * batch_size)
        '''
        pass

class Gaussian_random_features(feature_map):
    '''
    This class computes Gaussian random features. 
    When initializing a new instance, you should pass: 
    sigma: STD of the Gaussian kernel
    input_dim, features_num: size of projection matrix
    '''
    def __init__(self, input_dim, features_num, sigma):
        self.proj_mat=sigma*np.random.randn(features_num,input_dim) 
        self.features_num=features_num

    def transform(self, A):
        temp = self.proj_mat.dot(A)
        return np.concatenate((np.cos(temp),np.sin(temp)))

    

class Lighton_random_features(feature_map):
    '''
    This class computes optical random features with 
    the help of OPUs technology developed by LightOn company. 
    When initializing a new instance, you should pass: 
    input_dim, features_num: size of projection matrix
    '''
    def __init__(self, input_dim, features_num):
        self.features_num=features_num
        self.random_mapping = OPUMap(n_components=features_num)
        self.random_mapping.opu.open()
    def transform(self, A):
        A=np.uint8(A.T)
        train_random_features = self.random_mapping.transform(A)
        return train_random_features.astype('float32').T
    def close(self):
         self.random_mapping.opu.close()