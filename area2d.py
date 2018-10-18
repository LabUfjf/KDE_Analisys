# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:41:20 2017

@author: Rafael
"""
import numpy as np

def area2d(x,y):
    tbin = min(np.diff(x))
    area = np.sum(np.abs(y))*np.abs(tbin);
    
    return area