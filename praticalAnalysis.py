# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:16:25 2018

@author: Igor
"""

import numpy as np
#import scipy.stats as sp
import matplotlib.pyplot as plt
import scipy.io
#import someFunctions as sf

variavel = 1;

path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
## Control Variables
variavel = 1
eta=2;
et=4;
path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
#############################


def gethist(data,i):
    #import numpy as np
    data = np.array(data[:,i]) # For converting to numpy array
    y,x = np.histogram(data,bins = 1000,normed = True)
    x = np.mean(np.array([x[:-1],x[1:]]),0)
    
    return x,y

def plothist(x,y,i,tip):
    #import matplotlib.pyplot as plt
    ## Plot things
    #orange = '#ff7f0eff'

    hist, = ax1.plot(x,y)
    ax1.set_ylabel('Normalized Histogram', fontsize = 16)
    ax1.legend([hist], [tip])
    ax1.set_xlabel('Variable ' + str(i), fontsize = 16)
    
if __name__ == '__main__':
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    datas = mat.get(path2) 
    datab = mat.get(path3) 


    #for i in range(24):
    i=1;

    x1,y1=gethist(datas,i)
    x2,y2=gethist(datab,i)
    ## Plot things
    fig, ax1 = plt.subplots(figsize=(8,6),dpi=100)
    plothist(x1,y1,i,'Signal')
    plothist(x2,y2,i,'Background')