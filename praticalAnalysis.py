# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 12:16:25 2018

@author: Igor
"""

import numpy as np
#import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import someFunctions as sf
import funcoes as fc

path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
###############################################################################
## Control Variables
nroc = 10
npts = 100
doPlot = 0
variavel = 1
eta=2;
et=4;
path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
###############################################################################


def gethist(data,i):
    #import numpy as np
    data = np.array(data[:,i]) # For converting to numpy array
    x,y = sf.ash(data,m=10,tip='linear')
    ah=sf.area2d(x,y)
    y=np.true_divide(y,ah)
    #x = np.mean(np.array([x[:-1],x[1:]]),0)
    
    return x,y

def plothist(x,y,i,tip,c):
    #import matplotlib.pyplot as plt
    ## Plot things
    plt.bar(x, y, width = (np.diff(x)[0]), align='center', alpha = 0.5, color = c, label=tip)
    ax1.set_ylabel('Normalized Histogram', fontsize = 16)
    ax1.set_xlabel('Variable ' + str(i), fontsize = 16)
    
if __name__ == '__main__':
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    datas = mat.get(path2) 
    datab = mat.get(path3) 
    
    dataall = np.concatenate([datas,datab])
    target = np.concatenate([np.ones(np.size(datas,0)),np.zeros(np.size(datab,0))])
    
    
    # Vector Variables
    vector = np.array(([1, 3, 4, 9, 10, 14, 27, 36, 46, 50, 62, 65, 69, 73, 74, 75, 77, 79, 83, 87, 89, 91, 94, 95]),dtype='int')
    
    ## Plot the ring distribuition
    if doPlot == 1:
        
        for i in (vector-1):
        #i=95;
            x1,y1=gethist(datas,i)
            x2,y2=gethist(datab,i)
            ## Plot things
            fig, ax1 = plt.subplots(figsize=(8,6),dpi=100)
            plothist(x1,y1,(i+1),'Signal','Blue')
            plothist(x2,y2,(i+1),'Background', 'Red')
            plt.legend()
    ###########################################################################
    
    for v in (vector-1):
    
        for i in range(nroc):
            [indet, indev] = fc.crossValidation(np.arange(0,np.size(datas[:,v]),1),nroc,i)
            [indjt, indjv] = fc.crossValidation(np.arange(0,np.size(datab[:,v]),1),nroc,i)
            
            cdf_sig = fc.kernels(datas[indet,v],npts,1,'cdf')
            plt.plot(cdf_sig[0],cdf_sig[1],'-o',label = 'cdf_sig')
            