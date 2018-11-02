# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:42:34 2018

@author: Igor
"""
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
    
def plotASH(datas,i,leg,color):
    
    x1,y1=gethist(datas,i)
    plothist(x1,y1,(i+1),leg, color)

def plotHistogram(sinal = True, var = 1, kind = 'LINSPACE', npts = 10, eta = 2, et = 4, cv = 1):
    import numpy as np
    #import scipy.stats as sp
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import scipy.io
    import someFunctions as sf
    import funcoes as fc
    import KDEfunctions as KDE
    import os.path

    #path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    path='/media/atlas/Dados2/MedidasWerner/mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    ###############################################################################
    ## Control Variables
    path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    
    kind = kind.upper();
    ###############################################################################
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    kern = scipy.io.loadmat('./savedVariables/KDE/KDEv' + str(var) + 'cv' + str(cv) + kind + 'pts' + str(npts) + '.mat', matlab_compatible = True)
    
    if sinal:    
        data = mat.get(path2)
        pkde = kern.get('sig')
        labelset = 'KDE Signal'
    else:
        data = mat.get(path3)
        pkde = kern.get('bkg')
        labelset = 'KDE Background'
    
    fig, ax1 = plt.subplots(figsize=(8,6),dpi=100)
    plotASH(data[:,:],var,'Signal ASH', 'Blue')
    plt.plot(pkde[0][:npts],cdf_sig[1],'-ob',label = labelset, alpha = 0.7)
    #plt.plot(cdf_bg[0][:npts],cdf_bg[1],'-or', label = labelset, alpha = 0.7)
    plt.legend()
    plt.title('Method = %s - KernelPoints = %d - CV = %d' %(kind,npts,(i+1)))
    fig.savefig('./Figures/DIST/dist' + str(var) + 'cv' + str(cv) + kind + 'pts' + str(npts) + '.png', bbox_inches='tight')
    #plt.close(fig)
    

def plotROCs(kind = 'LINSPACE', npts = 10, eta = 2, et = 4):
    
    import numpy as np
    #import scipy.stats as sp
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import scipy.io
    import someFunctions as sf
    import funcoes as fc
    import KDEfunctions as KDE
    import os.path

    #path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    path='/media/atlas/Dados2/MedidasWerner/mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    ###############################################################################
    ## Control Variables
    path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    
    kind = kind.upper();
    ###############################################################################
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)