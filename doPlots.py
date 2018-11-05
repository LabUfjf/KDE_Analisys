# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:42:34 2018

@author: Igor
"""
def gethist(data,i):
    import someFunctions as sf
    import numpy as np
    data = np.array(data[:,i]) # For converting to numpy array
    x,y = sf.ash(data,m=2,tip='linear')
    ah=sf.area2d(x,y)
    y=np.true_divide(y,ah)
    #x = np.mean(np.array([x[:-1],x[1:]]),0)
    
    return x,y

def plothist(x,y,i,tip,c):
    import numpy as np
    import matplotlib.pyplot as plt
    ## Plot things
    plt.bar(x, y, width = (np.diff(x)[0]), align='center', alpha = 0.5, color = c, label=tip)
    plt.ylabel('Normalized Histogram', fontsize = 16)
    plt.xlabel('Variable ' + str(i), fontsize = 16)
    
def plotASH(datas,i,leg,color):
    
    x1,y1=gethist(datas,i)
    plothist(x1,y1,(i+1),leg, color)

def plotHistogram(sinal = True, var = 1, kind = 'LINSPACE', npts = 10, eta = 2, et = 4, cv = 1):
    import numpy as np
    #import scipy.stats as sp
    import matplotlib.pyplot as plt
    #import matplotlib.patches as mpatches
    import scipy.io
    #import someFunctions as sf
    #import funcoes as fc
    #import KDEfunctions as KDE
    #import os.path

    #path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    path='/media/atlas/Dados2/MedidasWerner/mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    ###############################################################################
    ## Control Variables
    path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    vector = np.array(([1, 3, 4, 9, 10, 14, 27, 36, 46, 50, 62, 65, 69, 73, 74, 75, 77, 79, 83, 87, 89, 91, 94, 95]),dtype='int')
    var=vector[var-1]-1
    
    kind = kind.upper();
    ###############################################################################
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    kern = scipy.io.loadmat('./savedVariables/KDE/KDEv' + str(var+1) + 'cv' + str(cv) + kind + 'pts' + str(npts) + '.mat', matlab_compatible = True)
    
    if sinal:    
        data = mat.get(path2)
        pkde = kern.get('sig')
        labelset = 'Signal'
        cl = 'Blue'
    else:
        data = mat.get(path3)
        pkde = kern.get('bkg')
        labelset = 'Background'
        cl = 'Red'
    
    fig, ax1 = plt.subplots(figsize=(8,6),dpi=100)
    #plotASH(data[:,:],var,'Signal ASH', 'Blue')
    plt.hist(data[:,var],bins = int(np.round(np.sqrt(np.size(data[:,var])))), density = True,label = ['Histogram ' + labelset], color = cl, alpha =  0.7)
    plt.plot(pkde[0][:npts],pkde[1],'-ok',label = 'KDE ' + labelset, alpha =  0.7)
    #plt.plot(cdf_bg[0][:npts],cdf_bg[1],'-or', label = labelset, alpha = 0.7)
    plt.ylabel('Probability (%)', fontsize = 16)
    plt.xlabel('Variable ' + str(var+1), fontsize = 16)
    plt.legend()
    plt.title('Method = %s - KernelPoints = %d - CV = %d' %(kind,npts,(cv)))
    fig.savefig('./Figures/DIST/dist' + labelset + str(var+1) + 'cv' + str(cv) + kind + 'pts' + str(npts) + '.png', bbox_inches='tight')
    #plt.close(fig)
    

def plotROCs(kind = 'LINSPACE', npts = 10, eta = 2, et = 4):
    
    import numpy as np
    #import scipy.stats as sp
    import matplotlib.pyplot as plt
    #import matplotlib.patches as mpatches
    import scipy.io
    import someFunctions as sf
    import funcoes as fc
    #import KDEfunctions as KDE
    #import os.path

    #path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    path='/media/atlas/Dados2/MedidasWerner/mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
    ###############################################################################
    ## Control Variables
    path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
    
    kind = kind.upper();
    ###############################################################################
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    datas = mat.get(path2) 
    datab = mat.get(path3) 
    
    datas = sf.downsamp(datas)
    datab = sf.downsamp(datab)
    
    targetv = np.concatenate([int(1e6)*np.ones(int((np.size(datas,0)/10))),np.zeros(int((np.size(datab,0)/10)))])
    
    fig2, ax1 = plt.subplots(figsize=(8,6),dpi=100)
    for i in range(10):
        dL1 = scipy.io.loadmat('./savedVariables/DL/DL' + kind + 'pts' + str(npts) + 'cv' + str(i+1) + '.mat', matlab_compatible = True)
        dL = np.transpose(dL1.get('dL'))
        
        y,x,auc = fc.roc(targetv,dL)
        
        plt.plot(x,1-y,label = ['ROC ' + str(i+1) + ' - AUC = ' + str(100*auc)])
    plt.title('Method = %s - KernelPoints = %d' %(kind,npts))
    plt.xlabel('Signal Efficiency', fontsize = 16)
    plt.ylabel('Background Rejection', fontsize = 16)
    plt.xlim(0.5, 1) 
    plt.ylim(0.5, 1) 
    plt.legend()
    fig2.savefig('./Figures/ROC/ROC' + kind + 'pts' + str(npts) + '.png', bbox_inches='tight')
    #plt.close(fig2)
    
    
    