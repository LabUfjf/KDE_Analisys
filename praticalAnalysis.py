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
import KDEfunctions as KDE

#path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
path='/media/atlas/Dados2/MedidasWerner/mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
###############################################################################
## Control Variables
nroc = 10
npts = 20
doPlot = 0
#variavel = 1
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
    
def plotASH(datas,i,leg,color):
    
    x1,y1=gethist(datas,i)
    plothist(x1,y1,(i+1),leg, color)
    
if __name__ == '__main__':
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    datas = mat.get(path2) 
    datab = mat.get(path3) 
    
    #dataall = np.concatenate([datas,datab])
    #target = np.concatenate([np.ones(np.size(datas,0)),np.zeros(np.size(datab,0))])
    
    
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
    
    for i in range(nroc):
    
        [indet, indev] = fc.crossValidation(np.arange(0,np.size(datas[:,0]),1),nroc,i)
        [indjt, indjv] = fc.crossValidation(np.arange(0,np.size(datab[:,0]),1),nroc,i)
        
        targett = np.concatenate([int(1e6)*np.ones(np.size(indet,0)),np.zeros(np.size(indjt,0))])
        targetv = np.concatenate([int(1e6)*np.ones(np.size(indev,0)),np.zeros(np.size(indjv,0))])
        
        Ls1=np.zeros(np.size(targetv,0))
        Lb1=np.zeros(np.size(targetv,0))
        
        for v in (vector-1):                                    
            kind = 'ipdf2'
            
            Xs = KDE.samplingMethod(datas[indet,v], npts, kind)
            Xb = KDE.samplingMethod(datab[indjt,v], npts, kind)
            
            
            cdf_sig = np.array(KDE.kdeClean(datas[indet,v],npts,1,Xs))
            cdf_bg = np.array(KDE.kdeClean(datab[indjt,v],npts,1,Xb))
            
            cdf_sig[0] = cdf_sig[0][:npts]
            cdf_sig[1] = cdf_sig[1][:npts]
            
            cdf_bg[0] = cdf_bg[0][:npts]
            cdf_bg[1] = cdf_bg[1][:npts]
            
            if doPlot == 1:
                fig, ax1 = plt.subplots(figsize=(8,6),dpi=100)
                plotASH(datas[indet,:],v,'Signal ASH', 'Blue')
                plotASH(datab[indjt,:],v,'Backgorund ASH', 'Red')
                plt.plot(cdf_sig[0][:npts],cdf_sig[1],'-ob',label = 'KDE Signal', alpha = 0.7)
                plt.plot(cdf_bg[0][:npts],cdf_bg[1],'-or', label = 'KDE Background', alpha = 0.7)
                plt.legend()
                plt.title('Method = %s - KernelPoints = %d - CV = %d' %(kind,npts,(i+1)))
                fig.savefig('./Figures/DIST/dist' + str(v+1) + 'cv' + str(i+1) + kind + 'pts' + str(npts) + '.png', bbox_inches='tight')
                plt.close(fig)
                
            Ls,Lb = fc.likelihood(cdf_sig[0],cdf_sig[1],cdf_bg[0],cdf_bg[1],datas[indev,v],datab[indjv,v])
            
            Ls1=(Ls1+np.log10(Ls));
            Lb1=(Lb1+np.log10(Lb));
            
        dL = fc.calcDL(Ls1,Lb1)
        y, x,auc = fc.roc(targetv,dL)
        
        plt.plot(x,1-y,label = ['ROC ' + str(i+1)])
        plt.title('Method = %s - KernelPoints = %d - CV = %d' %(kind,npts,(i+1)))
        plt.xlabel('Signal Efficiency', fontsize = 16)
        plt.ylabel('Background Rejection', fontsize = 16)
        plt.legend()