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
import os.path

#################### What the algorithm will do
doPlot = 0  #Do and save plots
save = 1    #Save variables
like = 1    #Do the likelihood
rocPlot = 0 #Do Roc, plot and save


#path='D:\PcLab\MedidasWerner\mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
path='/media/atlas/Dados2/MedidasWerner/mc16_13TeV.second.sgn.truth.bkg.truth.offline.binned.lhmedium.calo.mat'
###############################################################################
## Control Variables
nroc = 10
pts = np.concatenate([list(range(10,250,10)),list(range(250,550,50)),list(range(600,1100,100))])
#variavel = 1
eta=2;
et=4;
path2='signalPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
path3='backgroundPatterns_etBin_' + str(et) + '_etaBin_' + str(eta)
 ##################### DISCRETIZATION METHOD #######################
# 1- 'LINSPACE'
# 2- 'CDFM'
# 3- 'PDFM'
# 4- 'IPDF1'
# 5- 'IPDF2'
kinds = ['LINSPACE', 'CDFM', 'IPDF1', 'IPDF2']
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
    
def downsamp(data):
    if np.size(datas,0) > 200000:
        data = data[1:200000,:]
    return data
    
if __name__ == '__main__':
    mat = scipy.io.loadmat(path, variable_names = [path2, path3], matlab_compatible = True)
    datas = mat.get(path2) 
    datab = mat.get(path3) 
    
    datas = downsamp(datas)
    datab = downsamp(datab)
        
    #dataall = np.concatenate([datas,datab])
    #target = np.concatenate([np.ones(np.size(datas,0)),np.zeros(np.size(datab,0))])
    
    
    # Vector Variables
    vector = np.array(([1, 3, 4, 9, 10, 14, 27, 36, 46, 50, 62, 65, 69, 73, 74, 75, 77, 79, 83, 87, 89, 91, 94, 95]),dtype='int')
    
    ###########################################################################
    
    for kind in kinds:
        
        for npts in pts:    

            if rocPlot == 1 :
                fig2, ax2 = plt.subplots(figsize=(8,6),dpi=100)
                
            for i in range(nroc):
                if os.path.isfile('./savedVariables/DL/DL' + kind + 'pts' + str(npts) + 'cv' + str(i+1) + '.mat') :
                    print ('DL já salva..próxima\n')
                else:
                
                    print('#\nMethod = %s - KernelPoints = %d - CV = %d' %(kind,npts,(i+1)))
                    [indet, indev] = fc.crossValidation(np.arange(0,np.size(datas[:,0]),1),nroc,i)
                    [indjt, indjv] = fc.crossValidation(np.arange(0,np.size(datab[:,0]),1),nroc,i)
                    
                    targett = np.concatenate([int(1e6)*np.ones(np.size(indet,0)),np.zeros(np.size(indjt,0))])
                    targetv = np.concatenate([int(1e6)*np.ones(np.size(indev,0)),np.zeros(np.size(indjv,0))])
                    
                    Ls1=np.zeros(np.size(targetv,0))
                    Lb1=np.zeros(np.size(targetv,0))
                    
                    for v in (vector-1):
                        if os.path.isfile('./savedVariables/KDE/KDEv' + str(v+1) + 'cv' + str(i+1) + kind + 'pts' + str(npts) + '.mat') :
                            print ('KDE já salvo..próxima\n')
                            kern = scipy.io.loadmat('./savedVariables/KDE/KDEv' + str(v+1) + 'cv' + str(i+1) + kind + 'pts' + str(npts) + '.mat', matlab_compatible = True)
                            cdf_sig = kern.get('sig') 
                            cdf_bg = kern.get('bkg')
                            
                        else:
                                   
                            Xs = KDE.samplingMethod(datas[indet,v], npts, kind)
                            Xb = KDE.samplingMethod(datab[indjt,v], npts, kind)
                            
                            
                            cdf_sig = np.array(KDE.kdeClean(datas[indet,v],npts,1,Xs))
                            cdf_bg = np.array(KDE.kdeClean(datab[indjt,v],npts,1,Xb))
                            
                            cdf_sig[0] = cdf_sig[0][:npts]
                            cdf_sig[1] = cdf_sig[1][:npts]
                            
                            cdf_bg[0] = cdf_bg[0][:npts]
                            cdf_bg[1] = cdf_bg[1][:npts]
                            
                            if save == 1:
                                scipy.io.savemat('./savedVariables/KDE/KDEv' + str(v+1) + 'cv' + str(i+1) + kind + 'pts' + str(npts) + '.mat',{'sig':cdf_sig, 'bkg':cdf_bg})
                            
                            
                            if doPlot == 1:
                                
                                if os.path.isfile('./Figures/DIST/dist' + str(v+1) + 'cv' + str(i+1) + kind + 'pts' + str(npts) + '.png') :
                                    print ('DIST já salva..próxima\n')
                                else:
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
                        
                        if like == 1:
                            Ls1=(Ls1+np.log10(Ls));
                            Lb1=(Lb1+np.log10(Lb));

                    
                    dL = fc.calcDL(Ls1,Lb1)
                    if save == 1:
                            scipy.io.savemat('./savedVariables/DL/DL' + kind + 'pts' + str(npts) + 'cv' + str(i+1) + '.mat',{'dL':dL})
                                        
                    if rocPlot == 1:
                        y, x,auc = fc.roc(targetv,dL)
                        
                        
                        plt.plot(x,1-y,label = ['ROC ' + str(i+1) + ' - AUC = ' + str(100*auc)])
                        plt.title('Method = %s - KernelPoints = %d' %(kind,npts))
                        plt.xlabel('Signal Efficiency', fontsize = 16)
                        plt.ylabel('Background Rejection', fontsize = 16)
                        plt.xlim(0.5, 1) 
                        plt.ylim(0.5, 1) 
                        plt.legend()
                if rocPlot == 1:
                    fig2.savefig('./Figures/ROC/ROC' + kind + 'pts' + str(npts) + '.png', bbox_inches='tight')
                    plt.close(fig2)