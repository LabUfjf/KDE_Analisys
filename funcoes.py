# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:32:13 2017

@author: Rafael

In this file we have a few functions:
    - crossValidation
    - Kernels
    - Likelihood
    - Roc
    - calcDL


"""

def crossValidation(ind,numblock,aux):
    
    '''
    ind1 = eventos
    numblock = quantidade de blocos
    k = divisão dos blocos
    aux = rotação
    '''
    
    import numpy as np
#    from scipy.interpolate import interp1d
#    from scipy.interpolate import PchipInterpolator
#    from scipy.interpolate import Akima1DInterpolator
#    from scipy.interpolate import pchip_interpolate
    
    
    event = int(len(ind)/numblock)
    
    blocksort = np.array(np.roll(np.arange(0,numblock,1),(aux-1)),dtype='int')
    
    indet = []
    
    for i in blocksort[0:numblock-1]:
        indet.append(ind[event*(i):event*(i+1)])
        
    indev = []
    
    #for i in blocksort[numblock-1]:
    i = blocksort[numblock-1]
    indev.append(ind[event*(i):event*(i+1)])
        
   # return np.concatenate(indek),np.concatenate(indev)
    '''
    
    train = np.concatenate(indet)
    
    y,x = np.histogram(train, bins = 'fd', normed = True)
    
    xh = []
    for i in range(len(y)):
        xh.append((x[i]+x[i+1])/2.)
       
    x = np.array(xh)
    y = np.array(y)
    
    prob = interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = -1)
    '''
    return [np.concatenate(indet), np.concatenate(indev)]



def kernels(data,pts,f, kind2):
    from kernelClean import kernelClean
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    std = np.std(data, ddof = 1)

    #plt.close('all')
    #tic = time.time()
    LinX,LinP = kernelClean(data.T,pts,1,"linspace")
    th = LinP[np.where(LinX < 3*std)[-1][-1]]
    cdfX,cdfP = kernelClean(data.T,pts,1,"cdf")
    if kind2 == 'linspace':
       
        return LinX,LinP
    
   
    
    elif kind2 == 'cdf':
       
        return cdfX, cdfP

    
    
    elif kind2 == 'randLin':
    # =============================================================================
    # Retirada de pontos do linspace
    # =============================================================================
        LinX,LinP = kernelClean(data.T,pts,1,"linspace")
        th = LinP[np.where(LinX < 3*std)[-1][-1]]
        
        randLinX = {}
        randLinP = {}
        for i in range(pts):
            if cdfP[i] >= th:
                randLinX[len(randLinX)+1] = cdfX[i]
                randLinP[len(randLinP)+1] = cdfP[i]
            if LinP[i] < th:
                randLinX[len(randLinX)+1] = LinX[i]
                randLinP[len(randLinP)+1] = LinP[i]
             
        randLinX = list(randLinX.values())
        randLinP = list(randLinP.values())
        
        randLinP = interp1d(randLinX,randLinP)
        randLinX = list(np.sort(randLinX))
        randLinP = list(randLinP(randLinX))
        
        xpop = {}
        ppop = {}    
        
        i = 0
        while(np.size(randLinP) > pts):
            r = np.random.randint(np.size(cdfP))
            if (np.float16(randLinP[r]) < th):
                xpop[i] = randLinX.pop(r)
                ppop[i] = randLinP.pop(r)
                i = i + 1
                
        
        xpop = list(xpop.values())
        ppop = list(ppop.values())        
        
        return randLinX, randLinP
    
    elif kind2 == 'odd':
    # =============================================================================
    # Retirada dos termos ímpares
    # =============================================================================
        oddX = {}
        oddP = {}
        
        for i in range(pts*2-1):
            if i % 2:
                oddX[i] = [cdfX[i//2+1],cdfP[i//2+1]]
                oddP[i] = cdfP[i//2+1]
            else:
                oddX[i] = [LinX[i//2],LinP[i//2]]
                oddP[i] = LinP[i//2]
                
        oddX = sorted(oddX.items(), key = lambda value: value[1])
        
        xporandLinP = []
        pporandLinP = []
        k = 0
        for i in range(len(oddX)):
            r = np.random.randint(np.size(oddP))
            if i%2:
                xporandLinP.append(oddX.pop(k))
                k = k +1
        d = np.array(list(dict(oddX).values()))
        oddX = d[:,0]
        oddP = d[:,1]
        xporandLinP = np.array(list(dict(xporandLinP).values()))
        pporandLinP = xporandLinP[:,1]
        xporandLinP = xporandLinP[:,0]
        
        return oddX, oddP
    
    elif kind2 == 'rand':
    # =============================================================================
    # Retirada de pontos aleatórios
    # =============================================================================
        LinX,LinP = kernelClean(data.T,pts,1,"linspace")
        th = LinP[np.where(LinX < 3*std)[-1][-1]]
    
        randX = {}
        randP = {}
        
        for i in range(pts):
            if cdfP[i] >= th:
                randX[len(randX)+1] = cdfX[i]
                randP[len(randP)+1] = cdfP[i]
            if LinP[i] < th:
                randX[len(randX)+1] = LinX[i]
                randP[len(randP)+1] = LinP[i]
             
        randX = list(randX.values())
        randP = list(randP.values())
        
        randP = interp1d(randX,randP)
        randX = list(np.sort(randX))
        randP = list(randP(randX))
        
        xpop3 = {}
        ppop3 = {}    
        
        i = 0
        while(np.size(randP) > pts):
            r = np.random.randint(np.size(cdfP))
            xpop3[i] = randX.pop(r)
            ppop3[i] = randP.pop(r)
            i = i + 1
                
        
        xpop3 = list(xpop3.values())
        ppop3 = list(ppop3.values())
        
        return randX, randP

def likelihood(X_sig,Y_sig, X_bg, Y_bg, validation_sig_x, validation_bg_x):
    
     from scipy.interpolate import interp1d
     import numpy as np
     
    
     probs = interp1d(X_sig,Y_sig, kind = 'nearest', bounds_error = False, fill_value = 'extrapolate')
     probb = interp1d(X_bg,Y_bg, kind = 'nearest', bounds_error = False, fill_value = 'extrapolate')
     
     
     sig = probs(np.concatenate([validation_sig_x, validation_bg_x]))
# =============================================================================
#      sig = probs(np.linspace(np.min([validation_sig_x,validation_bg_x]),
#                              np.max([validation_sig_x,validation_bg_x]),
#                              len(validation_sig_x)+len(validation_bg_x)))
# =============================================================================
     
     bg = probb(np.concatenate([validation_sig_x, validation_bg_x]))
     
# =============================================================================
#      bg = probb(np.linspace(np.min([validation_sig_x,validation_bg_x])-1,
#                             np.max([validation_sig_x,validation_bg_x])+1,
#                             len(validation_sig_x)+len(validation_bg_x)))
# =============================================================================
     
# =============================================================================
#      maxbg =  np.where(bg == max(bg))[0][0]
#      
#  
#      for i in range(np.where(np.diff(np.where(bg == -1)) > 1)[1][0],-1,-1):
#          bg[i] = bg[i+1]*np.exp(-0.005)
# 
#      if np.where(bg == -1)[0][-1] == len(bg)-1:
#          for i in range(np.where(bg == -1)[0][0],len(bg),1):
#              bg[i] = bg[i-1]*np.exp(-0.005)
# 
#      if np.where(sig == -1)[0][0] == 0:
#          for i in range(np.where(np.diff(np.where(sig == -1)) > 1)[1][0],-1,-1):
#              sig[i] = sig[i+1]*np.exp(-0.005)
#     
#      if np.where(sig == -1)[0][-1] == len(sig)-1:
#          for i in range(np.where(sig == -1)[0][0],len(sig),1):
#              sig[i] = sig[i-1]*np.exp(-0.005)
# =============================================================================
     
        
     #DL = sig/(sig+bg)
     
     return sig,bg
 
def calcDL(sig,bg):
    dL = sig-bg
    return dL
     
def roc(test,pred,kind = '-'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test, pred,pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    #plt.figure()
    #plt.plot(1-fpr[1], tpr[1],kind)
    return fpr[1],tpr[1],roc_auc[1]
     