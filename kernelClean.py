# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:15:12 2017

@author: Rafael
"""

def kernelClean(data,nPoint,f,tipo):
    '''
    ==========================================================================
     KERNEL Optimizer para problemas de 1 Dimensão:
    =================================================
     Entradas: data: dados de Kernel
               nPoint: número de pontos do kernel
               f: fator de suavização do lambda ótimo
               e: elétron=1 jato=0
               v: número das variáveis da LH
    =================================================
     Saídas:   x: range X do Kernel
               p: probabilidades do Kernel
    '''
    
    import numpy as np
    from scipy.interpolate import InterpolatedUnivariateSpline #interp1d
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    #from kernelND import kernelND
    from area2d import area2d
    from numpy.matlib import repmat
    
    
    numDiv = 10
    numDiv1 = 2
    
# =============================================================================
#     Calculo do Valor ótimo da binagem
# =============================================================================
    yhko, xhko = np.histogram(data, bins = 'fd', normed = True)   
    optN = len(xhko) - 1                # Histograma Normalizado do Kernel
    xh = []
    
   
    
    for i in range(len(yhko)):
        xh.append((xhko[i]+xhko[i+1])/2.)
       
    ''' xh.append(xhko[-1])
    xh.insert(0,xhko[0])
    yhko = list(yhko)
    yhko.insert(0,yhko[0])
    yhko.append(yhko[-1])'''
    xhko = np.array(xh)
    yhko = np.array(yhko)
    
    ind_0 = np.where(np.array(yhko)!=0)
    xhko = xhko[ind_0]
    yhko = yhko[ind_0]
    
# =============================================================================
#     Calculando Lambda Ótimo
# =============================================================================
    prob = interp1d(xhko,yhko, kind = 'linear', fill_value = 'extrapolate')#, bounds_error = True, fill_value = 0)
    
    proby = prob(data)
    
    Lambda = np.exp((np.size(data)**-1)*np.sum(np.log(proby[proby>0])))
    
# =============================================================================
#     Cálculando KERNEL
# =============================================================================
    n = np.size(data)
    nd = 1;
    
    h = ((4./(nd+2))**(1/(nd+4)))*np.std(data, ddof=1)*n**(-1/(nd+4));
    h =h*f
    
    fp_pdf,x = np.histogram(data,optN, normed = True)
    x2 = []
    for i in range(len(fp_pdf)):
        x2.append((x[i]+x[i+1])/2.)
    '''x2.append(x[-1])
    x2.insert(0,x[0])
    fp_pdf = list(fp_pdf)
    fp_pdf.insert(0,fp_pdf[0])
    fp_pdf.append(fp_pdf[-1])'''
    x = x2
   # ah = area2d(x,y)
    
    #fp_pdf = y/ah
    fpi = interp1d(x,fp_pdf,kind = 'linear', fill_value = 'extrapolate')
    fpi = fpi(data)
        
    if tipo != "linspace":
        dataSort = np.sort(data)
        cdf = np.zeros([1,np.size(dataSort)])
        
        for j in range (np.size(dataSort)):
            th = dataSort[0,j]
            m = th>=dataSort
            cdf[0,j] = np.sum(m)
        k = 0;
        
        cdf1 = cdf/max(cdf[0])
       # plt.plot(dataSort[0],cdf1[0])
        for i in range(np.size(dataSort)):
            cdf[0,i] = cdf[0,i]+2*i
            
            
        cdf = cdf/max(cdf[0])
        div = np.linspace(0,1,nPoint)
        interp = interp1d(cdf[0],dataSort[0],kind = 'cubic', fill_value = 'extrapolate')
        X = interp(div)
        #plt.plot(dataSort[0],cdf[0])
       # X2 = np.linspace(np.min(data),np.max(data),nPoint)
# =============================================================================
# =============================================================================
#         X3 = []
#         for i in range(nPoint*2): 
#             if i%2:
#                 X3.append(X1[i//2])
#             else:
#                 X3.append(X2[(i-1)//2])
#         X3 = np.sort(X3)
#         X = []
#         k = 0;
#         for i in range(np.size(X3)):
#             if i%2:
#                 X.append(X3[i//2])
#                 k=k+1
#         X = np.array(X)
# =============================================================================
# =============================================================================
    else:
        X = np.linspace(np.min(data),np.max(data),nPoint)
    
    hi = (np.abs(h*np.sqrt(np.abs(Lambda/fpi))).T)
    
    Hi = (hi**2).T
    Kn = lambda u : np.dot(((2*np.pi)**(-0.5)),np.exp((-(u**2)/2.)))
    if len(data)>=70000:
        pdf = []
        for j in range(numDiv):
            pdf.append(((1/n)*np.sum((repmat((Hi**(-0.5)),nPoint//numDiv,1)*
                                  Kn(repmat((Hi**(-0.5)),nPoint//numDiv,1)*
                                     ((repmat(X[np.int16(np.arange(((nPoint//numDiv)*(j)),(nPoint//numDiv)*(j+1)))],np.size(data),1).T)
                                     -repmat(data,nPoint//numDiv,1)))),1)).T)
    else:
        pdf = []
        for j in range(numDiv1):
            pdf.append(((1/n)*np.sum((repmat((Hi**(-0.5)),nPoint//numDiv1,1)*
                                  Kn(repmat((Hi**(-0.5)),nPoint//numDiv1,1)*
                                     ((repmat(X[np.int16(np.arange(((nPoint//numDiv1)*(j)),(nPoint//numDiv1)*(j+1)))],np.size(data),1).T)
                                     -repmat(data,nPoint//numDiv1,1)))),1)).T)
            
    return X,np.concatenate(pdf)
