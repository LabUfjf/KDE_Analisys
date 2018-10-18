# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:30:10 2017

@author: Rafael
"""
# =============================================================================
# 
from IPython import get_ipython
#get_ipython().magic('reset -sf')
# =============================================================================

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
#from KDE_opt import KDE_opt
from kernelClean import kernelClean
from area2d import area2d
from funcoes import crossValidation
from funcoes import kernels
from funcoes import likelihood
from funcoes import roc
from scipy.interpolate import interp1d
import time

plt.close('all')
pts_size = 25
tempo = np.zeros((1,pts_size))
eventos = np.zeros((1,pts_size))
pontos = np.zeros((1,pts_size))

events = int(10e3)
mu = 1
sigma = 0.5
pts = 100
j = 0
i = 0
nroc = 10
npts = 100
#th = 0.05
#for events in np.int32(np.floor(np.linspace(100,int(5e6),pts_size))):
 #   i=i+1
   # for pts in np.int32(np.floor(np.linspace(100,1000,pts_size))):
   #     j = j+1
    #for i in range(10): 
signal = np.array(np.random.normal(0,8,[events,1]))
signal[0] = -100
signal[-1]= 100
BG = np.array(np.random.normal(2,0.4,[events,1]))
BG[0] = -100
BG[-1] = 100
signal = np.array(list(signal))
BG = np.array(list(BG))
data = np.concatenate([signal,BG])

#data = np.random.normal(mu,sigma,[events,1]) + np.random.normal(4,0.1,[events,1]) + np.random.normal(10,0.5,[events,1])
#data = np.concatenate(np.array([np.random.normal(4,0.5,[events,1]), np.random.normal(5,0.01   ,[events,1])]));
#data = np.load('data3.npy')
#data = np.random.normal(0,1,[events,1])
test = np.concatenate([np.ones(events//2),np.zeros(events//2)])

err1 = []
err2 = []
err3 = []
err4 = []
err5 = []
for i in range(nroc):
    err1.append([])
    err2.append([])
    err3.append([])
    err4.append([])
    err5.append([])
    
media1 = np.zeros(nroc)
media2 = np.zeros(nroc)
media3 = np.zeros(nroc)
media4 = np.zeros(nroc)
media5 = np.zeros(nroc)

for i in range(nroc):
    [treinoS, validacaoS] = crossValidation(signal,nroc,2,i)
    [treinoBG, validacaoBG] = crossValidation(BG,nroc,2,i)
    plt.figure(i)
    plt.hist(treinoBG,'fd',normed=True, alpha = 0.7)
    plt.hist(treinoS,'fd',normed=True, alpha = 0.7)
    
    cdf_sig = kernels(treinoS,pts,1,'cdf')
    plt.plot(cdf_sig[0],cdf_sig[1],'-o',label = 'cdf_sig')
    cdf_bg  = kernels(treinoBG,pts,1,'cdf')
    plt.plot(cdf_bg[0],cdf_bg[1],'o-', label = 'cdf_bg')
    sig,bg,DL= likelihood(cdf_sig[0],cdf_sig[1],cdf_bg[0],cdf_bg[1],validacaoS,validacaoBG)
    y, x, media1[i] = roc(test,DL)
    inter = interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    err1[i] = inter(np.linspace(0,1,npts))   
   
    lin_sig = kernels(treinoS,pts,1,'linspace')
    plt.plot(lin_sig[0],lin_sig[1],'-o',label = 'linspace_sig')
    lin_bg = kernels(treinoBG,pts,1,'linspace')
    plt.plot(lin_bg[0],lin_bg[1],'-o',label = 'linspace_bg')

    sig,bg,DL = likelihood(lin_sig[0],lin_sig[1],lin_bg[0],lin_bg[1],validacaoS,validacaoBG)
    y, x, media2[i] = roc(test,DL)
    inter = interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    err2[i] = inter(np.linspace(0,1,npts))
    
    randlin_sig = kernels(treinoS,pts,1,'randLin')
    plt.plot(randlin_sig[0],randlin_sig[1], '-o', label = 'randlin_sig')
    
    randlin_bg = kernels(treinoBG,pts,1,'randLin')
    plt.plot(randlin_bg[0],randlin_bg[1], '-o', label = 'randlin_bg')
    sig,bg,DL = likelihood(randlin_sig[0],randlin_sig[1],randlin_bg[0],randlin_bg[1],validacaoS,validacaoBG)
    y, x, media3[i] = roc(test,DL)
    inter = interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    err3[i] = inter(np.linspace(0,1,npts))
    
    odd_sig = kernels(treinoS,pts,1,'odd')
    plt.plot(odd_sig[0],odd_sig[1],'-o',label = 'odd_sig')
    
    odd_bg = kernels(treinoBG,pts,1,'odd')
    plt.plot(odd_bg[0],odd_bg[1],'-o',label = 'odd_bg')
    sig,bg,DL = likelihood(odd_sig[0],odd_sig[1],odd_bg[0],odd_bg[1],validacaoS,validacaoBG)
    y, x, media4[i] = roc(test,DL)
    inter = interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    err4[i] = inter(np.linspace(0,1,npts))
    
    rand_sig = kernels(treinoS,pts,1,'rand')
    plt.plot(rand_sig[0],rand_sig[1],'-o',label = 'rand_sig')
    
    rand_bg = kernels(treinoBG,pts,1,'rand')
    plt.plot(rand_bg[0],rand_bg[1],'-o',label = 'rand_bg')
    sig,bg,DL = likelihood(rand_sig[0],rand_sig[1],rand_bg[0],rand_bg[1],validacaoS,validacaoBG)
    y, x, media5[i] = roc(test,DL)
    inter = interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    err5[i] = inter(np.linspace(0,1,npts))
    plt.legend()

std1 = np.std(media1)/2
std2 = np.std(media2)/2
std3 = np.std(media3)/2
std4 = np.std(media4)/2
std5 = np.std(media5)/2
media1 = np.mean(media1)
media2 = np.mean(media2)
media3 = np.mean(media3)
media4 = np.mean(media4)
media5 = np.mean(media5)

plt.figure(11)
err1 = np.concatenate(err1)
err1 = np.reshape(err1,(npts,nroc),order='F')
DL1 = np.array([np.mean(err1,axis=1)])
x = np.linspace(0,1,npts)
yerr = np.std(err1, axis=1, ddof=1)/np.sqrt(nroc)
plt.plot(x,1-DL1[0],label = 'CDF (area = %.4f $\pm$ %.4f)' %(media1, std1))
plt.errorbar(x,1-DL1[0],yerr, ecolor = '#1f77b4ff', fmt = ' ')

err2 = np.concatenate(err2)
err2 = np.reshape(err2,(npts,nroc),order='F')
DL2 = np.array([np.mean(err2,axis=1)])
yerr = np.std(err2, axis=1, ddof=1)/np.sqrt(nroc)
plt.plot(x,1-DL2[0],label = 'Linspace (area = %.4f $\pm$ %.4f)' %(media2, std2))
plt.errorbar(x,1-DL2[0],yerr, ecolor = '#ff7f0eff', fmt = ' ')

err3 = np.concatenate(err3)
err3 = np.reshape(err3,(npts,nroc),order='F')
DL3 = np.array([np.mean(err3,axis=1)])
yerr = np.std(err3, axis=1, ddof=1)/np.sqrt(nroc)
plt.plot(x,1-DL3[0],label = 'randLin (area = %.4f $\pm$ %.4f)' %(media3, std3))
plt.errorbar(x,1-DL3[0],yerr, ecolor = '#2ca02cff', fmt = ' ')

err4 = np.concatenate(err4)
err4 = np.reshape(err4,(npts,nroc),order='F')
DL4 = np.array([np.mean(err4,axis=1)])
yerr = np.std(err4, axis=1, ddof=1)/np.sqrt(nroc)
plt.plot(x,1-DL4[0],label = 'odd (area = %.4f $\pm$ %.4f)' %(media4, std4))
plt.errorbar(x,1-DL4[0],yerr, ecolor = '#d62728ff', fmt = ' ')
   
err5 = np.concatenate(err5)
err5 = np.reshape(err5,(npts,nroc),order='F')
DL5 = np.array([np.mean(err5,axis=1)])
yerr = np.std(err5, axis=1, ddof=1)/np.sqrt(nroc)
plt.plot(x,1-DL5[0],label = 'rand (area = %.4f $\pm$ %.4f)' %(media5, std5))
plt.errorbar(x,1-DL5[0],yerr, ecolor = '#9467bdff', fmt = ' ')

plt.legend()
plt.show()


    #pporandLinP.append(oddP.pop(r))
    #cdfX, cdfP = KDE_opt(data,pts,f=1, KDE_type = 'both', Interp_type = 'linear');
#toc = time.time() - tic
#print(time.time() - tic)
# =============================================================================
# tempo[0,i-1] = time.time() - tic
# 
# eventos[0][i-1] = events
# pontos[0][i-1] = pts
#cdfX = np.sort(cdfX)
#cdfP = np.sort(cdfP)
#plt.plot(xpop,ppop,'rx')




# =============================================================================
        #eventos.append(events)
        #events = int(events*2)
        #ts = pts*2
    #j = 0
# =============================================================================
# pts = 100
#     
# i = 0
# =============================================================================
    #pts = 100
    #tempo.append([])
    #j = j+1
   
    #print(toc)
#plt.plot(data)
#plt.figure()
#(cdfX,y) = plt.hist(data,500, normed = True)
#tempo = np.array(tempo[0])
#media = np.mean(tempo)
#erro = media/np.sqrt(10)