"""
Created on Fri Oct 26 14:57:59 2018

@author: Rafael Mascarenha
@author2: Igor Abritta

In this file we have a few functions:
    - CDFm
    - PDFm
    - iPDF1
    - iPDF2

"""


def CDFm(data,nPoint):
    import numpy as np
    from scipy.interpolate import interp1d
    from statsmodels.distributions import ECDF
    eps = 5e-5
    
    yest = np.linspace(0+eps,1-eps,nPoint)
    ecdf = ECDF(data)
    inf,sup = min(data),max(data)
    xest = np.linspace(inf,sup,int(100e3))
    yest = ecdf(xest)
    interp = interp1d(yest,xest,fill_value = 'extrapolate', kind = 'nearest')
    y = np.linspace(eps,1-eps,nPoint)
    x = interp(y)
    
    return x

def PDFm(data,nPoint):
    import numpy as np
    from scipy.interpolate import interp1d
    eps = 5e-5
    
    yest,xest = np.histogram(data,bins = 'fd',normed = True)
    xest = np.mean(np.array([xest[:-1],xest[1:]]),0)
    M = np.where(yest == max(yest))[0][0]
    m = np.where(yest == min(yest))[0][0]
    
    if M:
        interpL = interp1d(yest[:M+1],xest[:M+1], fill_value = 'extrapolate')
        interpH = interp1d(yest[M:],xest[M:])
    
        y1 = np.linspace(yest[m]+eps,yest[M],nPoint//2+1)
        x1 = interpL(y1)
        
        y2 = np.flip(y1,0)
        x2 = interpH(y2)
           
            
        x = np.concatenate([x1[:-1],x2])
        y = np.concatenate([y1[:-1],y2])
    else:
        interp = interp1d(yest,xest,fill_value='extrapolate')
        if not nPoint%2:
            nPoint = nPoint+1
        y = np.linspace(yest[M],yest[m],nPoint)
        x = interp(y)
    
    return x

def iPDF1(data,nPoint):
    import numpy as np
    from scipy.interpolate import interp1d
    from methodDisc import mediaMovel
    eps = 5e-5
    n = 5
              
    y,x = np.histogram(data,bins = 'fd',normed = True)
    x = np.mean(np.array([x[:-1],x[1:]]),0)
  
    y = abs(np.diff(mediaMovel(y,n)))
    x = x[:-1]+np.diff(x)[0]/2
    
    cdf = np.cumsum(y)    
    cdf = cdf/max(cdf)
    
    interp = interp1d(cdf,x, fill_value = 'extrapolate')
    Y = np.linspace(eps,1-eps,nPoint)
    X = interp(Y)
    
    return X

def iPDF2(data,nPoint):
    import numpy as np
    from scipy.interpolate import interp1d
    eps = 5e-5
    n = 5
          
    y,x = np.histogram(data,bins = 'fd',normed = True)
    x = np.mean(np.array([x[:-1],x[1:]]),0)
  
    y = abs(np.diff(mediaMovel(y,n),2))
    x = x[:-2]+np.diff(x)[0]
    y = y/(np.diff(x)[0]*sum(y))
    
    cdf = np.cumsum(y)    
    cdf = cdf/max(cdf)
    
    interp = interp1d(cdf,x, fill_value = 'extrapolate')
    Y = np.linspace(eps,1-eps,nPoint)
    X = interp(Y)
    
    return X

def mediaMovel(x,n):
      from numpy import mean
      for i in range(len(x)):
            if i < n//2:
                  x[i] = mean(x[:n//2])
            else:
                  x[i] = mean(x[i-n//2:i+n//2])
      return x



