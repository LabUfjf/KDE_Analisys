# -*- coding: utf-8 -*-


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



