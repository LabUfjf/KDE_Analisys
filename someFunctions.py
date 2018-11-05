"""
Created on Wed Oct  17 10:32:13 2018

@author: Igor

In this file we have a few functions:
    - downsamp
    - ash
    - area2d
    - bin_fd


"""


def downsamp(data):
    import numpy as np
    
    if np.size(data,0) > 200000:
        data = data[0:(200000-1),:]
    return data

def ash(data,m=10,tip='nearest',normed=False):

    """
    #ASH  AVERAGE SHIFT HISTOGRAM.
     
    [X,Y] = ASH(DATA) bins the elements of Y into FD equally spaced
    containers with 10 SHIFT HISTOGRAMS and returns the mean number
    of elements in each container.
    
    [X,Y] = ASH(DATA,M), where M is a scalar, uses M SHIFT HISTOGRAMS.
    
    [X,Y] = ASH(DATA,X,TYPE), where TYPE is a defined, the interpolation
    has made with the definition. OBS. the default value is 'NEAREST'.
    
    ASH(...) without output arguments produces a histogram bar plot of
    the results.
    
    Class support for inputs Y, X: 
       float: double, single
    
     Parse possible Axes input
    
    """

    import numpy as np
    from scipy.interpolate import interp1d
    
    if np.size(data) > 100000 :
        binning = int(np.round(bin_fd(data)))
    else:
        binning=int(np.round(np.sqrt(np.size(data))))
            
    
    if binning < 10 :
        binning=int(np.round(np.sqrt(np.size(data))))
    
    h=(np.max(data)-np.min(data))/binning;
    
    ## AVERAGE SHIFT HISTOGRAM
    
    
    t0=np.linspace(0,(h-(h/m)),m)
    
    fa = np.zeros((m,binning-1),dtype=float)
    xa = np.zeros((m,binning-1),dtype=float)
    
    
    for j in range(m):
        gridx=np.arange((np.min(data)+t0[j]-h/2),(np.max(data)+t0[j]-h/2),h)
        for i in range(binning-1):
            fa[j,i]=np.size(np.where(np.logical_and(data>=gridx[i], data<=gridx[i+1])))
            xa[j,i]=(gridx[i]+gridx[i+1])/2;
    
    newgrid = np.sort(np.reshape(xa,(np.size(xa),1))[:,0])
    
    ##PAREI AQUI
    
    # probs = interp1d(X_sig,Y_sig, kind = 'nearest', bounds_error = False, fill_value = 'extrapolate')
    
    fsh = np.empty((m,),dtype=object)
    fshd = np.zeros((m,np.size(newgrid)),dtype=float)
    for k in range(m):
        fsh[k] = interp1d(xa[k,:],fa[k,:], kind = tip, bounds_error = False, fill_value = 0)
        fshd[k] = fsh[k](newgrid)
        #fsh(k,:)=interp1(xa(k,:),fa(k,:),newgrid,type,0);
    
    
    fash=np.mean(fshd,0);
    
    x=newgrid
    #ind = np.argsort(newgrid[:,0])
    y=fash
    if normed == False:
        return x,y
    elif normed == True:
        ah=area2d(x,y)
        y=np.true_divide(y,ah)
        return x,y
    



def area2d(x,y):
    import numpy as np
    
    tbin = min(np.diff(x))
    area = np.sum(np.abs(y))*np.abs(tbin);
    
    return area

def bin_fd(x):
    """
    The Freedman-Diaconis histogram bin estimator.

    The Freedman-Diaconis rule uses interquartile range (IQR) to
    estimate binwidth. It is considered a variation of the Scott rule
    with more robustness as the IQR is less affected by outliers than
    the standard deviation. However, the IQR depends on fewer points
    than the standard deviation, so it is less accurate, especially for
    long tailed distributions.

    If the IQR is 0, this function returns 1 for the number of bins.
    Binwidth is inversely proportional to the cube root of data size
    (asymptotically optimal).

    Parameters
    ----------
    x : array_like
        Input data that is to be histogrammed, trimmed to range. May not
        be empty.

    Returns
    -------
    h : An estimate of the optimal bin width for the given data.
    """
    import numpy as np
    
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)